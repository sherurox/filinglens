"""Cost meter: costs, sinks and spans (in-memory exporter); no database, no network."""

import asyncio
import json
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from filinglens.serve.backends import Reply
from filinglens.serve.cost_meter import (
    COLUMNS,
    DDL,
    INSERT,
    MemorySink,
    MeteredBackend,
    PostgresSink,
    Prices,
    error_type,
    pg_conninfo,
    request_cost,
)

PRICES = Prices(0.25, 1.50, 0.1817, {"vllm": 64.1, "gguf": 75.0})
CONV = [{"role": "user", "content": "x"}]


def reply(backend="gemini", prompt=1000, output=80, thinking=0, latency=2.0, error=None):
    if error:
        return Reply(backend, "m", "", 0, 0, 0, latency, None, error)
    return Reply(backend, "m", "{}", prompt, output, thinking, latency, "STOP")


class FakeBackend:
    def __init__(self, r: Reply, url: str | None = None) -> None:
        self.r, self.name, self.model = r, r.backend, r.model
        if url:
            self.url = url
        self.calls: list[tuple] = []

    async def chat(self, conv, schema=None, max_tokens=400):
        self.calls.append((conv, schema, max_tokens))
        return self.r


def tracer_and_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


def test_gemini_cost_prices_thinking_as_output():
    usd, basis = request_cost(reply(prompt=1_000_000, output=0, thinking=0), PRICES)
    assert usd == pytest.approx(0.25) and basis == "tokens x list price"
    usd, _ = request_cost(reply(prompt=0, output=600_000, thinking=400_000), PRICES)
    assert usd == pytest.approx(1.50)
    assert request_cost(reply(error="HTTP 503: busy"), PRICES)[0] == 0.0  # no tokens billed


def test_local_cost_is_power_times_latency():
    usd, basis = request_cost(reply("vllm", latency=0.4369), PRICES)
    assert usd == pytest.approx(64.1 * 0.4369 / 3.6e6 * 0.1817)  # T9's $1.4135e-6
    assert basis == "power x latency x electricity"
    # a failed local call still ran on the hardware: charged
    assert request_cost(reply("gguf", latency=1.0, error="HTTP 500: x"), PRICES)[0] > 0


def test_prices_come_from_router_inputs():
    path = Path("bench/router_inputs.json")
    if not path.is_file():
        pytest.skip("bench/router_inputs.json not present")
    prices = Prices.from_inputs(json.loads(path.read_text()))
    assert prices.gemini_usd_per_1m_input == 0.25 and prices.gemini_usd_per_1m_output == 1.50
    assert prices.usd_per_kwh == 0.1817 and prices.power_w == {"vllm": 64.1, "gguf": 75.0}


def test_metered_chat_records_span_and_row():
    tracer, exporter = tracer_and_exporter()
    sink = MemorySink()
    inner = FakeBackend(
        reply(prompt=1741, output=79, thinking=3),
        url="https://generativelanguage.googleapis.com/v1beta/models/m:generateContent",
    )
    meter = MeteredBackend(inner, PRICES, [sink], tracer)
    out = asyncio.run(meter.chat(CONV, {"type": "object"}, 400, route="extraction", user="demo"))
    assert out is inner.r and inner.calls == [(CONV, {"type": "object"}, 400)]

    (span,) = exporter.get_finished_spans()
    assert span.name == "chat m" and span.kind == SpanKind.CLIENT
    a = span.attributes
    assert a["gen_ai.operation.name"] == "chat" and a["gen_ai.provider.name"] == "gcp.gemini"
    assert a["gen_ai.usage.input_tokens"] == 1741
    assert a["gen_ai.usage.output_tokens"] == 82  # answer + thinking
    assert a["gen_ai.usage.reasoning.output_tokens"] == 3
    assert a["gen_ai.response.finish_reasons"] == ("STOP",)
    assert a["server.address"] == "generativelanguage.googleapis.com" and a["server.port"] == 443
    assert a["filinglens.route"] == "extraction" and a["filinglens.user"] == "demo"
    assert span.status.status_code != StatusCode.ERROR

    (row,) = sink.records
    assert set(row) == set(COLUMNS)
    assert row["cost_usd"] == pytest.approx((1741 * 0.25 + 82 * 1.50) / 1e6)
    assert row["cost_usd"] == a["filinglens.cost_usd"]
    assert row["trace_id"] == format(span.context.trace_id, "032x")


def test_failed_call_marks_the_span_and_is_still_recorded():
    tracer, exporter = tracer_and_exporter()
    sink = MemorySink()
    meter = MeteredBackend(
        FakeBackend(reply(error="HTTP 503: high demand")), PRICES, [sink], tracer
    )
    asyncio.run(meter.chat(CONV, route="short_qa", user="demo"))
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == "503"
    assert sink.records[0]["error"] == "HTTP 503: high demand"


class BrokenSink:
    def write(self, record):
        raise ConnectionError("database down")


def test_a_failing_sink_never_fails_the_request(capsys):
    tracer, _ = tracer_and_exporter()
    good = MemorySink()
    meter = MeteredBackend(FakeBackend(reply("vllm")), PRICES, [BrokenSink(), good], tracer)
    out = asyncio.run(meter.chat(CONV, route="extraction", user="demo"))
    assert out.error is None and meter.sink_errors == 1
    assert len(good.records) == 1  # the other sink still gets the row
    assert "database down" in capsys.readouterr().err


def test_error_type_is_low_cardinality():
    assert error_type("HTTP 429: quota") == "429"
    assert error_type("transport: ConnectTimeout") == "transport"
    assert error_type("quota: 500 / 500 requests already sent") == "quota"


class FakeConnection:
    def __init__(self) -> None:
        self.executed: list[tuple] = []
        self.closed = False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def close(self):
        self.closed = True


def test_postgres_sink_creates_the_table_once_and_inserts_parameters():
    con = FakeConnection()
    sink = PostgresSink(connect=lambda: con)
    record = {c: None for c in COLUMNS} | {"route": "extraction", "cost_usd": 1e-6}
    sink.write(record)
    sink.write(record)
    assert [sql for sql, _ in con.executed] == [DDL, INSERT, INSERT]
    assert con.executed[1][1]["route"] == "extraction"  # values go as parameters, never in SQL
    assert "%(route)s" in INSERT and "extraction" not in INSERT
    sink.close()
    assert con.closed


def test_pg_conninfo_prefers_database_url_then_host_and_port(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@neon.example/db?sslmode=require")
    assert pg_conninfo() == "postgresql://u:p@neon.example/db?sslmode=require"
    monkeypatch.delenv("DATABASE_URL")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pw")
    monkeypatch.delenv("POSTGRES_HOST", raising=False)
    monkeypatch.delenv("POSTGRES_PORT", raising=False)
    assert pg_conninfo().startswith("host=127.0.0.1 port=55432 ")
    monkeypatch.setenv("POSTGRES_HOST", "filinglens-postgres")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    assert pg_conninfo().startswith("host=filinglens-postgres port=5432 ")
    monkeypatch.delenv("POSTGRES_PASSWORD")
    with pytest.raises(SystemExit):
        pg_conninfo()


def test_pg_conninfo_refuses_an_empty_database_url(monkeypatch):
    # DATABASE_URL="$NEON_DATABASE_URL" with the variable unset must not fall back to local
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pw")
    with pytest.raises(SystemExit, match="set but empty"):
        pg_conninfo()
