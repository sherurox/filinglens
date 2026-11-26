"""Cost meter: tokens, latency and list-price cost per request, route and user (FL-M5 T11).

`MeteredBackend` wraps any serve.backends backend. Each chat() becomes one record, written to every
sink (Postgres table `serve.requests`, or memory in tests), and one OpenTelemetry span carrying the
same numbers. The backends and the router know nothing about metering; the API (T12) wraps them.

Cost, the same model as bench/router_inputs.json (actual spend is $0 throughout):
- Gemini: tokens x the paid-tier list price (input; answer and thinking tokens at the output price);
  a failed call reports no tokens, so it costs 0.
- local backends: power x the call's latency x the electricity price (vLLM: the GPU board power
  measured at c=1; GGUF: the 75 W CPU estimate). A failed local call still ran, so it is charged.

Span: the OpenTelemetry GenAI client inference span (semantic-conventions-genai, status
Development; checked 26 Nov 2026): name "chat {model}", kind CLIENT, gen_ai.operation.name "chat",
gen_ai.provider.name ("gcp.gemini" for the AI Studio endpoint; "vllm" / "llama.cpp" are this
project's own values, no well-known ones exist), gen_ai.request.model, gen_ai.request.max_tokens,
gen_ai.usage.input_tokens, gen_ai.usage.output_tokens (answer + thinking: the convention counts
reasoning as part of output), gen_ai.usage.reasoning.output_tokens, gen_ai.response.finish_reasons,
error.type, server.address, server.port; plus filinglens.route, filinglens.user,
filinglens.cost_usd, filinglens.cost_basis.

A sink that fails (database down) never fails the request: the error goes to stderr and is counted
in `sink_errors`; the reply is returned as usual.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from filinglens.serve.backends import Backend, Reply

SEMCONV = "semantic-conventions-genai (Development), gen_ai.client.inference, checked 2026-11-26"
PROVIDER = {"gemini": "gcp.gemini", "vllm": "vllm", "gguf": "llama.cpp"}
TRACER_NAME = "filinglens.serve.cost_meter"


@dataclass(frozen=True)
class Prices:
    gemini_usd_per_1m_input: float
    gemini_usd_per_1m_output: float
    usd_per_kwh: float
    power_w: dict[str, float]  # local backend -> watts

    @classmethod
    def from_inputs(cls, inputs: dict) -> Prices:
        """The prices and power of bench/router_inputs.json, so the meter and the LP agree."""
        price = inputs["sources"]["gemini_price"]
        return cls(
            gemini_usd_per_1m_input=price["usd_per_1m_input"],
            gemini_usd_per_1m_output=price["usd_per_1m_output"],
            usd_per_kwh=inputs["electricity_usd_per_kwh"],
            power_w={b: inputs["backends"][b]["power_w"] for b in ("vllm", "gguf")},
        )


def request_cost(reply: Reply, prices: Prices) -> tuple[float, str]:
    """(list-price USD, basis) of one reply."""
    if reply.backend == "gemini":
        usd = (
            reply.prompt_tokens * prices.gemini_usd_per_1m_input
            + (reply.output_tokens + reply.thinking_tokens) * prices.gemini_usd_per_1m_output
        ) / 1e6
        return usd, "tokens x list price"
    watts = prices.power_w[reply.backend]
    return watts * reply.latency_s / 3.6e6 * prices.usd_per_kwh, "power x latency x electricity"


# ---------- sinks ----------


class Sink(Protocol):
    def write(self, record: dict) -> None: ...


@dataclass
class MemorySink:
    records: list[dict] = field(default_factory=list)

    def write(self, record: dict) -> None:
        self.records.append(record)


DDL = """
CREATE SCHEMA IF NOT EXISTS serve;
CREATE TABLE IF NOT EXISTS serve.requests (
    id bigserial PRIMARY KEY,
    ts timestamptz NOT NULL DEFAULT now(),
    route text NOT NULL,
    user_id text NOT NULL,
    backend text NOT NULL,
    model text NOT NULL,
    prompt_tokens integer NOT NULL,
    output_tokens integer NOT NULL,
    thinking_tokens integer NOT NULL,
    latency_s double precision NOT NULL,
    cost_usd double precision NOT NULL,
    cost_basis text NOT NULL,
    finish_reason text,
    error text,
    trace_id text
);
CREATE INDEX IF NOT EXISTS requests_ts_idx ON serve.requests (ts);
COMMENT ON COLUMN serve.requests.cost_usd
    IS 'list price; actual spend is $0 (free tier, own laptop)';
"""
COLUMNS = (
    "route",
    "user_id",
    "backend",
    "model",
    "prompt_tokens",
    "output_tokens",
    "thinking_tokens",
    "latency_s",
    "cost_usd",
    "cost_basis",
    "finish_reason",
    "error",
    "trace_id",
)
INSERT = (
    f"INSERT INTO serve.requests ({', '.join(COLUMNS)}) "
    f"VALUES ({', '.join(f'%({c})s' for c in COLUMNS)})"
)


def pg_conninfo() -> str:
    """DATABASE_URL if set (Neon on Cloud Run, T13), else the local compose Postgres.

    POSTGRES_HOST / POSTGRES_PORT override 127.0.0.1:55432, e.g. from a container on the compose
    network (host filinglens-postgres, port 5432), where 127.0.0.1 is the container itself."""
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    password = os.environ.get("POSTGRES_PASSWORD")
    if not password:
        raise SystemExit("set POSTGRES_PASSWORD or DATABASE_URL (source .env)")
    host = os.environ.get("POSTGRES_HOST", "127.0.0.1")
    port = os.environ.get("POSTGRES_PORT", "55432")
    return f"host={host} port={port} dbname=filinglens user=filinglens password={password}"


class PostgresSink:
    """One row per request in serve.requests. `connect` is injectable for tests."""

    def __init__(self, connect: Callable[[], Any] | None = None) -> None:
        if connect is None:
            import psycopg

            def connect() -> Any:
                return psycopg.connect(pg_conninfo(), autocommit=True)

        self._connect = connect
        self._con: Any = None

    def _connection(self) -> Any:
        if self._con is None:
            self._con = self._connect()
            self._con.execute(DDL)
        return self._con

    def write(self, record: dict) -> None:
        self._connection().execute(INSERT, {c: record[c] for c in COLUMNS})

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None


# ---------- the wrapper ----------


def server_of(backend: Backend) -> tuple[str | None, int | None]:
    url = getattr(backend, "url", None)
    if not url:
        return None, None
    parsed = urlparse(url)
    return parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)


def span_attributes(record: dict, max_tokens: int, server: tuple) -> dict:
    attrs = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": PROVIDER.get(record["backend"], record["backend"]),
        "gen_ai.request.model": record["model"],
        "gen_ai.request.max_tokens": max_tokens,
        "gen_ai.usage.input_tokens": record["prompt_tokens"],
        "gen_ai.usage.output_tokens": record["output_tokens"] + record["thinking_tokens"],
        "gen_ai.usage.reasoning.output_tokens": record["thinking_tokens"],
        "filinglens.route": record["route"],
        "filinglens.user": record["user_id"],
        "filinglens.cost_usd": record["cost_usd"],
        "filinglens.cost_basis": record["cost_basis"],
        "filinglens.latency_s": record["latency_s"],
    }
    if record["finish_reason"]:
        attrs["gen_ai.response.finish_reasons"] = (str(record["finish_reason"]),)
    if server[0]:
        attrs["server.address"], attrs["server.port"] = server
    if record["error"]:
        attrs["error.type"] = error_type(record["error"])
    return attrs


def error_type(error: str) -> str:
    """A low-cardinality error.type from a Reply error ("HTTP 503: ..." -> "503")."""
    if error.startswith("HTTP "):
        return error[5:].split(":", 1)[0].strip()
    return error.split(":", 1)[0].strip() or "error"


class MeteredBackend:
    def __init__(
        self,
        inner: Backend,
        prices: Prices,
        sinks: list[Sink],
        tracer: trace.Tracer | None = None,
    ) -> None:
        self.inner, self.prices, self.sinks = inner, prices, sinks
        self.tracer = tracer or trace.get_tracer(TRACER_NAME)
        self.sink_errors = 0

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def model(self) -> str:
        return self.inner.model

    async def chat(
        self,
        conv: list[dict],
        schema: dict | None = None,
        max_tokens: int = 400,
        *,
        route: str,
        user: str,
    ) -> Reply:
        with self.tracer.start_as_current_span(
            f"chat {self.inner.model}", kind=SpanKind.CLIENT
        ) as span:
            reply = await self.inner.chat(conv, schema, max_tokens)
            usd, basis = request_cost(reply, self.prices)
            record = {
                "route": route,
                "user_id": user,
                "backend": reply.backend,
                "model": reply.model,
                "prompt_tokens": reply.prompt_tokens,
                "output_tokens": reply.output_tokens,
                "thinking_tokens": reply.thinking_tokens,
                "latency_s": reply.latency_s,
                "cost_usd": usd,
                "cost_basis": basis,
                "finish_reason": reply.finish_reason,
                "error": reply.error,
                "trace_id": format(span.get_span_context().trace_id, "032x"),
            }
            span.set_attributes(span_attributes(record, max_tokens, server_of(self.inner)))
            if reply.error:
                span.set_status(Status(StatusCode.ERROR, reply.error[:200]))
        for sink in self.sinks:
            try:
                sink.write(record)
            except Exception as error:  # metering must never fail a request
                self.sink_errors += 1
                print(f"cost_meter: {type(sink).__name__} failed: {error}", file=sys.stderr)
        return reply
