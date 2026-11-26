"""router_inputs helpers on fake backends and a temporary quota file: no network."""

import asyncio

import pytest

from filinglens.serve.backends import DailyQuota, Reply
from filinglens.serve.router_inputs import (
    bootstrap_mean,
    build_inputs,
    check_error_rates,
    collect,
    energy_usd,
    gemini_usd,
    normalize_answer,
    per_request,
    qa_conversation,
    qa_items,
    reply_summary,
    require_budget,
    token_f1,
)


def reply(latency, prompt=100, output=20, thinking=0, error=None, finish="STOP") -> Reply:
    if error:
        return Reply("gemini", "m", "", 0, 0, 0, latency, None, error)
    return Reply("gemini", "m", "{}", prompt, output, thinking, latency, finish)


class EchoBackend:
    """Records what it was asked, in order."""

    name, model = "fake", "fake"

    def __init__(self) -> None:
        self.seen: list[tuple] = []

    async def chat(self, conv, schema=None, max_tokens=400):
        self.seen.append((conv[-1]["content"], schema, max_tokens))
        return reply(0.1)


def test_collect_keeps_order_and_settings():
    backend = EchoBackend()
    convs = [[{"role": "user", "content": str(i)}] for i in range(5)]
    replies = asyncio.run(collect(backend, convs, {"type": "object"}, 400))
    assert len(replies) == 5
    assert [s[0] for s in backend.seen] == ["0", "1", "2", "3", "4"]
    assert all(s[1] == {"type": "object"} and s[2] == 400 for s in backend.seen)


class ScriptedBackend:
    """Returns the given replies in order."""

    name, model = "fake", "fake"

    def __init__(self, replies) -> None:
        self.replies = list(replies)

    async def chat(self, conv, schema=None, max_tokens=400):
        return self.replies.pop(0)


def test_collect_stops_if_the_first_request_never_reaches_the_server():
    backend = ScriptedBackend([reply(0.0, error="transport: ConnectError"), reply(0.1)])
    convs = [[{"role": "user", "content": "x"}]] * 2
    with pytest.raises(SystemExit, match="unreachable"):
        asyncio.run(collect(backend, convs, None, 100))


def test_collect_records_later_transport_errors_and_http_errors():
    backend = ScriptedBackend(
        [reply(0.1), reply(0.0, error="transport: ReadTimeout"), reply(0.1, error="HTTP 500: x")]
    )
    replies = asyncio.run(collect(backend, [[{"role": "user", "content": "x"}]] * 3, None, 100))
    assert [r.error for r in replies] == [None, "transport: ReadTimeout", "HTTP 500: x"]


def test_collect_does_not_stop_on_a_first_http_error():
    backend = ScriptedBackend([reply(0.1, error="HTTP 400: bad"), reply(0.1)])
    replies = asyncio.run(collect(backend, [[{"role": "user", "content": "x"}]] * 2, None, 100))
    assert len(replies) == 2


def test_reply_summary_excludes_failures_from_latency_and_tokens():
    replies = [reply(1.0, thinking=2), reply(3.0, prompt=300, thinking=0), reply(9.9, error="x")]
    s = reply_summary(replies)
    assert s["requests"] == 3 and s["errors"] == 1 and s["error_examples"] == ["x"]
    assert s["latency_s"]["n"] == 2 and s["latency_s"]["p50"] == pytest.approx(2.0)
    assert s["prompt_tokens_mean"] == 200.0 and s["prompt_tokens_max"] == 300
    assert s["thinking_tokens_total"] == 2 and s["finish_reasons"] == {"STOP": 2}


def test_reply_summary_all_failed():
    s = reply_summary([reply(1.0, error="quota: full")])
    assert s["latency_s"] == {"n": 0} and s["prompt_tokens_mean"] is None


def test_require_budget(tmp_path):
    quota = DailyQuota(10, tmp_path / "q.json", today=lambda: "2026-11-26")
    for _ in range(4):
        quota.take()
    assert require_budget(quota, 6) == 6
    with pytest.raises(SystemExit):
        require_budget(quota, 7)


def test_per_request_rows():
    rows = per_request(["a", "b"], [reply(0.123456), reply(2.0, error="HTTP 500: boom")])
    assert rows[0] == {
        "para_id": "a",
        "latency_s": 0.1235,
        "prompt_tokens": 100,
        "output_tokens": 20,
        "thinking_tokens": 0,
        "finish_reason": "STOP",
        "error": None,
    }
    assert rows[1]["error"] == "HTTP 500: boom"


# ---------- short Q&A ----------


def test_normalize_answer_follows_squad():
    words = normalize_answer("The Company's  costs, an increase!")
    assert words == ["companys", "costs", "increase"]


@pytest.mark.parametrize(
    ("pred", "ref", "expected"),
    [
        ("higher borrowing costs", "higher borrowing costs", 1.0),
        ("The higher borrowing costs.", "higher borrowing costs", 1.0),  # article, punctuation
        ("borrowing costs", "higher borrowing costs", 0.8),  # p 1, r 2/3
        ("costs costs costs", "costs", 0.5),  # a repeated token counts once per match
        ("nothing in common", "higher borrowing costs", 0.0),
        ("", "higher borrowing costs", 0.0),
        ("", "", 1.0),
    ],
)
def test_token_f1(pred, ref, expected):
    assert token_f1(pred, ref) == pytest.approx(expected)


def test_bootstrap_mean_is_deterministic_and_brackets_the_mean():
    values = [0.0, 0.5, 1.0, 0.25, 0.75] * 10
    ci = bootstrap_mean(values)
    assert ci == bootstrap_mean(values)  # seeded
    assert ci[0] < 0.5 < ci[1]


def test_qa_items_joins_and_refuses_gaps():
    queries = [{"query_id": "p1:q1", "question": "Q?", "source_para_id": "p1"}]
    refs = [{"qa_id": "p1:q1", "answer": "A."}, {"qa_id": "p2:q1", "answer": "B."}]
    paras = [{"para_id": "p1", "ticker": "XYZ", "period_year": 2024, "heading": "", "text": "T"}]
    items = qa_items(queries, refs, paras)
    assert items == [{"query_id": "p1:q1", "question": "Q?", "reference": "A.", "para": paras[0]}]
    with pytest.raises(SystemExit):
        qa_items(queries, refs, [])  # paragraph missing
    with pytest.raises(SystemExit):
        qa_items(queries, [], paras)  # reference answer missing


def test_qa_conversation_gives_the_paragraph_and_question():
    item = {
        "question": "What could raise XYZ's costs?",
        "para": {"ticker": "XYZ", "period_year": 2024, "heading": None, "text": "Rates may rise."},
    }
    system, user = qa_conversation(item, "RULES")
    assert system == {"role": "system", "content": "RULES"}
    assert "Ticker: XYZ" in user["content"] and "FY2024" in user["content"]
    assert "Heading: (none)" in user["content"]
    assert user["content"].endswith("QUESTION: What could raise XYZ's costs?")


# ---------- assemble ----------


def test_energy_usd():
    # 3,600 W for 1,000 s = 3.6e6 J = 1 kWh
    assert energy_usd(3600, 1000, 0.18) == pytest.approx(0.18)
    assert energy_usd(64.1, 0.4369, 0.1817) == pytest.approx(64.1 * 0.4369 / 3.6e6 * 0.1817)


def test_gemini_usd_prices_thinking_as_output():
    price = {"usd_per_1m_input": 0.25, "usd_per_1m_output": 1.50}
    assert gemini_usd(1_000_000, 0, 0, price) == pytest.approx(0.25)
    assert gemini_usd(0, 500_000, 500_000, price) == pytest.approx(1.50)


def fake_bench_files() -> dict:
    def load(c1_mean, c1_p95, watts):
        levels = [
            {
                "concurrency": c,
                "requests_per_s": 1.0 * c,
                "requests": 64,
                "errors": 0,
                "latency_s": {"mean": c1_mean * c, "p95": c1_p95 * c},
                "gpu": {"power_w_mean": watts},
            }
            for c in (1, 4, 16)
        ]
        return {"levels": levels, "_file": "bench/load.json"}

    def fields(f1):
        return {"all_rows": {"category_macro_f1": f1, "category_macro_f1_ci95": [0.5, 0.8]}}

    def summary(mean, p95, prompt, output, prompt_max=2000):
        return {
            "requests": 182,
            "errors": 0,
            "latency_s": {"mean": mean, "p95": p95},
            "prompt_tokens_mean": prompt,
            "prompt_tokens_max": prompt_max,
            "output_tokens_mean": output,
            "thinking_tokens_mean": 0.0,
        }

    def qa(f1, s):
        return {"token_f1_mean": f1, "token_f1_ci95": [f1 - 0.03, f1 + 0.03], "summary": s}

    files = {
        "fields_vllm": fields(0.73),
        "fields_gguf": fields(0.69),
        "fields_gemini": fields(0.72),
        "run_gemini_extract": {"summary": summary(6.0, 36.0, 1741.1, 79.4, 2369)},
        "load_vllm": load(0.44, 0.53, 64.1),
        "load_gguf": load(3.35, 4.84, 25.7),
        "qa_vllm": qa(0.55, summary(0.3, 0.4, 365, 25, 663)),
        "qa_gguf": qa(0.54, summary(1.26, 1.81, 365, 25, 663)),
        "qa_gemini": qa(0.60, summary(1.0, 9.0, 360, 30, 663)),
    }
    return {k: v | {"_file": f"bench/{k}.json"} for k, v in files.items()}


def test_build_inputs_values_and_eligibility():
    out = build_inputs(fake_bench_files())
    vllm, gguf, gem = (out["backends"][b] for b in ("vllm", "gguf", "gemini"))
    assert vllm["extraction"]["accuracy"] == 0.73 and vllm["extraction"]["latency_p95_s"] == 0.53
    # local cost = measured GPU watts x mean latency x price; GGUF uses the CPU estimate
    assert vllm["extraction"]["usd_per_request"] == pytest.approx(energy_usd(64.1, 0.44, 0.1817))
    assert gguf["short_qa"]["usd_per_request"] == pytest.approx(energy_usd(75.0, 1.26, 0.1817))
    assert gguf["gpu_idle_power_w_measured"] == 25.7
    price = {"usd_per_1m_input": 0.25, "usd_per_1m_output": 1.50}
    assert gem["extraction"]["usd_per_request"] == pytest.approx(gemini_usd(1741.1, 79.4, 0, price))
    assert gem["short_qa"]["latency_p95_s"] == 9.0
    # long synthesis: only a context window >= 16,384 qualifies
    assert vllm["long_synthesis"]["eligible"] is False
    assert gguf["long_synthesis"]["eligible"] is False
    assert gem["long_synthesis"]["eligible"] is True
    assert set(vllm["throughput_extraction"]) == {"1", "4", "16"}
    tpm = out["capacity"]["gemini_tpm_check"]
    assert tpm["largest_prompt_tokens_seen"] == 2369 and tpm["within_tpm"] is True
    assert out["spend"]["actual_usd"] == 0


def test_build_inputs_records_failures():
    files = fake_bench_files()
    files["qa_gemini"]["summary"]["errors"] = 1
    out = build_inputs(files)
    qa = out["backends"]["gemini"]["short_qa"]
    assert (qa["requests"], qa["failed"], qa["error_rate"]) == (182, 1, 0.0055)
    assert out["backends"]["vllm"]["extraction"]["error_rate"] == 0.0


def test_error_rate_gate():
    files = fake_bench_files()
    files["qa_gemini"]["summary"]["errors"] = 9  # 9 / 182 = 4.9%: kept
    check_error_rates(files)
    files["qa_gemini"]["summary"]["errors"] = 10  # 5.5%: refused
    with pytest.raises(SystemExit, match="rerun"):
        check_error_rates(files)


def test_collect_prints_progress(capsys):
    backend = EchoBackend()
    asyncio.run(collect(backend, [[{"role": "user", "content": "x"}]] * 45, None, 100))
    lines = capsys.readouterr().err.splitlines()
    assert lines == ["fake: 20/45 (0 failed)", "fake: 40/45 (0 failed)", "fake: 45/45 (0 failed)"]


def test_build_inputs_requires_c1_first():
    files = fake_bench_files()
    files["load_vllm"]["levels"].reverse()
    with pytest.raises(ValueError):
        build_inputs(files)
