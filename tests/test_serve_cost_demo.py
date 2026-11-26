"""Cost demo helpers: the plan, aggregation, span check and chart; no servers, no database."""

import json

import pytest

from filinglens.serve.cost_demo import aggregate, chart, check_spans, main, plan, user_for


def test_plan_splits_questions_without_reuse():
    lanes = plan(n_extraction=3, n_qa=10, vllm_qa=4, gguf_qa=3, gemini_qa=2)
    assert lanes["vllm"] == [("extraction", 0), ("extraction", 1), ("extraction", 2)] + [
        ("short_qa", i) for i in range(4)
    ]
    assert lanes["gguf"] == [("short_qa", i) for i in (4, 5, 6)]
    assert lanes["gemini"] == [("short_qa", 7), ("short_qa", 8)]
    with pytest.raises(SystemExit):
        plan(3, 10, 6, 3, 2)  # 11 questions asked, 10 exist


def test_users_alternate_under_the_tag():
    assert [user_for("t", i) for i in range(3)] == ["t/analyst-a", "t/analyst-b", "t/analyst-a"]


def row(route, backend, cost, latency=1.0, error=None, tid="a" * 32, user="t/analyst-a"):
    return {
        "route": route,
        "user_id": user,
        "backend": backend,
        "model": "m",
        "prompt_tokens": 100,
        "output_tokens": 20,
        "thinking_tokens": 1,
        "latency_s": latency,
        "cost_usd": cost,
        "error": error,
        "trace_id": tid,
    }


def test_aggregate_counts_costs_and_keeps_failures_out_of_latency():
    rows = [
        row("short_qa", "gemini", 1e-4, latency=1.0),
        row("short_qa", "gemini", 2e-4, latency=3.0),
        row("short_qa", "gemini", 0.0, latency=99.0, error="HTTP 503: x"),
        row("extraction", "vllm", 1e-6),
    ]
    out = aggregate(rows, ("route", "backend"))
    assert [(g["route"], g["backend"]) for g in out] == [
        ("extraction", "vllm"),
        ("short_qa", "gemini"),
    ]
    gem = out[1]
    assert gem["requests"] == 3 and gem["errors"] == 1
    assert gem["latency_s"]["n"] == 2 and gem["latency_s"]["p50"] == pytest.approx(2.0)
    assert gem["cost_usd"] == pytest.approx(3e-4)
    assert gem["usd_per_1k_requests"] == pytest.approx(1000 * 3e-4 / 3)  # failures count
    assert gem["output_tokens"] == 3 * 21  # answer + thinking


def test_check_spans_matches_trace_ids(tmp_path):
    rows = [row("x", "vllm", 0, tid="ab" * 16), row("x", "vllm", 0, tid="cd" * 16)]
    spans = tmp_path / "spans.jsonl"
    spans.write_text(
        "\n".join(json.dumps({"context": {"trace_id": "0x" + t}}) for t in ("cd" * 16, "ab" * 16))
        + "\n"
    )
    assert check_spans(rows, spans) == {
        "rows": 2,
        "spans": 2,
        "same_trace_ids": True,
        "zero_trace_ids": 0,
    }
    rows.append(row("x", "vllm", 0, tid="0" * 32))
    result = check_spans(rows, spans)
    assert result["same_trace_ids"] is False and result["zero_trace_ids"] == 1


def test_chart_writes_a_png(tmp_path):
    by_route = aggregate(
        [row("extraction", "vllm", 1e-6), row("short_qa", "gemini", 1e-4)], ("route", "backend")
    )
    out = tmp_path / "c.png"
    chart(by_route, out, "t")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("tag", ["a/b", "50%", "demo_1"])
def test_tag_may_not_contain_like_wildcards(tag):
    with pytest.raises(SystemExit, match="may not contain"):
        main(["report", "--tag", tag])
