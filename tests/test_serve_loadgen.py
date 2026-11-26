import asyncio

import pytest

from filinglens.serve.gpu_monitor import parse_gpu_line, summarize
from filinglens.serve.loadgen import (
    DONE,
    closed_loop,
    content_delta,
    level_summary,
    parse_sse,
    request_metrics,
    stream_request,
)
from filinglens.serve.vllm_server import serve_argv


def test_stream_request_has_eval_decoding_and_usage():
    body = stream_request([{"role": "user", "content": "x"}], {"type": "object"}, "m")
    assert body["model"] == "m" and body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["temperature"] == 0.0 and body["max_tokens"] == 400
    assert body["response_format"]["json_schema"] == {
        "name": "risk_fields",
        "schema": {"type": "object"},
    }
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_parse_sse_lines():
    assert parse_sse('data: {"a": 1}') == {"a": 1}
    assert parse_sse("data: [DONE]") == DONE
    assert parse_sse("") is None
    assert parse_sse(": keep-alive") is None
    assert parse_sse("event: message") is None


def test_content_delta():
    assert content_delta({"choices": [{"delta": {"content": "{"}}]}) == "{"
    assert content_delta({"choices": [{"delta": {"role": "assistant"}}]}) == ""
    assert content_delta({"choices": [], "usage": {"completion_tokens": 5}}) == ""


def test_request_metrics():
    m = request_metrics(started=10.0, first=10.5, ended=12.5, tokens=41)
    assert m["ttft_s"] == 0.5 and m["latency_s"] == 2.5
    assert m["decode_tok_s"] == 20.0  # 40 tokens after the first, in 2.0 s
    assert request_metrics(0.0, None, 1.0, None)["decode_tok_s"] is None


def test_closed_loop_keeps_exactly_k_in_flight_and_finishes_n():
    state = {"in_flight": 0, "peak": 0, "seen": []}

    async def send(body):
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        state["seen"].append(body["i"])
        await asyncio.sleep(0.01)
        state["in_flight"] -= 1
        return {"ttft_s": 0.001, "latency_s": 0.01, "completion_tokens": 2, "decode_tok_s": 100.0}

    bodies = [{"i": i} for i in range(5)]
    rows, wall = asyncio.run(closed_loop(send, bodies, concurrency=4, n=10, start=3))
    assert len(rows) == 10 and state["peak"] == 4 and wall > 0
    assert sorted(state["seen"]) == sorted((3 + i) % 5 for i in range(10))


def test_closed_loop_rejects_bad_levels():
    async def send(body):
        return {}

    with pytest.raises(ValueError):
        asyncio.run(closed_loop(send, [{}], concurrency=0, n=1, start=0))


def test_level_summary_counts_errors_and_rates():
    rows = [
        {"ttft_s": 0.1, "latency_s": 1.0, "completion_tokens": 10, "decode_tok_s": 10.0},
        {"ttft_s": 0.3, "latency_s": 2.0, "completion_tokens": 30, "decode_tok_s": 20.0},
        {"error": "HTTP 500"},
    ]
    s = level_summary(rows, wall_s=4.0, concurrency=2)
    assert s["requests"] == 3 and s["errors"] == 1
    assert s["output_tok_per_s"] == 10.0 and s["requests_per_s"] == 0.5
    assert s["decode_tok_per_s_per_request_mean"] == 15.0
    assert s["ttft_s"]["n"] == 2 and s["latency_s"]["p50"] == 1.5


def test_parse_gpu_line_and_summary():
    assert parse_gpu_line("2199, 18.36, 45.30, 3") == {
        "memory.used": 2199.0,
        "power.draw": 18.36,
        "temperature.gpu": 45.3,
        "utilization.gpu": 3.0,
    }
    assert parse_gpu_line("2199, [N/A], 45, 3") is None
    assert parse_gpu_line("") is None
    s = summarize([parse_gpu_line("2000, 10, 40, 0"), parse_gpu_line("7000, 50, 60, 100")])
    assert s["memory_added_mib_peak"] == 5000.0 and s["temperature_c_max"] == 60.0
    assert s["power_w_mean"] == 30.0
    assert summarize([]) == {"samples": 0}


def test_serve_argv_fixed_flags():
    argv = serve_argv("/v/bin/vllm", "gptq-w4a16")
    assert argv[:3] == ["/v/bin/vllm", "serve", "data/interim/serve/gptq-w4a16"]
    assert argv[argv.index("--served-model-name") + 1] == "gptq-w4a16"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--max-num-seqs") + 1] == "16"
    assert "--enforce-eager" not in argv
    assert "--enforce-eager" in serve_argv("v", "b", eager=True)
