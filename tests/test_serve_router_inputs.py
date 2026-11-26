"""router_inputs helpers on fake backends and a temporary quota file: no network."""

import asyncio

import pytest

from filinglens.serve.backends import DailyQuota, Reply
from filinglens.serve.router_inputs import collect, per_request, reply_summary, require_budget


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
