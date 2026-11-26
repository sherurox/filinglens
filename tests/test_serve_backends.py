"""Backends on a fake transport and a fake clock: no network, no real waiting."""

import asyncio

import pytest
from fakes import FakeClock

from filinglens.extract.silver_label import schema as risk_schema
from filinglens.ingest.rate_limit import max_in_window
from filinglens.serve.backends import (
    DailyQuota,
    GeminiBackend,
    HttpResult,
    OpenAICompatBackend,
    TransportError,
    gemini_body,
    gemini_schema,
    parse_gemini,
    retry_delay,
    to_gemini,
)

CONV = [
    {"role": "system", "content": "rules"},
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "q2"},
]
KEY = "test-key-not-real"


def ok_gemini(text="{}", prompt=100, answer=20, thoughts=None) -> HttpResult:
    usage = {"promptTokenCount": prompt, "candidatesTokenCount": answer}
    if thoughts is not None:
        usage["thoughtsTokenCount"] = thoughts
    return HttpResult(
        200,
        {
            "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
            "usageMetadata": usage,
        },
    )


def too_many(delay="7s") -> HttpResult:
    detail = {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}
    return HttpResult(429, {"error": {"code": 429, "message": "quota", "details": [detail]}})


class FakeTransport:
    """Returns the queued results in order (repeating the last) and records every call."""

    def __init__(self, clock: FakeClock, results: list, seconds: float = 0.5) -> None:
        self.clock, self.results, self.seconds = clock, list(results), seconds
        self.calls: list[dict] = []

    async def post(self, url, headers, body, timeout):
        self.calls.append({"t": self.clock.now(), "url": url, "headers": headers, "body": body})
        await self.clock.sleep(self.seconds)  # the call "takes" this long
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(result, Exception):
            raise result
        return result


def gemini(tmp_path, results, rpm=15, rpd=100, clock=None):
    clock = clock or FakeClock()
    transport = FakeTransport(clock, results)
    quota = DailyQuota(rpd, tmp_path / "quota.json", today=lambda: "2026-11-27")
    backend = GeminiBackend("m", KEY, rpm, quota, transport=transport, clock=clock)
    return backend, transport, quota


def test_to_gemini_moves_system_and_renames_assistant():
    instruction, contents = to_gemini(CONV)
    assert instruction == {"parts": [{"text": "rules"}]}
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[1]["parts"] == [{"text": "a1"}]
    assert to_gemini([{"role": "user", "content": "x"}])[0] is None


def test_gemini_schema_types_enum_nodes_and_keeps_the_input():
    original = risk_schema()
    out = gemini_schema(original)
    props = out["properties"]
    assert props["severity"] == {"enum": [1, 2, 3, None], "type": ["integer", "null"]}
    assert props["category"]["type"] == "string"
    assert props["rationale"] == {"type": "string"}  # typed nodes are unchanged
    assert out["required"] == original["required"]
    assert "type" not in original["properties"]["severity"]  # input not mutated


def test_gemini_body_settings():
    body = gemini_body(CONV, {"type": "object"}, 400, "minimal")
    config = body["generationConfig"]
    assert config["maxOutputTokens"] == 400 and config["seed"] == 0
    assert config["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert "temperature" not in config  # Gemini 3: Google's default, on purpose
    assert config["responseFormat"]["text"]["mimeType"] == "APPLICATION_JSON"
    assert "responseFormat" not in gemini_body(CONV, None, 80, "minimal")["generationConfig"]


def test_parse_gemini_skips_thoughts_and_counts_thinking():
    body = {
        "candidates": [
            {
                "content": {"parts": [{"text": "plan", "thought": True}, {"text": "answer"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 9,
            "candidatesTokenCount": 3,
            "thoughtsTokenCount": 5,
        },
    }
    assert parse_gemini(body) == ("answer", 9, 3, 5, "STOP")
    blocked = {"promptFeedback": {"blockReason": "SAFETY"}}
    assert parse_gemini(blocked) == ("", 0, 0, 0, "SAFETY")


def test_retry_delay():
    assert retry_delay(too_many("12s").body) == 12.0
    assert retry_delay(too_many("0.5s").body) == 0.5
    assert retry_delay({"error": {"details": []}}) is None
    assert retry_delay({}) is None


def test_gemini_reply_and_key_only_in_header(tmp_path):
    backend, transport, quota = gemini(tmp_path, [ok_gemini('{"a": 1}', thoughts=4)])
    reply = asyncio.run(backend.chat(CONV, {"type": "object"}))
    assert (reply.text, reply.prompt_tokens, reply.output_tokens) == ('{"a": 1}', 100, 20)
    assert reply.thinking_tokens == 4 and reply.error is None
    assert reply.latency_s == pytest.approx(0.5)
    call = transport.calls[0]
    assert call["headers"]["x-goog-api-key"] == KEY
    assert KEY not in call["url"] and KEY not in repr(backend) and KEY not in repr(reply)
    assert quota.used() == 1


def test_429_is_retried_once_after_the_server_delay(tmp_path):
    backend, transport, quota = gemini(tmp_path, [too_many("7s"), ok_gemini()])
    reply = asyncio.run(backend.chat(CONV))
    assert reply.error is None and len(transport.calls) == 2
    assert backend.clock.sleeps.count(7.0) == 1
    assert quota.used() == 2  # both attempts count against the daily cap


def test_second_429_fails_the_row(tmp_path):
    backend, transport, _ = gemini(tmp_path, [too_many(), too_many()])
    reply = asyncio.run(backend.chat(CONV))
    assert reply.error.startswith("HTTP 429") and len(transport.calls) == 2


def test_rpm_is_never_exceeded(tmp_path):
    clock = FakeClock()
    backend, transport, _ = gemini(tmp_path, [ok_gemini()], rpm=15, rpd=1000, clock=clock)
    transport.seconds = 0.0  # instant calls: the limiter alone sets the pace

    async def run():
        for _ in range(60):
            await backend.chat(CONV)

    asyncio.run(run())
    times = [c["t"] for c in transport.calls]
    assert len(times) == 60
    assert max_in_window(times, 60.0) <= 15


def test_daily_quota_stops_sending_and_resets_next_day(tmp_path):
    day = {"d": "2026-11-27"}
    path = tmp_path / "quota.json"
    quota = DailyQuota(2, path, today=lambda: day["d"])
    clock = FakeClock()
    transport = FakeTransport(clock, [ok_gemini()])
    backend = GeminiBackend("m", KEY, 15, quota, transport=transport, clock=clock)
    replies = [asyncio.run(backend.chat(CONV)) for _ in range(3)]
    assert [r.error is None for r in replies] == [True, True, False]
    assert replies[2].error.startswith("quota:") and len(transport.calls) == 2
    assert DailyQuota(2, path, today=lambda: day["d"]).used() == 2  # shared through the file
    day["d"] = "2026-11-28"
    assert quota.used() == 0 and asyncio.run(backend.chat(CONV)).error is None


def test_transport_error_is_a_failed_reply(tmp_path):
    backend, _, _ = gemini(tmp_path, [TransportError("ConnectTimeout")])
    reply = asyncio.run(backend.chat(CONV))
    assert reply.error == "transport: ConnectTimeout" and reply.text == ""


def test_openai_backend_body_and_reply():
    clock = FakeClock()
    body = {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 2},
    }
    transport = FakeTransport(clock, [HttpResult(200, body)])
    backend = OpenAICompatBackend("vllm", "http://x:1/", "gptq", transport=transport, clock=clock)
    reply = asyncio.run(backend.chat(CONV, {"type": "object"}, max_tokens=80))
    sent = transport.calls[0]
    assert sent["url"] == "http://x:1/v1/chat/completions"
    assert sent["body"]["temperature"] == 0.0 and sent["body"]["max_tokens"] == 80
    assert sent["body"]["response_format"]["json_schema"]["name"] == "answer"
    assert sent["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert (reply.text, reply.prompt_tokens, reply.output_tokens, reply.thinking_tokens) == (
        "hi",
        12,
        2,
        0,
    )


def test_openai_backend_http_error_is_a_failed_reply():
    clock = FakeClock()
    transport = FakeTransport(clock, [HttpResult(500, {"error": {"message": "boom"}})])
    backend = OpenAICompatBackend("gguf", "http://x:1", "gguf", transport=transport, clock=clock)
    reply = asyncio.run(backend.chat(CONV))
    assert reply.error == "HTTP 500: boom" and reply.text == ""
