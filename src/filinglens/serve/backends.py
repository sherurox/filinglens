"""The router's three model backends behind one async interface (FL-M5 T9).

    vLLM student     OpenAICompatBackend("vllm", "http://127.0.0.1:58000", model=<build>)
    GGUF on CPU      OpenAICompatBackend("gguf", "http://127.0.0.1:58081", model="gguf")
    Gemini API       GeminiBackend.from_env(model=..., rpm=..., rpd=...)

Each takes an OpenAI-style conversation (system / user / assistant messages), an optional JSON
schema and a token budget, and returns a `Reply`: the text, token counts, the HTTP call's latency
and the finish reason. A failed call returns a Reply with `error` set instead of raising, so a
measurement run records the failure and the row scores as invalid (as serve.llama_client does).

Local decoding is the eval's: temperature 0, seed 0, thinking off. Gemini 3 decoding follows
Google's guidance instead: temperature left at its default (Google warns that lowering it can cause
looping), seed 0, thinking level "minimal" (Flash-Lite's default, stated explicitly so a default
change can't move the numbers). Gemini bills thinking tokens as output, so they are counted apart.

Free-tier limits are enforced on the client, before a request is sent:
- requests per minute: `TokenBucket(rate=(rpm - 1) / 60, capacity=1)`. A bucket lets through at
  most capacity + rate * T requests in any T seconds, so at most 1 + (rpm - 1) = rpm per minute.
- requests per day: `DailyQuota`, a counter in a small JSON file keyed by the Pacific-time date
  (Google resets RPD at midnight Pacific), so separate runs on the same day share one count.
  Every attempt counts, including retried ones, so the count can only be too high, never too low.
A 429 is retried once after the server's RetryInfo delay (at most 60 s); a second 429 fails the row.

The API key goes only into the `x-goog-api-key` header: never into a URL, a Reply or a log line.
HTTP sits behind `Transport` so tests run without a network; `HttpxTransport` is the real one.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

from filinglens.ingest.rate_limit import Clock, MonotonicClock, TokenBucket

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"
PACIFIC = ZoneInfo("America/Los_Angeles")
QUOTA_FILE = Path("data/interim/serve/gemini_quota.json")
MAX_RETRY_WAIT_S = 60.0
TIMEOUT_S = 120.0


@dataclass(frozen=True)
class Reply:
    backend: str
    model: str
    text: str
    prompt_tokens: int
    output_tokens: int  # the answer's tokens
    thinking_tokens: int  # Gemini thinking tokens (billed as output); 0 for local backends
    latency_s: float  # the HTTP call only; waiting for the rate limiter is not included
    finish_reason: str | None
    error: str | None = None


@dataclass(frozen=True)
class HttpResult:
    status: int
    body: dict


class TransportError(Exception):
    """The request never got an HTTP response (connection refused, timeout, ...)."""


class Transport(Protocol):
    async def post(
        self, url: str, headers: dict[str, str], body: dict, timeout: float
    ) -> HttpResult: ...


class HttpxTransport:
    """The real transport. httpx is imported here, so the pure helpers need no network library."""

    def __init__(self) -> None:
        import httpx

        self._httpx = httpx
        self._client = httpx.AsyncClient()

    async def post(self, url: str, headers: dict[str, str], body: dict, timeout: float):
        try:
            response = await self._client.post(url, headers=headers, json=body, timeout=timeout)
        except self._httpx.HTTPError as error:
            raise TransportError(type(error).__name__) from None  # no URL or headers in the text
        try:
            data = response.json()
        except ValueError:
            data = {"non_json_body": response.text[:300]}
        return HttpResult(response.status_code, data if isinstance(data, dict) else {"body": data})

    async def aclose(self) -> None:
        await self._client.aclose()


class Backend(Protocol):
    name: str
    model: str

    async def chat(
        self, conv: list[dict], schema: dict | None = None, max_tokens: int = 400
    ) -> Reply: ...


# ---------- OpenAI-compatible servers: vLLM and llama-server ----------


def openai_body(conv: list[dict], schema: dict | None, max_tokens: int, model: str) -> dict:
    """The eval's decoding as a chat request (`json_schema.name` is required by vLLM)."""
    body = {
        "model": model,
        "messages": conv,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema},
        }
    return body


def parse_openai(body: dict) -> tuple[str, int, int, str | None]:
    """(text, prompt tokens, completion tokens, finish reason) of a chat completion."""
    choice = (body.get("choices") or [{}])[0]
    usage = body.get("usage") or {}
    text = (choice.get("message") or {}).get("content") or ""
    return (
        text,
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
        choice.get("finish_reason"),
    )


class OpenAICompatBackend:
    def __init__(
        self,
        name: str,
        url: str,
        model: str,
        transport: Transport | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.name, self.url, self.model = name, url.rstrip("/"), model
        self.transport = transport or HttpxTransport()
        self.clock = clock or MonotonicClock()

    async def chat(
        self, conv: list[dict], schema: dict | None = None, max_tokens: int = 400
    ) -> Reply:
        body = openai_body(conv, schema, max_tokens, self.model)
        headers = {"Content-Type": "application/json"}
        started = self.clock.now()
        try:
            result = await self.transport.post(
                self.url + "/v1/chat/completions", headers, body, TIMEOUT_S
            )
        except TransportError as error:
            return self._failed(f"transport: {error}", self.clock.now() - started)
        latency = self.clock.now() - started
        if result.status != 200:
            return self._failed(f"HTTP {result.status}: {short(result.body)}", latency)
        text, prompt, completion, finish = parse_openai(result.body)
        return Reply(self.name, self.model, text, prompt, completion, 0, latency, finish)

    def _failed(self, error: str, latency: float) -> Reply:
        return Reply(self.name, self.model, "", 0, 0, 0, latency, None, error)


# ---------- Gemini API (generateContent, REST) ----------


def to_gemini(conv: list[dict]) -> tuple[dict | None, list[dict]]:
    """OpenAI-style messages -> (systemInstruction, contents). Roles: assistant -> model."""
    system = [m["content"] for m in conv if m["role"] == "system"]
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in conv
        if m["role"] != "system"
    ]
    instruction = {"parts": [{"text": "\n\n".join(system)}]} if system else None
    return instruction, contents


JSON_TYPES = {bool: "boolean", int: "integer", float: "number", str: "string", type(None): "null"}


def gemini_schema(schema: dict) -> dict:
    """The same schema with an explicit `type` on enum-only nodes ({"enum": [1, 2, None]} ->
    {"type": ["integer", "null"], "enum": [1, 2, None]}). Same constraint; Gemini's schema support
    is documented per type, so nothing is left for it to infer. The input is not changed."""
    out = dict(schema)
    if "properties" in out:
        out["properties"] = {k: gemini_schema(v) for k, v in out["properties"].items()}
    for key in ("items", "additionalProperties"):
        if isinstance(out.get(key), dict):
            out[key] = gemini_schema(out[key])
    if "enum" in out and "type" not in out:
        types = list(dict.fromkeys(JSON_TYPES[type(v)] for v in out["enum"]))
        out["type"] = types[0] if len(types) == 1 else types
    return out


def gemini_body(
    conv: list[dict], schema: dict | None, max_tokens: int, thinking_level: str, seed: int = 0
) -> dict:
    instruction, contents = to_gemini(conv)
    config: dict = {
        "maxOutputTokens": max_tokens,
        "seed": seed,
        "thinkingConfig": {"thinkingLevel": thinking_level},
    }
    if schema is not None:
        config["responseFormat"] = {
            # an enum constant (TextResponseFormat.MimeType), not a MIME string: smoke test 26 Nov
            "text": {"mimeType": "APPLICATION_JSON", "schema": gemini_schema(schema)}
        }
    body: dict = {"contents": contents, "generationConfig": config}
    if instruction:
        body["systemInstruction"] = instruction
    return body


def parse_gemini(body: dict) -> tuple[str, int, int, int, str | None]:
    """(text, prompt tokens, answer tokens, thinking tokens, finish reason). Thought parts are
    left out of the text; a blocked prompt (no candidates) reports promptFeedback.blockReason."""
    usage = body.get("usageMetadata") or {}
    candidates = body.get("candidates") or []
    if candidates:
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        finish = candidates[0].get("finishReason")
    else:
        text, finish = "", (body.get("promptFeedback") or {}).get("blockReason", "NO_CANDIDATES")
    return (
        text,
        int(usage.get("promptTokenCount") or 0),
        int(usage.get("candidatesTokenCount") or 0),
        int(usage.get("thoughtsTokenCount") or 0),
        finish,
    )


DURATION = re.compile(r"^(\d+(?:\.\d+)?)s$")


def retry_delay(body: dict) -> float | None:
    """Seconds from a 429's google.rpc.RetryInfo detail ("retryDelay": "12s"), if present."""
    for detail in (body.get("error") or {}).get("details") or []:
        if str(detail.get("@type", "")).endswith("RetryInfo"):
            match = DURATION.match(str(detail.get("retryDelay", "")))
            if match:
                return float(match.group(1))
    return None


def short(body: dict, limit: int = 200) -> str:
    message = (body.get("error") or {}).get("message") if isinstance(body, dict) else None
    return (message or json.dumps(body))[:limit]


class QuotaExhausted(RuntimeError):
    pass


def pacific_today() -> str:
    return datetime.now(PACIFIC).date().isoformat()


class DailyQuota:
    """Requests sent per Pacific-time day, kept in a JSON file shared by every run that day."""

    def __init__(
        self, limit: int, path: Path = QUOTA_FILE, today: Callable[[], str] = pacific_today
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        self.limit, self.path, self.today = limit, path, today

    def used(self) -> int:
        if not self.path.is_file():
            return 0
        return int(json.loads(self.path.read_text()).get(self.today(), 0))

    def take(self) -> None:
        """Count one request, or raise QuotaExhausted without counting it."""
        day, used = self.today(), self.used()
        if used >= self.limit:
            raise QuotaExhausted(f"{used} / {self.limit} requests already sent on {day} (PT)")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({day: used + 1}) + "\n")  # only today's count is kept


class GeminiBackend:
    name = "gemini"

    def __init__(
        self,
        model: str,
        api_key: str,
        rpm: int,
        quota: DailyQuota,
        transport: Transport | None = None,
        clock: Clock | None = None,
        thinking_level: str = "minimal",
        retries_on_429: int = 1,
    ) -> None:
        if rpm < 1:
            raise ValueError("rpm must be >= 1")
        if not api_key:
            raise ValueError("empty Gemini API key")
        self.model, self.thinking_level, self.retries = model, thinking_level, retries_on_429
        self.quota = quota
        self.clock = clock or MonotonicClock()
        self.transport = transport or HttpxTransport()
        # rate (rpm - 1) / 60 with capacity 1: at most rpm requests in any 60-second window
        self.bucket = TokenBucket(rate=max(rpm - 1, 0.5) / 60, capacity=1, clock=self.clock)
        self._headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        self.url = f"{GEMINI_URL}/models/{model}:generateContent"

    @classmethod
    def from_env(cls, model: str, rpm: int, rpd: int, **kwargs) -> GeminiBackend:
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise SystemExit("GEMINI_API_KEY is not set: `set -a; source .env; set +a`")
        return cls(model, key, rpm, DailyQuota(rpd), **kwargs)

    def __repr__(self) -> str:  # the key is never part of the representation
        return f"GeminiBackend(model={self.model!r}, thinking_level={self.thinking_level!r})"

    async def chat(
        self, conv: list[dict], schema: dict | None = None, max_tokens: int = 400
    ) -> Reply:
        body = gemini_body(conv, schema, max_tokens, self.thinking_level)
        for attempt in range(self.retries + 1):
            try:
                self.quota.take()
            except QuotaExhausted as error:
                return self._failed(f"quota: {error}", 0.0)
            await self.bucket.acquire()
            started = self.clock.now()
            try:
                result = await self.transport.post(self.url, self._headers, body, TIMEOUT_S)
            except TransportError as error:
                return self._failed(f"transport: {error}", self.clock.now() - started)
            latency = self.clock.now() - started
            if result.status == 429 and attempt < self.retries:
                wait = min(retry_delay(result.body) or MAX_RETRY_WAIT_S, MAX_RETRY_WAIT_S)
                await self.clock.sleep(wait)
                continue
            if result.status != 200:
                return self._failed(f"HTTP {result.status}: {short(result.body)}", latency)
            text, prompt, answer, thinking, finish = parse_gemini(result.body)
            return Reply(self.name, self.model, text, prompt, answer, thinking, latency, finish)
        raise AssertionError("unreachable: the last attempt always returns")

    def _failed(self, error: str, latency: float) -> Reply:
        return Reply(self.name, self.model, "", 0, 0, 0, latency, None, error)
