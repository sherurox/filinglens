"""EdgarClient tests with httpx.MockTransport and a fake clock: no network, no real waiting."""

import asyncio
import random

import httpx
import pytest
from fakes import FakeClock

from filinglens.ingest import edgar_client
from filinglens.ingest.edgar_client import EdgarClient, load_user_agent
from filinglens.ingest.rate_limit import TokenBucket

UA = "FilingLens Test test@example.org"
URL = "https://data.sec.gov/submissions/CIK0000320193.json"


class CountingBucket(TokenBucket):
    """A real TokenBucket that also counts how many tokens were taken."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__(clock=clock)
        self.acquired = 0

    async def acquire(self) -> None:
        self.acquired += 1
        await super().acquire()


def make_client(*outcomes, **kwargs):
    """Client whose fake server replies with `outcomes` in order (last one repeats).

    An outcome is a status code, a (status, headers) tuple, or an exception to raise.
    """
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        outcome = outcomes[min(len(calls), len(outcomes)) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        status, headers = outcome if isinstance(outcome, tuple) else (outcome, {})
        return httpx.Response(status, headers=headers, json={"status": status})

    clock = FakeClock()
    bucket = CountingBucket(clock)
    client = EdgarClient(
        UA,
        bucket=bucket,
        clock=clock,
        transport=httpx.MockTransport(handler),
        rng=random.Random(0),
        **kwargs,
    )
    return client, calls, clock, bucket


def fetch(client: EdgarClient) -> httpx.Response:
    async def go() -> httpx.Response:
        async with client:
            return await client.get(URL)

    return asyncio.run(go())


def test_sends_sec_user_agent():
    client, calls, _, _ = make_client(200)
    assert fetch(client).status_code == 200
    assert calls[0].headers["User-Agent"] == UA


def test_retries_429_and_honours_retry_after():
    client, calls, clock, _ = make_client((429, {"Retry-After": "2"}), 200)
    assert fetch(client).status_code == 200
    assert len(calls) == 2
    assert clock.sleeps == [2.0]  # waited exactly what the server asked; the bucket had a token


def test_retries_transport_errors():
    client, calls, _, _ = make_client(httpx.ConnectError("connection reset"), 200)
    assert fetch(client).status_code == 200
    assert len(calls) == 2


def test_gives_up_after_max_attempts_on_5xx():
    client, calls, _, bucket = make_client(503, max_attempts=4)
    with pytest.raises(httpx.HTTPStatusError):
        fetch(client)
    assert len(calls) == 4
    assert bucket.acquired == 4  # every retry paid for a rate-limit token


def test_does_not_retry_404():
    client, calls, _, _ = make_client(404)
    with pytest.raises(httpx.HTTPStatusError):
        fetch(client)
    assert len(calls) == 1


def test_backoff_is_full_jitter_and_capped():
    client, *_ = make_client(200, backoff_base=0.5, backoff_cap=30.0)
    for attempt in range(1, 12):
        delay = client.backoff_delay(attempt, None)
        assert 0.0 <= delay <= min(30.0, 0.5 * 2 ** (attempt - 1))
    asyncio.run(client.aclose())


def test_retry_after_wins_over_cap():
    client, *_ = make_client(200, backoff_cap=30.0)
    assert client.backoff_delay(1, httpx.Response(429, headers={"Retry-After": "600"})) == 600.0
    asyncio.run(client.aclose())


def test_load_user_agent_rejects_missing_or_template(monkeypatch):
    monkeypatch.setattr(edgar_client, "load_dotenv", lambda: None)  # never read the real .env
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError):
        load_user_agent()
    monkeypatch.setenv("SEC_USER_AGENT", "FilingLens Your Name your.email@example.com")
    with pytest.raises(RuntimeError):
        load_user_agent()
    monkeypatch.setenv("SEC_USER_AGENT", UA)
    assert load_user_agent() == UA
