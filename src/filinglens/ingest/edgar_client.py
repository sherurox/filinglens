"""Async SEC EDGAR client: rate-limited, retries 429/5xx with exponential backoff and jitter."""

import asyncio
import os
import random
from typing import Any, Self

import httpx
from dotenv import load_dotenv

from filinglens.ingest.rate_limit import Clock, MonotonicClock, TokenBucket

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
TEMPLATE_EMAIL = "your.email@example.com"


def load_user_agent() -> str:
    """Read SEC_USER_AGENT from the environment or .env. SEC requires a name and an email."""
    load_dotenv()
    user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not user_agent or "@" not in user_agent or TEMPLATE_EMAIL in user_agent:
        raise RuntimeError('Set SEC_USER_AGENT in .env to "FilingLens <your name> <your email>"')
    return user_agent


class EdgarClient:
    """One shared HTTP connection pool and one shared rate limiter for all EDGAR requests."""

    def __init__(
        self,
        user_agent: str,
        *,
        bucket: TokenBucket | None = None,
        clock: Clock | None = None,
        max_attempts: int = 5,
        backoff_base: float = 0.5,
        backoff_cap: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.clock = clock or MonotonicClock()
        self.bucket = bucket or TokenBucket(clock=self.clock)
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.rng = rng or random.Random()
        self._http = httpx.AsyncClient(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
            transport=transport,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    def backoff_delay(self, attempt: int, response: httpx.Response | None) -> float:
        """Seconds to wait before the next try, after failed attempt number `attempt` (1-based).

        A server's Retry-After (in seconds) wins, even above the cap: SEC can block an
        address for minutes, and retrying sooner only extends the block. Otherwise use
        full jitter: a random wait in [0, min(cap, base * 2^(attempt - 1))].
        """
        if response is not None:
            retry_after = response.headers.get("Retry-After", "")
            if retry_after.isdigit():
                return float(retry_after)
        return self.rng.uniform(0, min(self.backoff_cap, self.backoff_base * 2 ** (attempt - 1)))

    async def get(self, url: str) -> httpx.Response:
        """GET with retries. Every attempt, retries included, takes a rate-limit token."""
        for attempt in range(1, self.max_attempts + 1):
            await self.bucket.acquire()
            response: httpx.Response | None = None
            try:
                response = await self._http.get(url)
            except httpx.TransportError:
                if attempt == self.max_attempts:
                    raise
            else:
                if response.status_code not in RETRY_STATUSES:
                    return response.raise_for_status()  # 2xx/3xx returns; other 4xx raises
                if attempt == self.max_attempts:
                    response.raise_for_status()
            await self.clock.sleep(self.backoff_delay(attempt, response))
        raise AssertionError("unreachable: the loop either returns or raises")

    async def get_json(self, url: str) -> Any:
        return (await self.get(url)).json()

    async def company_tickers(self) -> dict[str, Any]:
        """{"0": {"cik_str": ..., "ticker": ..., "title": ...}, ...} for every listed company."""
        return await self.get_json(TICKERS_URL)

    async def submissions(self, cik: int) -> dict[str, Any]:
        """Company metadata (name, SIC code, ...) plus its recent filings."""
        return await self.get_json(SUBMISSIONS_URL.format(cik=cik))


async def _smoke() -> None:
    """Two live requests to SEC: the ticker list, then the first company's submissions."""
    async with EdgarClient(load_user_agent()) as client:
        tickers = await client.company_tickers()
        first = tickers["0"]
        print(f"company_tickers.json: {len(tickers)} entries; first: {first}")
        subs = await client.submissions(int(first["cik_str"]))
        recent = subs["filings"]["recent"]
        print(f"submissions: {subs['name']}, SIC {subs.get('sic')}, {len(recent['form'])} recent")


if __name__ == "__main__":
    asyncio.run(_smoke())
