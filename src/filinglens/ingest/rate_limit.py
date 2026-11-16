"""Token-bucket rate limiter for SEC EDGAR (fair access: at most 10 requests per second)."""

import asyncio
import time
from bisect import bisect_left
from typing import Protocol


class Clock(Protocol):
    """What the limiter needs from time. Injected so tests can use a fake clock."""

    def now(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


class MonotonicClock:
    """Real clock. Monotonic time never jumps backwards when the wall clock changes."""

    def now(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class TokenBucket:
    """Allows `rate` requests per second on average, with bursts of at most `capacity`.

    In any window of T seconds at most `capacity + rate * T` requests get through,
    so the defaults (rate=8, capacity=1) allow at most 9 per second; the SEC limit is 10.
    """

    def __init__(
        self, rate: float = 8.0, capacity: float = 1.0, clock: Clock | None = None
    ) -> None:
        if rate <= 0 or capacity < 1:
            raise ValueError("rate must be > 0 and capacity must be >= 1")
        self.rate = rate
        self.capacity = capacity
        self.clock = clock or MonotonicClock()
        self._tokens = capacity  # start full
        self._last = self.clock.now()  # time of the last refill
        self._lock = asyncio.Lock()  # one waiter at a time, first come first served

    def _refill(self) -> None:
        """Add the tokens earned since the last refill, never above capacity."""
        now = self.clock.now()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now

    async def acquire(self) -> None:
        """Wait until a token is available, then take it."""
        async with self._lock:
            self._refill()
            if self._tokens < 1.0:
                wait_time = (1.0 - self._tokens) / self.rate
                await self.clock.sleep(wait_time)
                self._refill()
            self._tokens -= 1.0


def max_in_window(times: list[float], window: float = 1.0) -> int:
    """Most events inside any half-open window [t, t + window). `times` need not be sorted."""
    ordered = sorted(times)
    return max((bisect_left(ordered, t + window) - i for i, t in enumerate(ordered)), default=0)
