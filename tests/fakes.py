"""Test doubles shared across test files."""

import asyncio


class FakeClock:
    """Deterministic clock: time moves only when someone sleeps, so tests run instantly.

    sleep() moves time forward to the sleeper's deadline and never backwards, so a
    sleeper always wakes at or after the time it asked for (like a real sleep).
    """

    def __init__(self, start: float = 0.0) -> None:
        self.t = start
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError(f"negative sleep: {seconds}")
        deadline = self.t + seconds
        self.sleeps.append(seconds)
        await asyncio.sleep(0)  # yield to other tasks, as a real sleep would
        self.t = max(self.t, deadline)
