"""Rate-limit tests on a fake clock: no real waiting, fully deterministic."""

import asyncio

import pytest
from fakes import FakeClock

from filinglens.ingest.rate_limit import TokenBucket, max_in_window

SEC_LIMIT_PER_SECOND = 10


def grant_times(bucket: TokenBucket, clock: FakeClock, n: int) -> list[float]:
    """Start n concurrent acquirers; return the sorted fake times at which each got a token."""

    async def main() -> list[float]:
        times: list[float] = []

        async def worker() -> None:
            await bucket.acquire()
            times.append(clock.now())

        await asyncio.gather(*(worker() for _ in range(n)))
        return times

    # Real-time guard: a limiter that spins forever fails the test instead of hanging it.
    return sorted(asyncio.run(asyncio.wait_for(main(), timeout=5)))


def test_default_bucket_never_exceeds_sec_limit():
    clock = FakeClock()
    times = grant_times(TokenBucket(clock=clock), clock, n=500)
    assert len(times) == 500
    assert max_in_window(times, 1.0) <= SEC_LIMIT_PER_SECOND


def test_burst_is_capped_by_capacity_plus_rate():
    clock = FakeClock()
    times = grant_times(TokenBucket(rate=5, capacity=5, clock=clock), clock, n=200)
    assert times[:5] == [0.0] * 5  # the full bucket is spent at once
    assert times[5] > 0.0  # the 6th request has to wait
    assert max_in_window(times, 1.0) <= 5 + 5  # capacity + rate * 1 s


def test_sustained_rate_matches_configured_rate():
    clock = FakeClock()
    times = grant_times(TokenBucket(rate=8, capacity=1, clock=clock), clock, n=81)
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(10.0)  # 80 more tokens at 8 per second


@pytest.mark.parametrize(("rate", "capacity"), [(0, 1), (-1, 1), (8, 0.5)])
def test_rejects_bad_config(rate, capacity):
    with pytest.raises(ValueError):
        TokenBucket(rate=rate, capacity=capacity)


def test_max_in_window_counts_half_open_windows():
    assert max_in_window([0.0, 0.5, 1.0, 1.5], 1.0) == 2  # [0, 1) holds 0.0 and 0.5
    assert max_in_window([], 1.0) == 0
