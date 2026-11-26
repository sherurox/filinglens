"""Tests for the hand-written greedy baseline (serve/greedy.py)."""

import pytest

from filinglens.serve.greedy import greedy_route


def test_spills_to_the_next_cheapest_when_capacity_runs_out():
    traffic = {"extraction": 500.0}
    # "dear" is listed first on purpose, so the test also checks the cheapest-first sort.
    options = {
        "extraction": [
            {"backend": "dear", "accuracy": 0.90, "usd_per_request": 0.01},
            {"backend": "cheap", "accuracy": 0.80, "usd_per_request": 0.001},
        ]
    }
    floors = {"extraction": None}
    capacity = {"cheap": 300.0, "dear": 1000.0}
    usage = {("extraction", "cheap"): 1.0, ("extraction", "dear"): 1.0}

    result = greedy_route(traffic, options, floors, capacity, usage)

    assert result["assigned"][("extraction", "cheap")] == pytest.approx(300)
    assert result["assigned"][("extraction", "dear")] == pytest.approx(200)
    assert result["unrouted"]["extraction"] == pytest.approx(0)
    assert result["cost_usd"] == pytest.approx(300 * 0.001 + 200 * 0.01)
    # The caller's capacity dict must be left untouched.
    assert capacity == {"cheap": 300.0, "dear": 1000.0}


def test_skips_below_floor_backends_and_reports_unrouted():
    traffic = {"short_qa": 200.0}
    options = {
        "short_qa": [
            {"backend": "cheap", "accuracy": 0.50, "usd_per_request": 0.0001},
            {"backend": "good", "accuracy": 0.60, "usd_per_request": 0.01},
        ]
    }
    floors = {"short_qa": 0.55}
    capacity = {"cheap": 1_000_000.0, "good": 150.0}
    usage = {("short_qa", "cheap"): 1.0, ("short_qa", "good"): 1.0}

    result = greedy_route(traffic, options, floors, capacity, usage)

    assert ("short_qa", "cheap") not in result["assigned"]
    assert result["assigned"][("short_qa", "good")] == pytest.approx(150)
    assert result["unrouted"]["short_qa"] == pytest.approx(50)
    assert result["cost_usd"] == pytest.approx(150 * 0.01)
