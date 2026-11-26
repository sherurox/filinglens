"""LP router tests: brute force, the hand-written greedy, blending, infeasibility, the rule."""

import hashlib
import itertools
import json
import random

import numpy as np
import pytest

from filinglens.serve.router import (
    Option,
    Problem,
    check_assignment,
    diagnose_infeasible,
    eligible_options,
    inequality_rows,
    load_rule_and_inputs,
    rule_floors,
    run_greedy,
    solve_lp,
)


def simplex_grid(n: int, step: float = 0.1) -> np.ndarray:
    """All share vectors of length n in multiples of `step` that sum to 1."""
    k = int(round(1 / step))
    points = [c for c in itertools.product(range(k + 1), repeat=n) if sum(c) == k]
    return np.array(points, dtype=float) / k


def brute_force(p: Problem) -> float | None:
    """Cheapest grid point that meets every LP row, or None if no grid point does."""
    classes = list(p.traffic)
    idx = {c: [i for i, o in enumerate(p.options) if o.cls == c] for c in classes}
    grids = [simplex_grid(len(idx[c])) for c in classes]
    combos = list(itertools.product(*[range(len(g)) for g in grids]))
    x = np.zeros((len(combos), len(p.options)))
    for row, combo in enumerate(combos):
        for c, g, k in zip(classes, grids, combo, strict=True):
            x[row, idx[c]] = g[k]
    rows = inequality_rows(p)
    ok = np.ones(len(x), dtype=bool)
    if rows.a:
        ok = np.all(x @ np.array(rows.a).T <= np.array(rows.b) + 1e-9, axis=1)
    if not ok.any():
        return None
    cost = np.array([p.traffic[o.cls] * o.usd_per_request for o in p.options])
    return float((x[ok] @ cost).min())


def random_problem(rng: random.Random) -> Problem:
    backends = ["vllm", "gguf", "gemini"][: rng.choice([2, 3])]
    rng.shuffle(backends)
    classes = ["extraction", "short_qa"]
    options = []
    for cls in classes:
        for b in backends:
            options.append(
                Option(
                    cls,
                    b,
                    round(rng.uniform(0.45, 0.8), 3),
                    round(rng.uniform(1e-6, 1e-3), 7),
                    round(rng.uniform(0.2, 5.0), 3),
                    round(rng.uniform(300, 2500), 1),
                )
            )
    traffic = {c: float(rng.randint(20, 600)) for c in classes}
    floors = {c: rng.choice([None, round(rng.uniform(0.5, 0.75), 3)]) for c in classes}
    gemini = (
        {"rpd": float(rng.randint(50, 600)), "rpm": 15.0, "tpm": 250000.0}
        if "gemini" in backends
        else None
    )
    busy = {b: float(rng.randint(200, 3000)) for b in backends if b != "gemini"}
    return Problem(traffic, options, floors, gemini, busy)


def test_lp_matches_brute_force_and_satisfies_every_row():
    rng = random.Random(0)
    solved = 0
    for _ in range(120):
        p = random_problem(rng)
        lp, grid = solve_lp(p), brute_force(p)
        if grid is not None:  # the grid found a feasible point, so the LP must too
            assert lp["status"] == "optimal"
            assert lp["cost_usd_per_day"] <= grid + 1e-9
        if lp["status"] == "optimal":
            solved += 1
            requests = {tuple(k.split("/")): v for k, v in lp["requests"].items()}
            assert check_assignment(p, requests) == []
            for cls, t in p.traffic.items():  # all traffic routed
                assert sum(v for (c, _), v in requests.items() if c == cls) == pytest.approx(t)
    assert solved > 30  # the random cases aren't all infeasible


def test_lp_never_costs_more_than_a_fully_routed_greedy():
    rng = random.Random(1)
    compared = 0
    for _ in range(300):
        p = random_problem(rng)
        g = run_greedy(p)
        assert g["violates_lp_rows"] == []  # greedy is held to the same rows
        if g["status"] == "all routed":
            compared += 1
            lp = solve_lp(p)
            assert lp["status"] == "optimal"
            assert lp["cost_usd_per_day"] <= g["cost_usd_per_day"] + 1e-9
    assert compared > 50


def two_backend_extraction(floor: float, rpd: float = 500.0) -> Problem:
    """GGUF (cheap, 0.69) and Gemini (dear, 0.72) for one class of 100 requests/day."""
    options = [
        Option("extraction", "gguf", 0.69, 1e-5, 3.0, None),
        Option("extraction", "gemini", 0.72, 5e-4, 6.0, 1700.0),
    ]
    return Problem(
        {"extraction": 100.0},
        options,
        {"extraction": floor},
        {"rpd": rpd, "rpm": 15.0, "tpm": 250000.0},
        {"gguf": 86400.0},
    )


def test_lp_blends_to_meet_the_floor_exactly_and_beats_greedy():
    p = two_backend_extraction(floor=0.705)
    lp, g = solve_lp(p), run_greedy(p)
    # 0.69 (1 - s) + 0.72 s >= 0.705  ->  s >= 0.5: the cheapest blend is exactly half Gemini
    assert lp["shares"]["extraction/gemini"] == pytest.approx(0.5)
    assert lp["binding"] == ["floor:extraction"]
    assert lp["cost_usd_per_day"] == pytest.approx(50 * 1e-5 + 50 * 5e-4)
    # greedy can't blend: GGUF is below the floor on its own, so all 100 go to Gemini
    assert g["requests"] == {"extraction/gemini": pytest.approx(100.0)}
    assert g["cost_usd_per_day"] > lp["cost_usd_per_day"]


def test_infeasible_lp_names_the_constraint():
    p = two_backend_extraction(floor=0.705, rpd=30.0)  # needs 50 Gemini requests, cap is 30
    assert solve_lp(p)["status"] == "infeasible"
    # either relaxation alone fixes it: more Gemini requests, or a floor GGUF meets by itself
    assert diagnose_infeasible(p) == ["floor:extraction", "gemini_rpd"]
    g = run_greedy(p)
    assert g["status"] == "unrouted" and g["unrouted"]["extraction"] == pytest.approx(70.0)


def test_long_synthesis_over_the_daily_cap_blames_only_rpd():
    options = [Option("long_synthesis", "gemini", None, 4.6e-3, None, 15984.0)]
    p = Problem(
        {"long_synthesis": 600.0},
        options,
        {"long_synthesis": None},
        {"rpd": 500.0, "rpm": 15.0, "tpm": 250000.0},
    )
    assert solve_lp(p)["status"] == "infeasible"
    assert diagnose_infeasible(p) == ["gemini_rpd"]


def test_no_eligible_backend_is_infeasible():
    p = Problem({"long_synthesis": 10.0}, [], {"long_synthesis": None})
    assert solve_lp(p)["status"] == "infeasible"


SMALL_INPUTS = {
    "classes": {
        "long_synthesis": {"required_context_tokens": 16384, "nominal_tokens": {"prompt": 15984}}
    },
    "backends": {
        "vllm": {
            "context_tokens": 4608,
            "extraction": {
                "accuracy": 0.73,
                "usd_per_request": 1e-6,
                "latency_mean_s": 0.4,
                "latency_p95_s": 0.5,
            },
            "short_qa": {
                "accuracy": 0.55,
                "usd_per_request": 1e-6,
                "latency_mean_s": 0.2,
                "latency_p95_s": 0.3,
            },
            "long_synthesis": {"eligible": False},
        },
        "gemini": {
            "context_tokens": 1048576,
            "extraction": {
                "accuracy": 0.72,
                "usd_per_request": 5e-4,
                "latency_mean_s": 6.0,
                "latency_p95_s": 36.0,
                "tokens_mean": {"prompt": 1741.1},
            },
            "short_qa": {
                "accuracy": 0.50,
                "usd_per_request": 1e-4,
                "latency_mean_s": 3.9,
                "latency_p95_s": 20.4,
                "tokens_mean": {"prompt": 364.6},
            },
            "long_synthesis": {"eligible": True, "usd_per_request_nominal": 4.6e-3},
        },
    },
}
SMALL_RULE = {"eligibility": {"p95_latency_max_s": {"short_qa": 5.0, "extraction": 60.0}}}
SMALL_RULE["eligibility"]["p95_latency_max_s"]["long_synthesis"] = None


def test_eligibility_filters_latency_and_context():
    opts = eligible_options(SMALL_RULE, SMALL_INPUTS, ["vllm", "gemini"])
    pairs = {(o.cls, o.backend) for o in opts}
    assert ("short_qa", "gemini") not in pairs  # p95 20.4 s > 5 s
    assert ("extraction", "gemini") in pairs  # 36 s <= 60 s
    assert ("long_synthesis", "vllm") not in pairs  # 4,608 < 16,384 tokens
    assert ("long_synthesis", "gemini") in pairs


def test_rule_floors_are_best_eligible_minus_001():
    opts = eligible_options(SMALL_RULE, SMALL_INPUTS, ["vllm", "gemini"])
    floors = rule_floors(opts)
    assert floors["extraction"] == pytest.approx(0.73 - 0.01)
    assert floors["short_qa"] == pytest.approx(0.55 - 0.01)  # Gemini isn't eligible here
    assert floors["long_synthesis"] is None


def test_inputs_must_match_the_pinned_sha256(tmp_path):
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps(SMALL_INPUTS))
    rule = tmp_path / "rule.json"
    sha = hashlib.sha256(inputs.read_bytes()).hexdigest()
    rule.write_text(json.dumps({"inputs": {"sha256": sha}}))
    load_rule_and_inputs(rule, inputs)  # matches: fine
    inputs.write_text(json.dumps(SMALL_INPUTS) + " ")
    with pytest.raises(SystemExit, match="sha256"):
        load_rule_and_inputs(rule, inputs)
