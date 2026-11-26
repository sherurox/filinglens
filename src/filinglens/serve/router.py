"""LP router: which backend serves each request class, at least list-price cost (FL-M5 T10).

    uv run python -m filinglens.serve.router            # -> bench/router_results.json

Everything follows bench/m5_router_rule.json, committed before this file existed; the inputs are
bench/router_inputs.json (dev only), checked against the SHA-256 the rule pins.

Variables: x[c, b] = share of class c's daily traffic sent to backend b, in [0, 1]; each class's
shares sum to 1 (all traffic is routed, or the LP is infeasible). Only eligible (c, b) pairs get a
variable: a backend takes long synthesis only if its context window holds the required tokens,
and takes a class only if its measured p95 is within that class's limit. p95 is not linear in the
shares (the tail of a mix is not the share-weighted mean of the tails), so latency filters instead
of entering the LP.

Objective: minimize sum of x[c, b] * traffic[c] * usd_per_request[c, b] (list prices; the actual
spend is $0). Constraints, each a named row so the report can say which ones bind:
- floor:<class>  sum_b x[c, b] * accuracy[c, b] >= floor[c]  (expected accuracy is linear)
- gemini_rpd     sum_c x[c, gemini] * traffic[c] <= RPD
- gemini_rpm     sum_c x[c, gemini] * traffic[c] / 1440 <= RPM     (arrivals spread over 24 h)
- gemini_tpm     sum_c x[c, gemini] * traffic[c] / 1440 * prompt_tokens[c] <= TPM
- busy:<backend> sum_c x[c, b] * traffic[c] * mean_latency[c, b] <= 86,400 s  (local backends)

The greedy baseline is the user's hand-written serve.greedy.greedy_route, given the same eligible
backends and floors, Gemini's RPD and each local backend's busy seconds; its result is then checked
against every LP row (RPM and TPM included), so both are held to the same constraints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from filinglens.serve.greedy import greedy_route

RULE = Path("bench/m5_router_rule.json")
INPUTS = Path("bench/router_inputs.json")
OUT = Path("bench/router_results.json")
CLASSES = ("extraction", "short_qa", "long_synthesis")
LOCAL_BACKENDS = ("vllm", "gguf")
DAY_S = 86400.0
MINUTES_PER_DAY = 1440.0
FLOOR_MARGIN = 0.01  # rule: floor = best eligible dev accuracy - 0.01
SWEEP_STEP = 0.0025
TOL = 1e-9


@dataclass(frozen=True)
class Option:
    """One (class, backend) pair the LP may use."""

    cls: str
    backend: str
    accuracy: float | None
    usd_per_request: float
    mean_latency_s: float | None
    prompt_tokens: float | None  # Gemini only (TPM)


@dataclass
class Problem:
    traffic: dict[str, float]
    options: list[Option]
    floors: dict[str, float | None]
    gemini: dict[str, float] | None = None  # {"rpd", "rpm", "tpm"} if Gemini is in the scenario
    busy_s: dict[str, float] = field(default_factory=dict)  # local backend -> seconds per day


@dataclass
class Rows:
    """linprog's inequality rows (A_ub x <= b_ub) with a name per row."""

    names: list[str] = field(default_factory=list)
    a: list[list[float]] = field(default_factory=list)
    b: list[float] = field(default_factory=list)

    def add(self, name: str, coefs: list[float], bound: float) -> None:
        if any(coefs):
            self.names.append(name)
            self.a.append(coefs)
            self.b.append(bound)


def inequality_rows(p: Problem) -> Rows:
    rows = Rows()
    opts = p.options
    for cls, floor in p.floors.items():
        if floor is not None:  # sum x*acc >= floor  ->  -sum x*acc <= -floor
            coefs = [-(o.accuracy or 0.0) if o.cls == cls else 0.0 for o in opts]
            rows.add(f"floor:{cls}", coefs, -floor)
    if p.gemini:
        gem = [o.backend == "gemini" for o in opts]
        t = [p.traffic[o.cls] for o in opts]
        rows.add("gemini_rpd", [t[i] if gem[i] else 0.0 for i in range(len(opts))], p.gemini["rpd"])
        rows.add(
            "gemini_rpm",
            [t[i] / MINUTES_PER_DAY if gem[i] else 0.0 for i in range(len(opts))],
            p.gemini["rpm"],
        )
        rows.add(
            "gemini_tpm",
            [
                t[i] / MINUTES_PER_DAY * (o.prompt_tokens or 0.0) if gem[i] else 0.0
                for i, o in enumerate(opts)
            ],
            p.gemini["tpm"],
        )
    for backend, seconds in p.busy_s.items():
        coefs = [
            p.traffic[o.cls] * (o.mean_latency_s or 0.0) if o.backend == backend else 0.0
            for o in opts
        ]
        rows.add(f"busy:{backend}", coefs, seconds)
    return rows


def solve_lp(p: Problem, skip: frozenset[str] = frozenset()) -> dict:
    """Solve; `skip` drops named rows (used to diagnose infeasibility)."""
    classes = list(p.traffic)
    missing = [c for c in classes if not any(o.cls == c for o in p.options)]
    if missing:
        return {"status": "infeasible", "reason": f"no eligible backend for {missing}"}
    rows = inequality_rows(p)
    keep = [i for i, n in enumerate(rows.names) if n not in skip]
    cost = np.array([p.traffic[o.cls] * o.usd_per_request for o in p.options])
    a_eq = [[1.0 if o.cls == c else 0.0 for o in p.options] for c in classes]
    result = linprog(
        cost,
        A_ub=np.array([rows.a[i] for i in keep]) if keep else None,
        b_ub=np.array([rows.b[i] for i in keep]) if keep else None,
        A_eq=np.array(a_eq),
        b_eq=np.ones(len(classes)),
        bounds=[(0.0, 1.0)] * len(p.options),
        method="highs",
    )
    if result.status != 0:
        return {"status": "infeasible" if result.status == 2 else "error", "reason": result.message}
    names = [rows.names[i] for i in keep]
    slack = dict(zip(names, (float(s) for s in result.ineqlin.residual), strict=True))
    duals = dict(zip(names, (float(d) for d in result.ineqlin.marginals), strict=True))
    return {
        "status": "optimal",
        "cost_usd_per_day": float(result.fun),
        "shares": {
            f"{o.cls}/{o.backend}": float(x) for o, x in zip(p.options, result.x, strict=True)
        },
        "requests": {
            f"{o.cls}/{o.backend}": float(x) * p.traffic[o.cls]
            for o, x in zip(p.options, result.x, strict=True)
        },
        "binding": sorted(n for n, s in slack.items() if s <= 1e-7),
        "slack": slack,
        "duals_usd_per_unit": duals,
    }


def diagnose_infeasible(p: Problem) -> list[str]:
    """Named rows whose removal alone makes the LP feasible."""
    rows = inequality_rows(p)
    return [n for n in rows.names if solve_lp(p, frozenset({n}))["status"] == "optimal"]


def greedy_inputs(p: Problem) -> tuple[dict, dict, dict, dict, dict]:
    """The same problem in greedy_route's terms: Gemini capacity in requests (RPD), local
    capacity in busy seconds."""
    options: dict[str, list[dict]] = {c: [] for c in p.traffic}
    usage: dict[tuple[str, str], float] = {}
    for o in p.options:
        options[o.cls].append(
            {
                "backend": o.backend,
                "accuracy": o.accuracy or 0.0,
                "usd_per_request": o.usd_per_request,
            }
        )
        usage[(o.cls, o.backend)] = 1.0 if o.backend == "gemini" else (o.mean_latency_s or 0.0)
    capacity = dict(p.busy_s)
    if p.gemini:
        capacity["gemini"] = p.gemini["rpd"]
    floors = {c: p.floors.get(c) for c in p.traffic}
    return dict(p.traffic), options, floors, capacity, usage


def check_assignment(p: Problem, requests: dict[tuple[str, str], float]) -> list[str]:
    """LP rows a (possibly partial) assignment breaks. Floors are checked on routed traffic."""
    shares = np.array([requests.get((o.cls, o.backend), 0.0) / p.traffic[o.cls] for o in p.options])
    rows = inequality_rows(p)
    broken = []
    for name, coefs, bound in zip(rows.names, rows.a, rows.b, strict=True):
        if name.startswith("floor:"):
            cls = name.split(":", 1)[1]
            routed = sum(v for (c, _), v in requests.items() if c == cls)
            if routed <= 0:
                continue
            value = float(np.dot(coefs, shares)) * p.traffic[cls] / routed
        else:
            value = float(np.dot(coefs, shares))
        if value > bound + 1e-6:
            broken.append(name)
    return broken


def run_greedy(p: Problem) -> dict:
    g = greedy_route(*greedy_inputs(p))
    unrouted = {c: float(v) for c, v in g["unrouted"].items()}
    return {
        "status": "all routed" if all(v <= 1e-6 for v in unrouted.values()) else "unrouted",
        "cost_usd_per_day": float(g["cost_usd"]),
        "requests": {f"{c}/{b}": float(v) for (c, b), v in g["assigned"].items()},
        "unrouted": unrouted,
        "violates_lp_rows": check_assignment(p, g["assigned"]),
    }


def compare(lp: dict, greedy: dict) -> dict:
    if lp["status"] != "optimal":
        return {"savings": None, "note": f"LP {lp['status']}; greedy {greedy['status']}"}
    if greedy["status"] != "all routed":
        left = sum(greedy["unrouted"].values())
        return {
            "savings": None,
            "note": f"greedy leaves {left:g} requests/day unrouted; the LP routes all",
        }
    saved = greedy["cost_usd_per_day"] - lp["cost_usd_per_day"]
    pct = 100 * saved / greedy["cost_usd_per_day"] if greedy["cost_usd_per_day"] else 0.0
    return {"savings_usd_per_day": saved, "savings_pct": pct}


# ---------- building problems from the rule and the inputs ----------


def load_rule_and_inputs(rule_path: Path = RULE, inputs_path: Path = INPUTS) -> tuple[dict, dict]:
    rule = json.loads(rule_path.read_text())
    raw = inputs_path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != rule["inputs"]["sha256"]:
        raise SystemExit(f"{inputs_path} sha256 {sha[:12]}... is not the one the rule pins")
    return rule, json.loads(raw)


def eligible_options(rule: dict, inputs: dict, backends: list[str]) -> list[Option]:
    limits = rule["eligibility"]["p95_latency_max_s"]
    need = inputs["classes"]["long_synthesis"]["required_context_tokens"]
    out = []
    for cls in CLASSES:
        for name in backends:
            b = inputs["backends"][name]
            if cls == "long_synthesis":
                if b["context_tokens"] < need:
                    continue
                ls = b["long_synthesis"]
                out.append(
                    Option(
                        cls,
                        name,
                        None,
                        ls["usd_per_request_nominal"],
                        None,
                        inputs["classes"]["long_synthesis"]["nominal_tokens"]["prompt"],
                    )
                )
                continue
            m = b[cls]
            if limits[cls] is not None and m["latency_p95_s"] > limits[cls]:
                continue
            prompt = (m.get("tokens_mean") or {}).get("prompt")
            out.append(
                Option(cls, name, m["accuracy"], m["usd_per_request"], m["latency_mean_s"], prompt)
            )
    return out


def rule_floors(options: list[Option]) -> dict[str, float | None]:
    floors: dict[str, float | None] = {}
    for cls in CLASSES:
        accs = [o.accuracy for o in options if o.cls == cls and o.accuracy is not None]
        floors[cls] = max(accs) - FLOOR_MARGIN if accs and cls != "long_synthesis" else None
    return floors


def build_problem(rule: dict, inputs: dict, scenario: str, traffic: dict) -> Problem:
    backends = rule["scenarios"][scenario]["backends"]
    options = eligible_options(rule, inputs, backends)
    free = inputs["capacity"]["gemini_free_tier"]
    return Problem(
        traffic={c: float(traffic[c]) for c in CLASSES},
        options=options,
        floors=rule_floors(options),
        gemini={k: float(free[k]) for k in ("rpd", "rpm", "tpm")} if "gemini" in backends else None,
        busy_s={b: DAY_S for b in backends if b in LOCAL_BACKENDS},
    )


def sweep(p: Problem, cls: str) -> list[dict]:
    """LP and greedy cost as `cls`'s floor moves from its lowest to highest eligible accuracy."""
    accs = sorted({o.accuracy for o in p.options if o.cls == cls and o.accuracy is not None})
    if len(accs) < 2:
        return []
    points = []
    steps = int(round((accs[-1] - accs[0]) / SWEEP_STEP))
    for k in range(steps + 1):
        floor = round(accs[0] + k * SWEEP_STEP, 6)
        q = Problem(p.traffic, p.options, p.floors | {cls: floor}, p.gemini, p.busy_s)
        lp, g = solve_lp(q), run_greedy(q)
        points.append(
            {
                "floor": floor,
                "lp_status": lp["status"],
                "lp_cost_usd_per_day": lp.get("cost_usd_per_day"),
                "greedy_status": g["status"],
                "greedy_cost_usd_per_day": g["cost_usd_per_day"],
                "greedy_unrouted": sum(g["unrouted"].values()),
            }
        )
    return points


def options_table(p: Problem) -> list[dict]:
    return [o.__dict__ for o in p.options]


def run(rule_path: Path = RULE, inputs_path: Path = INPUTS) -> dict:
    rule, inputs = load_rule_and_inputs(rule_path, inputs_path)
    cases = {
        "main": rule["traffic_per_day"]["main"],
        "infeasible_case": rule["traffic_per_day"]["infeasible_case"],
    }
    results: dict = {}
    for scenario in rule["scenarios"]:
        for case, traffic in cases.items():
            p = build_problem(rule, inputs, scenario, traffic)
            lp, g = solve_lp(p), run_greedy(p)
            entry = {
                "traffic_per_day": p.traffic,
                "eligible": options_table(p),
                "floors": p.floors,
                "lp": lp,
                "greedy": g,
                "comparison": compare(lp, g),
            }
            if lp["status"] != "optimal":
                entry["infeasible_because"] = diagnose_infeasible(p)
            if case == "main":
                entry["sweep"] = {c: sweep(p, c) for c in ("extraction", "short_qa")}
            results[f"{scenario}/{case}"] = entry
    return {
        "rule": rule_path.as_posix(),
        "inputs": {"file": inputs_path.as_posix(), "sha256": rule["inputs"]["sha256"]},
        "spend": inputs["spend"],
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    body = run()
    result = {
        "command": " ".join(["python -m filinglens.serve.router", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **body,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for key, r in result["results"].items():
        lp, g = r["lp"], r["greedy"]
        print(f"{key}: floors {r['floors']}")
        cost, binding = lp.get("cost_usd_per_day"), lp.get("binding")
        print(f"  LP     {lp['status']}: {cost} $/day; binding {binding}")
        print(f"         requests {lp.get('requests')}")
        if "infeasible_because" in r:
            print(f"         infeasible because of: {r['infeasible_because']}")
        print(
            f"  greedy {g['status']}: {g['cost_usd_per_day']} $/day; unrouted {g['unrouted']};"
            f" violates {g['violates_lp_rows']}"
        )
        print(f"         requests {g['requests']}")
        print(f"  {r['comparison']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
