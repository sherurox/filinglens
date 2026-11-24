"""Offline A/B: dense top-10 vs dense top-50 + reranker, paired per query (FL-M3 T9/T10).

Both arms are scored on the same queries, so the unit of analysis is the per-query difference
d = metric(reranked) - metric(dense). The primary metric is NDCG@10 (fixed before any test
result); recall@10 and MRR@10 are reported as secondary metrics, without deciding anything.

Queries of one company share boilerplate, so they aren't independent: every bootstrap here
resamples companies (all of a company's queries together), and the power analysis inflates the
independent-query sample size by the design effect 1 + (m - 1) * ICC, with m queries per company
and the intra-company correlation ICC of d estimated on dev (one-way ANOVA estimator). With C
companies the needed n solves n = n_t * (1 + (n / C - 1) * ICC); it has no solution when
ICC >= C / n_t: then more queries from the same companies can't reach the power.

1. `pilot` (ret-dev): from the dev differences, their standard deviation, the company ICC and a
   power analysis for the smallest effect worth detecting (`--mde`, absolute NDCG@10) at
   two-sided alpha 0.05 and power 0.8:
   - independent queries: closed form n = ((z_{1-alpha/2} + z_{1-beta}) * sd / mde)^2 and the
     exact paired t-test (smallest n whose noncentral-t power reaches 0.8);
   - clustered: the n that the design effect asks for with ret-test's companies, the queries
     per company that means, and how many new ret-test queries that needs;
   - a parametric simulation of the actual rule (company-cluster bootstrap CI lower bound > 0)
     at the current and at the needed queries per company.
   It writes the decision rule to bench/ab_rule.json, with the latency budget (the reranker's
   p95, `--latency-budget-ms`). Commit that file before step 2: the commit time proves the rule
   came before the test result.
2. `test` (ret-test, once, --allow-test): refuses unless bench/ab_rule.json is committed and
   unchanged and ret-test has at least the queries the rule asks for; company-cluster bootstrap
   of the mean difference (95% percentile CI, share of resamples where the reranker is not
   better, effect size d_z = mean / sd) on all ret-test queries; applies the rule; writes
   bench/ab_test.json and one MLflow run.

Rule: ship the rerank stage if the 95% company-cluster CI of the NDCG@10 difference lies above 0
and the reranker's p95 latency on ret-test is within the budget. Otherwise keep dense-only.

    uv run python -m filinglens.retrieval.ab pilot --mde 0.03 --latency-budget-ms 500
    git add bench/ab_rule.json bench/ab_power.json && git commit -m "..."
    uv run python -m filinglens.retrieval.ab test --allow-test --note "..."
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from filinglens.retrieval.config import RET_DIR
from filinglens.retrieval.queries import OUT_DIR as QUERY_DIR
from filinglens.retrieval.queries import load_queries
from filinglens.retrieval.runlog import log_run

ALPHA, POWER = 0.05, 0.80
N_BOOT = 10_000
N_SIM, SIM_BOOT = 500, 1000
SEED = 0
METRICS = ("ndcg", "recall", "mrr")
PRIMARY = "ndcg"
RULE = Path("bench/ab_rule.json")


# ---- statistics (pure, tested) ----------------------------------------------------------------


def differences(rows: Sequence[dict], metric: str) -> np.ndarray:
    """Per-query after - before for one metric, from rerank_<split>_per_query.jsonl rows."""
    return np.array([r[f"{metric}_after"] - r[f"{metric}_before"] for r in rows], dtype=float)


def bootstrap_means(
    d: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED, clusters: Sequence | None = None
) -> np.ndarray:
    """Means of n_boot resamples of the per-query differences.

    Without `clusters`, queries are resampled with replacement. With `clusters` (one label per
    query, e.g. the CIK), clusters are resampled with replacement and each brings all its
    queries: mean = sum of the drawn clusters' sums / their total query count.
    """
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        return d[idx].mean(axis=1)
    labels, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    sums = np.bincount(inverse, weights=d, minlength=len(labels))
    counts = np.bincount(inverse, minlength=len(labels)).astype(float)
    idx = rng.integers(0, len(labels), size=(n_boot, len(labels)))
    return sums[idx].sum(axis=1) / counts[idx].sum(axis=1)


def paired_bootstrap(
    d: Sequence[float], n_boot: int = N_BOOT, seed: int = SEED, clusters: Sequence | None = None
) -> dict:
    """Observed mean difference, 95% percentile CI, share of resamples <= 0, effect size d_z."""
    arr = np.asarray(d, dtype=float)
    means = bootstrap_means(arr, n_boot, seed, clusters)
    low, high = np.percentile(means, [2.5, 97.5])
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sd = sd if sd > 1e-12 else 0.0  # identical differences: rounding leaves ~1e-17, not 0
    return {
        "n": int(len(arr)),
        "clusters": int(len(set(clusters))) if clusters is not None else None,
        "mean_diff": round(float(arr.mean()), 4),
        "ci95": [round(float(low), 4), round(float(high), 4)],
        "share_not_better": round(float(np.mean(means <= 0)), 4),
        "sd_diff": round(sd, 4),
        "effect_size_dz": round(float(arr.mean()) / sd, 4) if sd > 0 else None,
        "queries_better": int((arr > 0).sum()),
        "queries_worse": int((arr < 0).sum()),
        "queries_tied": int((arr == 0).sum()),
    }


def icc_oneway(d: Sequence[float], clusters: Sequence) -> float:
    """Intra-cluster correlation of d (one-way ANOVA estimator, unequal sizes), clipped at 0."""
    arr = np.asarray(d, dtype=float)
    labels, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    k, n = len(labels), len(arr)
    if k < 2 or n <= k:
        raise ValueError("need at least 2 clusters and more queries than clusters")
    sizes = np.bincount(inverse, minlength=k).astype(float)
    means = np.bincount(inverse, weights=arr, minlength=k) / sizes
    grand = arr.mean()
    ms_between = float((sizes * (means - grand) ** 2).sum() / (k - 1))
    ms_within = float(((arr - means[inverse]) ** 2).sum() / (n - k))
    m0 = (n - (sizes**2).sum() / n) / (k - 1)
    denom = ms_between + (m0 - 1) * ms_within
    return max(0.0, (ms_between - ms_within) / denom) if denom > 0 else 0.0


def n_clustered(n_independent: int, icc: float, n_clusters: int) -> int | None:
    """Queries needed when they come from n_clusters companies: n = n_t * (1 + (n/C - 1) * ICC).

    None if no n works (ICC >= C / n_t): extra queries from the same companies can't make up for
    their correlation.
    """
    denom = 1 - n_independent * icc / n_clusters
    if denom <= 0:
        return None
    return math.ceil(n_independent * (1 - icc) / denom)


def icc_from_bootstraps(cluster: dict, query: dict) -> float:
    """ICC implied by how much wider the company-cluster CI is than the query CI:
    design effect = (width ratio)^2 = 1 + (m - 1) * ICC, with m = queries per company."""
    m = cluster["n"] / cluster["clusters"]
    width = lambda r: r["ci95"][1] - r["ci95"][0]  # noqa: E731
    if m <= 1 or width(query) <= 0:
        return 0.0
    return max(0.0, ((width(cluster) / width(query)) ** 2 - 1) / (m - 1))


def power_clustered(sd: float, icc: float, effect: float, n_clusters: int, per_cluster: float,
                    alpha: float = ALPHA) -> float:  # fmt: skip
    """Normal-approximation power of a two-sided test of the mean difference when n_clusters
    companies contribute per_cluster queries each: var = sd^2 * (ICC + (1 - ICC) / m) / C."""
    from scipy.stats import norm

    se = math.sqrt(sd**2 * (icc + (1 - icc) / per_cluster) / n_clusters)
    z = norm.ppf(1 - alpha / 2)
    return float(norm.cdf(effect / se - z) + norm.cdf(-effect / se - z))


def n_normal(sd: float, mde: float, alpha: float = ALPHA, power: float = POWER) -> int:
    """Closed-form sample size for a two-sided paired test (normal approximation)."""
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return math.ceil((z * sd / mde) ** 2)


def power_t(n: int, sd: float, effect: float, alpha: float = ALPHA) -> float:
    """Exact power of a two-sided paired t-test with n pairs (noncentral t)."""
    from scipy.stats import nct, norm, t

    if n < 2:
        return 0.0
    crit = t.ppf(1 - alpha / 2, n - 1)
    nc = effect / sd * math.sqrt(n)
    power = float(nct.sf(crit, n - 1, nc) + nct.cdf(-crit, n - 1, nc))
    if not math.isfinite(
        power
    ):  # scipy's noncentral t can return NaN at large df; t ~ normal there
        power = float(norm.sf(crit - nc) + norm.cdf(-crit - nc))
    return power


def n_t(
    sd: float, mde: float, alpha: float = ALPHA, power: float = POWER, cap: int = 100_000
) -> int:
    """Smallest n whose paired t-test reaches the power (searched upward from 2)."""
    n = max(2, n_normal(sd, mde, alpha, power) - 2)
    while power_t(n, sd, mde, alpha) < power:
        n += 1
        if n > cap:
            raise ValueError(f"no n <= {cap} reaches power {power}")
    while n > 2 and power_t(n - 1, sd, mde, alpha) >= power:
        n -= 1
    return n


def mde_at(n: int, sd: float, alpha: float = ALPHA, power: float = POWER) -> float:
    """Smallest effect the paired t-test detects with the given power at n pairs."""
    from scipy.optimize import brentq
    from scipy.stats import norm

    # bracket from the normal approximation: 3x its MDE has power ~1, and keeps the
    # noncentrality small enough for scipy's noncentral t (it returns NaN far out)
    upper = 3 * (norm.ppf(1 - alpha / 2) + norm.ppf(power)) * sd / math.sqrt(n)
    return float(brentq(lambda e: power_t(n, sd, e, alpha) - power, 1e-9, upper))


def simulated_power(
    d: np.ndarray, effect: float, n: int, n_sim: int = N_SIM, n_boot: int = SIM_BOOT,
    seed: int = SEED,
) -> float:  # fmt: skip
    """Share of simulated A/Bs with n queries in which the rule (CI lower bound > 0) ships, when
    the true mean difference is `effect` and the spread is that of the pilot differences."""
    rng = np.random.default_rng(seed)
    shifted = d - d.mean() + effect
    ships = 0
    for _ in range(n_sim):
        sample = shifted[rng.integers(0, len(shifted), n)]
        means = sample[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
        ships += np.percentile(means, 2.5) > 0
    return ships / n_sim


def simulated_power_clustered(
    sd: float, icc: float, effect: float, n_clusters: int, per_cluster: int,
    n_sim: int = N_SIM, n_boot: int = SIM_BOOT, seed: int = SEED,
) -> float:  # fmt: skip
    """Power of the cluster-bootstrap rule (CI lower bound > 0) under a random-intercept model:
    d = effect + u_company + e, var(u) = ICC * sd^2, var(e) = (1 - ICC) * sd^2. A parametric
    cross-check of the closed form (real differences are not normal: many are exactly 0)."""
    rng = np.random.default_rng(seed)
    clusters = np.repeat(np.arange(n_clusters), per_cluster)
    ships = 0
    for i in range(n_sim):
        u = rng.normal(0, math.sqrt(icc) * sd, n_clusters)[clusters]
        d = effect + u + rng.normal(0, math.sqrt(1 - icc) * sd, len(clusters))
        means = bootstrap_means(d, n_boot, seed + i + 1, clusters)
        ships += np.percentile(means, 2.5) > 0
    return ships / n_sim


def decide(primary: dict, p95_ms: float, budget_ms: float) -> dict:
    significant = primary["ci95"][0] > 0
    fast_enough = p95_ms <= budget_ms
    return {
        "ci_lower_above_zero": significant,
        "p95_within_budget": fast_enough,
        "ship_rerank": bool(significant and fast_enough),
    }


# ---- steps ------------------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_rows(rows: Sequence[dict], split: str, query_dir: Path, allow_test: bool) -> None:
    expected = [q["query_id"] for q in load_queries(split, query_dir, allow_test=allow_test)]
    if [r["query_id"] for r in rows] != expected:
        raise ValueError(f"per-query file doesn't cover queries.{split}.jsonl in order")


def rule_is_committed(path: Path) -> bool:
    """Tracked by git and identical to the committed version."""

    def ok(*cmd: str) -> bool:
        return subprocess.run(["git", *cmd], capture_output=True).returncode == 0

    return ok("ls-files", "--error-unmatch", str(path)) and ok(
        "diff", "--quiet", "HEAD", "--", str(path)
    )


def test_companies(query_dir: Path) -> list[int]:
    """CIKs of the ret-test queries, read as metadata only (no results exist for them yet)."""
    lines = (query_dir / "queries.test.jsonl").read_text().splitlines()
    return [json.loads(line)["cik"] for line in lines if line.strip()]


def run_pilot(args: argparse.Namespace, argv: list[str]) -> None:
    rows = read_jsonl(args.ret_dir / "rerank_dev_per_query.jsonl")
    check_rows(rows, "dev", args.query_dir, allow_test=False)
    dev_cik = [q["cik"] for q in load_queries("dev", args.query_dir)]
    test_cik = test_companies(args.query_dir)
    n_test, c_test = len(test_cik), len(set(test_cik))
    d = differences(rows, PRIMARY)
    sd = float(d.std(ddof=1))
    icc = icc_oneway(d, dev_cik)
    by_company, by_query = paired_bootstrap(d, clusters=dev_cik), paired_bootstrap(d)
    icc_boot = icc_from_bootstraps(by_company, by_query)
    need_t = n_t(sd, args.mde)
    need = n_clustered(need_t, icc, c_test)
    per_company = math.ceil(need / c_test) if need else None
    table = {
        f"{name} {value:.4f}": {
            f"{m:g}_per_company": round(power_clustered(sd, value, args.mde, c_test, m), 4)
            for m in sorted({round(n_test / c_test, 2), 5, 9, 15})
        }
        for name, value in (("icc_anova", icc), ("icc_bootstrap_implied", icc_boot))
    }
    power = {
        "metric": f"{PRIMARY}_at_10",
        "pilot_split": "dev",
        "pilot_queries": len(rows),
        "pilot_companies": len(set(dev_cik)),
        "pilot_sd_diff": round(sd, 4),
        "pilot_icc_company": round(icc, 4),
        "pilot_icc_bootstrap_implied": round(icc_boot, 4),
        "pilot_effect_cluster_bootstrap": by_company,
        "pilot_effect_query_bootstrap": by_query,
        "alpha_two_sided": ALPHA,
        "power": POWER,
        "mde": args.mde,
        "n_required_normal_independent": n_normal(sd, args.mde),
        "n_required_t_independent": need_t,
        "test_companies": c_test,
        "n_test_available": n_test,
        "n_required_clustered": need,
        "queries_per_company_needed": per_company,
        "feasible_with_test_companies": need is not None,
        "new_test_queries_needed": max(0, need - n_test) if need else None,
        "design_effect_at_need": round(1 + (need / c_test - 1) * icc, 4) if need else None,
        "power_t_at_n_test_independent": round(power_t(n_test, sd, args.mde), 4),
        "power_clustered_normal_approx": table,
        "simulated_power_cluster_rule": {
            f"{c_test}x{math.ceil(n_test / c_test)}": simulated_power_clustered(
                sd, icc, args.mde, c_test, math.ceil(n_test / c_test)
            ),
            **(
                {
                    f"{c_test}x{per_company}": simulated_power_clustered(
                        sd, icc, args.mde, c_test, per_company
                    )
                }
                if per_company
                else {}
            ),
        },
        "secondary_pilot_effects": {
            m: paired_bootstrap(differences(rows, m), clusters=dev_cik) for m in METRICS[1:]
        },
    }
    rule = {
        "fixed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": "python -m filinglens.retrieval.ab " + shlex.join(argv),
        "primary_metric": f"{PRIMARY}_at_10",
        "mde": args.mde,
        "alpha_two_sided": ALPHA,
        "power": POWER,
        "icc_company_dev": round(icc, 4),
        "icc_bootstrap_implied_dev": round(icc_boot, 4),
        # the power analysis's n if it exists and ret-test has it; else every ret-test query
        # (user decision 24 Nov 2026: grow ret-test to ~15 per company; the ICC is uncertain)
        "n_test_required": need if need is not None and need <= n_test else n_test,
        "n_test_basis": "power analysis" if need is not None and need <= n_test else "all ret-test",
        "expected_power_at_n_test": {
            k: round(power_clustered(sd, v, args.mde, c_test, n_test / c_test), 4)
            for k, v in (("icc_anova", icc), ("icc_bootstrap_implied", icc_boot))
        },
        "test_companies": c_test,
        "latency_budget_p95_ms": args.latency_budget_ms,
        "bootstrap": {"resamples": N_BOOT, "seed": SEED, "ci": "95% percentile", "unit": "company"},
        "rule": (
            "use every ret-test query (at least n_test_required); ship the rerank stage iff the "
            "95% company-cluster bootstrap CI of the per-query NDCG@10 difference (reranked - "
            "dense) lies above 0 AND the reranker's p95 latency on ret-test <= "
            "latency_budget_p95_ms; otherwise keep dense-only"
        ),
    }
    (args.bench / "ab_power.json").write_text(json.dumps(power, indent=2) + "\n")
    args.rule.write_text(json.dumps(rule, indent=2) + "\n")
    print(json.dumps({"power": power, "rule": rule}, indent=2))


def run_test(args: argparse.Namespace, argv: list[str]) -> None:
    if not args.allow_test:
        raise PermissionError("the A/B test step reads ret-test: pass --allow-test (once)")
    if not rule_is_committed(args.rule):
        raise SystemExit(f"{args.rule} must be committed (and unchanged) before the test run")
    rule = json.loads(args.rule.read_text())
    per_query = args.ret_dir / "rerank_test_per_query.jsonl"
    rows = read_jsonl(per_query)
    check_rows(rows, "test", args.query_dir, allow_test=True)
    need = rule["n_test_required"]
    if need is None or len(rows) < need:
        raise SystemExit(f"{len(rows)} ret-test queries; the rule needs at least {need}")
    ciks = [q["cik"] for q in load_queries("test", args.query_dir, allow_test=True)]
    reranker = json.loads((args.bench / "reranker_test.json").read_text())
    results = {m: paired_bootstrap(differences(rows, m), clusters=ciks) for m in METRICS}
    p95 = reranker["rerank_latency"]["p95_ms"]
    record = {
        "command": "python -m filinglens.retrieval.ab " + shlex.join(argv),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": "test",
        "labels": "synthetic queries, automatic grades (no human judgment)",
        "rule_file": str(args.rule),
        "rule": rule,
        "arms": {
            "A": "dense top-10 (FAISS Flat, CIK filter)",
            "B": "dense top-50 + bge-reranker-base",
        },
        "means": reranker["metrics"],
        "primary": {f"{PRIMARY}_at_10": results[PRIMARY]},
        "primary_query_bootstrap_for_reference": paired_bootstrap(differences(rows, PRIMARY)),
        "icc_company_test": round(icc_oneway(differences(rows, PRIMARY), ciks), 4),
        "secondary": {f"{m}_at_10": results[m] for m in METRICS[1:]},
        "rerank_latency_test": reranker["rerank_latency"],
        "decision": decide(results[PRIMARY], p95, rule["latency_budget_p95_ms"]),
    }
    metrics = {
        "ndcg_diff": results["ndcg"]["mean_diff"],
        "ndcg_ci_low": results["ndcg"]["ci95"][0],
        "ndcg_ci_high": results["ndcg"]["ci95"][1],
        "rerank_p95_ms": p95,
        "ship_rerank": float(record["decision"]["ship_rerank"]),
    }
    params = {"mde": rule["mde"], "n": len(rows), "companies": len(set(ciks)),
              "budget_ms": rule["latency_budget_p95_ms"]}  # fmt: skip
    record["mlflow_run_id"] = log_run("ab_test", record, params, metrics, per_query, args.note)
    (args.bench / "ab_test.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    pilot, test = sub.add_parser("pilot"), sub.add_parser("test")
    pilot.add_argument("--mde", type=float, required=True, help="smallest NDCG@10 gain worth it")
    pilot.add_argument("--latency-budget-ms", type=float, required=True, help="reranker p95 cap")
    test.add_argument("--allow-test", action="store_true")
    test.add_argument("--note", required=True, help="power source and Windows power mode")
    for p in (pilot, test):
        p.add_argument("--ret-dir", type=Path, default=RET_DIR)
        p.add_argument("--query-dir", type=Path, default=QUERY_DIR)
        p.add_argument("--bench", type=Path, default=Path("bench"))
        p.add_argument("--rule", type=Path, default=RULE)
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    (run_pilot if args.step == "pilot" else run_test)(args, argv)


if __name__ == "__main__":
    main()
