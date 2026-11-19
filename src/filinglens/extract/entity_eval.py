"""Evaluate the company-name resolver against SEC identifiers (CIKs), with no hand labels.

Queries are the (cik, name) pairs filers used in FSDS sub.txt, 2018q1-2026q3. The resolver
sees only the name and matches it against the current names in company_tickers.json; the
filer's own CIK is the answer key. Filers that aren't listed today are in the queries too:
for them the right answer is no match, so they test false merges (a subsidiary resolved to
its listed parent, say).

Companies are split in half by a hash of the CIK. The fuzzy threshold and margin are chosen
on the tune half (highest F0.5: precision counts more than recall, because a false merge
corrupts every record joined through it). The eval half is scored once with that setting,
and 100 of its matches are drawn at random for the D4 precision check.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filinglens.extract.entities import (
    MAX_DF,
    Resolver,
    normalize,
    read_csv,
    ticker_rows,
)

SEED = 20261119
THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95
MARGINS = [0.0, 0.05, 0.1, 0.2]
ACCEPTED = ("exact", "compact", "fuzzy")
PAIR_FIELDS = [
    "sample",
    "query",
    "gold_cik",
    "pred_cik",
    "matched_name",
    "method",
    "score",
    "margin",
    "gold_in_reference",
    "correct",
]


def half(cik: int, seed: int = SEED) -> str:
    """'tune' or 'eval', fixed per company: every name of one CIK lands in the same half."""
    digest = hashlib.sha256(f"{seed}:{cik}".encode()).digest()
    return "tune" if digest[0] % 2 == 0 else "eval"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """95% Wilson score interval for k successes in n: sane near 0 and 1, unlike p ± 1.96·SE."""
    if n == 0:
        return None, None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(max(0.0, centre - half_width), 4), round(min(1.0, centre + half_width), 4)


def raw_matches(
    resolver: Resolver, queries: list[dict[str, Any]], reference_ciks: set[int]
) -> list[dict[str, Any]]:
    """Resolve every query once with no threshold; settings are applied afterwards by decide()."""
    ids_by_cik: dict[int, set[int]] = {}
    for i, (cik, _, _) in enumerate(resolver.entries):
        ids_by_cik.setdefault(cik, set()).add(i)
    out = []
    for q in queries:
        gold = int(q["cik"])
        m = resolver.resolve(q["name"], threshold=0.0, min_margin=0.0)
        tokens = set(normalize(q["name"]))
        # Blocking check: does any rare token of the query lead to the true company at all?
        reachable = any(
            ids_by_cik.get(gold, set()) & set(resolver.postings.get(t, ())) for t in tokens
        )
        out.append(
            {
                "cik": gold,
                "name": q["name"],
                "half": half(gold),
                "gold_in_reference": gold in reference_ciks,
                "best_cik": m.cik,
                "best_name": m.name,
                "method": m.method,  # exact | compact | ambiguous | fuzzy | none
                "score": m.score,
                "margin": m.margin,
                "candidates": m.candidates,
                "reachable": reachable,
            }
        )
    return out


def decide(r: dict[str, Any], threshold: float, margin: float) -> int | None:
    """The CIK the resolver returns at this setting, or None (no match)."""
    if r["method"] in ("exact", "compact"):
        return r["best_cik"]
    if r["method"] == "fuzzy" and r["score"] >= threshold and r["margin"] >= margin:
        return r["best_cik"]
    return None


def metrics(rows: list[dict[str, Any]], threshold: float, margin: float) -> dict[str, Any]:
    matched = correct = 0
    for r in rows:
        pred = decide(r, threshold, margin)
        if pred is not None:
            matched += 1
            correct += pred == r["cik"]
    in_ref = sum(r["gold_in_reference"] for r in rows)
    p = correct / matched if matched else 0.0
    rec = correct / in_ref if in_ref else 0.0
    f05 = 1.25 * p * rec / (0.25 * p + rec) if p + rec else 0.0
    return {
        "threshold": threshold,
        "margin": margin,
        "queries": len(rows),
        "gold_in_reference": in_ref,
        "matched": matched,
        "correct": correct,
        "precision": round(p, 4),
        "recall": round(rec, 4),
        "f0_5": round(f05, 4),
    }


def choose(sweep: list[dict[str, Any]]) -> dict[str, Any]:
    """Highest F0.5; ties go to the stricter setting (higher threshold, then higher margin)."""
    return max(sweep, key=lambda m: (m["f0_5"], m["threshold"], m["margin"]))


def by_method(rows: list[dict[str, Any]], threshold: float, margin: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for method in ACCEPTED:
        picked = [
            r for r in rows if r["method"] == method and decide(r, threshold, margin) is not None
        ]
        k = sum(decide(r, threshold, margin) == r["cik"] for r in picked)
        lo, hi = wilson(k, len(picked))
        out[method] = {
            "matched": len(picked),
            "correct": k,
            "precision": round(k / len(picked), 4) if picked else None,
            "ci95": [lo, hi],
        }
    return out


def errors(rows: list[dict[str, Any]], threshold: float, margin: float, n: int = 15) -> dict:
    wrong = [
        r for r in rows if (pred := decide(r, threshold, margin)) is not None and pred != r["cik"]
    ]
    kinds = Counter(
        "true company listed (wrong CIK chosen)"
        if r["gold_in_reference"]
        else "true company not listed (false merge)"
        for r in wrong
    )
    return {
        "wrong": len(wrong),
        "kinds": dict(kinds),
        "methods": dict(Counter(r["method"] for r in wrong)),
        "examples": [
            {k: r[k] for k in ("name", "cik", "best_name", "best_cik", "method", "score")}
            for r in sorted(wrong, key=lambda r: (-r["score"], r["name"]))[:n]
        ],
    }


def blocking(rows: list[dict[str, Any]], entries: int) -> dict[str, Any]:
    """How much blocking saved, and how many true companies it put out of reach."""
    fuzzy = [r for r in rows if r["method"] in ("fuzzy", "none")]
    compared = sum(r["candidates"] for r in fuzzy)
    listed = [r for r in fuzzy if r["gold_in_reference"]]
    reach = sum(r["reachable"] for r in listed)
    return {
        "max_df": MAX_DF,
        "fuzzy_queries": len(fuzzy),
        "comparisons": compared,
        "all_pairs": len(fuzzy) * entries,
        "reduction_ratio": round(1 - compared / (len(fuzzy) * entries), 6) if fuzzy else None,
        "listed_fuzzy_queries": len(listed),
        "pair_completeness": round(reach / len(listed), 4) if listed else None,
    }


def sample_pairs(
    rows: list[dict[str, Any]], threshold: float, margin: float, n: int, name: str, seed: int
) -> list[dict[str, Any]]:
    matched = [r for r in rows if decide(r, threshold, margin) is not None]
    picked = random.Random(seed).sample(matched, min(n, len(matched)))
    return [
        {
            "sample": name,
            "query": r["name"],
            "gold_cik": r["cik"],
            "pred_cik": r["best_cik"],
            "matched_name": r["best_name"],
            "method": r["method"],
            "score": r["score"],
            "margin": r["margin"],
            "gold_in_reference": r["gold_in_reference"],
            "correct": r["best_cik"] == r["cik"],
        }
        for r in sorted(picked, key=lambda r: (r["cik"], r["name"]))
    ]


def precision_of(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    k = sum(p["correct"] for p in pairs)
    lo, hi = wilson(k, len(pairs))
    return {
        "pairs": len(pairs),
        "correct": k,
        "precision": round(k / len(pairs), 4) if pairs else None,
        "wilson_ci95": [lo, hi],
        "methods": dict(Counter(p["method"] for p in pairs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickers", type=Path, default=Path("data/raw/company_tickers.json"))
    parser.add_argument(
        "--fsds-names", type=Path, default=Path("data/interim/entities/fsds_names.csv")
    )
    parser.add_argument("--stats", type=Path, default=Path("bench/entity_eval.json"))
    parser.add_argument("--precision", type=Path, default=Path("bench/entity_precision.json"))
    parser.add_argument("--pairs", type=Path, default=Path("bench/entity_pairs_100.csv"))
    args = parser.parse_args()
    started = time.perf_counter()

    tickers = ticker_rows(json.loads(args.tickers.read_text()))
    resolver = Resolver((r["cik"], r["name"]) for r in tickers)
    reference_ciks = {r["cik"] for r in tickers}
    rows = raw_matches(resolver, read_csv(args.fsds_names), reference_ciks)
    resolve_seconds = round(time.perf_counter() - started, 1)
    tune = [r for r in rows if r["half"] == "tune"]
    test = [r for r in rows if r["half"] == "eval"]

    sweep = [metrics(tune, t, m) for t in THRESHOLDS for m in MARGINS]
    best = choose(sweep)
    t, m = best["threshold"], best["margin"]
    result = metrics(test, t, m)

    # D4: 100 random matches from the eval half, checked against the filer's CIK. A second
    # sample of 100 leaves out exact-name matches, which are the easy cases.
    everything = sample_pairs(test, t, m, 100, "all_matches", SEED)
    hard = [r for r in test if r["method"] in ("compact", "fuzzy")]
    non_exact = sample_pairs(hard, t, m, 100, "non_exact_matches", SEED + 1)

    report = {
        "command": "python -m filinglens.extract.entity_eval",
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "reference": {"source": str(args.tickers), "ciks": len(reference_ciks)},
        "reference_names": len(resolver.entries),
        "queries": {
            "source": str(args.fsds_names),
            "total": len(rows),
            "tune": len(tune),
            "eval": len(test),
            "methods_eval": dict(Counter(r["method"] for r in test)),
        },
        "split": f"sha256('{SEED}:<cik>') first byte even = tune, odd = eval",
        "selection_rule": "max F0.5 on tune; ties -> higher threshold, then higher margin",
        "sweep_tune": sweep,
        "chosen": {"threshold": t, "margin": m, "tune": best},
        "eval": result,
        "eval_by_method": by_method(test, t, m),
        "eval_errors": errors(test, t, m),
        "blocking_eval": blocking(test, len(resolver.entries)),
        "resolve_seconds": resolve_seconds,
        "seconds": round(time.perf_counter() - started, 1),
    }
    precision = {
        "command": report["command"],
        "created": report["created"],
        "check": "pred_cik == filer CIK in FSDS sub.txt (SEC identifier); no hand labels",
        "setting": {"threshold": t, "margin": m},
        "all_matches": precision_of(everything),
        "non_exact_matches": precision_of(non_exact),
        "pairs_file": str(args.pairs),
    }
    for path, data in ((args.stats, report), (args.precision, precision)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n")
    with args.pairs.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(everything + non_exact)

    print(
        json.dumps(
            {
                "chosen": report["chosen"]["tune"],
                "eval": result,
                "eval_by_method": report["eval_by_method"],
                "eval_errors": {k: report["eval_errors"][k] for k in ("wrong", "kinds", "methods")},
                "blocking_eval": report["blocking_eval"],
                "d4_all_matches": precision["all_matches"],
                "d4_non_exact_matches": precision["non_exact_matches"],
                "seconds": report["seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
