"""Agreement between the two silver labelers: Cohen's kappa with bootstrap CIs (FL-M2 T12).

This replaces the spec's human test-retest kappa (decision of 19 Nov 2026, see gold/MANIFEST.json):
it measures how often two models from different families agree beyond chance, not how reliable
a human labeler is.

Kappa compares observed agreement with the agreement expected if both raters labeled at random
with their own label frequencies: kappa = 1 - (weighted observed disagreement) / (weighted
disagreement expected by chance). 1 is perfect agreement, 0 is chance level, below 0 is worse
than chance. Weighted kappa (linear or quadratic) gives partial credit to near misses on ordered
labels (severity 1-3, years), so a 2-vs-3 disagreement costs less than 1-vs-3.

    uv run python -m filinglens.extract.agreement      # writes bench/kappa.json
"""

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

RAW = Path("gold/silver_raw")
FILES = {
    "primary": "ibm-granite-granite-4-1-3b.jsonl",
    "check": "qwen-qwen3-4b-awq.jsonl",
}
BOOTSTRAP = 2000
SEED = 20261119


# ---------------------------------------------------------------------------------------------
# Hand-written for FL-M2 (live-coding practice): cohen_kappa below.
# ---------------------------------------------------------------------------------------------
def cohen_kappa(a: Sequence, b: Sequence, weights: str | None = None) -> float:
    """Cohen's kappa between two raters' labels for the same items.

    a, b: equal-length sequences of labels (any sortable, hashable values).
    weights: None (every disagreement costs 1), "linear" (cost |i - j|) or "quadratic"
    (cost (i - j)**2), where i and j are the positions of the two labels in the sorted list of
    all labels either rater used.
    Returns nan when the expected disagreement is 0 (both raters always gave the same single
    label), like sklearn. Raises ValueError if the lengths differ or are 0, or weights is unknown.
    """
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("a and b must be non-empty sequences of equal length")
    if weights not in (None, "linear", "quadratic"):
        raise ValueError(f"unknown weights: {weights!r}")

    labels = sorted(set(a) | set(b))
    k = len(labels)
    index = {label: i for i, label in enumerate(labels)}

    ia = np.fromiter((index[x] for x in a), dtype=np.intp, count=len(a))
    ib = np.fromiter((index[y] for y in b), dtype=np.intp, count=len(b))
    observed = np.bincount(ia * k + ib, minlength=k * k).reshape(k, k)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / len(a)

    grid = np.arange(k)
    diff = np.abs(grid[:, None] - grid[None, :])
    if weights is None:
        w = (diff != 0).astype(float)
    elif weights == "linear":
        w = diff.astype(float)
    else:
        w = (diff**2).astype(float)

    denom = float(np.sum(w * expected))
    if denom == 0.0:
        return float("nan")
    return 1.0 - float(np.sum(w * observed)) / denom


# ---------------------------------------------------------------------------------------------


def bootstrap_ci(a: list, b: list, weights: str | None, n: int = BOOTSTRAP, seed: int = SEED):
    """Percentile 95% CI of kappa over n resamples of the items (with replacement)."""
    rng = np.random.default_rng(seed)
    a_arr, b_arr = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(a_arr), len(a_arr))
        value = cohen_kappa(list(a_arr[idx]), list(b_arr[idx]), weights)
        if not math.isnan(value):  # a resample where both raters used one label has no kappa
            values.append(value)
    if not values:
        return None, None
    low, high = np.percentile(values, [2.5, 97.5])
    return round(float(low), 3), round(float(high), 3)


def summary(a: list, b: list, weights: str | None) -> dict:
    kappa = cohen_kappa(a, b, weights)
    return {
        "n": len(a),
        "observed_agreement": round(sum(x == y for x, y in zip(a, b, strict=True)) / len(a), 3),
        "kappa": None if math.isnan(kappa) else round(kappa, 3),  # null in JSON, not NaN
        "ci95": bootstrap_ci(a, b, weights),
        "weights": weights or "none",
    }


def paired(primary: list[dict], check: list[dict]) -> list[tuple[dict, dict]]:
    """Answer pairs for paragraphs where both labelers gave a rule-following answer."""
    other = {r["para_id"]: r for r in check}
    return [
        (p, other[p["para_id"]]) for p in primary if p["valid"] and other[p["para_id"]]["valid"]
    ]


def agreement_report(pairs: list[tuple[dict, dict]]) -> dict:
    cat_a = [p["category"] for p, _ in pairs]
    cat_b = [q["category"] for _, q in pairs]
    risks = [(p, q) for p, q in pairs if "not_a_risk" not in (p["category"], q["category"])]
    sev_a, sev_b = [p["severity"] for p, _ in risks], [q["severity"] for _, q in risks]
    year_a = [p["first_seen_year"] for p, _ in risks]
    year_b = [q["first_seen_year"] for _, q in risks]
    confusions = Counter(tuple(sorted((x, y))) for x, y in zip(cat_a, cat_b, strict=True) if x != y)
    return {
        "category": summary(cat_a, cat_b, None),
        "severity": summary(sev_a, sev_b, None),
        "severity_quadratic": summary(sev_a, sev_b, "quadratic"),
        "first_seen_year_quadratic": summary(year_a, year_b, "quadratic"),
        "top_category_disagreements": [
            {"pair": list(pair), "count": n} for pair, n in confusions.most_common(8)
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--out", type=Path, default=Path("bench/kappa.json"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    def read(name: str) -> list[dict]:
        return [json.loads(line) for line in (args.raw / name).read_text().splitlines() if line]

    pairs = paired(read(FILES["primary"]), read(FILES["check"]))
    report = {
        "command": " ".join(["uv run python -m filinglens.extract.agreement", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "what": "agreement between two model labelers (silver), not human test-retest",
        "raters": {"a": "ibm-granite/granite-4.1-3b (FP8)", "b": "Qwen/Qwen3-4B-AWQ"},
        "bootstrap": {"resamples": BOOTSTRAP, "seed": SEED},
        **agreement_report(pairs),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
