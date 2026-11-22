"""Paired bootstrap comparison of two prediction files on the same silver rows (FL-M4 T10).

Two models scored on the same 71 dev rows are strongly correlated: most rows are right (or wrong)
for both. Resampling the rows once and scoring both models on the same resample gives the
distribution of the *difference*, which is much tighter than comparing two separate confidence
intervals. Reported: observed difference (b - a) in category macro-F1 and accuracy, its 95%
bootstrap interval, and the share of resamples where b is not better than a.

Predictions are the `fields generate` outputs, named by their suffix ("" = FL-M2 baseline file,
"base-s1a", a run name, ...). Same scorer, same label set and same treatment of invalid answers as
fields.scores.

    uv run python -m filinglens.train.compare --a base-s1a --b teacher-n1500-r16-lr0.0002-e1
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from filinglens.extract.fields import OUT_DIR, RISK_CATEGORIES, load_rows, macro_f1

N_BOOT = 2000


def predicted_categories(rows: list[dict], preds: dict[str, dict]) -> list[str]:
    """The category per row, 'invalid' for missing or rule-breaking answers (as fields.scores)."""
    out = []
    for r in rows:
        p = preds.get(r["para_id"], {})
        out.append(p.get("category") if p.get("valid") else "invalid")
    return out


def accuracy(gold: list[str], pred: list[str]) -> float:
    return sum(g == p for g, p in zip(gold, pred, strict=True)) / len(gold)


def paired_bootstrap(
    gold: list[str], a: list[str], b: list[str], n: int = N_BOOT, seed: int = 0
) -> dict:
    """Observed b - a and its bootstrap distribution, for macro-F1 and accuracy."""
    import numpy as np

    rng = np.random.default_rng(seed)
    g, pa, pb = (np.asarray(x, dtype=object) for x in (gold, a, b))
    result = {}
    for name, metric in (
        ("category_macro_f1", lambda gg, pp: macro_f1(gg, pp, RISK_CATEGORIES)),
        ("category_accuracy", accuracy),
    ):
        observed = metric(gold, b) - metric(gold, a)
        diffs = []
        for _ in range(n):
            idx = rng.integers(0, len(g), len(g))
            gs = list(g[idx])
            diffs.append(metric(gs, list(pb[idx])) - metric(gs, list(pa[idx])))
        low, high = np.percentile(diffs, [2.5, 97.5])
        result[name] = {
            "a": round(metric(gold, a), 4),
            "b": round(metric(gold, b), 4),
            "diff_b_minus_a": round(observed, 4),
            "diff_ci95": [round(float(low), 4), round(float(high), 4)],
            "share_b_not_better": round(float(np.mean(np.asarray(diffs) <= 0)), 4),
        }
    result["rows"] = len(gold)
    triples = list(zip(a, gold, b, strict=True))
    result["rows_where_only_a_right"] = sum(x == y and z != y for x, y, z in triples)
    result["rows_where_only_b_right"] = sum(z == y and x != y for x, y, z in triples)
    return result


def load_preds(split: str, tag: str, variant: str = "zero") -> dict[str, dict]:
    suffix = f"_{tag}" if tag else ""
    path = OUT_DIR / f"fields_{variant}_{split}{suffix}.jsonl"
    return {r["para_id"]: r for r in map(json.loads, path.read_text().splitlines())}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", required=True, help="prediction suffix of the reference model")
    parser.add_argument("--b", required=True, help="prediction suffix of the model compared")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--n", type=int, default=N_BOOT)
    parser.add_argument("--bench", type=Path, default=Path("bench"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    rows = load_rows(args.split, args.allow_test)
    gold = [r["category"] for r in rows]
    a = predicted_categories(rows, load_preds(args.split, args.a))
    b = predicted_categories(rows, load_preds(args.split, args.b))
    result = {
        "command": "python -m filinglens.train.compare " + " ".join(argv),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "labels": "silver (agreement with ibm-granite/granite-4.1-3b, not correctness)",
        "split": args.split,
        "a": args.a,
        "b": args.b,
        "bootstrap_resamples": args.n,
        **paired_bootstrap(gold, a, b, n=args.n),
    }
    out = args.bench / f"compare_{args.split}_{args.a or 'fl-m2'}_vs_{args.b}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
