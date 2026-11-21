"""Filter the teacher's pool labels into the teacher SFT arm, and report on dev (FL-M4 T6).

The keep rule was fixed on 20 Nov 2026, before any filtered result was looked at. A pool row is
kept only if, in this order:
1. rule_valid: the answer passed silver_label.check (else "rule_invalid");
2. it is a risk: not_a_risk rows are dropped ("not_a_risk"), as in the silver arm;
3. first_token_consistent: the generated token at the category step is the first token computed
   for the chosen category, so the confidence is about the right token ("inconsistent_first_token");
4. confidence of the chosen category >= TAU = 0.9 ("low_confidence"). TAU is fixed, not tuned:
   dev has 71 rows, so a tuned threshold would mostly fit noise. The report shows how many rows
   fall in each confidence band, on the pool and on dev, so the effect of 0.9 is visible;
5. the keyword heuristic does not disagree ("heuristic_disagrees"). It follows the labeling
   guide's precedence rules, applied to the heading (or the first 40 words if there is none):
   a pandemic, climate/greenhouse rules, tariffs/sanctions/war/export controls, or a
   cyberattack/breach named there points to that category. If exactly one pattern fires and the
   teacher chose another category, the row is dropped; otherwise the heuristic abstains.

Dev (silver labels, not used to choose anything): the teacher's scores with the FL-M2 scorer
(fields.scores), agreement of kept rows at several thresholds, and how often the heuristic fires
and agrees with the silver label. The test split is not read here: it is scored once, in T12.

Guards before anything is written: every kept row is from a train-split company and none is a
gold-sample paragraph.

    uv run python -m filinglens.train.filter_teacher

Outputs: data/interim/teacher/filtered.jsonl (rows for `sft_data --teacher`) and
bench/teacher_filter.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from filinglens.extract.fields import scores
from filinglens.extract.gold_set import load_split

TAU = 0.9
BANDS = (0.0, 0.5, 0.9, 0.99, 0.999, 0.9999)
TEACHER_DIR = Path("data/interim/teacher/kaggle_out/teacher")
POOL_FILES = ("pool_first7200_shard0of2.jsonl", "pool_first7200_shard1of2.jsonl")
DEV_FILE = "dev_shard0of1.jsonl"
OUT = Path("data/interim/teacher/filtered.jsonl")
ANSWER_KEYS = ("rationale", "category", "severity", "first_seen_year", "unsure")
HEURISTICS = {
    "health_pandemic": r"covid|coronavirus|pandemic|epidemic",
    "climate_environment": r"climate change|climate-related|greenhouse|carbon emission",
    "geopolitical_international": r"tariff|sanction|\bwars?\b|export control",
    "technology_cyber": r"cyber|data breach|security breach|ransomware|hack",
}
PATTERNS = {category: re.compile(rx) for category, rx in HEURISTICS.items()}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chosen_confidence(record: dict) -> float | None:
    """The teacher's probability for the category it chose, or None if unavailable."""
    confidence = record.get("confidence") or {}
    return confidence.get(record.get("category"))


def heuristic(task: dict) -> str | None:
    """The category the guide's precedence rules point to, or None (no or several matches)."""
    heading = task.get("heading")
    if not isinstance(heading, str) or not heading.strip():
        heading = " ".join(task["text"].split()[:40])
    text = heading.lower()
    fired = [category for category, rx in PATTERNS.items() if rx.search(text)]
    return fired[0] if len(fired) == 1 else None


def drop_reason(record: dict, tau: float = TAU) -> str | None:
    """None if the row is kept, else the first rule it fails (in the documented order)."""
    if not record.get("valid"):
        return "rule_invalid"
    if record.get("category") == "not_a_risk":
        return "not_a_risk"
    if record.get("first_token_consistent") is not True:
        return "inconsistent_first_token"
    confidence = chosen_confidence(record)
    if confidence is None or confidence < tau:
        return "low_confidence"
    expected = heuristic(record["task"])
    if expected is not None and expected != record["category"]:
        return "heuristic_disagrees"
    return None


def answer(record: dict) -> dict:
    return {k: record[k] for k in ANSWER_KEYS}


def filtered_row(record: dict) -> dict:
    """A row in the format train.sft_data.teacher_examples reads."""
    return {
        "para_id": record["para_id"],
        "task": record["task"],
        "answer": answer(record),
        "confidence": chosen_confidence(record),
        "teacher": record.get("teacher"),
    }


def bands(values: list[float | None]) -> dict[str, int]:
    """Counts per confidence band [lo, next lo); None counts as 'none'."""
    out: Counter = Counter()
    for v in values:
        if v is None:
            out["none"] += 1
            continue
        lo = max(b for b in BANDS if v >= b)
        hi = BANDS[BANDS.index(lo) + 1] if lo != BANDS[-1] else 1.0
        out[f"[{lo}, {hi}{']' if hi == 1.0 else ')'}"] += 1
    return dict(sorted(out.items()))


def dev_report(dev_rows: list[dict], dev_records: list[dict]) -> dict:
    """Teacher vs silver on dev: FL-M2 scores, agreement by threshold, heuristic hit rate."""
    preds = {r["para_id"]: r for r in dev_records}
    agreed = [r for r in dev_rows if r["models_agree_category"]]
    by_threshold = {}
    for tau in (0.0, 0.5, 0.9, 0.99):
        kept = [
            r
            for r in dev_rows
            if preds.get(r["para_id"], {}).get("valid")
            and (chosen_confidence(preds[r["para_id"]]) or 0.0) >= tau
        ]
        same = sum(preds[r["para_id"]]["category"] == r["category"] for r in kept)
        by_threshold[str(tau)] = {
            "kept": len(kept),
            "category_agreement_with_silver": round(same / len(kept), 4) if kept else None,
        }
    fired = [(r, heuristic(r)) for r in dev_rows]
    fired = [(r, h) for r, h in fired if h is not None]
    return {
        "labels": "silver (agreement with ibm-granite/granite-4.1-3b, not correctness)",
        "teacher_vs_silver_all_rows": scores(dev_rows, preds),
        "teacher_vs_silver_rows_where_silver_labelers_agree": scores(agreed, preds),
        "kept_by_threshold": by_threshold,
        "confidence_bands": bands([chosen_confidence(r) for r in dev_records if r.get("valid")]),
        "heuristic_fires_on_silver_rows": len(fired),
        "heuristic_agrees_with_silver": sum(h == r["category"] for r, h in fired),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--teacher-dir", type=Path, default=TEACHER_DIR)
    parser.add_argument("--gold", type=Path, default=Path("gold"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--stats", type=Path, default=Path("bench/teacher_filter.json"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    pool_paths = [args.teacher_dir / name for name in POOL_FILES]
    pool = sorted(
        (r for p in pool_paths for r in read_jsonl(p)), key=lambda r: r["task"]["task_no"]
    )
    ids = [r["para_id"] for r in pool]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate para_id across the pool shards")

    reasons = Counter()
    kept = []
    for record in pool:
        reason = drop_reason(record)
        if reason is None:
            kept.append(filtered_row(record))
        else:
            reasons[reason] += 1

    splits = pd.read_csv(args.gold / "splits.csv")
    train_ciks = set(splits.loc[splits["split"] == "train", "cik"].astype(int))
    sample_ids = set(pd.read_csv(args.gold / "risk_sample.csv")["para_id"])
    leaks = {
        "kept_rows_not_from_train_company": sum(
            int(r["task"]["cik"]) not in train_ciks for r in kept
        ),
        "kept_rows_in_gold_sample": sum(r["para_id"] in sample_ids for r in kept),
    }
    if any(leaks.values()):
        raise SystemExit(f"leakage check failed, nothing written: {leaks}")

    dev_path = args.teacher_dir / DEV_FILE
    dev = dev_report(load_split("dev", gold=args.gold), read_jsonl(dev_path))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    valid_risks = [r for r in pool if r.get("valid") and r.get("category") != "not_a_risk"]
    stats = {
        "command": " ".join(["uv run python -m filinglens.train.filter_teacher", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "rule": "valid, not not_a_risk, first token consistent, confidence >= tau, heuristic ok",
        "tau": TAU,
        "tau_chosen": "fixed in advance (20 Nov 2026), not tuned on dev",
        "heuristics": HEURISTICS,
        "pool_rows": len(pool),
        "kept": len(kept),
        "dropped": dict(sorted(reasons.items())),
        "kept_categories": dict(Counter(r["answer"]["category"] for r in kept).most_common()),
        "pool_confidence_bands_valid_risks": bands([chosen_confidence(r) for r in valid_risks]),
        "pool_heuristic_fires_on_valid_risks": sum(
            heuristic(r["task"]) is not None for r in valid_risks
        ),
        "leakage": leaks,
        "dev": dev,
        "inputs": {p.name: sha256_file(p) for p in [*pool_paths, dev_path]},
        "output": args.out.as_posix(),
        "output_sha256": sha256_file(args.out),
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "inputs"}, indent=2))


if __name__ == "__main__":
    main()
