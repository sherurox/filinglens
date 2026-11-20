"""Inputs for the FL-M4 teacher run (Qwen/Qwen3-8B-AWQ on Kaggle) (FL-M4 T3).

Pool: unlabeled paragraphs of train-split companies only, from the same eligible set as the gold
sample (gold_sample.eligible: 40-400 words, the sampled years, no summary bullets), minus
- the 330 gold-sample paragraphs: they are the silver arm, and the teacher arm must not repeat them;
- any paragraph whose text (case/space-normalized) also appears at a dev or test company
  (synthetic_qa.text_key), so copied boilerplate can't carry test text into training;
- repeats of the same text (a company's boilerplate copied year to year): the first one drawn wins;
- more than PER_FILING paragraphs from one filing, so long filings don't dominate a cell.

Rows are written round-robin over the 30 sector x year cells, so every prefix of pool.jsonl is
still balanced: the teacher run labels the first N rows, with N chosen after its smoke test.
Each row carries the same earlier-year candidates as the gold tasks (same rules as
gold_sample.first_seen), so the teacher sees exactly the prompt the silver labeler and the
FL-M2 baseline saw (silver_label.user_message).

Also written: the dev tasks (to choose the filter threshold) and, only with --allow-test, the test
tasks, which the teacher labels once for the final comparison. Before anything is written, a
leakage check must find 0 pool rows from a non-train company, 0 with text found at a dev/test
company and 0 from the gold sample.

    uv run python -m filinglens.train.teacher_inputs --allow-test

Outputs: data/interim/teacher/inputs/{pool,dev,test}.jsonl, silver_prompt.md and MANIFEST.json
(the folder that becomes a private Kaggle dataset), and bench/teacher_inputs.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from filinglens.extract.gold_sample import (
    CONTEXT_WORDS,
    FIRST_YEAR,
    SUGGEST_THRESHOLD,
    _context,
    content_words,
    eligible,
    overlap,
)
from filinglens.extract.gold_set import load_split
from filinglens.extract.synthetic_qa import text_key

SEED = 20261120
PER_CELL = 300  # 6 sectors x 5 years x 300 = 9,000 rows; the run labels a prefix
PER_FILING = 10
OUT = Path("data/interim/teacher/inputs")
PROMPT = Path("docs/labeling/silver_prompt.md")
TASKS = Path("data/interim/label_tasks/risk_tasks.json")
ROW_KEYS = ("para_id", "cik", "ticker", "sector", "period_year", "accession", "heading", "text")
OUTPUT_FILES = ("pool.jsonl", "dev.jsonl", "test.jsonl", "silver_prompt.md", "MANIFEST.json")


# ---------- earlier-year candidates ----------


def candidate_index(company: pd.DataFrame) -> dict[int, list[dict]]:
    """One company's paragraphs by year, in corpus order, with their context words computed once
    (gold_sample.first_seen recomputes them for every query: too slow for thousands of rows)."""
    index: dict[int, list[dict]] = {}
    for r in company[["para_id", "period_year", "heading", "text"]].to_dict("records"):
        index.setdefault(int(r["period_year"]), []).append(
            r | {"_words": content_words(_context(r))}
        )
    return index


def first_seen(row: dict, index: dict[int, list[dict]]) -> dict:
    """Same result as gold_sample.first_seen(row, company): the best match in each earlier year
    (first one on ties, like idxmax) and the earliest year at or above the threshold."""
    query = content_words(_context(row))
    candidates = []
    for year in range(FIRST_YEAR, int(row["period_year"])):
        same_year = index.get(year)
        if not same_year:
            continue
        best, best_score = same_year[0], overlap(query, same_year[0]["_words"])
        for r in same_year[1:]:
            score = overlap(query, r["_words"])
            if score > best_score:
                best, best_score = r, score
        candidates.append(
            {
                "year": year,
                "score": round(float(best_score), 3),
                "para_id": best["para_id"],
                "heading": best["heading"],
                "text": " ".join(best["text"].split()[:CONTEXT_WORDS]),
            }
        )
    above = [c["year"] for c in candidates if c["score"] >= SUGGEST_THRESHOLD]
    return {"suggested": min(above, default=int(row["period_year"])), "candidates": candidates}


# ---------- pool and draw ----------


def build_pool(
    corpus: pd.DataFrame, splits: pd.DataFrame, sample_ids: set[str]
) -> tuple[pd.DataFrame, dict, set[int], set[str]]:
    """Eligible train-company paragraphs without gold-sample rows or dev/test-shared text.
    Returns the pool, its counts, the train CIKs and the text keys found at dev/test companies."""
    train = set(splits.loc[splits["split"] == "train", "cik"].astype(int))
    held = set(splits.loc[splits["split"] != "train", "cik"].astype(int))
    held_keys = set(corpus.loc[corpus["cik"].isin(held), "text"].map(text_key))
    pool = eligible(corpus)
    pool = pool[pool["cik"].isin(train)]
    n_eligible = len(pool)
    in_sample = pool["para_id"].isin(sample_ids)
    pool = pool[~in_sample]
    keys = pool["text"].map(text_key)
    shared = keys.isin(held_keys)
    pool = pool[~shared].assign(text_key=keys[~shared])
    stats = {
        "train_companies": len(train),
        "held_out_companies": len(held),
        "eligible_train_paragraphs": n_eligible,
        "dropped_in_gold_sample": int(in_sample.sum()),
        "dropped_text_also_at_dev_or_test_company": int(shared.sum()),
        "pool_paragraphs": len(pool),
    }
    return pool, stats, train, held_keys


def draw(pool: pd.DataFrame, per_cell: int, per_filing: int, seed: int) -> tuple[list[dict], dict]:
    """Up to per_cell rows per (sector, year), seeded; no repeated text, at most per_filing per
    filing; returned round-robin over the cells (cell order sorted) so every prefix is balanced."""
    rng = random.Random(seed)
    cells: dict[tuple[str, int], list[dict]] = {}
    for r in sorted(pool[[*ROW_KEYS, "text_key"]].to_dict("records"), key=lambda r: r["para_id"]):
        cells.setdefault((r["sector"], int(r["period_year"])), []).append(r)
    seen: set[str] = set()
    per_accession: Counter = Counter()
    skipped: Counter = Counter()
    picks: dict[tuple[str, int], list[dict]] = {}
    for cell in sorted(cells):
        rows = cells[cell][:]
        rng.shuffle(rows)
        chosen: list[dict] = []
        for r in rows:
            if len(chosen) == per_cell:
                break
            if r["text_key"] in seen:
                skipped["repeated_text"] += 1
                continue
            if per_accession[r["accession"]] >= per_filing:
                skipped["filing_cap"] += 1
                continue
            seen.add(r["text_key"])
            per_accession[r["accession"]] += 1
            chosen.append(r)
        picks[cell] = chosen
    order = [
        picks[cell][i]
        for i in range(max((len(c) for c in picks.values()), default=0))
        for cell in sorted(picks)
        if i < len(picks[cell])
    ]
    stats = {
        "cells": len(picks),
        "cells_short_of_per_cell": {
            f"{s} {y}": len(c) for (s, y), c in sorted(picks.items()) if len(c) < per_cell
        },
        "skipped_while_drawing": dict(sorted(skipped.items())),
    }
    return order, stats


def leakage(rows: list[dict], train: set[int], held_keys: set[str], sample_ids: set[str]) -> dict:
    """Counts that must all be 0 before anything is written (text keys recomputed from text)."""
    return {
        "rows_not_from_train_company": sum(int(r["cik"]) not in train for r in rows),
        "rows_with_text_at_dev_or_test_company": sum(
            text_key(r["text"]) in held_keys for r in rows
        ),
        "rows_in_gold_sample": sum(r["para_id"] in sample_ids for r in rows),
    }


def pool_tasks(order: list[dict], corpus: pd.DataFrame) -> list[dict]:
    """Task records in draw order, each with its earlier-year candidates."""
    by_company = {}
    tasks = []
    for task_no, r in enumerate(order, start=1):
        cik = int(r["cik"])
        if cik not in by_company:
            by_company[cik] = candidate_index(corpus[corpus["cik"] == cik])
        row = {k: r[k] for k in ROW_KEYS} | {"cik": cik, "period_year": int(r["period_year"])}
        tasks.append(
            {"task_no": task_no, **row, "text_key": r["text_key"], "split": "train"}
            | {"first_seen": first_seen(row, by_company[cik])}
        )
    return tasks


def held_out_tasks(split: str, tasks: dict[str, dict], gold: Path, allow_test: bool) -> list[dict]:
    """The gold tasks (with their candidates) of the silver rows of one split."""
    rows = load_split(split, gold=gold, allow_test=allow_test)
    return [tasks[r["para_id"]] | {"split": split} for r in rows]


# ---------- output ----------


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def describe(tasks: list[dict]) -> dict:
    n_words = [len(t["text"].split()) for t in tasks]
    return {
        "rows": len(tasks),
        "companies": len({t["cik"] for t in tasks}),
        "filings": len({t["accession"] for t in tasks}),
        "by_sector": dict(sorted(Counter(t["sector"] for t in tasks).items())),
        "by_year": dict(sorted(Counter(str(t["period_year"]) for t in tasks).items())),
        "words_median": statistics.median(n_words) if n_words else None,
        "words_p95": sorted(n_words)[int(0.95 * (len(n_words) - 1))] if n_words else None,
        "with_earlier_year_candidates": sum(bool(t["first_seen"]["candidates"]) for t in tasks),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=Path("data/interim/risk_paragraphs.parquet"))
    parser.add_argument("--gold", type=Path, default=Path("gold"))
    parser.add_argument("--tasks", type=Path, default=TASKS)
    parser.add_argument("--prompt", type=Path, default=PROMPT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--stats", type=Path, default=Path("bench/teacher_inputs.json"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--per-cell", type=int, default=PER_CELL)
    parser.add_argument("--per-filing", type=int, default=PER_FILING)
    parser.add_argument("--allow-test", action="store_true", help="also write test.jsonl")
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    corpus = pd.read_parquet(args.corpus)
    splits = pd.read_csv(args.gold / "splits.csv")
    sample_ids = set(pd.read_csv(args.gold / "risk_sample.csv")["para_id"])
    pool, pool_stats, train, held_keys = build_pool(corpus, splits, sample_ids)
    order, draw_stats = draw(pool, args.per_cell, args.per_filing, args.seed)
    leaks = leakage(order, train, held_keys, sample_ids)
    if any(leaks.values()):
        raise SystemExit(f"leakage check failed, nothing written: {leaks}")
    tasks = pool_tasks(order, corpus)

    gold_tasks = {t["para_id"]: t for t in json.loads(args.tasks.read_text())}
    held = {"dev": held_out_tasks("dev", gold_tasks, args.gold, allow_test=False)}
    if args.allow_test:
        held["test"] = held_out_tasks("test", gold_tasks, args.gold, allow_test=True)

    args.out.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILES:  # never leave a stale test.jsonl or manifest behind
        (args.out / name).unlink(missing_ok=True)
    write_jsonl(tasks, args.out / "pool.jsonl")
    for split, rows in held.items():
        write_jsonl(rows, args.out / f"{split}.jsonl")
    shutil.copy(args.prompt, args.out / "silver_prompt.md")
    files = sorted(p for p in args.out.iterdir() if p.name in OUTPUT_FILES)

    command = " ".join(["uv run python -m filinglens.train.teacher_inputs", *argv])
    summary = {
        "command": command,
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": args.seed,
        "per_cell": args.per_cell,
        "per_filing": args.per_filing,
        "pool": pool_stats,
        "draw": draw_stats,
        "teacher_pool": describe(tasks),
        "held_out_rows": {split: len(rows) for split, rows in held.items()},
        "test_written": args.allow_test,
        "leakage": leaks,
    }
    manifest = summary | {"files": {p.name: sha256_file(p) for p in files}}
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stats = manifest | {"manifest_sha256": sha256_file(args.out / "MANIFEST.json")}
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
