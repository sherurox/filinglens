"""More ret-test queries from unused paragraphs of the ret-test companies (FL-M3 T10a).

The power analysis (bench/ab_power.json) showed that 175 ret-test queries can't detect a 0.03
NDCG@10 gain, and that more companies aren't available (FL-M2's dev and test companies must not
be used for prompts). So the A/B gets more queries per ret-test company: about TARGET_PER_COMPANY
each. This module is FL-M2's synthetic Q&A pipeline (extract.synthetic_qa) pointed at a new pool:
same generator, prompt v2, greedy schema-constrained decoding, pair checks and leakage test. Only
the sampling is new, and every output goes to new paths: FL-M2's files are never touched.

Pool, in this order:
- paragraphs of the 75 ret-test companies (the CIKs in gold/retrieval/queries.test.jsonl), passing
  FL-M2's `eligible()` filter;
- minus any paragraph whose text also appears at an FL-M2 dev or test company (FL-M2's rule);
- minus every paragraph the existing synthetic Q&A was written from, and any other paragraph with
  the same text (a copy in another year would give a second query with the same answer);
- one paragraph per distinct text (duplicates within the pool would do the same).

Per company: ceil((TARGET_PER_COMPANY - queries it already has) / YIELD) paragraphs, at most one
per filing per round, round-robin over its filings in a seeded order, so a company's new queries
spread over its years. YIELD is FL-M2's measured share of paragraphs that gave at least one kept
pair (357 of 420).

    uv run python -m filinglens.retrieval.extra_queries prepare
    ~/venvs/fl-gpu/bin/python -m filinglens.retrieval.extra_queries generate \\
        --model Qwen/Qwen3-4B-AWQ --gpu-memory-utilization 0.80 --enforce-eager
    uv run python -m filinglens.retrieval.extra_queries finalize

Top-level imports are standard library only: `generate` runs in the GPU environment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from filinglens.extract import synthetic_qa as sq

TARGET_PER_COMPANY = 15
YIELD = 357 / 420  # FL-M2: paragraphs with at least one kept pair / paragraphs drawn
SEED = 20261124
BATCH = "ret_test_extra_v1"
WORK = Path("data/interim/synthetic_ret_test")
TASKS = WORK / "qa_tasks.jsonl"
OUTPUT = Path("gold/synthetic/risk_qa.ret_test_extra.jsonl")
EXISTING_QA = Path("gold/synthetic/risk_qa.train.jsonl")
TEST_QUERIES = Path("gold/retrieval/queries.test.jsonl")
BENCH = Path("bench/synthetic_qa_ret_test_extra.json")
BENCH_RUN = Path("bench/synthetic_qa_ret_test_extra_run.json")
COLUMNS = ["para_id", "cik", "ticker", "sector", "period_year", "accession", "heading", "text"]


def per_company_draws(have: dict[int, int], target: int = TARGET_PER_COMPANY) -> dict[int, int]:
    """Paragraphs to draw per company so that, at FL-M2's yield, it reaches about `target`."""
    return {cik: max(0, math.ceil((target - n) / YIELD)) for cik, n in have.items()}


def spread_draw(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Up to n rows of one company: round-robin over its filings (seeded order), one per filing
    per round, so the draws cover its years before taking a second paragraph from any filing."""
    by_filing: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["para_id"]):
        by_filing[row["accession"]].append(row)
    for filing_rows in by_filing.values():
        rng.shuffle(filing_rows)
    filings = sorted(by_filing)
    rng.shuffle(filings)
    picked: list[dict] = []
    while len(picked) < n and any(by_filing[f] for f in filings):
        for filing in filings:
            if by_filing[filing] and len(picked) < n:
                picked.append(by_filing[filing].pop())
    return picked


def build_pool(corpus, splits, test_ciks: set[int], used_para_ids: set[str]) -> tuple[list, dict]:
    """Eligible, leakage-free, not-yet-used, text-unique paragraphs of the ret-test companies."""
    from filinglens.extract.gold_sample import eligible

    held = set(splits.loc[splits["split"] != "train", "cik"].astype(int))
    held_keys = set(corpus.loc[corpus["cik"].isin(held), "text"].map(sq.text_key))
    used_keys = set(corpus.loc[corpus["para_id"].isin(used_para_ids), "text"].map(sq.text_key))
    pool = eligible(corpus)
    pool = pool[pool["cik"].isin(test_ciks)].copy()
    pool["text_key"] = pool["text"].map(sq.text_key)
    stats = {"eligible_test_company_paragraphs": len(pool)}
    shared = pool["text_key"].isin(held_keys)
    stats["dropped_text_also_at_m2_dev_or_test_company"] = int(shared.sum())
    pool = pool[~shared]
    reused = pool["text_key"].isin(used_keys)
    stats["dropped_text_of_an_existing_qa_paragraph"] = int(reused.sum())
    pool = pool[~reused]
    before = len(pool)
    pool = pool.sort_values("para_id").drop_duplicates("text_key", keep="first")
    stats["dropped_duplicate_text_within_pool"] = before - len(pool)
    stats["pool_paragraphs"] = len(pool)
    return pool[[*COLUMNS, "text_key"]].to_dict("records"), stats


def prepare(corpus, splits, test_queries: list[dict], existing_qa: list[dict], seed: int = SEED):
    have = Counter(int(q["cik"]) for q in test_queries)
    used = {r["para_id"] for r in existing_qa}
    pool, stats = build_pool(corpus, splits, set(have), used)
    by_company: dict[int, list[dict]] = defaultdict(list)
    for row in pool:
        by_company[int(row["cik"])].append(row)
    wanted = per_company_draws(dict(have))
    rng = random.Random(seed)
    picked = []
    for cik in sorted(wanted):
        picked.extend(spread_draw(by_company.get(cik, []), wanted[cik], rng))
    tasks = [
        {
            "task_no": i,
            **{k: (int(r[k]) if k in ("cik", "period_year") else r[k]) for k in COLUMNS},
            "text_key": r["text_key"],
        }
        for i, r in enumerate(picked, start=1)
    ]
    short = {c: wanted[c] - sum(t["cik"] == c for t in tasks) for c in wanted}
    stats.update(
        {
            "test_companies": len(have),
            "existing_test_queries": sum(have.values()),
            "target_per_company": TARGET_PER_COMPANY,
            "assumed_yield": round(YIELD, 4),
            "paragraphs_wanted": sum(wanted.values()),
            "tasks": len(tasks),
            "companies_short_of_paragraphs": sum(v > 0 for v in short.values()),
            "by_sector": dict(sorted(Counter(t["sector"] for t in tasks).items())),
            "by_year": dict(sorted(Counter(str(t["period_year"]) for t in tasks).items())),
        }
    )
    return tasks, stats


def finalize_rows(tasks, raws, generator, splits, corpus, test_ciks, existing_qa):
    """FL-M2's checks and leakage test, plus: only ret-test companies, no reused paragraph."""
    kept, checked = sq.build_rows(tasks, raws, generator)
    held = set(splits.loc[splits["split"] != "train", "cik"].astype(int))
    held_keys = set(corpus.loc[corpus["cik"].isin(held), "text"].map(sq.text_key))
    train = set(splits.loc[splits["split"] == "train", "cik"].astype(int))
    leaks = sq.leakage(kept, train, held_keys, {t["para_id"]: t for t in tasks})
    used = {r["para_id"] for r in existing_qa}
    leaks["rows_from_non_ret_test_companies"] = sum(r["cik"] not in test_ciks for r in kept)
    leaks["rows_reusing_an_existing_qa_paragraph"] = sum(r["para_id"] in used for r in kept)
    for row in kept:
        row["batch"] = BATCH
    return kept, checked, leaks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "finalize"):
        p = sub.add_parser(name)
        p.add_argument("--corpus", type=Path, default=Path("data/interim/risk_paragraphs.parquet"))
        p.add_argument("--splits", type=Path, default=Path("gold/splits.csv"))
    sub.choices["finalize"].add_argument("--model", default="Qwen/Qwen3-4B-AWQ")
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--quantization", default=None)
    g.add_argument("--limit", type=int, default=None, help="first N tasks (smoke test)")
    g.add_argument("--max-model-len", type=int, default=4096)
    g.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    g.add_argument("--enforce-eager", action="store_true")
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    command = " ".join(["python -m filinglens.retrieval.extra_queries", *argv])

    if args.command == "prepare":
        import pandas as pd

        tasks, stats = prepare(
            pd.read_parquet(args.corpus),
            pd.read_csv(args.splits),
            sq.read_jsonl(TEST_QUERIES),  # CIKs only: metadata, no results
            sq.read_jsonl(EXISTING_QA),
        )
        sq.write_jsonl(tasks, TASKS)
        stats = {"command": command, "seed": SEED, **stats}
        (WORK / "prepare.json").write_text(json.dumps(stats, indent=2) + "\n")
        print(json.dumps(stats, indent=2))
        return

    if args.command == "generate":
        from filinglens.extract.silver_label import PeakGpuMemory, slug

        tasks = sq.read_jsonl(TASKS)[: args.limit]
        started = time.perf_counter()
        with PeakGpuMemory() as peak:
            raws = sq.generate(tasks, args, sq.PROMPT.read_text())
        elapsed = round(time.perf_counter() - started, 1)
        name = slug(args.model) + (f"-first{args.limit}" if args.limit else "")
        rows = [{"task_no": t["task_no"], "raw": r} for t, r in zip(tasks, raws, strict=True)]
        sq.write_jsonl(rows, WORK / f"qa_raw_{name}.jsonl")
        run = {
            "command": command,
            "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "model": args.model,
            "prompt_version": sq.PROMPT_VERSION,
            "quantization": args.quantization,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "tasks": len(tasks),
            "elapsed_s_incl_model_load": elapsed,
            "peak_gpu_memory_mib": peak.peak_mib,
        }
        if not args.limit:
            BENCH_RUN.write_text(json.dumps(run, indent=2) + "\n")
        print(json.dumps(run, indent=2))
        return

    import pandas as pd

    from filinglens.extract.silver_label import slug

    tasks = sq.read_jsonl(TASKS)
    raws = {
        r["task_no"]: r["raw"] for r in sq.read_jsonl(WORK / f"qa_raw_{slug(args.model)}.jsonl")
    }
    test_ciks = {int(q["cik"]) for q in sq.read_jsonl(TEST_QUERIES)}
    kept, checked, leaks = finalize_rows(
        tasks,
        raws,
        args.model,
        pd.read_csv(args.splits),
        pd.read_parquet(args.corpus, columns=["para_id", "cik", "text"]),
        test_ciks,
        sq.read_jsonl(EXISTING_QA),
    )
    if any(leaks.values()):
        raise SystemExit(f"leakage test failed, nothing written: {leaks}")
    sq.write_jsonl(checked, WORK / "qa_checked.jsonl")
    sq.write_jsonl(kept, OUTPUT)
    per_company = Counter(r["cik"] for r in kept)
    stats = {
        "command": command,
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "generator": args.model,
        "prompt": str(sq.PROMPT),
        "prompt_version": sq.PROMPT_VERSION,
        "batch": BATCH,
        "prepare": json.loads((WORK / "prepare.json").read_text()),
        "pairs_generated": len(checked),
        "pairs_kept": len(kept),
        "kept_rate": round(len(kept) / len(checked), 4) if checked else None,
        "problems": dict(Counter(p for c in checked for p in c["problems"]).most_common()),
        "kept_paragraphs": len({r["para_id"] for r in kept}),
        "paragraph_yield": round(len({r["para_id"] for r in kept}) / len(tasks), 4)
        if tasks
        else None,
        "kept_companies": len(per_company),
        "kept_by_type": dict(Counter(r["type"] for r in kept).most_common()),
        "leakage": leaks,
        "output": str(OUTPUT),
        "output_sha256": sq.hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    }
    BENCH.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "prepare"}, indent=2))


if __name__ == "__main__":
    main()
