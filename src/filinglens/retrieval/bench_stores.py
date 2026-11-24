"""Compare the four vector stores on 100 ret-dev queries: quality and latency (FL-M3 T7).

Two modes per store, same 100 queries (a seeded sample of the 182 ret-dev queries), k = 10:
- filtered (the task): search within the query's company; recall@10, MRR@10 and NDCG@10 against
  the graded chunks, overlap@10 with FAISS Flat filtered (exact), the share of the top 10 that
  are grade-1 copies, and how many searches returned fewer than 10 results;
- unfiltered (index quality): search all 203,200 chunks; overlap@10 with FAISS Flat (exact).
  Not scored against the grades: T4b showed cross-company search isn't the task.

Latency: per store and mode, 2 warm-up passes, then 5 timed passes over the 100 queries
(500 samples) -> p50 / p95 / p99 / mean in ms, client-observed (FAISS in process; pgvector and
Qdrant through their client libraries over localhost). Query embedding isn't included (T4).
The garbage collector is paused during timed passes.

Check: FAISS Flat filtered, scored on all 182 ret-dev queries, must reproduce T4b's exact
within-company recall@10 (bench/retrieval_dev_diagnostics.json); the run stops otherwise.

Outputs: bench/vector_stores.csv, bench/vector_stores.json, per-query top-10s in
data/interim/retrieval/store_results.jsonl, one MLflow run.

    uv run python -m filinglens.retrieval.bench_stores --note "AC power, Best performance"
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import shlex
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from filinglens.retrieval.build_indexes import FAISS_DIR
from filinglens.retrieval.chunks import sha256_file
from filinglens.retrieval.config import CHUNKS, EMB_DIR, RET_DIR
from filinglens.retrieval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from filinglens.retrieval.queries import OUT_DIR as QUERY_DIR
from filinglens.retrieval.queries import load_queries
from filinglens.retrieval.runlog import log_run
from filinglens.retrieval.stores import FaissStore, PgVectorStore, QdrantStore, overlap_at_k

K = 10
SAMPLE, SAMPLE_SEED = 100, 0
WARMUP, PASSES = 2, 5
MODES = ("filtered", "unfiltered")
STORES = ("faiss-flat", "faiss-hnsw", "pgvector", "qdrant")
REFERENCE = "faiss-flat"
REPRODUCE_TOLERANCE = 1e-4
CSV_COLUMNS = [
    "store",
    "mode",
    "queries",
    "recall_at_10",
    "mrr_at_10",
    "ndcg_at_10",
    "overlap_at_10_exact",
    "grade1_share_top10",
    "short_results",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "mean_ms",
    "samples",
]


# ---- pure helpers -----------------------------------------------------------------------------


def sample_queries(queries: Sequence[dict], n: int = SAMPLE, seed: int = SAMPLE_SEED) -> list:
    """A seeded sample of n queries, kept in their file order."""
    if n >= len(queries):
        return list(queries)
    keep = set(random.Random(seed).sample(range(len(queries)), n))
    return [q for i, q in enumerate(queries) if i in keep]


def search_all(store, vectors: np.ndarray, queries: Sequence[dict], mode: str, k: int = K):
    """One top-k list per query; filtered mode passes the query's CIK."""
    return [
        store.search(vec, k, cik=q["cik"] if mode == "filtered" else None)
        for q, vec in zip(queries, vectors, strict=True)
    ]


def time_searches(
    store,
    vectors: np.ndarray,
    queries: Sequence[dict],
    mode: str,
    k: int = K,
    warmup: int = WARMUP,
    passes: int = PASSES,
) -> list[float]:
    """Milliseconds per search over `passes` timed passes, after `warmup` untimed passes."""
    for _ in range(warmup):
        search_all(store, vectors, queries, mode, k)
    samples: list[float] = []
    gc.disable()
    try:
        for _ in range(passes):
            for q, vec in zip(queries, vectors, strict=True):
                cik = q["cik"] if mode == "filtered" else None
                start = time.perf_counter_ns()
                store.search(vec, k, cik=cik)
                samples.append((time.perf_counter_ns() - start) / 1e6)
    finally:
        gc.enable()
    return samples


def latency_summary(ms: Sequence[float]) -> dict:
    arr = np.asarray(ms, dtype=float)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    return {
        "p50_ms": round(float(p50), 3),
        "p95_ms": round(float(p95), 3),
        "p99_ms": round(float(p99), 3),
        "mean_ms": round(float(arr.mean()), 3),
        "samples": int(arr.size),
    }


def quality(
    results: Sequence[Sequence[str]],
    reference: Sequence[Sequence[str]],
    queries: Sequence[dict],
    mode: str,
    k: int = K,
) -> dict:
    """Mean metrics over the queries; graded metrics only for the filtered (task) mode."""
    out = {
        "queries": len(queries),
        "overlap_at_10_exact": round(
            float(
                np.mean([overlap_at_k(r, e, k) for r, e in zip(results, reference, strict=True)])
            ),
            4,
        ),
        "short_results": sum(len(r) < k for r in results),
    }
    if mode == "filtered":
        grades = [q["grades"] for q in queries]
        pairs = list(zip(results, grades, strict=True))
        out["recall_at_10"] = round(float(np.mean([recall_at_k(r, g, k) for r, g in pairs])), 4)
        out["mrr_at_10"] = round(float(np.mean([mrr_at_k(r, g, k) for r, g in pairs])), 4)
        out["ndcg_at_10"] = round(float(np.mean([ndcg_at_k(r, g, k) for r, g in pairs])), 4)
        share = [sum(g.get(c) == 1 for c in r[:k]) / k for r, g in pairs]
        out["grade1_share_top10"] = round(float(np.mean(share)), 4)
    return out


def csv_rows(table: dict) -> list[dict]:
    rows = []
    for store, modes in table.items():
        for mode, values in modes.items():
            rows.append(
                {"store": store, "mode": mode, **{c: values.get(c, "") for c in CSV_COLUMNS[2:]}}
            )
    return rows


# ---- run --------------------------------------------------------------------------------------


def open_store(name: str, ids: list[str], ciks: np.ndarray, faiss_dir: Path):
    if name.startswith("faiss-"):
        kind = name.removeprefix("faiss-")
        return FaissStore.load(kind, faiss_dir / f"{kind}.index", ids, ciks)
    if name == "pgvector":
        return PgVectorStore()
    if name == "qdrant":
        return QdrantStore.load(ids)
    raise ValueError(f"unknown store {name!r}")


def load_dev(emb_dir: Path, query_dir: Path) -> tuple[list[dict], np.ndarray]:
    queries = load_queries("dev", query_dir)
    ids = (emb_dir / "query_ids.dev.txt").read_text().splitlines()
    if ids != [q["query_id"] for q in queries]:
        raise ValueError("query vectors and query file are in different orders")
    return queries, np.load(emb_dir / "queries.dev.npy")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stores", nargs="+", choices=STORES, default=list(STORES))
    parser.add_argument("--emb-dir", type=Path, default=EMB_DIR)
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--query-dir", type=Path, default=QUERY_DIR)
    parser.add_argument("--faiss-dir", type=Path, default=FAISS_DIR)
    parser.add_argument(
        "--diagnostics", type=Path, default=Path("bench/retrieval_dev_diagnostics.json")
    )
    parser.add_argument("--bench", type=Path, default=Path("bench"))
    parser.add_argument("--results", type=Path, default=RET_DIR / "store_results.jsonl")
    parser.add_argument("--note", required=True, help="power source and Windows power mode")
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    if REFERENCE not in args.stores:
        raise SystemExit(f"{REFERENCE} is the exact reference and must be among --stores")

    chunk_ids = (args.emb_dir / "chunk_ids.txt").read_text().splitlines()
    meta = pd.read_parquet(args.chunks, columns=["chunk_id", "cik"])
    if meta["chunk_id"].tolist() != chunk_ids:
        raise ValueError("chunk_ids.txt and chunks.parquet are in different orders")
    ciks = meta["cik"].to_numpy()
    all_dev, all_vecs = load_dev(args.emb_dir, args.query_dir)
    picked = sample_queries(all_dev)
    picked_ids = {q["query_id"] for q in picked}
    vecs = all_vecs[[i for i, q in enumerate(all_dev) if q["query_id"] in picked_ids]]

    stores = {name: open_store(name, chunk_ids, ciks, args.faiss_dir) for name in args.stores}
    results = {
        name: {m: search_all(s, vecs, picked, m) for m in MODES} for name, s in stores.items()
    }

    # ties T7 to T4b: exact within-company recall@10 on all 182 dev queries
    full = search_all(stores[REFERENCE], all_vecs, all_dev, "filtered")
    reproduced = float(
        np.mean([recall_at_k(r, q["grades"], K) for r, q in zip(full, all_dev, strict=True)])
    )
    expected = json.loads(args.diagnostics.read_text())["recall_at_10"]["company"]

    table: dict = {}
    for name, store in stores.items():
        table[name] = {}
        for mode in MODES:
            row = quality(results[name][mode], results[REFERENCE][mode], picked, mode)
            row.update(latency_summary(time_searches(store, vecs, picked, mode)))
            table[name][mode] = row
            print(name, mode, json.dumps(row), flush=True)
    for store in stores.values():
        if hasattr(store, "close"):
            store.close()

    record = {
        "command": "python -m filinglens.retrieval.bench_stores " + shlex.join(argv),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "power_note": args.note,
        "split": "dev",
        "queries_sampled": len(picked),
        "sample_seed": SAMPLE_SEED,
        "k": K,
        "warmup_passes": WARMUP,
        "timed_passes": PASSES,
        "latency": "client-observed; FAISS in process (1 thread), pgvector + Qdrant over localhost",
        "labels": "synthetic queries, automatic grades (no human judgment)",
        "reference": REFERENCE,
        "reproduce_t4b": {
            "queries": len(all_dev),
            "faiss_flat_filtered_recall_at_10": round(reproduced, 4),
            "diagnostics_company_recall_at_10": expected,
            "match": abs(reproduced - expected) <= REPRODUCE_TOLERANCE,
        },
        "inputs_sha256": {
            "chunks.npy": sha256_file(args.emb_dir / "chunks.npy"),
            "queries.dev.npy": sha256_file(args.emb_dir / "queries.dev.npy"),
            "queries.dev.jsonl": sha256_file(args.query_dir / "queries.dev.jsonl"),
        },
        "table": table,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("w") as f:
        for name in results:
            for mode in MODES:
                for q, top in zip(picked, results[name][mode], strict=True):
                    f.write(
                        json.dumps(
                            {"store": name, "mode": mode, "query_id": q["query_id"], "top": top}
                        )
                        + "\n"
                    )
    metrics = {
        f"{name}_{mode}_{key}": value
        for name, modes in table.items()
        for mode, row in modes.items()
        for key, value in row.items()
        if key in ("recall_at_10", "ndcg_at_10", "overlap_at_10_exact", "p95_ms")
    }
    params = {"stores": ",".join(args.stores), "queries": len(picked), "k": K, "passes": PASSES}
    record["mlflow_run_id"] = log_run(
        "vector_stores", record, params, metrics, args.query_dir / "queries.dev.jsonl", args.note
    )
    (args.bench / "vector_stores.json").write_text(json.dumps(record, indent=2) + "\n")
    with (args.bench / "vector_stores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows(table))
    if not record["reproduce_t4b"]["match"]:
        raise SystemExit(f"FAISS Flat filtered recall {reproduced:.4f} != T4b {expected}")


if __name__ == "__main__":
    main()
