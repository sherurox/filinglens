"""Load the 203,200 chunk embeddings into the four vector stores (FL-M3 T5).

Builds FAISS Flat and HNSW (saved under data/interim/retrieval/faiss/), the pgvector table and
indexes, and the Qdrant collection, all from the same vectors and IDs as T4 wrote them. For each
store it records build time, threads or workers, size and parameters; for pgvector also the
plans the planner picks for one filtered and one unfiltered search, and any build NOTICE (graph
outgrew maintenance_work_mem). A self-check per store: a chunk's own vector, searched within its
company, must return that chunk in the top 10.

Output: bench/index_build.json and one MLflow run in fl-m3-retrieval.

    uv run python -m filinglens.retrieval.build_indexes --note "AC power, Best performance"
    uv run python -m filinglens.retrieval.build_indexes --stores pgvector --note "..."  # one store
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from filinglens.retrieval.chunks import sha256_file
from filinglens.retrieval.config import CHUNKS, EMB_DIR, RET_DIR
from filinglens.retrieval.runlog import log_run
from filinglens.retrieval.stores import (
    EF_CONSTRUCTION,
    EF_SEARCH,
    HNSW_M,
    FaissStore,
    PgVectorStore,
    QdrantStore,
)

STORES = ("faiss-flat", "faiss-hnsw", "pgvector", "qdrant")
FAISS_DIR = RET_DIR / "faiss"
SELF_CHECK_ROW = 0


def load_inputs(emb_dir: Path, chunks_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Chunk IDs, vectors and CIKs, in T4's order; stops if any of them disagree."""
    ids = (emb_dir / "chunk_ids.txt").read_text().splitlines()
    meta = pd.read_parquet(chunks_path, columns=["chunk_id", "cik"])
    if meta["chunk_id"].tolist() != ids:
        raise ValueError("chunk_ids.txt and chunks.parquet are in different orders")
    vectors = np.load(emb_dir / "chunks.npy")
    if len(vectors) != len(ids):
        raise ValueError(f"{len(vectors)} vectors for {len(ids)} chunk IDs")
    return ids, vectors, meta["cik"].to_numpy()


def self_check(store, ids: Sequence[str], vectors: np.ndarray, ciks: np.ndarray) -> bool:
    """A chunk's own vector, searched within its company, finds that chunk in the top 10."""
    row = SELF_CHECK_ROW
    return ids[row] in store.search(vectors[row], 10, cik=int(ciks[row]))


def build_one(name: str, ids, vectors, ciks, args: argparse.Namespace) -> dict:
    if name.startswith("faiss-"):
        kind = name.removeprefix("faiss-")
        store = FaissStore(kind)
        stats = store.build(ids, vectors, ciks)
        path = args.faiss_dir / f"{kind}.index"
        stats["index_bytes"] = store.save(path)
        stats["path"] = str(path)
        stats["search_threads"] = 1
    elif name == "pgvector":
        store = PgVectorStore()
        stats = store.build(
            ids,
            vectors,
            ciks,
            maintenance_work_mem=args.pg_maintenance_work_mem,
            workers=args.pg_workers,
        )
        row = SELF_CHECK_ROW
        stats["plan_filtered"] = store.explain(vectors[row], 10, cik=int(ciks[row]))
        stats["plan_unfiltered"] = store.explain(vectors[row], 10)
    elif name == "qdrant":
        store = QdrantStore()
        stats = store.build(ids, vectors, ciks)
    else:
        raise ValueError(f"unknown store {name!r}")
    stats["self_check_top10"] = self_check(store, ids, vectors, ciks)
    if hasattr(store, "close"):
        store.close()
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stores", nargs="+", choices=STORES, default=list(STORES))
    parser.add_argument("--emb-dir", type=Path, default=EMB_DIR)
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--faiss-dir", type=Path, default=FAISS_DIR)
    parser.add_argument("--bench", type=Path, default=Path("bench"))
    parser.add_argument("--pg-maintenance-work-mem", default="1GB")
    parser.add_argument("--pg-workers", type=int, default=4)
    parser.add_argument("--note", required=True, help="power source and Windows power mode")
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    ids, vectors, ciks = load_inputs(args.emb_dir, args.chunks)
    record = {
        "command": "python -m filinglens.retrieval.build_indexes " + shlex.join(argv),
        "power_note": args.note,
        "cpu_count": os.cpu_count(),
        "rows": len(ids),
        "dim": int(vectors.shape[1]),
        "shared_params": {"M": HNSW_M, "ef_construction": EF_CONSTRUCTION, "ef_search": EF_SEARCH},
        "inputs_sha256": {
            "chunks.npy": sha256_file(args.emb_dir / "chunks.npy"),
            "chunk_ids.txt": sha256_file(args.emb_dir / "chunk_ids.txt"),
        },
        "stores": {},
    }
    for name in args.stores:
        print(f"building {name} ...", flush=True)
        record["stores"][name] = build_one(name, ids, vectors, ciks, args)
        print(json.dumps(record["stores"][name], indent=2), flush=True)
    failed = [n for n, s in record["stores"].items() if not s["self_check_top10"]]
    record["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    metrics = {f"{n}_build_seconds": s["build_seconds"] for n, s in record["stores"].items()}
    params = {
        "stores": ",".join(args.stores),
        **record["shared_params"],
        "pg_maintenance_work_mem": args.pg_maintenance_work_mem,
        "pg_workers": args.pg_workers,
    }
    record["mlflow_run_id"] = log_run(
        "index_build", record, params, metrics, args.emb_dir / "chunk_ids.txt", args.note
    )
    out = args.bench / "index_build.json"
    if args.stores != list(STORES) and out.exists():  # a partial rebuild updates its stores only
        previous = json.loads(out.read_text())
        record["stores"] = {**previous.get("stores", {}), **record["stores"]}
    out.write_text(json.dumps(record, indent=2) + "\n")
    if failed:
        raise SystemExit(f"self-check failed for: {failed}")


if __name__ == "__main__":
    main()
