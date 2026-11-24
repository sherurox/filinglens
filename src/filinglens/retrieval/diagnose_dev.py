"""Why is exact recall@10 on ret-dev what it is? Brute-force diagnostics on dev only (FL-M3 T4b).

T4's sanity number (exact recall@10 over all 203,200 chunks) was low. Two explanations predict
different numbers here:
- a bug (vectors not aligned with their IDs, wrong pooling) leaves recall low even when the
  search is limited to the query's own company, and the source chunk ranks nowhere;
- company identity: the questions name a company and fiscal year ("NDAQ's FY2019 10-K"), the
  chunk text says "we", and risk boilerplate is shared across companies. Then other companies'
  copies fill the top-10, and recall rises sharply inside the query's company.

Reported, all exact (inner product over every candidate), dev queries only:
- recall@10 over all chunks (must equal bench/embed_queries_dev.json), within the query's
  company, and within its company and fiscal year;
- where the first relevant chunk ranks over all chunks (median, share in the top 10/50/100/1000);
- what the global top-10 is made of: share from the query's company, and from its company in
  another filing;
- how often the query's ticker appears in its grade-2 chunk's text.

    uv run python -m filinglens.retrieval.diagnose_dev
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from filinglens.retrieval.chunks import sha256_file
from filinglens.retrieval.config import CHUNKS, EMB_DIR
from filinglens.retrieval.queries import OUT_DIR as QUERY_DIR
from filinglens.retrieval.queries import load_queries

K = 10
RANK_BANDS = (10, 50, 100, 1000)


def recall_in(scores: np.ndarray, candidates: np.ndarray, relevant: np.ndarray, k: int) -> bool:
    """Is a relevant row among the k best-scoring candidate rows?"""
    pool = np.flatnonzero(candidates)
    if pool.size == 0:
        return False
    top = pool[np.argsort(-scores[pool], kind="stable")[:k]]
    return bool(relevant[top].any())


def first_relevant_rank(scores: np.ndarray, relevant: np.ndarray) -> int:
    """1-based rank of the best-scoring relevant row among all rows (ties count against it)."""
    best = scores[relevant].max()
    return int((scores > best).sum()) + 1


def diagnose(
    query_vecs: np.ndarray,
    chunk_vecs: np.ndarray,
    chunk_ids: Sequence[str],
    chunk_cik: np.ndarray,
    chunk_year: np.ndarray,
    queries: Sequence[dict],
    k: int = K,
) -> dict:
    position = {cid: i for i, cid in enumerate(chunk_ids)}
    found = {"all": 0, "company": 0, "company_year": 0}
    ranks, own_share, own_other_filing_share = [], [], []
    for q, vec in zip(queries, query_vecs, strict=True):
        scores = chunk_vecs @ vec
        relevant = np.zeros(len(chunk_ids), dtype=bool)
        relevant[[position[c] for c in q["grades"]]] = True
        own = chunk_cik == q["cik"]
        found["all"] += recall_in(scores, np.ones_like(own), relevant, k)
        found["company"] += recall_in(scores, own, relevant, k)
        found["company_year"] += recall_in(
            scores, own & (chunk_year == q["period_year"]), relevant, k
        )
        ranks.append(first_relevant_rank(scores, relevant))
        top = np.argsort(-scores, kind="stable")[:k]
        own_share.append(own[top].mean())
        grade1 = np.zeros_like(relevant)
        grade1[[position[c] for c, g in q["grades"].items() if g == 1]] = True
        own_other_filing_share.append(grade1[top].mean())
    n = len(queries)
    ranks_arr = np.asarray(ranks)
    return {
        "queries": n,
        "k": k,
        f"recall_at_{k}": {name: round(v / n, 4) for name, v in found.items()},
        "first_relevant_rank_all_chunks": {
            "median": float(np.median(ranks_arr)),
            **{f"share_le_{b}": round(float((ranks_arr <= b).mean()), 4) for b in RANK_BANDS},
        },
        f"top_{k}_share_same_company": round(float(np.mean(own_share)), 4),
        f"top_{k}_share_grade1_copies": round(float(np.mean(own_other_filing_share)), 4),
    }


def ticker_in_text(queries: Sequence[dict], text_of: dict[str, str]) -> float:
    """Share of queries whose ticker appears as a whole word in a grade-2 chunk's text."""
    hits = 0
    for q in queries:
        pattern = re.compile(rf"\b{re.escape(q['ticker'])}\b")
        texts = [text_of[c] for c, g in q["grades"].items() if g == 2]
        hits += any(pattern.search(t) for t in texts)
    return round(hits / len(queries), 4) if queries else 0.0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--emb-dir", type=Path, default=EMB_DIR)
    parser.add_argument("--query-dir", type=Path, default=QUERY_DIR)
    parser.add_argument("--bench", type=Path, default=Path("bench"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    queries = load_queries("dev", args.query_dir)
    query_ids = (args.emb_dir / "query_ids.dev.txt").read_text().splitlines()
    if query_ids != [q["query_id"] for q in queries]:
        raise ValueError("query vectors and query file are in different orders")
    chunk_ids = (args.emb_dir / "chunk_ids.txt").read_text().splitlines()
    meta = pd.read_parquet(args.chunks, columns=["chunk_id", "cik", "period_year", "text"])
    if meta["chunk_id"].tolist() != chunk_ids:
        raise ValueError("chunk vectors and chunks.parquet are in different orders")
    chunk_vecs = np.load(args.emb_dir / "chunks.npy")
    query_vecs = np.load(args.emb_dir / "queries.dev.npy")
    result = {
        "command": "python -m filinglens.retrieval.diagnose_dev " + shlex.join(argv),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": "dev",
        "search": "exact inner product over every candidate chunk",
        **diagnose(
            query_vecs,
            chunk_vecs,
            chunk_ids,
            meta["cik"].to_numpy(),
            meta["period_year"].to_numpy(),
            queries,
        ),
        "ticker_in_grade2_text": ticker_in_text(
            queries, dict(zip(chunk_ids, meta["text"], strict=True))
        ),
        "inputs_sha256": {
            "chunks.npy": sha256_file(args.emb_dir / "chunks.npy"),
            "queries.dev.npy": sha256_file(args.emb_dir / "queries.dev.npy"),
        },
    }
    (args.bench / "retrieval_dev_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
