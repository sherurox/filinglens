"""Rerank the filtered dense top-50 with a cross-encoder (FL-M3 T8).

Two steps, in the two environments that have what they need:

1. `candidates` (uv env, faiss): for each query of a split, the top DEPTH chunks of its company
   from FAISS Flat (exact; T7 showed pgvector and Qdrant return the same quality, and Flat
   breaks ties the same way every run). Written to data/interim/retrieval/candidates.<split>.jsonl.
2. `score` (fl-train venv, torch): BAAI/bge-reranker-base at its pinned revision scores every
   (question, chunk) pair; candidates are reordered by score, ties kept in dense order.
   Before = the dense top 10, after = the reranked top 10; recall@10, MRR@10 and NDCG@10 for
   both, per query and averaged, plus the candidates' recall@DEPTH (the most reranking can find).

Checks in `score`: the model's config at that revision is a sequence classifier with one output;
fp16 on the GPU vs fp32 on the CPU for the first queries' candidates, largest logit difference
<= MAX_LOGIT_DIFF, else stop; pairs longer than 512 reranker tokens are counted (the chunk text is
truncated, never the question); on dev, the "before" recall must reproduce T4b's exact
within-company recall@10.

Latency: the rerank stage per query (tokenize + forward pass over DEPTH pairs + GPU sync), after
WARMUP untimed queries; p50/p95/p99 in ms; peak VRAM. Query embedding and dense search come on
top of it (T4, T7).

ret-test is touched once, in T10, with --allow-test on both steps.

    uv run python -m filinglens.retrieval.rerank candidates --split dev
    ~/venvs/fl-train/bin/python -m filinglens.retrieval.rerank score --split dev --note "..."
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from filinglens.retrieval.chunks import sha256_file
from filinglens.retrieval.config import (
    CHUNKS,
    EMB_DIR,
    MAX_SEQ_TOKENS,
    RERANK_MODEL,
    RERANK_REVISION,
    RET_DIR,
)
from filinglens.retrieval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from filinglens.retrieval.queries import OUT_DIR as QUERY_DIR
from filinglens.retrieval.queries import load_queries
from filinglens.retrieval.runlog import log_run

DEPTH = 50
K = 10
BATCH = DEPTH  # one query's candidates per forward pass
WARMUP = 3
PRECISION_QUERIES = 3
MAX_LOGIT_DIFF = 0.1
REPRODUCE_TOLERANCE = 1e-4


# ---- pure helpers (tested without torch or faiss) ---------------------------------------------


def rerank_order(candidates: Sequence[str], scores: Sequence[float]) -> list[str]:
    """Candidates by score, highest first; equal scores keep their dense order."""
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
    return [candidates[i] for i in order]


def per_query_metrics(dense: Sequence[str], reranked: Sequence[str], grades: dict) -> dict:
    return {
        "recall_before": recall_at_k(dense, grades, K),
        "recall_after": recall_at_k(reranked, grades, K),
        "mrr_before": mrr_at_k(dense, grades, K),
        "mrr_after": mrr_at_k(reranked, grades, K),
        "ndcg_before": ndcg_at_k(dense, grades, K),
        "ndcg_after": ndcg_at_k(reranked, grades, K),
        "recall_at_depth": recall_at_k(dense, grades, len(dense)) if dense else 0.0,
    }


def mean_metrics(rows: Sequence[dict]) -> dict:
    keys = [k for k in rows[0] if k != "query_id"] if rows else []
    return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}


def latency_summary(ms: Sequence[float]) -> dict:
    arr = np.asarray(ms, dtype=float)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    return {
        "p50_ms": round(float(p50), 2),
        "p95_ms": round(float(p95), 2),
        "p99_ms": round(float(p99), 2),
        "mean_ms": round(float(arr.mean()), 2),
        "samples": int(arr.size),
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def load_split(split: str, query_dir: Path, emb_dir: Path, allow_test: bool):
    queries = load_queries(split, query_dir, allow_test=allow_test)
    ids = (emb_dir / f"query_ids.{split}.txt").read_text().splitlines()
    if ids != [q["query_id"] for q in queries]:
        raise ValueError(f"query vectors and queries.{split}.jsonl are in different orders")
    return queries, np.load(emb_dir / f"queries.{split}.npy")


# ---- step 1: candidates (uv env) --------------------------------------------------------------


def run_candidates(args: argparse.Namespace, argv: list[str]) -> None:
    from filinglens.retrieval.build_indexes import FAISS_DIR
    from filinglens.retrieval.stores import FaissStore

    queries, vecs = load_split(args.split, args.query_dir, args.emb_dir, args.allow_test)
    chunk_ids = (args.emb_dir / "chunk_ids.txt").read_text().splitlines()
    meta = pd.read_parquet(args.chunks, columns=["chunk_id", "cik"])
    if meta["chunk_id"].tolist() != chunk_ids:
        raise ValueError("chunk_ids.txt and chunks.parquet are in different orders")
    store = FaissStore.load("flat", FAISS_DIR / "flat.index", chunk_ids, meta["cik"].to_numpy())
    rows = [
        {"query_id": q["query_id"], "candidates": store.search(v, DEPTH, cik=q["cik"])}
        for q, v in zip(queries, vecs, strict=True)
    ]
    out = args.ret_dir / f"candidates.{args.split}.jsonl"
    write_jsonl(out, rows)
    print(
        json.dumps(
            {
                "command": "python -m filinglens.retrieval.rerank " + shlex.join(argv),
                "split": args.split,
                "queries": len(rows),
                "depth": DEPTH,
                "short_lists": sum(len(r["candidates"]) < DEPTH for r in rows),
                "out": str(out),
                "out_sha256": sha256_file(out),
            },
            indent=2,
        )
    )


# ---- step 2: score (fl-train venv) ------------------------------------------------------------


def check_model_files() -> dict:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(RERANK_MODEL, "config.json", revision=RERANK_REVISION)
    config = json.loads(Path(path).read_text())
    archs = config.get("architectures", [])
    labels = len(config.get("id2label", {"0": "LABEL_0"}))
    if not any(a.endswith("ForSequenceClassification") for a in archs) or labels != 1:
        raise ValueError(f"not a one-output sequence classifier: {archs}, {labels} labels")
    return {"architectures": archs, "num_labels": labels}


class Reranker:
    """bge-reranker-base: one relevance logit per (question, chunk) pair."""

    def __init__(self, device: str = "cuda", fp16: bool = True) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL, revision=RERANK_REVISION)
        dtype = torch.float16 if fp16 else torch.float32
        model = AutoModelForSequenceClassification.from_pretrained(
            RERANK_MODEL, revision=RERANK_REVISION, dtype=dtype
        )
        self.model = model.to(device).eval()
        self.truncated = 0

    def count_long_pairs(self, question: str, texts: Sequence[str]) -> int:
        ids = self.tokenizer([question] * len(texts), list(texts))["input_ids"]
        return sum(len(x) > MAX_SEQ_TOKENS for x in ids)

    def __call__(self, question: str, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            enc = self.tokenizer(
                [question] * len(texts),
                list(texts),
                padding=True,
                truncation="only_second",  # cut the chunk, never the question
                max_length=MAX_SEQ_TOKENS,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits.view(-1).float()
            if self.device == "cuda":
                torch.cuda.synchronize()
            return logits.cpu().numpy()


def precision_check(gpu: Reranker, pairs: Sequence[tuple[str, list[str]]]) -> float:
    cpu = Reranker(device="cpu", fp16=False)
    worst = 0.0
    for question, texts in pairs:
        worst = max(worst, float(np.abs(gpu(question, texts) - cpu(question, texts)).max()))
    if worst > MAX_LOGIT_DIFF:
        raise ValueError(f"fp16 vs fp32 logits differ by {worst:.4f} > {MAX_LOGIT_DIFF}")
    return worst


def run_score(args: argparse.Namespace, argv: list[str]) -> None:
    import torch

    from filinglens.train.tracking import gpu_snapshot

    queries = load_queries(args.split, args.query_dir, allow_test=args.allow_test)
    by_id = {q["query_id"]: q for q in queries}
    cand_path = args.ret_dir / f"candidates.{args.split}.jsonl"
    cands = read_jsonl(cand_path)
    if [c["query_id"] for c in cands] != [q["query_id"] for q in queries]:
        raise ValueError(f"{cand_path} doesn't match queries.{args.split}.jsonl")
    texts = pd.read_parquet(args.chunks, columns=["chunk_id", "embed_text"])
    text_of = dict(zip(texts["chunk_id"], texts["embed_text"], strict=True))

    record = {
        "command": "python -m filinglens.retrieval.rerank " + shlex.join(argv),
        "split": args.split,
        "model": f"{RERANK_MODEL}@{RERANK_REVISION}",
        "model_files": check_model_files(),
        "depth": DEPTH,
        "k": K,
        "dtype": "float16 weights, fp32 logits",
        "max_pair_tokens": MAX_SEQ_TOKENS,
        "power_note": args.note,
        "labels": "synthetic queries, automatic grades (no human judgment)",
    }
    gpu = Reranker()
    first = [
        (by_id[c["query_id"]]["question"], [text_of[i] for i in c["candidates"]])
        for c in cands[:PRECISION_QUERIES]
    ]
    record["fp16_vs_fp32_max_logit_diff"] = round(precision_check(gpu, first), 5)
    for question, chunk_texts in first[:WARMUP]:
        gpu(question, chunk_texts)  # warm-up: CUDA kernels, allocator
    record["gpu_before"] = gpu_snapshot()
    torch.cuda.reset_peak_memory_stats()

    rows, latencies, long_pairs = [], [], 0
    for c in cands:
        q = by_id[c["query_id"]]
        chunk_texts = [text_of[i] for i in c["candidates"]]
        long_pairs += gpu.count_long_pairs(q["question"], chunk_texts)
        start = time.perf_counter()
        scores = gpu(q["question"], chunk_texts)
        latencies.append((time.perf_counter() - start) * 1000)
        reranked = rerank_order(c["candidates"], scores.tolist())
        rows.append(
            {"query_id": q["query_id"], **per_query_metrics(c["candidates"], reranked, q["grades"])}
        )
    record["gpu_after"] = gpu_snapshot()
    record.update(
        {
            "queries": len(rows),
            "pairs": sum(len(c["candidates"]) for c in cands),
            "pairs_truncated": long_pairs,
            "metrics": mean_metrics(rows),
            "rerank_latency": latency_summary(latencies),
            "peak_vram_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
            "peak_vram_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
            "inputs_sha256": {
                "candidates": sha256_file(cand_path),
                f"queries.{args.split}.jsonl": sha256_file(
                    args.query_dir / f"queries.{args.split}.jsonl"
                ),
            },
        }
    )
    if args.split == "dev":
        expected = json.loads(args.diagnostics.read_text())["recall_at_10"]["company"]
        got = record["metrics"]["recall_before"]
        record["reproduce_t4b"] = {
            "recall_before": got,
            "diagnostics": expected,
            "match": abs(got - expected) <= REPRODUCE_TOLERANCE,
        }
    record["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    write_jsonl(args.ret_dir / f"rerank_{args.split}_per_query.jsonl", rows)
    metrics = {
        **record["metrics"],
        **{f"latency_{k}": v for k, v in record["rerank_latency"].items()},
    }
    params = {"split": args.split, "depth": DEPTH, "k": K, "model": record["model"]}
    record["mlflow_run_id"] = log_run(
        f"rerank_{args.split}", record, params, metrics, cand_path, args.note
    )
    (args.bench / f"reranker_{args.split}.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.split == "dev":
        write_table(args.bench / "reranker.csv", record)
    print(json.dumps(record, indent=2))
    if args.split == "dev" and not record["reproduce_t4b"]["match"]:
        raise SystemExit("dense 'before' recall doesn't reproduce T4b")


def write_table(path: Path, record: dict) -> None:
    """The D1 reranker table: dense top-10 vs reranked top-10 on the same queries."""
    m, lat = record["metrics"], record["rerank_latency"]
    rows = [
        {"variant": "dense (FAISS Flat, CIK filter)", "recall_at_10": m["recall_before"],
         "mrr_at_10": m["mrr_before"], "ndcg_at_10": m["ndcg_before"], "rerank_p50_ms": "",
         "rerank_p95_ms": "", "rerank_p99_ms": ""},
        {"variant": f"dense top-{DEPTH} + bge-reranker-base", "recall_at_10": m["recall_after"],
         "mrr_at_10": m["mrr_after"], "ndcg_at_10": m["ndcg_after"],
         "rerank_p50_ms": lat["p50_ms"], "rerank_p95_ms": lat["p95_ms"],
         "rerank_p99_ms": lat["p99_ms"]},
    ]  # fmt: skip
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) + ["queries", "recall_at_depth"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {**row, "queries": record["queries"], "recall_at_depth": m["recall_at_depth"]}
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    for name in ("candidates", "score"):
        p = sub.add_parser(name)
        p.add_argument("--split", choices=["dev", "test"], required=True)
        p.add_argument("--allow-test", action="store_true")
        p.add_argument("--query-dir", type=Path, default=QUERY_DIR)
        p.add_argument("--emb-dir", type=Path, default=EMB_DIR)
        p.add_argument("--chunks", type=Path, default=CHUNKS)
        p.add_argument("--ret-dir", type=Path, default=RET_DIR)
    score = sub.choices["score"]
    score.add_argument("--bench", type=Path, default=Path("bench"))
    score.add_argument(
        "--diagnostics", type=Path, default=Path("bench/retrieval_dev_diagnostics.json")
    )
    score.add_argument("--note", required=True, help="power source and Windows power mode")
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    (run_candidates if args.step == "candidates" else run_score)(args, argv)


if __name__ == "__main__":
    main()
