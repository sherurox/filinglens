"""Embed the chunks and the queries with bge-base-en-v1.5 on the GPU (FL-M3 T4).

Pooling is the [CLS] hidden state, L2-normalized: what the model's own sentence-transformers
config (`1_Pooling/config.json` at the pinned revision) specifies. Queries get the model card's
retrieval instruction; passages get none. Both are checked against the files at the pinned
revision before anything is embedded, so the code can't drift from the model card unnoticed.

Weights run in fp16 on the GPU; pooled vectors are normalized in fp32 and saved as float32 .npy
with their IDs, one per line, in the same order. Texts are batched longest first (less padding,
and an out-of-memory batch shows up at once rather than at the end).

Checks built into the run:
- precision: 64 chunks embedded twice, fp16 on the GPU and fp32 on the CPU; every pair's cosine
  must be >= 0.999, else the run stops (catches a pooling or precision bug before 200k rows);
- every vector's norm within 1 +/- 1e-3, no NaN, no input truncated (T2 kept chunks <= 510);
- for dev queries only: exact (brute-force) recall@10 against the graded chunks, a sanity number
  that T7's FAISS Flat row must reproduce. Test queries get no metric here.

Outputs under data/interim/retrieval/bge-base/, bench/embed_<what>.json, and one MLflow run each
in `fl-m3-retrieval` (a tracking outage is recorded in the bench file, not fatal).

    TRAIN=~/venvs/fl-train/bin/python
    $TRAIN -m filinglens.retrieval.embed corpus --note "AC power, Best performance"
    $TRAIN -m filinglens.retrieval.embed queries --split dev --note "AC power, Best performance"
    $TRAIN -m filinglens.retrieval.embed queries --split test --allow-test --note "..."  # T10
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from filinglens.retrieval.chunks import sha256_file
from filinglens.retrieval.config import (
    CHUNKS,
    EMB_DIR,
    EMBED_DIM,
    EMBED_MODEL,
    EMBED_REVISION,
    MAX_SEQ_TOKENS,
    QUERY_INSTRUCTION,
)
from filinglens.retrieval.queries import OUT_DIR as QUERY_DIR
from filinglens.retrieval.queries import load_queries

BATCH = 128
PRECISION_ROWS = 64
MIN_COSINE_FP16_FP32 = 0.999
NORM_TOLERANCE = 1e-3


# ---- pure helpers (tested without torch) -------------------------------------------------------


def length_order(texts: Sequence[str]) -> np.ndarray:
    """Indices that sort texts longest first (stable, so ties keep their order)."""
    return np.argsort(-np.fromiter((len(t) for t in texts), dtype=np.int64), kind="stable")


def restore_order(order: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Put rows computed in `order` back in the original order."""
    out = np.empty_like(vectors)
    out[order] = vectors
    return out


def check_vectors(vectors: np.ndarray, rows: int, dim: int = EMBED_DIM) -> dict:
    """Shape, NaN and unit-norm checks; raises on failure, returns the norm range."""
    if vectors.shape != (rows, dim):
        raise ValueError(f"expected {(rows, dim)} vectors, got {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("NaN or inf in the embeddings")
    norms = np.linalg.norm(vectors, axis=1)
    if np.abs(norms - 1).max() > NORM_TOLERANCE:
        raise ValueError(f"norms outside 1 +/- {NORM_TOLERANCE}: {norms.min()}..{norms.max()}")
    return {"norm_min": float(norms.min()), "norm_max": float(norms.max())}


def exact_recall_at_k(
    query_vecs: np.ndarray,
    chunk_vecs: np.ndarray,
    chunk_ids: Sequence[str],
    grades: Sequence[Mapping[str, int]],
    k: int = 10,
    block: int = 64,
) -> float:
    """Share of queries with >= 1 graded chunk among their exact top-k by inner product."""
    ids = np.asarray(chunk_ids)
    hits = 0
    for start in range(0, len(query_vecs), block):
        scores = query_vecs[start : start + block] @ chunk_vecs.T
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        for row, g in zip(top, grades[start : start + block], strict=True):
            hits += any(g.get(cid, 0) >= 1 for cid in ids[row])
    return hits / len(query_vecs) if len(query_vecs) else 0.0


def read_ids(path: Path) -> list[str]:
    return path.read_text().splitlines()


def write_ids(path: Path, ids: Sequence[str]) -> None:
    path.write_text("".join(f"{i}\n" for i in ids))


# ---- model ------------------------------------------------------------------------------------


def check_model_files() -> dict:
    """The pinned revision's pooling config says CLS, and its card has our query instruction."""
    from huggingface_hub import hf_hub_download

    pooling_path = hf_hub_download(EMBED_MODEL, "1_Pooling/config.json", revision=EMBED_REVISION)
    pooling = json.loads(Path(pooling_path).read_text())
    if not pooling.get("pooling_mode_cls_token") or pooling.get("pooling_mode_mean_tokens"):
        raise ValueError(f"the model's pooling config isn't CLS pooling: {pooling}")
    card = Path(hf_hub_download(EMBED_MODEL, "README.md", revision=EMBED_REVISION)).read_text()
    if QUERY_INSTRUCTION.strip() not in card:
        raise ValueError("the query instruction isn't in the model card at this revision")
    return {"pooling": pooling, "query_instruction_in_card": True}


class Embedder:
    """bge-base at the pinned revision: tokenize, run, take [CLS], L2-normalize in fp32."""

    def __init__(self, device: str = "cuda", fp16: bool = True) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, revision=EMBED_REVISION)
        dtype = torch.float16 if fp16 else torch.float32
        model = AutoModel.from_pretrained(EMBED_MODEL, revision=EMBED_REVISION, dtype=dtype)
        self.model = model.to(device).eval()
        self.truncated = 0

    def __call__(self, texts: Sequence[str], batch: int = BATCH) -> np.ndarray:
        torch = self.torch
        order = length_order(texts)
        out = np.empty((len(texts), EMBED_DIM), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(texts), batch):
                part = [texts[i] for i in order[start : start + batch]]
                lengths = [len(x) for x in self.tokenizer(part)["input_ids"]]
                self.truncated += sum(n > MAX_SEQ_TOKENS for n in lengths)
                enc = self.tokenizer(
                    part,
                    padding=True,
                    truncation=True,
                    max_length=MAX_SEQ_TOKENS,
                    return_tensors="pt",
                ).to(self.device)
                cls = self.model(**enc).last_hidden_state[:, 0].float()
                vecs = torch.nn.functional.normalize(cls, dim=1)
                out[start : start + len(part)] = vecs.cpu().numpy()
        return restore_order(order, out)


def precision_check(texts: Sequence[str], gpu: Embedder) -> float:
    """Lowest cosine between fp16-GPU and fp32-CPU embeddings of the same texts."""
    cpu = Embedder(device="cpu", fp16=False)
    a, b = gpu(texts), cpu(texts)
    lowest = float((a * b).sum(axis=1).min())
    if lowest < MIN_COSINE_FP16_FP32:
        raise ValueError(f"fp16 vs fp32 cosine {lowest:.5f} < {MIN_COSINE_FP16_FP32}")
    return lowest


# ---- runs -------------------------------------------------------------------------------------


def log_mlflow(run_name: str, result: dict, data_manifest: Path, note: str) -> str | None:
    """One run in fl-m3-retrieval; returns its ID, or None (with the error in `result`)."""
    try:
        import mlflow

        from filinglens.train.tracking import run_tags, start_run

        with start_run(
            "retrieval", run_name, tags=run_tags(data_manifest=data_manifest, note=note)
        ):
            mlflow.log_params({k: v for k, v in result.items() if k in PARAM_KEYS})
            mlflow.log_metrics({k: float(v) for k, v in result.items() if k in METRIC_KEYS})
            mlflow.log_dict(result, f"{run_name}.json")
            return mlflow.active_run().info.run_id
    except Exception as error:  # tracking is a side channel: record the failure, keep the work
        result["mlflow_error"] = f"{type(error).__name__}: {error}"
        return None


PARAM_KEYS = ("model", "what", "split", "batch", "dtype", "max_seq_tokens", "rows")
METRIC_KEYS = (
    "seconds",
    "rows_per_second",
    "peak_vram_allocated_mib",
    "peak_vram_reserved_mib",
    "fp16_vs_fp32_min_cosine",
    "exact_recall_at_10",
)


def base_record(what: str, note: str, argv: list[str]) -> dict:
    return {
        "command": "python -m filinglens.retrieval.embed " + shlex.join(argv),
        "model": f"{EMBED_MODEL}@{EMBED_REVISION}",
        "what": what,
        "dtype": "float16 weights, fp32 pooling + norm, float32 .npy",
        "batch": BATCH,
        "max_seq_tokens": MAX_SEQ_TOKENS,
        "power_note": note,
    }


def run_corpus(args: argparse.Namespace, argv: list[str]) -> dict:
    import torch

    from filinglens.train.tracking import gpu_snapshot

    result = base_record("corpus", args.note, argv)
    result["model_files"] = check_model_files()
    chunks = pd.read_parquet(args.chunks, columns=["chunk_id", "embed_text"])
    texts = chunks["embed_text"].tolist()
    gpu = Embedder()
    result["fp16_vs_fp32_min_cosine"] = precision_check(texts[:PRECISION_ROWS], gpu)
    gpu.truncated = 0
    result["gpu_before"] = gpu_snapshot()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    vectors = gpu(texts)
    seconds = time.perf_counter() - start
    result["gpu_after"] = gpu_snapshot()
    result.update(check_vectors(vectors, len(texts)))
    if gpu.truncated:
        raise ValueError(f"{gpu.truncated} chunks were longer than {MAX_SEQ_TOKENS} tokens")
    args.emb_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.emb_dir / "chunks.npy", vectors)
    write_ids(args.emb_dir / "chunk_ids.txt", chunks["chunk_id"].tolist())
    result.update(
        {
            "rows": len(texts),
            "dim": EMBED_DIM,
            "truncated": gpu.truncated,
            "seconds": round(seconds, 1),
            "rows_per_second": round(len(texts) / seconds, 1),
            "peak_vram_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
            "peak_vram_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
            "chunks_sha256": sha256_file(args.chunks),
            "chunks_npy_sha256": sha256_file(args.emb_dir / "chunks.npy"),
            "chunk_ids_sha256": sha256_file(args.emb_dir / "chunk_ids.txt"),
        }
    )
    return result


def run_queries(args: argparse.Namespace, argv: list[str]) -> dict:
    queries = load_queries(args.split, args.query_dir, allow_test=args.allow_test)
    result = base_record("queries", args.note, argv)
    result["split"] = args.split
    result["model_files"] = check_model_files()
    gpu = Embedder()
    start = time.perf_counter()
    vectors = gpu([QUERY_INSTRUCTION + q["question"] for q in queries])
    seconds = time.perf_counter() - start
    result.update(check_vectors(vectors, len(queries)))
    args.emb_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.emb_dir / f"queries.{args.split}.npy", vectors)
    write_ids(args.emb_dir / f"query_ids.{args.split}.txt", [q["query_id"] for q in queries])
    result.update(
        {
            "rows": len(queries),
            "dim": EMBED_DIM,
            "truncated": gpu.truncated,
            "seconds": round(seconds, 2),
            "rows_per_second": round(len(queries) / seconds, 1),
        }
    )
    if args.split == "dev":  # a sanity number only; ret-test is scored once, in T10
        chunk_vecs = np.load(args.emb_dir / "chunks.npy", mmap_mode="r")
        chunk_ids = read_ids(args.emb_dir / "chunk_ids.txt")
        grades = [q["grades"] for q in queries]
        recall = exact_recall_at_k(vectors, np.asarray(chunk_vecs), chunk_ids, grades)
        result["exact_recall_at_10"] = round(recall, 4)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="what", required=True)
    corpus = sub.add_parser("corpus", help="embed every chunk")
    corpus.add_argument("--chunks", type=Path, default=CHUNKS)
    queries = sub.add_parser("queries", help="embed one split's queries")
    queries.add_argument("--split", choices=["dev", "test"], required=True)
    queries.add_argument("--allow-test", action="store_true")
    queries.add_argument("--query-dir", type=Path, default=QUERY_DIR)
    for p in (corpus, queries):
        p.add_argument("--emb-dir", type=Path, default=EMB_DIR)
        p.add_argument("--bench", type=Path, default=Path("bench"))
        p.add_argument("--note", required=True, help="power source and Windows power mode")
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    result = run_corpus(args, argv) if args.what == "corpus" else run_queries(args, argv)
    name = "embed_corpus" if args.what == "corpus" else f"embed_queries_{args.split}"
    result["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    manifest = (
        args.chunks if args.what == "corpus" else args.query_dir / f"queries.{args.split}.jsonl"
    )
    result["mlflow_run_id"] = log_mlflow(name, result, manifest, args.note)
    (args.bench / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
