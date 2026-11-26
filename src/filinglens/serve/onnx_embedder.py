r"""bge-base-en-v1.5 query encoder to ONNX for the Cloud Run API: export, optimize, int8, dev check.

Steps (FL-M5 T12), each in the venv that has what it needs; files under data/interim/serve/onnx/:
    ~/venvs/fl-train/bin/python -m filinglens.serve.onnx_embedder export
    PYTHONPATH=src ~/venvs/fl-onnx/bin/python -m filinglens.serve.onnx_embedder optimize
    uv run --group api python -m filinglens.serve.onnx_embedder check --split dev

`export`: bge-base at FL-M3's pinned revision, fp32, eager attention (T8: the eager graph is the
one ONNX Runtime's attention fusion recognizes), inputs input_ids / attention_mask (dynamic batch
and length), output `cls` = the [CLS] hidden state (normalized at run time, as FL-M3 did after the
model). Also copies the revision's tokenizer.json next to the graph and checks that the
`tokenizers` library (what the API uses) gives the same token IDs as transformers' tokenizer
(what FL-M3 used) on every dev query; saves torch fp32 [CLS] vectors of 8 dev queries.

`optimize`: ONNX Runtime's transformer optimizer (BERT: 12 heads, hidden 768; opt_level=0 as in
T8) -> bge-base.eager-opt.onnx (fp32), then dynamic int8 quantization of the weights
(onnxruntime.quantization.quantize_dynamic, QInt8) -> bge-base.eager-opt-int8.onnx.

`check`: for each graph, through api.embedder.OnnxEmbedder on the CPU with ONE thread (a Cloud Run
instance has one vCPU): the 182 dev queries one at a time (latency per query, tokenization
included), their lowest cosine to FL-M3's stored dev query vectors (fp16 GPU, the vectors the
0.2967 was measured with), how many keep the same exact top 10, and exact recall@10 against the
stored corpus vectors (FL-M3's own function). The decision follows bench/m5_embed_rule.json,
committed before this step runs on any graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from filinglens.retrieval.config import (
    EMB_DIR,
    EMBED_MODEL,
    EMBED_REVISION,
    MAX_SEQ_TOKENS,
    QUERY_INSTRUCTION,
)

ONNX_DIR = Path("data/interim/serve/onnx")
GRAPHS = {
    "eager": ONNX_DIR / "bge-base.eager.onnx",
    "opt-fp32": ONNX_DIR / "bge-base.eager-opt.onnx",
    "int8": ONNX_DIR / "bge-base.eager-opt-int8.onnx",
}
TOKENIZER = ONNX_DIR / "bge-base.tokenizer.json"
REFERENCE = ONNX_DIR / "bge-base.torch-fp32-cls.npz"
RULE = Path("bench/m5_embed_rule.json")
NUM_HEADS, HIDDEN, OPSET = 12, 768, 17
REFERENCE_QUERIES = 8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dev_questions() -> tuple[list[str], list[dict]]:
    from filinglens.retrieval.queries import OUT_DIR, load_queries

    queries = load_queries("dev", OUT_DIR)
    return [QUERY_INSTRUCTION + q["question"] for q in queries], queries


def latency_summary(ms: list[float]) -> dict:
    a = np.asarray(ms)
    return {
        "n": len(ms),
        "p50_ms": round(float(np.percentile(a, 50)), 2),
        "p95_ms": round(float(np.percentile(a, 95)), 2),
        "mean_ms": round(float(a.mean()), 2),
    }


def min_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Lowest row-wise cosine of two sets of unit vectors."""
    return float(np.min(np.sum(a * b, axis=1)))


def same_top_k(q_a: np.ndarray, q_b: np.ndarray, chunks: np.ndarray, k: int = 10) -> int:
    """Queries whose exact top-k chunk set is the same under both query vectors."""
    same = 0
    for start in range(0, len(q_a), 64):
        sa = q_a[start : start + 64] @ chunks.T
        sb = q_b[start : start + 64] @ chunks.T
        ta = np.argpartition(-sa, k - 1, axis=1)[:, :k]
        tb = np.argpartition(-sb, k - 1, axis=1)[:, :k]
        same += sum(set(x) == set(y) for x, y in zip(ta, tb, strict=True))
    return int(same)


def rule_decision(rule: dict, results: dict) -> dict:
    """Apply the committed rule to the measured results."""
    parity = rule["pipeline_check"]
    fp32 = results["opt-fp32"]
    pipeline_ok = fp32["min_cosine_vs_fl_m3"] >= parity["min_cosine"]
    accept = rule["int8_accept"]
    int8 = results["int8"]
    drop = rule["fl_m3_dev_recall_at_10"] - int8["exact_recall_at_10"]
    int8_ok = pipeline_ok and drop <= accept["max_recall_drop"]
    return {
        "pipeline_ok": pipeline_ok,
        "int8_recall_drop": round(drop, 4),
        "int8_ok": int8_ok,
        "ship": "int8" if int8_ok else ("opt-fp32" if pipeline_ok else None),
    }


# ---- export (fl-train venv: torch + transformers) ----------------------------------------------


def run_export(args: argparse.Namespace) -> dict:
    import onnx
    import torch
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from transformers import AutoModel, AutoTokenizer

    hf_tok = AutoTokenizer.from_pretrained(EMBED_MODEL, revision=EMBED_REVISION)
    model = AutoModel.from_pretrained(
        EMBED_MODEL, revision=EMBED_REVISION, dtype=torch.float32, attn_implementation="eager"
    ).eval()

    class Cls(torch.nn.Module):  # plain tensors in, the [CLS] hidden state out
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            out = self.inner(input_ids=input_ids, attention_mask=attention_mask)
            return out.last_hidden_state[:, 0]

    wrapper = Cls(model).eval()
    sample = hf_tok(["a short query", "a somewhat longer second query"], padding=True,
                    return_tensors="pt")  # fmt: skip
    inputs = (sample["input_ids"], sample["attention_mask"])
    axes = {"input_ids": {0: "batch", 1: "tokens"}, "attention_mask": {0: "batch", 1: "tokens"},
            "cls": {0: "batch"}}  # fmt: skip
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    out_file = GRAPHS["eager"]
    started = time.perf_counter()
    torch.onnx.export(wrapper, inputs, str(out_file), input_names=["input_ids", "attention_mask"],
                      output_names=["cls"], dynamic_axes=axes, opset_version=OPSET,
                      dynamo=False)  # fmt: skip
    export_s = time.perf_counter() - started
    onnx.checker.check_model(str(out_file))

    shutil.copyfile(
        hf_hub_download(EMBED_MODEL, "tokenizer.json", revision=EMBED_REVISION), TOKENIZER
    )
    fast = Tokenizer.from_file(str(TOKENIZER))
    fast.enable_truncation(max_length=MAX_SEQ_TOKENS)
    texts, _ = dev_questions()
    hf_ids = hf_tok(texts, truncation=True, max_length=MAX_SEQ_TOKENS)["input_ids"]
    fast_ids = [e.ids for e in fast.encode_batch(texts)]
    differ = [i for i, (a, b) in enumerate(zip(hf_ids, fast_ids, strict=True)) if a != b]

    ref_texts = texts[:REFERENCE_QUERIES]
    enc = hf_tok(ref_texts, padding=True, truncation=True, max_length=MAX_SEQ_TOKENS,
                 return_tensors="pt")  # fmt: skip
    with torch.inference_mode():
        cls = wrapper(enc["input_ids"], enc["attention_mask"]).numpy()
    np.savez(REFERENCE, cls=cls, texts=np.array(ref_texts))
    graph = onnx.load(str(out_file), load_external_data=False).graph
    return {
        "model": f"{EMBED_MODEL}@{EMBED_REVISION}",
        "attn_implementation": "eager",
        "opset": OPSET,
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "export_s": round(export_s, 1),
        "inputs": [i.name for i in graph.input],
        "outputs": [o.name for o in graph.output],
        "nodes": len(graph.node),
        "bytes": out_file.stat().st_size,
        "sha256": sha256(out_file),
        "tokenizer": {"file": TOKENIZER.as_posix(), "sha256": sha256(TOKENIZER)},
        "tokenizer_parity": {"queries": len(texts), "ids_differ": len(differ), "first": differ[:5]},
        "reference": {"file": REFERENCE.as_posix(), "queries": len(ref_texts)},
    }


# ---- optimize + int8 (fl-onnx venv: onnxruntime + onnx) ----------------------------------------


def run_optimize(args: argparse.Namespace) -> dict:
    import onnx
    import onnxruntime
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.transformers.optimizer import optimize_model

    started = time.perf_counter()
    # opt_level=0: only the Python fusion passes; higher levels import torch (T8)
    model = optimize_model(str(GRAPHS["eager"]), model_type="bert", num_heads=NUM_HEADS,
                           hidden_size=HIDDEN, opt_level=0)  # fmt: skip
    model.save_model_to_file(str(GRAPHS["opt-fp32"]))
    fused = {k: v for k, v in model.get_fused_operator_statistics().items() if v}
    opt_s = time.perf_counter() - started
    started = time.perf_counter()
    # The fused graph uses ONNX Runtime's own operators (com.microsoft Attention, SkipLayerNorm,
    # ...), which standard shape inference can't type, so the quantizer can't find the type of
    # the MatMul after the first fused attention (26 Nov). Every tensor here is float apart from
    # the int64 token inputs, so float is the right default.
    quantize_dynamic(str(GRAPHS["opt-fp32"]), str(GRAPHS["int8"]), weight_type=QuantType.QInt8,
                     extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT})  # fmt: skip
    quant_s = time.perf_counter() - started
    return {
        "onnxruntime": onnxruntime.__version__,
        "fused_operators": fused,
        "nodes_opt": len(model.model.graph.node),
        "graphs": {
            name: {"bytes": GRAPHS[name].stat().st_size, "sha256": sha256(GRAPHS[name])}
            for name in ("opt-fp32", "int8")
        },
        "optimize_s": round(opt_s, 1),
        "quantize_s": round(quant_s, 1),
        "quantization": "dynamic, weights QInt8, activations quantized at run time",
        "quantize_extra_options": {"DefaultTensorType": "FLOAT"},
    }


# ---- check (main env, uv --group api: onnxruntime + tokenizers) --------------------------------


def run_check(args: argparse.Namespace) -> dict:
    import onnxruntime

    from filinglens.api.embedder import OnnxEmbedder
    from filinglens.retrieval.embed import exact_recall_at_k, read_ids

    if args.split != "dev":
        raise SystemExit("the embedder check runs on dev only")
    rule = json.loads(RULE.read_text())
    texts, queries = dev_questions()
    grades = [q["grades"] for q in queries]
    stored = np.load(EMB_DIR / "queries.dev.npy")
    stored_ids = read_ids(EMB_DIR / "query_ids.dev.txt")
    if stored_ids != [q["query_id"] for q in queries]:
        raise SystemExit("FL-M3's stored dev query vectors are not in the query order")
    chunks = np.load(EMB_DIR / "chunks.npy", mmap_mode="r")
    chunk_ids = read_ids(EMB_DIR / "chunk_ids.txt")
    chunk_arr = np.asarray(chunks)
    ref = np.load(REFERENCE)

    results: dict = {}
    for name in ("opt-fp32", "int8"):
        embedder = OnnxEmbedder(GRAPHS[name], TOKENIZER, threads=1)
        for t in texts[:5]:  # warm-up, untimed
            embedder.embed_texts([t])
        vectors, ms = [], []
        for t in texts:
            started = time.perf_counter()
            vectors.append(embedder.embed_texts([t])[0])
            ms.append((time.perf_counter() - started) * 1000)
        vecs = np.stack(vectors)
        ref_ids, ref_mask = embedder.encode_ids([str(x) for x in ref["texts"]])
        (ref_cls,) = embedder.session.run(
            ["cls"], {"input_ids": ref_ids, "attention_mask": ref_mask}
        )
        results[name] = {
            "bytes": GRAPHS[name].stat().st_size,
            "latency_1_thread": latency_summary(ms),
            "min_cosine_vs_fl_m3": round(min_cosine(vecs, stored), 6),
            "same_top10_vs_fl_m3": same_top_k(vecs, stored, chunk_arr),
            "exact_recall_at_10": round(exact_recall_at_k(vecs, chunk_arr, chunk_ids, grades), 4),
            "max_abs_cls_diff_vs_torch_fp32": round(float(np.abs(ref_cls - ref["cls"]).max()), 6),
        }
    return {
        "onnxruntime": onnxruntime.__version__,
        "queries": len(texts),
        "fl_m3_dev_recall_at_10": rule["fl_m3_dev_recall_at_10"],
        "results": results,
        "decision": rule_decision(rule, results),
        "rule": RULE.as_posix(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["export", "optimize", "check"])
    parser.add_argument("--split", default="dev")
    parser.add_argument("--bench", type=Path, default=Path("bench"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    body = {"export": run_export, "optimize": run_optimize, "check": run_check}[args.command](args)
    result = {
        "command": " ".join(["python -m filinglens.serve.onnx_embedder", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **body,
    }
    out = args.bench / f"embed_onnx_{args.command}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
