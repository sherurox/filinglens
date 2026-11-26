r"""The FL-M3 reranker (bge-reranker-base@2cfc18c9) exported to ONNX, against PyTorch (FL-M5 T8).

Steps, each in the venv that has what it needs (data under data/interim/serve/onnx/):
    ~/venvs/fl-train/bin/python -m filinglens.serve.onnx_reranker export
    ~/venvs/fl-train/bin/python -m filinglens.serve.onnx_reranker inputs --split dev
    ~/venvs/fl-train/bin/python -m filinglens.serve.onnx_reranker score --split dev \
        --engine torch --device cuda --dtype fp16 --note "..."      # FL-M3's production setting
    ~/venvs/fl-train/bin/python -m filinglens.serve.onnx_reranker score --split dev \
        --engine torch --device cpu --dtype fp32 --note "..."
    PYTHONPATH=src ~/venvs/fl-onnx/bin/python -m filinglens.serve.onnx_reranker score --split dev \
        --engine ort --device cpu --note "..."                       # and --device cuda
    PYTHONPATH=src ~/venvs/fl-onnx/bin/python -m filinglens.serve.onnx_reranker optimize
    PYTHONPATH=src ~/venvs/fl-onnx/bin/python -m filinglens.serve.onnx_reranker score --split dev \
        --engine ort --device cuda --graph opt-fp16 --note "..."     # and cpu --graph opt-fp32
    uv run python -m filinglens.serve.onnx_reranker compare --split dev

`export`: fp32 ONNX graph (inputs input_ids / attention_mask, dynamic batch and length; output one
logit per pair), checked with onnx.checker. `inputs`: each query's DEPTH (question, chunk) pairs
from FL-M3's candidates file, tokenized exactly as retrieval.rerank does (padding to the longest
pair, only the chunk truncated, 512 tokens), saved once so every engine scores identical token IDs;
tokenization time per query is recorded here, once. `score`: forward pass only, one query's pairs
per call, WARMUP untimed queries first, logits saved per query. `compare`: largest logit difference
against both PyTorch references, queries whose top 10 is identical, NDCG@10 / recall@10 / MRR@10
with FL-M3's own rerank_order and metrics, and the latency table (bench/reranker_onnx_<split>.json,
dev also bench/reranker_onnx.csv). The test split needs --allow-test and runs once, under
bench/m5_onnx_rule.json.

`export --attn eager` (26 Nov): the optimizer fused SkipLayerNormalization (24) and BiasGelu (12)
but no Attention: transformers 5 runs this model with SDPA attention, whose TorchScript export is
not the MatMul -> scale -> mask -> Softmax -> MatMul subgraph the fusion matches. The eager
attention implementation is that subgraph, so it is exported to *.eager.onnx and optimized apart.

`optimize` (26 Nov): the plain export is slower than PyTorch (TorchScript export splits attention
into hundreds of small ops) and the GPU comparison was fp32 ORT vs fp16 PyTorch. ONNX Runtime's own
transformer optimizer fuses attention, LayerNorm and GELU (BERT-style graph; bge-reranker-base is
XLM-RoBERTa base: 12 heads, hidden 768) and writes an fp32 and an fp16 graph (fp16 inside, fp32
inputs/outputs), so ort-cuda-fp16-opt is the like-for-like match for PyTorch's fp16 GPU setting.

Module-level imports are standard library + numpy only, so `score --engine ort` runs in a venv with
just onnxruntime and numpy.
"""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from filinglens.retrieval.config import MAX_SEQ_TOKENS, RERANK_MODEL, RERANK_REVISION

ONNX_DIR = Path("data/interim/serve/onnx")
MODEL_FILE = ONNX_DIR / "bge-reranker-base.onnx"
GRAPHS = {
    "plain": MODEL_FILE,
    "opt-fp32": ONNX_DIR / "bge-reranker-base.opt-fp32.onnx",
    "opt-fp16": ONNX_DIR / "bge-reranker-base.opt-fp16.onnx",
    "eager": ONNX_DIR / "bge-reranker-base.eager.onnx",
    "eager-opt-fp32": ONNX_DIR / "bge-reranker-base.eager.opt-fp32.onnx",
    "eager-opt-fp16": ONNX_DIR / "bge-reranker-base.eager.opt-fp16.onnx",
}
SOURCES = {"plain": "sdpa", "eager": "eager"}  # graph -> attention implementation at export
NUM_HEADS, HIDDEN = 12, 768  # XLM-RoBERTa base (bge-reranker-base config.json)
OPSET = 17
WARMUP = 3
REFERENCES = ("torch-cpu-fp32", "torch-cuda-fp16")  # what every variant is compared against


def variant_name(engine: str, device: str, dtype: str, graph: str = "plain") -> str:
    """torch-cuda-fp16, ort-cpu-fp32, ort-cuda-fp16-opt, ort-cuda-fp16-eager-opt, ..."""
    suffix = ("-eager" if graph.startswith("eager") else "") + ("-opt" if "opt" in graph else "")
    return f"{engine}-{device}-{dtype}{suffix}"


def graph_dtype(graph: str) -> str:
    return "fp16" if graph.endswith("fp16") else "fp32"


def guard_split(split: str, allow_test: bool) -> None:
    if split == "test" and not allow_test:
        raise PermissionError(
            "ret-test is touched once, under bench/m5_onnx_rule.json: --allow-test"
        )


def save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def query_keys(n: int) -> list[str]:
    return [f"q{i:05d}" for i in range(n)]


def latency_summary(ms: list[float]) -> dict:
    arr = np.asarray(ms, dtype=float)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    return {
        "p50_ms": round(float(p50), 2),
        "p95_ms": round(float(p95), 2),
        "p99_ms": round(float(p99), 2),
        "mean_ms": round(float(arr.mean()), 2),
        "samples": int(arr.size),
    }


def max_abs_diff(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    if a.keys() != b.keys():
        raise ValueError("score files cover different queries")
    return float(max(np.abs(a[k].astype(np.float64) - b[k]).max() for k in a))


def same_top_k(scores_a: np.ndarray, scores_b: np.ndarray, k: int) -> bool:
    """Same top-k list in the same order (stable sort: equal scores keep candidate order)."""
    order_a = np.argsort(-scores_a, kind="stable")[:k]
    order_b = np.argsort(-scores_b, kind="stable")[:k]
    return bool(np.array_equal(order_a, order_b))


# ---- export / inputs (fl-train venv: torch + transformers) -------------------------------------


def run_export(args: argparse.Namespace) -> dict:
    import onnx
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL, revision=RERANK_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANK_MODEL, revision=RERANK_REVISION, dtype=torch.float32, attn_implementation=args.attn
    ).eval()
    out_file = MODEL_FILE if args.attn == "sdpa" else GRAPHS["eager"]

    class Logits(torch.nn.Module):  # plain tensors in, one logit per pair out
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask).logits[:, 0]

    wrapper = Logits(model).eval()
    sample = tokenizer(["a question"] * 2, ["one chunk", "a longer second chunk"], padding=True,
                       return_tensors="pt")  # fmt: skip
    inputs = (sample["input_ids"], sample["attention_mask"])
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    axes = {"input_ids": {0: "pairs", 1: "tokens"}, "attention_mask": {0: "pairs", 1: "tokens"},
            "logits": {0: "pairs"}}  # fmt: skip
    started = time.perf_counter()
    exporter, errors = None, []
    for dynamo in (False, True):  # the TorchScript exporter first; the dynamo one if it's gone
        try:
            torch.onnx.export(wrapper, inputs, str(out_file), input_names=list(axes)[:2],
                              output_names=["logits"], dynamic_axes=axes, opset_version=OPSET,
                              dynamo=dynamo)  # fmt: skip
            exporter = "dynamo" if dynamo else "torchscript"
            break
        except Exception as error:  # recorded; the other exporter is tried
            errors.append(f"dynamo={dynamo}: {type(error).__name__}: {str(error)[:300]}")
    if exporter is None:
        raise RuntimeError("both exporters failed: " + " | ".join(errors))
    export_s = time.perf_counter() - started
    onnx.checker.check_model(str(out_file))
    graph = onnx.load(str(out_file), load_external_data=False).graph
    with torch.inference_mode():
        reference = wrapper(*inputs).numpy()
    return {
        "model": f"{RERANK_MODEL}@{RERANK_REVISION}",
        "attn_implementation": args.attn,
        "exporter": exporter,
        "exporter_errors": errors,
        "opset": OPSET,
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "export_s": round(export_s, 1),
        "inputs": [i.name for i in graph.input],
        "outputs": [o.name for o in graph.output],
        "nodes": len(graph.node),
        "bytes": out_file.stat().st_size,
        "sample_logits_torch_fp32": [round(float(x), 6) for x in reference],
        "out": out_file.as_posix(),
    }


def run_inputs(args: argparse.Namespace) -> dict:
    import pandas as pd
    from transformers import AutoTokenizer

    from filinglens.retrieval.config import CHUNKS, RET_DIR
    from filinglens.retrieval.queries import OUT_DIR as QUERY_DIR
    from filinglens.retrieval.queries import load_queries

    queries = load_queries(args.split, QUERY_DIR, allow_test=args.allow_test)
    cand_path = RET_DIR / f"candidates.{args.split}.jsonl"
    cands = [json.loads(line) for line in cand_path.read_text().splitlines() if line.strip()]
    if [c["query_id"] for c in cands] != [q["query_id"] for q in queries]:
        raise ValueError(f"{cand_path} doesn't match the {args.split} queries")
    texts = pd.read_parquet(CHUNKS, columns=["chunk_id", "embed_text"])
    text_of = dict(zip(texts["chunk_id"], texts["embed_text"], strict=True))
    tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL, revision=RERANK_REVISION)

    arrays, tok_ms, pairs = {}, [], 0
    for key, q, c in zip(query_keys(len(cands)), queries, cands, strict=True):
        chunk_texts = [text_of[i] for i in c["candidates"]]
        started = time.perf_counter()
        enc = tokenizer(
            [q["question"]] * len(chunk_texts),
            chunk_texts,
            padding=True,
            truncation="only_second",  # as retrieval.rerank: cut the chunk, never the question
            max_length=MAX_SEQ_TOKENS,
            return_tensors="np",
        )
        tok_ms.append((time.perf_counter() - started) * 1000)
        arrays[f"{key}_ids"] = enc["input_ids"].astype(np.int64)
        arrays[f"{key}_mask"] = enc["attention_mask"].astype(np.int64)
        pairs += len(chunk_texts)
    out = ONNX_DIR / f"inputs_{args.split}.npz"
    save_arrays(out, arrays)
    meta = {"query_ids": [q["query_id"] for q in queries], "candidates": str(cand_path)}
    (ONNX_DIR / f"inputs_{args.split}.json").write_text(json.dumps(meta) + "\n")
    return {
        "queries": len(cands),
        "pairs": pairs,
        "tokenize_latency": latency_summary(tok_ms),
        "out": out.as_posix(),
    }


def run_optimize(args: argparse.Namespace) -> dict:
    """Fused fp32 and fp16 graphs from the plain export (fl-onnx venv: onnxruntime + onnx)."""
    import onnxruntime
    from onnxruntime.transformers.optimizer import optimize_model

    source = GRAPHS[args.source]
    prefix = "" if args.source == "plain" else f"{args.source}-"
    out = {"onnxruntime": onnxruntime.__version__, "num_heads": NUM_HEADS, "hidden": HIDDEN,
           "source": source.name}  # fmt: skip
    for graph, fp16 in ((f"{prefix}opt-fp32", False), (f"{prefix}opt-fp16", True)):
        started = time.perf_counter()
        # opt_level=0: only the Python fusion passes (Attention, SkipLayerNorm, Gelu, ...). Higher
        # levels first run onnxruntime's own optimizer, whose code imports torch, which this venv
        # doesn't have (26 Nov); onnxruntime still applies its graph optimizations at load time.
        model = optimize_model(
            str(source), model_type="bert", num_heads=NUM_HEADS, hidden_size=HIDDEN, opt_level=0
        )
        if fp16:
            model.convert_float_to_float16(keep_io_types=True)  # fp32 in/out, fp16 inside
        model.save_model_to_file(str(GRAPHS[graph]))
        out[graph] = {
            "fused_operators": {
                k: v for k, v in model.get_fused_operator_statistics().items() if v
            },
            "nodes": len(model.model.graph.node),
            "bytes": GRAPHS[graph].stat().st_size,
            "seconds": round(time.perf_counter() - started, 1),
        }
    return out


# ---- score (torch in fl-train; onnxruntime in fl-onnx) ------------------------------------------


def torch_scorer(device: str, dtype: str):
    import torch
    from transformers import AutoModelForSequenceClassification

    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANK_MODEL, revision=RERANK_REVISION, dtype=torch_dtype
    )
    model = model.to(device).eval()

    def score(ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            logits = model(
                input_ids=torch.from_numpy(ids).to(device),
                attention_mask=torch.from_numpy(mask).to(device),
            ).logits[:, 0]
            if device == "cuda":
                torch.cuda.synchronize()
            return logits.float().cpu().numpy()

    info = {"torch": torch.__version__, "threads": torch.get_num_threads()}
    return score, info


def ort_scorer(device: str, model: Path = MODEL_FILE):
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()  # the CUDA / cuDNN libraries from the pip packages, if installed
    wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else [
        "CPUExecutionProvider"
    ]  # fmt: skip
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model), sess_options=options, providers=wanted)
    used = session.get_providers()
    if device == "cuda" and used[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA execution provider not used; session providers: {used}")

    def score(ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return session.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]

    info = {
        "onnxruntime": ort.__version__,
        "graph": model.name,
        "provider_options": session.get_provider_options(),
        "available_providers": ort.get_available_providers(),
        "session_providers": used,
        "intra_op_threads": options.intra_op_num_threads,
    }
    return score, info


def run_score(args: argparse.Namespace) -> dict:
    inputs = load_arrays(ONNX_DIR / f"inputs_{args.split}.npz")
    n = len(inputs) // 2
    keys = query_keys(n)
    if args.split == "test":  # touched once, under bench/m5_onnx_rule.json
        planned = variant_name(
            args.engine,
            args.device,
            args.dtype if args.engine == "torch" else graph_dtype(args.graph),
            args.graph,
        )
        if (ONNX_DIR / f"scores_test_{planned}.npz").exists():
            raise FileExistsError(f"ret-test was already scored with {planned}")
    if args.engine == "torch":
        if args.graph != "plain":
            raise ValueError("--graph applies to --engine ort only")
        dtype = args.dtype
        score, info = torch_scorer(args.device, dtype)
    else:
        dtype = graph_dtype(args.graph)
        score, info = ort_scorer(args.device, GRAPHS[args.graph])
    for key in keys[:WARMUP]:  # untimed: kernels, allocator, graph optimisation
        score(inputs[f"{key}_ids"], inputs[f"{key}_mask"])
    scores, ms = {}, []
    for key in keys:
        started = time.perf_counter()
        scores[key] = score(inputs[f"{key}_ids"], inputs[f"{key}_mask"]).astype(np.float32)
        ms.append((time.perf_counter() - started) * 1000)
    name = variant_name(args.engine, args.device, dtype, args.graph)
    save_arrays(ONNX_DIR / f"scores_{args.split}_{name}.npz", scores)
    from filinglens.train.tracking import gpu_snapshot

    return {
        "variant": name,
        "queries": n,
        "forward_latency": latency_summary(ms),
        "engine_info": info,
        "gpu": gpu_snapshot() if args.device == "cuda" else {},
        "cpu": platform.processor() or platform.machine(),
        "power_note": args.note,
    }


# ---- compare (uv env) ------------------------------------------------------------------------


def run_compare(args: argparse.Namespace) -> dict:
    from filinglens.retrieval.queries import OUT_DIR as QUERY_DIR
    from filinglens.retrieval.queries import load_queries
    from filinglens.retrieval.rerank import K, mean_metrics, per_query_metrics, rerank_order

    queries = load_queries(args.split, QUERY_DIR, allow_test=args.allow_test)
    meta = json.loads((ONNX_DIR / f"inputs_{args.split}.json").read_text())
    if meta["query_ids"] != [q["query_id"] for q in queries]:
        raise ValueError("inputs were built for different queries")
    cands_path = Path(meta["candidates"])
    cands = [json.loads(line) for line in cands_path.read_text().splitlines() if line.strip()]
    keys = query_keys(len(queries))
    runs = {}
    for path in sorted(ONNX_DIR.glob(f"scores_{args.split}_*.npz")):
        name = path.stem.removeprefix(f"scores_{args.split}_")
        record_path = args.bench / f"reranker_onnx_score_{args.split}_{name}.json"
        runs[name] = {
            "scores": load_arrays(path),
            "record": json.loads(record_path.read_text()) if record_path.is_file() else {},
        }
    variants = {}
    for name, run in runs.items():
        scores = run["scores"]
        rows = []
        for key, q, c in zip(keys, queries, cands, strict=True):
            order = rerank_order(c["candidates"], scores[key].tolist())
            rows.append(per_query_metrics(c["candidates"], order, q["grades"]))
        entry = {
            "metrics": mean_metrics(rows),
            "forward_latency": run["record"].get("forward_latency"),
            "engine_info": run["record"].get("engine_info"),
        }
        for ref in REFERENCES:
            if ref in runs and ref != name:
                other = runs[ref]["scores"]
                entry[f"max_abs_logit_diff_vs_{ref}"] = round(max_abs_diff(scores, other), 6)
                entry[f"same_top{K}_vs_{ref}"] = sum(
                    same_top_k(scores[k], other[k], K) for k in keys
                )
        variants[name] = entry
    inputs_record = args.bench / f"reranker_onnx_inputs_{args.split}.json"
    tokenize = json.loads(inputs_record.read_text()).get("tokenize_latency")
    result = {"queries": len(queries), "k": K, "tokenize_latency": tokenize, "variants": variants}
    if args.split == "test":
        rule_path = args.bench / "m5_onnx_rule.json"
        rule = json.loads(rule_path.read_text())
        fl_m3 = json.loads((args.bench / "reranker_test.json").read_text())
        result["rule"] = rule_path.as_posix()
        result["rule_checks"] = rule_checks(rule, variants, fl_m3["metrics"]["ndcg_after"])
    return result


def rule_checks(rule: dict, variants: dict, fl_m3_ndcg: float) -> dict:
    """The two checks of bench/m5_onnx_rule.json on the test variants' mean metrics."""
    ref = variants[rule["reference_variant"]]["metrics"]
    cand = variants[rule["candidate_variant"]]["metrics"]
    reproduce_gap = abs(ref["ndcg_after"] - fl_m3_ndcg)
    gaps = {m: round(abs(cand[m] - ref[m]), 6) for m in rule["match_metrics"]}
    return {
        "fl_m3_test_ndcg_at_10": fl_m3_ndcg,
        "reference_ndcg_at_10": ref["ndcg_after"],
        "reproduce_gap": round(reproduce_gap, 6),
        "pipeline_reproduces_fl_m3": reproduce_gap <= rule["reproduce_tolerance"],
        "candidate_vs_reference_gaps": gaps,
        "export_matches": all(g <= rule["match_tolerance"] for g in gaps.values()),
    }


def write_csv(result: dict, path: Path) -> None:
    import csv

    refs = [f"max_abs_logit_diff_vs_{r}" for r in REFERENCES]
    same = [f"same_top{result['k']}_vs_{r}" for r in REFERENCES]
    fields = ["variant", "ndcg_at_10", "recall_at_10", "mrr_at_10", "forward_p50_ms",
              "forward_p95_ms", "forward_p99_ms", "tokenize_p50_ms", *refs, *same,
              "queries"]  # fmt: skip
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, v in sorted(result["variants"].items()):
            lat = v.get("forward_latency") or {}
            writer.writerow({
                "variant": name,
                "ndcg_at_10": v["metrics"]["ndcg_after"],
                "recall_at_10": v["metrics"]["recall_after"],
                "mrr_at_10": v["metrics"]["mrr_after"],
                "forward_p50_ms": lat.get("p50_ms"),
                "forward_p95_ms": lat.get("p95_ms"),
                "forward_p99_ms": lat.get("p99_ms"),
                "tokenize_p50_ms": (result.get("tokenize_latency") or {}).get("p50_ms"),
                **{k: v.get(k, "") for k in refs + same},
                "queries": result["queries"],
            })  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    export = sub.add_parser("export")
    export.add_argument("--attn", choices=sorted(SOURCES.values()), default="sdpa")
    optimize = sub.add_parser("optimize")
    optimize.add_argument("--source", choices=sorted(SOURCES), default="plain")
    for name in ("inputs", "score", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--split", choices=["dev", "test"], required=True)
        p.add_argument("--allow-test", action="store_true")
    score = sub.choices["score"]
    score.add_argument("--engine", choices=["torch", "ort"], required=True)
    score.add_argument("--device", choices=["cuda", "cpu"], required=True)
    score.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32", help="torch only")
    score.add_argument("--graph", choices=sorted(GRAPHS), default="plain", help="ort only")
    score.add_argument("--note", required=True, help="power source and Windows power mode")
    for p in sub.choices.values():
        p.add_argument("--bench", type=Path, default=Path("bench"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    if args.step not in ("export", "optimize"):
        guard_split(args.split, args.allow_test)

    body = {"export": run_export, "optimize": run_optimize, "inputs": run_inputs,
            "score": run_score, "compare": run_compare}[args.step](args)  # fmt: skip
    result = {
        "command": "python -m filinglens.serve.onnx_reranker " + shlex.join(argv),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **body,
    }
    split = getattr(args, "split", None)
    name = {
        "export": "reranker_onnx_export.json"
        if getattr(args, "attn", "sdpa") == "sdpa"
        else f"reranker_onnx_export_{args.attn}.json",
        "optimize": "reranker_onnx_optimize.json"
        if getattr(args, "source", "plain") == "plain"
        else f"reranker_onnx_optimize_{args.source}.json",
        "inputs": f"reranker_onnx_inputs_{split}.json",
        "score": f"reranker_onnx_score_{split}_{body.get('variant')}.json",
        "compare": f"reranker_onnx_{split}.json",
    }[args.step]
    (args.bench / name).write_text(json.dumps(result, indent=2) + "\n")
    if args.step == "compare" and split == "dev":
        write_csv(result, args.bench / "reranker_onnx.csv")
    print(json.dumps(result, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
