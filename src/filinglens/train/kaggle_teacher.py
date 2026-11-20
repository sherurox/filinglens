"""Teacher labels for the FL-M4 distillation arm, run on Kaggle (Qwen/Qwen3-8B-AWQ, 2x T4) (T4).

Same task as the silver labeler and the FL-M2 baseline: the rules in silver_prompt.md, the same
user message with earlier-year candidates (silver_label.messages), the same JSON schema, greedy
decoding, thinking off. The teacher is not Granite (the silver labeler), so the student is not
trained on the answers it is scored against.

Confidence: at the generation step where the category value starts, vLLM returns the top-20
log-probabilities. Each category's first token at that step is found in context (the text of
that token before the value, plus the category name, tokenized from that token boundary), and
train.confidence.category_confidence renormalizes over the 12 allowed categories. The raw top-20
and first-token ids are stored too, so the filter (T6) can recompute. `first_token_consistent`
records whether the token the teacher actually generated equals the first token computed for the
category it chose: a check that the in-context tokenization is right.

T4 (Turing) has no bf16 and no Marlin kernels: dtype float16, quantization picked by vLLM (AWQ).
One process per GPU (data parallel, --shard/--num-shards): the 8B AWQ model fits one 16 GB T4.

    CUDA_VISIBLE_DEVICES=0 python -m filinglens.train.kaggle_teacher --split pool --limit 200

Top-level imports are standard library only (plus filinglens modules that are), so CI can test
the helpers; vLLM is imported in generate().
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from filinglens.extract.silver_label import CATEGORIES, messages, parse, schema
from filinglens.train.confidence import category_confidence

MODEL = "Qwen/Qwen3-8B-AWQ"
SPLITS = ("pool", "dev", "test")
TOP_K = 20
CATEGORY_KEY = re.compile(r'"category"\s*:\s*"')


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select(rows: list[dict], limit: int | None, shard: int, num_shards: int) -> list[dict]:
    """The first `limit` rows (a balanced prefix of pool.jsonl), then every num_shards-th."""
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard {shard} not in [0, {num_shards})")
    head = rows if limit is None else rows[:limit]
    return [r for i, r in enumerate(head) if i % num_shards == shard]


def token_starts(ids: list[int], decode: Callable[[list[int]], str]) -> list[int]:
    """Character offset where each generated token starts in the decoded text."""
    return [len(decode(ids[:i])) for i in range(len(ids))]


def category_step(
    ids: list[int],
    decode: Callable[[list[int]], str],
    encode: Callable[[str], list[int]],
) -> dict | None:
    """The generation step whose token starts the category value, and each category's first
    token at that step. None if the output has no category field."""
    text = decode(ids)
    match = CATEGORY_KEY.search(text)
    if match is None or not ids:
        return None
    value_start = match.end()
    starts = token_starts(ids, decode)
    index = max(i for i, s in enumerate(starts) if s <= value_start)
    lead = text[starts[index] : value_start]  # '' at a clean boundary, else e.g. '"'
    first_tokens = {c: encode(lead + c)[0] for c in CATEGORIES}
    return {
        "index": index,
        "lead": lead,
        "generated_token": ids[index],
        "first_tokens": first_tokens,
    }


def build_record(
    task: dict,
    raw: str,
    ids: list[int],
    step_logprobs: list[dict[int, float]],
    decode: Callable[[list[int]], str],
    encode: Callable[[str], list[int]],
    model: str,
) -> dict:
    """One output row: the parsed answer with rule checks, plus the category-step confidence."""
    record = parse(raw, task) | {"split": task.get("split"), "teacher": model}
    record["task"] = task
    step = category_step(ids, decode, encode)
    record["category_step"] = None
    record["confidence"] = None
    record["confidence_error"] = None
    record["first_token_consistent"] = None
    if step is None:
        record["confidence_error"] = "no category field in the output"
        return record
    top = step_logprobs[step["index"]] if step["index"] < len(step_logprobs) else {}
    record["category_step"] = step | {"top_logprobs": {str(k): v for k, v in top.items()}}
    chosen = record.get("category")
    if chosen in step["first_tokens"]:
        record["first_token_consistent"] = step["first_tokens"][chosen] == step["generated_token"]
    try:
        record["confidence"] = category_confidence(top, step["first_tokens"])
    except ValueError as exc:  # two categories share a first token at this step
        record["confidence_error"] = str(exc)
    return record


def summarize(records: list[dict]) -> dict:
    valid = [r for r in records if r["valid"]]
    chosen_conf = [
        r["confidence"][r["category"]]
        for r in valid
        if r["confidence"] and r.get("category") in r["confidence"]
    ]
    return {
        "rows": len(records),
        "rule_valid": len(valid),
        "invalid_problems": dict(Counter(p for r in records for p in r["problems"])),
        "categories": dict(Counter(r["category"] for r in valid).most_common()),
        "first_token_consistent": sum(r["first_token_consistent"] is True for r in records),
        "first_token_inconsistent": sum(r["first_token_consistent"] is False for r in records),
        "confidence_errors": dict(
            Counter(r["confidence_error"] for r in records if r["confidence_error"])
        ),
        "chosen_category_confidence_median": (
            sorted(chosen_conf)[len(chosen_conf) // 2] if chosen_conf else None
        ),
        "unsure": sum(bool(r.get("unsure")) for r in valid),
    }


def generate(tasks: list[dict], args: argparse.Namespace, system_prompt: str) -> tuple:
    """Greedy, schema-constrained answers with top-20 logprobs per generated token."""
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(
        model=args.model,
        quantization=args.quantization,
        dtype="float16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        seed=0,
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=400,
        logprobs=TOP_K,
        structured_outputs=StructuredOutputsParams(json=schema()),
    )
    outputs = llm.chat(
        [messages(t, system_prompt) for t in tasks],
        params,
        chat_template_kwargs={"enable_thinking": False},
    )
    tokenizer = llm.get_tokenizer()
    results = []
    for out in outputs:
        completion = out.outputs[0]
        step_logprobs = [
            {int(tid): float(lp.logprob) for tid, lp in step.items()}
            for step in (completion.logprobs or [])
        ]
        results.append((completion.text, list(completion.token_ids), step_logprobs))
    return results, tokenizer, vllm.__version__


def find_inputs(root: Path) -> Path:
    """The folder holding pool.jsonl (Kaggle mounts datasets under /kaggle/input/...)."""
    if (root / "pool.jsonl").exists():
        return root
    hits = sorted(root.glob("**/pool.jsonl"))
    if not hits:
        raise FileNotFoundError(f"no pool.jsonl under {root}")
    return hits[0].parent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--split", choices=SPLITS, default="pool")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="first N rows (balanced prefix)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("/kaggle/working/teacher"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--quantization", default=None, help="default: from the model config")
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action="store_true")
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    if args.split == "test" and not args.allow_test:
        raise SystemExit("the test split is labeled once, for the final comparison: --allow-test")

    inputs = find_inputs(args.inputs)
    rows = read_jsonl(inputs / f"{args.split}.jsonl")
    tasks = select(rows, args.limit, args.shard, args.num_shards)
    system_prompt = (inputs / "silver_prompt.md").read_text()

    started = time.perf_counter()
    results, tokenizer, vllm_version = generate(tasks, args, system_prompt)
    elapsed = time.perf_counter() - started

    def decode(ids: list[int]) -> str:
        return tokenizer.decode(ids)

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    records = [
        build_record(task, raw, ids, lps, decode, encode, args.model)
        for task, (raw, ids, lps) in zip(tasks, results, strict=True)
    ]
    limit = f"_first{args.limit}" if args.limit else ""
    name = f"{args.split}{limit}_shard{args.shard}of{args.num_shards}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{name}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))

    import torch

    stats = {
        "command": " ".join(["python -m filinglens.train.kaggle_teacher", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": args.model,
        "vllm": vllm_version,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "inputs_manifest": (
            json.loads((inputs / "MANIFEST.json").read_text()).get("finished_utc")
            if (inputs / "MANIFEST.json").exists()
            else None
        ),
        "split": args.split,
        "limit": args.limit,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "elapsed_s_incl_model_load": round(elapsed, 1),
        "rows_per_s_incl_load": round(len(records) / elapsed, 3) if elapsed else None,
        "output": out.name,
        **summarize(records),
    }
    (args.out_dir / f"{name}_run.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
