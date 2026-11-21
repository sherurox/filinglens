"""Prompted baseline: risk category, severity and first-seen year from a small model (FL-M2 T11).

The model is Qwen3-1.7B, the base of the student that M4 fine-tunes, so M4 compares like with like.
It gets the same rules as the silver labeler (docs/labeling/silver_prompt.md), the same paragraph,
heading and earlier-year candidates, and the same JSON schema. Two variants:
- zero: rules only;
- fewshot: plus one worked example per category, taken from train rows where both silver labelers
  agreed (the example answers are the silver labels; the rationale is the primary labeler's).

The labels being predicted are silver (model-made), so the scores measure agreement with the silver
labeler, reported on all rows and on the rows where the two silver labelers agreed.

Generation runs in the GPU venv; scoring runs in the main environment (scikit-learn):
    ~/venvs/fl-gpu/bin/python -m filinglens.extract.fields generate --variant fewshot --split dev
    uv run python -m filinglens.extract.fields score --variant fewshot --split dev
The test split needs --allow-test in both.

FL-M4 (T9): `generate --lora data/interim/adapters/<run>/adapter` serves the same base model with a
LoRA adapter (vLLM enable_lora) on exactly the same prompts; outputs get a suffix (the run name,
or --tag), so the FL-M2 baseline files are never overwritten. Without --lora and --tag, names and
behavior are unchanged.

Batch-dependent numerics (found 21 Nov): the same base model on the same 71 dev prompts gave
category macro-F1 0.5792 (19 Nov) and 0.6255 (21 Nov), with 33/71 identical raw outputs. vLLM sizes
its KV cache from the GPU memory free at start, so batching (and bf16 rounding at near-ties)
varied between runs. FL-M4 comparisons therefore run with --max-num-seqs 1.
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from filinglens.extract.silver_label import (
    CATEGORIES,
    PeakGpuMemory,
    messages,
    parse,
    schema,
    user_message,
)

GOLD = Path("gold")
TASKS = Path("data/interim/label_tasks/risk_tasks.json")
RATIONALES = Path("gold/silver_raw/ibm-granite-granite-4-1-3b.jsonl")
PROMPT = Path("docs/labeling/silver_prompt.md")
OUT_DIR = Path("data/interim/fields")
MODEL = "Qwen/Qwen3-1.7B"
RISK_CATEGORIES = [c for c in CATEGORIES if c != "not_a_risk"]
EXAMPLE_WORDS = 90


def load_rows(split: str, allow_test: bool, gold: Path = GOLD) -> list[dict]:
    if split == "test" and not allow_test:
        raise PermissionError("the test split is reserved for final evaluation: --allow-test")
    path = gold / f"risk_paragraphs.{split}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def task_view(row: dict, tasks: dict[str, dict]) -> dict:
    """The fields user_message needs: the silver row plus its first-seen candidates."""
    return row | {"first_seen": tasks[row["para_id"]]["first_seen"]}


def pick_examples(train: list[dict]) -> list[dict]:
    """One train row per category, the lowest task number among rows both labelers agreed on."""
    chosen = {}
    for row in sorted(train, key=lambda r: r["task_no"]):
        if row["models_agree_category"] and row["category"] not in chosen:
            chosen[row["category"]] = row
    return [chosen[c] for c in RISK_CATEGORIES if c in chosen]


def cut(text: str, words: int) -> str:
    parts = text.split()
    return " ".join(parts[:words]) + (" ..." if len(parts) > words else "")


def example_view(row: dict, tasks: dict[str, dict]) -> dict:
    """A short version of a train row for the prompt: the paragraph cut to EXAMPLE_WORDS and only
    the candidate from the labeled first-seen year (cut to 30 words), to keep the prompt small."""
    first_seen = tasks[row["para_id"]]["first_seen"]
    match = [c for c in first_seen["candidates"] if c["year"] == row["first_seen_year"]][:1]
    match = [c | {"text": cut(c["text"], 30)} for c in match]
    return row | {
        "text": cut(row["text"], EXAMPLE_WORDS),
        "first_seen": {"suggested": first_seen["suggested"], "candidates": match},
    }


def example_turns(examples: list[dict], tasks: dict, rationales: dict) -> list[dict]:
    """User/assistant turns showing the expected answer for each example."""
    turns = []
    for row in examples:
        answer = {
            "rationale": rationales.get(row["para_id"], ""),
            "category": row["category"],
            "severity": row["severity"],
            "first_seen_year": row["first_seen_year"],
            "unsure": False,
        }
        turns.append({"role": "user", "content": user_message(example_view(row, tasks))})
        turns.append({"role": "assistant", "content": json.dumps(answer)})
    return turns


def conversations(rows, tasks, system_prompt, shots: list[dict]) -> list[list[dict]]:
    out = []
    for row in rows:
        system, user = messages(task_view(row, tasks), system_prompt)
        out.append([system, *shots, user])
    return out


def generate(convs: list[list[dict]], args: argparse.Namespace) -> list[str]:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    lora = getattr(args, "lora", None)
    extra = (
        {"enable_lora": True, "max_lora_rank": args.max_lora_rank, "max_loras": 1} if lora else {}
    )
    if getattr(args, "max_num_seqs", None):  # 1 = no batching: outputs can't depend on the batch
        extra["max_num_seqs"] = args.max_num_seqs
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        seed=0,
        **extra,
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=400,
        structured_outputs=StructuredOutputsParams(json=schema()),
    )
    request = {}
    if lora:
        from vllm.lora.request import LoRARequest

        request = {"lora_request": LoRARequest(output_suffix(args).lstrip("_"), 1, str(lora))}
    outputs = llm.chat(convs, params, chat_template_kwargs={"enable_thinking": False}, **request)
    return [o.outputs[0].text for o in outputs]


def run_generate(args: argparse.Namespace) -> dict:
    rows = load_rows(args.split, args.allow_test)
    tasks = {t["para_id"]: t for t in json.loads(TASKS.read_text())}
    shots: list[dict] = []
    example_ids: list[str] = []
    if args.variant == "fewshot":
        rationales = {
            r["para_id"]: r.get("rationale", "")
            for r in (json.loads(line) for line in RATIONALES.read_text().splitlines() if line)
        }
        examples = pick_examples(load_rows("train", allow_test=False))
        example_ids = [e["para_id"] for e in examples]
        shots = example_turns(examples, tasks, rationales)
    convs = conversations(rows, tasks, PROMPT.read_text(), shots)

    started = time.perf_counter()
    with PeakGpuMemory() as peak:
        raws = generate(convs, args)
    elapsed = time.perf_counter() - started
    records = [parse(raw, task_view(row, tasks)) for raw, row in zip(raws, rows, strict=True)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"fields_{args.variant}_{args.split}{output_suffix(args)}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    return {
        "model": args.model,
        "lora": str(args.lora) if getattr(args, "lora", None) else None,
        "max_num_seqs": getattr(args, "max_num_seqs", None),
        "variant": args.variant,
        "examples": example_ids,
        "rows": len(rows),
        "valid": sum(r["valid"] for r in records),
        "elapsed_s_incl_model_load": round(elapsed, 1),
        "peak_gpu_memory_mib": peak.peak_mib,
        "predictions": str(out),
    }


def macro_f1(gold: list, pred: list, labels: list) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0))


def bootstrap_f1(gold: list, pred: list, labels: list, n: int = 1000, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    g, p = np.asarray(gold, dtype=object), np.asarray(pred, dtype=object)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(g), len(g))
        values.append(macro_f1(list(g[idx]), list(p[idx]), labels))
    low, high = np.percentile(values, [2.5, 97.5])
    return round(float(low), 3), round(float(high), 3)


def scores(rows: list[dict], preds: dict[str, dict]) -> dict:
    """Macro-F1 for category and severity, first-seen accuracy, rule-valid rate."""
    pairs = [(r, preds.get(r["para_id"], {})) for r in rows]
    gold_cat = [r["category"] for r, _ in pairs]
    pred_cat = [p.get("category") if p.get("valid") else "invalid" for _, p in pairs]
    gold_sev = [r["severity"] for r, _ in pairs]
    # invalid answers and not_a_risk answers (severity null) both count as a wrong severity
    pred_sev = [p.get("severity") if p.get("valid") and p.get("severity") else -1 for _, p in pairs]
    years = [(r["first_seen_year"], p.get("first_seen_year")) for r, p in pairs]
    exact = sum(g == q for g, q in years)
    near = sum(q is not None and abs(g - q) <= 1 for g, q in years)
    return {
        "rows": len(rows),
        "rule_valid": sum(bool(p.get("valid")) for _, p in pairs),
        "category_macro_f1": round(macro_f1(gold_cat, pred_cat, RISK_CATEGORIES), 4),
        "category_macro_f1_ci95": bootstrap_f1(gold_cat, pred_cat, RISK_CATEGORIES),
        "category_accuracy": round(
            sum(g == q for g, q in zip(gold_cat, pred_cat, strict=True)) / len(rows), 4
        ),
        "severity_macro_f1": round(macro_f1(gold_sev, pred_sev, [1, 2, 3]), 4),
        "first_seen_exact": round(exact / len(rows), 4),
        "first_seen_within_1y": round(near / len(rows), 4),
    }


def run_score(args: argparse.Namespace) -> dict:
    rows = load_rows(args.split, args.allow_test)
    path = OUT_DIR / f"fields_{args.variant}_{args.split}{output_suffix(args)}.jsonl"
    preds = {r["para_id"]: r for r in map(json.loads, path.read_text().splitlines())}
    agreed = [r for r in rows if r["models_agree_category"]]
    return {
        "labels": "silver (agreement with ibm-granite/granite-4.1-3b, not correctness)",
        "predictions": str(path),
        "all_rows": scores(rows, preds),
        "rows_where_silver_labelers_agree": scores(agreed, preds),
    }


def output_suffix(args: argparse.Namespace) -> str:
    """'' for the FL-M2 baseline; '_<tag>' or '_<run name>' (the adapter's parent folder)."""
    tag = getattr(args, "tag", None)
    lora = getattr(args, "lora", None)
    if tag:
        return f"_{tag}"
    return f"_{Path(lora).parent.name}" if lora else ""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["generate", "score"])
    parser.add_argument("--variant", choices=["zero", "fewshot"], required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--bench", type=Path, default=Path("bench"))
    parser.add_argument("--lora", type=Path, default=None, help="LoRA adapter folder (FL-M4)")
    parser.add_argument("--max-lora-rank", type=int, default=16)
    parser.add_argument("--tag", default=None, help="output suffix (default: the run name)")
    parser.add_argument(
        "--max-num-seqs", type=int, default=None, help="1 = one request at a time (FL-M4 evals)"
    )
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    body = run_generate(args) if args.command == "generate" else run_score(args)
    suffix = "_run" if args.command == "generate" else ""
    result = {
        "command": " ".join(["python -m filinglens.extract.fields", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": args.split,
        **body,
    }
    out = args.bench / f"fields_{args.variant}_{args.split}{output_suffix(args)}{suffix}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
