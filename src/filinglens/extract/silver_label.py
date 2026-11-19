"""Silver labels for the risk-paragraph sample from a local open-weight model (FL-M2 T5).

Decision (19 Nov 2026): the 330 sampled paragraphs are labeled by a model, not by hand. These are
"silver" labels: F1 measured against them means agreement with the labeling model, and the docs
say so. Two different model families label every paragraph; their agreement (Cohen's kappa)
replaces the human test-retest check. The labeling model must not be the M2 baseline model or the
M4 teacher, or those would be graded against their own answers.

The rules are docs/labeling/silver_prompt.md, a condensed copy of docs/LABELING_GUIDE.md v1.0.
Decoding is greedy (temperature 0) and constrained to a JSON schema, so every answer parses;
rule checks (severity and year present for risks, year not after the paragraph's own year) run
afterwards and mark violations as invalid instead of fixing them.

Runs in the separate GPU environment (vLLM), as a module so WSL's spawn start method works:
    ~/venvs/fl-gpu/bin/python -m filinglens.extract.silver_label \
        --model ibm-granite/granite-4.1-3b --quantization fp8 \
        --gpu-memory-utilization 0.80 --enforce-eager --limit 5

VRAM: the laptop GPU has 8 GB and Windows' display already holds about 2 GB, so 8B models don't
fit even in 4 bits once the KV cache is counted (vLLM 0.34 also dropped in-flight bitsandbytes).
The labelers are therefore 3-4B models: Granite 4.1 3B quantized to FP8 while loading, and
Qwen3-4B-AWQ. The full runs used --gpu-memory-utilization 0.80 --enforce-eager: at 0.70, the
weights, activations and CUDA graphs left no room for the KV cache (measured in the smoke test).
"""

import argparse
import json
import re
import sys
import threading
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

CATEGORIES = [
    "not_a_risk",
    "macro_market",
    "competition_demand",
    "regulatory_legal",
    "technology_cyber",
    "operations_supply",
    "financial_capital",
    "people",
    "health_pandemic",
    "climate_environment",
    "geopolitical_international",
    "strategy_corporate",
]
FIRST_YEAR = 2018
PROMPT = Path("docs/labeling/silver_prompt.md")
TASKS = Path("data/interim/label_tasks/risk_tasks.json")
OUT_DIR = Path("data/interim/silver")


def schema(last_year: int = 2025) -> dict:
    """JSON schema for one answer; rationale first, so the model states a reason before labels."""
    return {
        "type": "object",
        "properties": {
            "rationale": {"type": "string"},
            "category": {"enum": CATEGORIES},
            "severity": {"enum": [1, 2, 3, None]},
            "first_seen_year": {"enum": [*range(FIRST_YEAR, last_year + 1), None]},
            "unsure": {"type": "boolean"},
        },
        "required": ["rationale", "category", "severity", "first_seen_year", "unsure"],
        "additionalProperties": False,
    }


def user_message(task: dict) -> str:
    lines = [
        f"Company: {task['ticker']} ({task['sector']}). Paragraph's fiscal year: "
        f"{task['period_year']}.",
        f"Heading (context only): {task['heading'] or '(none)'}",
        f"Paragraph: {task['text']}",
        "",
        "Most similar paragraph of the same company in each earlier year (year, overlap score):",
    ]
    candidates = task["first_seen"]["candidates"]
    lines += [f"- {c['year']} ({c['score']:.2f}): {c['heading']} | {c['text']}" for c in candidates]
    if not candidates:
        lines.append("- none (no earlier 10-K in the corpus)")
    return "\n".join(lines)


def messages(task: dict, system_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message(task)},
    ]


def check(answer: dict, period_year: int) -> list[str]:
    """Rule violations of one parsed answer (empty list = valid)."""
    problems = []
    if answer["category"] == "not_a_risk":
        if answer["severity"] is not None or answer["first_seen_year"] is not None:
            problems.append("not_a_risk with severity or year")
    else:
        if answer["severity"] is None:
            problems.append("risk without severity")
        if answer["first_seen_year"] is None:
            problems.append("risk without first_seen_year")
        elif answer["first_seen_year"] > period_year:
            problems.append("first_seen_year after the paragraph's year")
    return problems


def parse(raw: str, task: dict) -> dict:
    """One output record: the labels, the raw text and whether the answer follows the rules."""
    record = {"task_no": task["task_no"], "para_id": task["para_id"], "raw": raw}
    try:
        answer = json.loads(raw)
    except json.JSONDecodeError as exc:
        return record | {"valid": False, "problems": [f"invalid JSON: {exc}"]}
    problems = check(answer, task["period_year"])
    return record | answer | {"valid": not problems, "problems": problems}


def slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def summarize(records: list[dict]) -> dict:
    valid = [r for r in records if r["valid"]]
    return {
        "tasks": len(records),
        "valid": len(valid),
        "invalid_problems": dict(Counter(p for r in records for p in r["problems"])),
        "categories": dict(Counter(r["category"] for r in valid).most_common()),
        "severity": dict(sorted(Counter(str(r["severity"]) for r in valid).items())),
        "unsure": sum(bool(r.get("unsure")) for r in valid),
    }


class PeakGpuMemory:
    """Samples used GPU memory with NVML in a background thread (vLLM's engine runs in another
    process, so torch's own counters in this process would read zero)."""

    def __init__(self, interval: float = 0.5):
        self.peak_mib = None
        self._stop = threading.Event()
        self._interval = interval

    def __enter__(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # no NVML: report null, don't fail the labeling run
            return self
        self.peak_mib = 0

        def sample():
            while not self._stop.is_set():
                used = pynvml.nvmlDeviceGetMemoryInfo(handle).used // 2**20
                self.peak_mib = max(self.peak_mib, used)
                self._stop.wait(self._interval)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        return False


def generate(tasks: list[dict], args: argparse.Namespace, system_prompt: str) -> list[str]:
    """Greedy, schema-constrained answers from vLLM, in task order."""
    from vllm import LLM, SamplingParams

    try:  # vLLM >= 0.11
        from vllm.sampling_params import StructuredOutputsParams

        constrained = {"structured_outputs": StructuredOutputsParams(json=schema())}
    except ImportError:  # older vLLM
        from vllm.sampling_params import GuidedDecodingParams

        constrained = {"guided_decoding": GuidedDecodingParams(json=schema())}

    llm = LLM(
        model=args.model,
        quantization=args.quantization,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        seed=0,
    )
    params = SamplingParams(temperature=0.0, max_tokens=400, **constrained)
    outputs = llm.chat(
        [messages(t, system_prompt) for t in tasks],
        params,
        chat_template_kwargs={"enable_thinking": False},  # Qwen3: answer directly
    )
    return [o.outputs[0].text for o in outputs]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--quantization", default=None, help="e.g. fp8, awq_marlin")
    parser.add_argument("--tasks", type=Path, default=TASKS)
    parser.add_argument("--prompt", type=Path, default=PROMPT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--stats-dir", type=Path, default=Path("bench"))
    parser.add_argument("--limit", type=int, default=None, help="first N tasks (smoke test)")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--enforce-eager", action="store_true", help="no CUDA graphs: less VRAM")
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    tasks = json.loads(args.tasks.read_text())[: args.limit]
    system_prompt = args.prompt.read_text()
    started = time.perf_counter()
    with PeakGpuMemory() as peak:
        raws = generate(tasks, args, system_prompt)
    elapsed = time.perf_counter() - started
    records = [parse(raw, task) for raw, task in zip(raws, tasks, strict=True)]

    name = slug(args.model) + (f"-first{args.limit}" if args.limit else "")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{name}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    stats = {
        "command": " ".join(["python -m filinglens.extract.silver_label", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": args.model,
        "quantization": args.quantization,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "elapsed_s_incl_model_load": round(elapsed, 1),
        "peak_gpu_memory_mib": peak.peak_mib,
        "output": str(out),
        **summarize(records),
    }
    if not args.limit:  # only full runs are evidence
        (args.stats_dir / f"silver_{slug(args.model)}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
