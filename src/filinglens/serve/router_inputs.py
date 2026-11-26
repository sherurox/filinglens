"""Dev measurements for the FL-M5 router: accuracy, latency and tokens per backend and class (T9).

    set -a; source .env; set +a
    uv run python -m filinglens.serve.router_inputs extract --rpm 15 --rpd 500
    uv run python -m filinglens.extract.fields score --variant zero --split dev \
        --tag gemini-3.1-flash-lite
    uv run python -m filinglens.serve.router_inputs qa --backend gemini --rpm 15 --rpd 500
    uv run python -m filinglens.serve.router_inputs qa --backend vllm --model gptq-w4a16
    uv run python -m filinglens.serve.router_inputs qa --backend gguf

`extract`: the 71 dev extraction rows through Gemini with exactly the local builds' inputs (FL-M2's
zero-shot prompt, the same JSON schema, 400 output tokens), one request at a time under the
free-tier limits. Records go through the same parser and file names as `extract.fields`, so
`fields score --tag <model>` scores them unchanged; the local builds' dev scores already exist
(bench/fields_zero_dev_<build>.json). The run file adds per-request latency, tokens (thinking
counted apart) and finish reasons, and `loadgen.percentiles` for latency, as the local builds'
bench/load_*.json use.

`qa` (short Q&A): the 182 ret-dev questions (gold/retrieval/queries.dev.jsonl, FL-M3), each
with its source paragraph given (data/interim/synthetic/qa_tasks.jsonl, joined on para_id) and
scored by token-F1 against FL-M2's synthetic reference answer (gold/synthetic/risk_qa.train.jsonl,
joined on qa_id = query_id). Same prompt for every backend (docs/labeling/short_qa_prompt.md: one
sentence, at most 40 words, the paragraph only: the rules the reference answers were held to), no
schema, 100 output tokens. Token-F1 is SQuAD's: lower case, punctuation and a/an/the removed,
overlap of the token multisets; its mean gets a bootstrap CI over questions. The references were
written by Qwen3-4B-AWQ, so the score is agreement with that model's wording, not correctness, and
it may favor the Qwen-family student.

`assemble`: bench/router_inputs.json, the T10 linear program's inputs, from committed bench files
only, every value with the file or URL it came from:
- accuracy per (backend, class): extraction = dev category macro-F1 (bench/fields_zero_dev_*.json),
  short Q&A = dev token-F1 (bench/qa_dev_*.json); long synthesis = eligibility: a backend can take
  it only if its context window holds REQUIRED_CONTEXT tokens (M6's largest context budget, 16k),
  so the local backends (4,608 tokens, as served) are out and Gemini (1,048,576) is in. Long
  synthesis accuracy and latency are not measured.
- latency per (backend, class): p95 and mean at one request at a time (local extraction: c=1 of
  bench/load_*.json, the same dev prompts streamed; everything else: the run files here).
- cost per request at list prices. Gemini: tokens x the paid-tier list price (input; output and
  thinking at the output price). Local: power x mean latency x the electricity price. vLLM's power
  is the GPU board power measured at c=1 (nvidia-smi; reused for Q&A, where it wasn't measured);
  GGUF's is an estimate, the CPU's 75 W configurable TDP (an upper bound, not a measurement). Both
  leave out the rest of the laptop, so local costs are lower bounds on wall-socket energy.
  Actual spend on every backend is $0 (free tier; own laptop): the costs say what the same traffic
  would cost at list prices, which is what the router minimizes.
- failures: requests and failed requests per (backend, class). A failed row scored 0 in its
  accuracy (the rule stated before the runs); failed rows are not retried after the fact, which
  would make a backend look more available than it was. A run with more than 5% failed rows is
  refused (a broken run to repeat, not a property of the backend).
- capacity: Gemini's free-tier RPM / TPM / RPD (AI Studio), shared by all classes; local
  throughput at c=1 / 4 / 16 from bench/load_*.json (extraction prompts).

Dev only: the router's inputs never touch the test split, and Gemini never sees it. Before sending
anything, the run checks that today's remaining free-tier requests (serve.backends.DailyQuota)
cover every row, and it refuses to overwrite an earlier run's predictions unless --force, so a
repeated command can't silently spend the day's quota.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import string
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from filinglens.serve.backends import (
    Backend,
    DailyQuota,
    GeminiBackend,
    OpenAICompatBackend,
    Reply,
)
from filinglens.serve.loadgen import percentiles

GEMINI_MODEL = "gemini-3.1-flash-lite"
EXTRACT_MAX_TOKENS = 400  # as extract.fields.generate
PROGRESS_EVERY = 20  # a line on stderr every N requests: long Gemini runs print nothing else
MAX_ERROR_RATE = 0.05  # assemble refuses a run with more failed rows than this: rerun it
QA_MAX_TOKENS = 100  # the answer is one sentence of at most 40 words
QUERIES = Path("gold/retrieval/queries.dev.jsonl")
REFERENCES = Path("gold/synthetic/risk_qa.train.jsonl")
PARAGRAPHS = Path("data/interim/synthetic/qa_tasks.jsonl")
QA_PROMPT = Path("docs/labeling/short_qa_prompt.md")
QA_OUT = Path("data/interim/serve")
VLLM_BUILD = "gptq-w4a16"  # decided 26 Nov on dev: F1 within noise of BF16, ~half its c=1 p95
LOCAL_CONTEXT = 4608  # --max-model-len / -c as served (vllm_server.py, llama_client.py)
REQUIRED_CONTEXT = 16384  # long synthesis: M6's largest context-builder budget (16k)
LONG_OUTPUT_TOKENS = 400  # nominal answer size for the long-synthesis cost estimate
CHECKED = "2026-11-26"
SOURCES = {
    "gemini_price": {
        "model": GEMINI_MODEL,
        "usd_per_1m_input": 0.25,
        "usd_per_1m_output": 1.50,
        "note": "paid-tier Standard list price (output includes thinking); free tier: free",
        "url": "https://ai.google.dev/gemini-api/docs/pricing",
        "checked": CHECKED,
    },
    "gemini_context": {
        "input_tokens": 1048576,
        "output_tokens": 65536,
        "url": "https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite",
        "checked": CHECKED,
    },
    "gemini_free_tier": {
        "rpm": 15,
        "tpm": 250000,
        "rpd": 500,
        "note": "Google AI Studio > Rate Limit, project filinglens (personal account, Free tier);"
        " RPD resets at midnight Pacific",
        "checked": CHECKED,
    },
    "electricity": {
        "usd_per_kwh": 0.1817,
        "note": "US average residential price, September 2026 (EIA Electric Power Monthly"
        " Table 5.3, released 24 Nov 2026)",
        "url": "https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_3",
        "checked": CHECKED,
    },
    "cpu_power": {
        "watts": 75.0,
        "kind": "estimate (upper bound), not measured",
        "note": "Ryzen 9 8940HX: 55 W base TDP, officially configurable up to 75 W",
        "url": "https://www.notebookcheck.net/R9-8940HX-vs-R9-8945HX_19595_19594.247596.0.html",
        "checked": CHECKED,
    },
}
LOCAL = {  # the T5 / T4 servers; vLLM serves the build folder name as its model name
    "vllm": "http://127.0.0.1:58000",
    "gguf": "http://127.0.0.1:58081",
}


async def collect(
    backend: Backend, convs: list[list[dict]], schema: dict | None, max_tokens: int
) -> list[Reply]:
    """One request at a time, in order: latency is then the backend's, not a queue's.

    If the first request can't reach the server at all (nothing listening, DNS, timeout), the run
    stops before any file is written: a backend that was never reached has no score, not zero.
    A transport error later in the run is a failure of that row and is recorded as such."""
    replies: list[Reply] = []
    for conv in convs:
        reply = await backend.chat(conv, schema, max_tokens)
        if not replies and reply.error and reply.error.startswith("transport:"):
            raise SystemExit(f"{backend.name} unreachable ({reply.error}); nothing written")
        replies.append(reply)
        if len(replies) % PROGRESS_EVERY == 0 or len(replies) == len(convs):
            failed = sum(r.error is not None for r in replies)
            print(f"{backend.name}: {len(replies)}/{len(convs)} ({failed} failed)", file=sys.stderr)
    return replies


def reply_summary(replies: list[Reply]) -> dict:
    """Latency percentiles and token means over successful replies; failures counted."""
    ok = [r for r in replies if r.error is None]

    def mean(values: list[int]) -> float | None:
        return round(sum(values) / len(values), 1) if values else None

    return {
        "requests": len(replies),
        "errors": len(replies) - len(ok),
        "error_examples": [r.error for r in replies if r.error][:5],
        "finish_reasons": dict(Counter(str(r.finish_reason) for r in ok)),
        "latency_s": percentiles([r.latency_s for r in ok]),
        "prompt_tokens_mean": mean([r.prompt_tokens for r in ok]),
        "prompt_tokens_max": max((r.prompt_tokens for r in ok), default=None),
        "output_tokens_mean": mean([r.output_tokens for r in ok]),
        "thinking_tokens_mean": mean([r.thinking_tokens for r in ok]),
        "thinking_tokens_total": sum(r.thinking_tokens for r in ok),
    }


def require_budget(quota: DailyQuota, needed: int) -> int:
    """Remaining requests today, or exit before sending anything if they don't cover the run."""
    remaining = quota.limit - quota.used()
    if remaining < needed:
        raise SystemExit(
            f"{needed} requests needed, {remaining} left today (PT); run after Pacific midnight"
        )
    return remaining


def per_request(ids: list[str], replies: list[Reply]) -> list[dict]:
    return [
        {
            "para_id": i,
            "latency_s": round(r.latency_s, 4),
            "prompt_tokens": r.prompt_tokens,
            "output_tokens": r.output_tokens,
            "thinking_tokens": r.thinking_tokens,
            "finish_reason": r.finish_reason,
            "error": r.error,
        }
        for i, r in zip(ids, replies, strict=True)
    ]


def run_extract(args: argparse.Namespace) -> dict:
    from filinglens.extract.fields import (
        OUT_DIR,
        PROMPT,
        TASKS,
        conversations,
        load_rows,
        task_view,
    )
    from filinglens.extract.silver_label import parse, schema

    rows = load_rows("dev", allow_test=False)
    tasks = {t["para_id"]: t for t in json.loads(TASKS.read_text())}
    convs = conversations(rows, tasks, PROMPT.read_text(), [])
    out = OUT_DIR / f"fields_zero_dev_{args.model}.jsonl"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists: score it, or rerun with --force (spends quota again)")

    async def go() -> tuple[list[Reply], int]:
        backend = GeminiBackend.from_env(args.model, args.rpm, args.rpd)
        remaining = require_budget(backend.quota, len(convs))
        try:
            return await collect(backend, convs, schema(), EXTRACT_MAX_TOKENS), remaining
        finally:
            await backend.transport.aclose()

    started = datetime.now(UTC)
    replies, remaining = asyncio.run(go())
    wall = (datetime.now(UTC) - started).total_seconds()
    records = [parse(r.text, task_view(row, tasks)) for r, row in zip(replies, rows, strict=True)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    return {
        "backend": "gemini",
        "model": args.model,
        "limits": {"rpm": args.rpm, "rpd": args.rpd, "remaining_before_run": remaining},
        "decoding": {
            "temperature": "API default (not sent)",
            "seed": 0,
            "thinking_level": "minimal",
            "max_output_tokens": EXTRACT_MAX_TOKENS,
        },
        "variant": "zero",
        "rows": len(rows),
        "valid": sum(r["valid"] for r in records),
        "wall_s": round(wall, 1),
        "summary": reply_summary(replies),
        "predictions": str(out),
        "requests": per_request([row["para_id"] for row in rows], replies),
    }


# ---------- short Q&A ----------

ARTICLES = re.compile(r"\b(a|an|the)\b")
PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> list[str]:
    """SQuAD normalization: lower case, punctuation and articles removed, split on whitespace."""
    return ARTICLES.sub(" ", text.lower().translate(PUNCT)).split()


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = normalize_answer(prediction), normalize_answer(reference)
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def bootstrap_mean(values: list[float], n: int = 1000, seed: int = 0) -> list[float]:
    import numpy as np

    rng = np.random.default_rng(seed)
    a = np.asarray(values, dtype=float)
    means = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    low, high = np.percentile(means, [2.5, 97.5])
    return [round(float(low), 4), round(float(high), 4)]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def qa_items(queries: list[dict], references: list[dict], paragraphs: list[dict]) -> list[dict]:
    """Question + reference answer + source paragraph for every query; stops if any is missing."""
    answers = {r["qa_id"]: r["answer"] for r in references}
    paras = {p["para_id"]: p for p in paragraphs}
    missing = [
        q["query_id"]
        for q in queries
        if q["query_id"] not in answers or q["source_para_id"] not in paras
    ]
    if missing:
        raise SystemExit(f"{len(missing)} queries without an answer or paragraph: {missing[:3]}")
    return [
        {
            "query_id": q["query_id"],
            "question": q["question"],
            "reference": answers[q["query_id"]],
            "para": paras[q["source_para_id"]],
        }
        for q in queries
    ]


def qa_conversation(item: dict, system_prompt: str) -> list[dict]:
    para = item["para"]
    user = (
        f"Ticker: {para['ticker']}\nFiscal year: FY{para['period_year']}\n"
        f"Heading: {para.get('heading') or '(none)'}\n\nPARAGRAPH:\n{para['text']}\n\n"
        f"QUESTION: {item['question']}"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


def make_backend(args: argparse.Namespace) -> Backend:
    if args.backend == "gemini":
        if args.rpm is None or args.rpd is None:
            raise SystemExit("--rpm and --rpd (from AI Studio) are required for gemini")
        return GeminiBackend.from_env(args.model or GEMINI_MODEL, args.rpm, args.rpd)
    if args.backend == "vllm" and not args.model:
        raise SystemExit("--model <build> is required for vllm (the served model name)")
    return OpenAICompatBackend(args.backend, args.url or LOCAL[args.backend], args.model or "gguf")


def qa_tag(args: argparse.Namespace) -> str:
    """File tag: gemini-<model id>, vllm-<build>, gguf-q4_k_m."""
    default = GEMINI_MODEL if args.backend == "gemini" else "q4_k_m"
    return f"{args.backend}-{args.model or default}"


def run_qa(args: argparse.Namespace) -> dict:
    items = qa_items(read_jsonl(QUERIES), read_jsonl(REFERENCES), read_jsonl(PARAGRAPHS))
    system = QA_PROMPT.read_text()
    convs = [qa_conversation(item, system) for item in items]
    out = QA_OUT / f"qa_dev_{qa_tag(args)}.jsonl"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists: rerun with --force to replace it")

    async def go() -> list[Reply]:
        backend = make_backend(args)
        if isinstance(backend, GeminiBackend):
            require_budget(backend.quota, len(convs))
        try:
            return await collect(backend, convs, None, QA_MAX_TOKENS)
        finally:
            await backend.transport.aclose()

    started = datetime.now(UTC)
    replies = asyncio.run(go())
    wall = (datetime.now(UTC) - started).total_seconds()
    scores = [
        token_f1(r.text, item["reference"]) if r.error is None else 0.0
        for r, item in zip(replies, items, strict=True)
    ]
    rows = [
        {"query_id": item["query_id"], "answer": r.text, "token_f1": round(f1, 4)}
        for r, item, f1 in zip(replies, items, scores, strict=True)
    ]
    QA_OUT.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    words = [len(r.text.split()) for r in replies if r.error is None]
    return {
        "backend": args.backend,
        "model": replies[0].model if replies else None,
        "prompt": str(QA_PROMPT),
        "max_output_tokens": QA_MAX_TOKENS,
        "questions": len(items),
        "references": "synthetic (Qwen/Qwen3-4B-AWQ, FL-M2); agreement with its wording",
        "token_f1_mean": round(sum(scores) / len(scores), 4),
        "token_f1_ci95": bootstrap_mean(scores),
        "failed_rows_scored_zero": sum(r.error is not None for r in replies),
        "answer_words_mean": round(sum(words) / len(words), 1) if words else None,
        "answers_over_40_words": sum(w > 40 for w in words),
        "wall_s": round(wall, 1),
        "summary": reply_summary(replies),
        "answers": str(out),
        "requests": per_request([item["query_id"] for item in items], replies),
    }


# ---------- assemble ----------


def energy_usd(watts: float, seconds: float, usd_per_kwh: float) -> float:
    """Electricity cost of drawing `watts` for `seconds`: W x s = J; 3.6e6 J = 1 kWh."""
    return watts * seconds / 3.6e6 * usd_per_kwh


def gemini_usd(prompt: float, output: float, thinking: float, price: dict) -> float:
    return (
        prompt * price["usd_per_1m_input"] + (output + thinking) * price["usd_per_1m_output"]
    ) / 1e6


def load_bench(bench: Path, name: str) -> dict:
    path = bench / name
    if not path.is_file():
        raise SystemExit(f"missing {path}: run its step first")
    return json.loads(path.read_text()) | {"_file": path.as_posix()}


def failures(requests: int, failed: int) -> dict:
    return {
        "requests": requests,
        "failed": failed,
        "error_rate": round(failed / requests, 4) if requests else None,
    }


def local_class(
    acc: float,
    ci: list,
    mean_s: float,
    p95_s: float,
    watts: float,
    fails: dict,
    src: dict,
) -> dict:
    usd = energy_usd(watts, mean_s, SOURCES["electricity"]["usd_per_kwh"])
    return {
        "accuracy": acc,
        "accuracy_ci95": ci,
        **fails,
        "latency_mean_s": mean_s,
        "latency_p95_s": p95_s,
        "usd_per_request": usd,
        "usd_per_1k_requests": round(usd * 1000, 6),
        "sources": src,
    }


def gemini_class(acc: float | None, ci: list | None, summary: dict, src: dict) -> dict:
    price = SOURCES["gemini_price"]
    usd = gemini_usd(
        summary["prompt_tokens_mean"],
        summary["output_tokens_mean"],
        summary["thinking_tokens_mean"],
        price,
    )
    return {
        "accuracy": acc,
        "accuracy_ci95": ci,
        **failures(summary["requests"], summary["errors"]),
        "latency_mean_s": summary["latency_s"]["mean"],
        "latency_p95_s": summary["latency_s"]["p95"],
        "tokens_mean": {
            "prompt": summary["prompt_tokens_mean"],
            "output": summary["output_tokens_mean"],
            "thinking": summary["thinking_tokens_mean"],
        },
        "usd_per_request": usd,
        "usd_per_1k_requests": round(usd * 1000, 6),
        "sources": src,
    }


def throughput(load: dict) -> dict:
    """Requests/s and p95 latency per concurrency level (extraction prompts, dev)."""
    return {
        str(level["concurrency"]): {
            "requests_per_s": level["requests_per_s"],
            "latency_p95_s": level["latency_s"]["p95"],
        }
        for level in load["levels"]
    }


def build_inputs(f: dict) -> dict:
    """The router's inputs from already-loaded bench files (`f`: name -> parsed JSON)."""
    elec = SOURCES["electricity"]["usd_per_kwh"]
    cpu_w = SOURCES["cpu_power"]["watts"]
    vllm_c1 = f["load_vllm"]["levels"][0]
    gguf_c1 = f["load_gguf"]["levels"][0]
    if vllm_c1["concurrency"] != 1 or gguf_c1["concurrency"] != 1:
        raise ValueError("the first load level must be c=1")
    gpu_w = vllm_c1["gpu"]["power_w_mean"]

    def acc(name: str) -> tuple[float, list]:
        rows = f[name]["all_rows"]
        return rows["category_macro_f1"], rows["category_macro_f1_ci95"]

    def qa(name: str) -> tuple[float, list, dict]:
        return f[name]["token_f1_mean"], f[name]["token_f1_ci95"], f[name]["summary"]["latency_s"]

    def qa_fails(name: str) -> dict:
        return failures(f[name]["summary"]["requests"], f[name]["summary"]["errors"])

    def load_fails(level: dict) -> dict:
        return failures(level["requests"], level["errors"])

    vllm_qa, gguf_qa = qa("qa_vllm"), qa("qa_gguf")
    gem_qa = f["qa_gemini"]
    long_prompt = REQUIRED_CONTEXT - LONG_OUTPUT_TOKENS
    long_usd = gemini_usd(long_prompt, LONG_OUTPUT_TOKENS, 0, SOURCES["gemini_price"])
    gem_prompt_max = max(
        f["run_gemini_extract"]["summary"]["prompt_tokens_max"],
        gem_qa["summary"]["prompt_tokens_max"],
    )
    free = SOURCES["gemini_free_tier"]
    return {
        "split": "dev",
        "classes": {
            "extraction": {
                "accuracy_metric": "category macro-F1 vs silver labels (71 dev rows)",
                "max_output_tokens": EXTRACT_MAX_TOKENS,
            },
            "short_qa": {
                "accuracy_metric": "token-F1 vs synthetic Qwen3-4B answers (182 ret-dev questions)",
                "max_output_tokens": QA_MAX_TOKENS,
            },
            "long_synthesis": {
                "accuracy_metric": "none: eligibility only (context window >= required context)",
                "required_context_tokens": REQUIRED_CONTEXT,
                "nominal_tokens": {"prompt": long_prompt, "output": LONG_OUTPUT_TOKENS},
            },
        },
        "backends": {
            "vllm": {
                "model": VLLM_BUILD,
                "context_tokens": LOCAL_CONTEXT,
                "power_w": gpu_w,
                "power_basis": "GPU board power, mean at c=1 during extraction load (measured)",
                "extraction": local_class(
                    *acc("fields_vllm"),
                    vllm_c1["latency_s"]["mean"],
                    vllm_c1["latency_s"]["p95"],
                    gpu_w,
                    load_fails(vllm_c1),
                    {"accuracy": f["fields_vllm"]["_file"], "latency": f["load_vllm"]["_file"]},
                ),
                "short_qa": local_class(
                    vllm_qa[0],
                    vllm_qa[1],
                    vllm_qa[2]["mean"],
                    vllm_qa[2]["p95"],
                    gpu_w,
                    qa_fails("qa_vllm"),
                    {"accuracy_latency": f["qa_vllm"]["_file"], "power": f["load_vllm"]["_file"]},
                ),
                "long_synthesis": {"eligible": LOCAL_CONTEXT >= REQUIRED_CONTEXT},
                "throughput_extraction": throughput(f["load_vllm"]),
            },
            "gguf": {
                "model": "student-q4_k_m.gguf (llama.cpp, CPU)",
                "context_tokens": LOCAL_CONTEXT,
                "power_w": cpu_w,
                "power_basis": SOURCES["cpu_power"]["kind"],
                "gpu_idle_power_w_measured": gguf_c1["gpu"]["power_w_mean"],
                "extraction": local_class(
                    *acc("fields_gguf"),
                    gguf_c1["latency_s"]["mean"],
                    gguf_c1["latency_s"]["p95"],
                    cpu_w,
                    load_fails(gguf_c1),
                    {"accuracy": f["fields_gguf"]["_file"], "latency": f["load_gguf"]["_file"]},
                ),
                "short_qa": local_class(
                    gguf_qa[0],
                    gguf_qa[1],
                    gguf_qa[2]["mean"],
                    gguf_qa[2]["p95"],
                    cpu_w,
                    qa_fails("qa_gguf"),
                    {"accuracy_latency": f["qa_gguf"]["_file"]},
                ),
                "long_synthesis": {"eligible": LOCAL_CONTEXT >= REQUIRED_CONTEXT},
                "throughput_extraction": throughput(f["load_gguf"]),
            },
            "gemini": {
                "model": GEMINI_MODEL,
                "context_tokens": SOURCES["gemini_context"]["input_tokens"],
                "extraction": gemini_class(
                    *acc("fields_gemini"),
                    f["run_gemini_extract"]["summary"],
                    {
                        "accuracy": f["fields_gemini"]["_file"],
                        "latency_tokens": f["run_gemini_extract"]["_file"],
                    },
                ),
                "short_qa": gemini_class(
                    gem_qa["token_f1_mean"],
                    gem_qa["token_f1_ci95"],
                    gem_qa["summary"],
                    {"accuracy_latency_tokens": gem_qa["_file"]},
                ),
                "long_synthesis": {
                    "eligible": SOURCES["gemini_context"]["input_tokens"] >= REQUIRED_CONTEXT,
                    "usd_per_request_nominal": long_usd,
                    "latency": "not measured",
                },
            },
        },
        "capacity": {
            "gemini_free_tier": free,
            "gemini_tpm_check": {
                "largest_prompt_tokens_seen": gem_prompt_max,
                "tokens_per_minute_at_rpm": gem_prompt_max * free["rpm"],
                "within_tpm": gem_prompt_max * free["rpm"] <= free["tpm"],
                "long_synthesis_tokens_per_minute_at_rpm": REQUIRED_CONTEXT * free["rpm"],
                "long_synthesis_within_tpm": REQUIRED_CONTEXT * free["rpm"] <= free["tpm"],
            },
        },
        "spend": {
            "actual_usd": 0,
            "why": "Gemini free tier (no billing linked); local backends on the author's laptop",
            "modelled": "list prices (Gemini) and electricity at the EIA US average (local)",
        },
        "electricity_usd_per_kwh": elec,
        "sources": SOURCES,
    }


BENCH_FILES = {
    "fields_vllm": f"fields_zero_dev_{VLLM_BUILD}.json",
    "fields_gguf": "fields_zero_dev_gguf-q4_k_m.json",
    "fields_gemini": f"fields_zero_dev_{GEMINI_MODEL}.json",
    "run_gemini_extract": f"fields_zero_dev_{GEMINI_MODEL}_run.json",
    "load_vllm": f"load_{VLLM_BUILD}.json",
    "load_gguf": "load_gguf-q4_k_m.json",
    "qa_vllm": f"qa_dev_vllm-{VLLM_BUILD}.json",
    "qa_gguf": "qa_dev_gguf-q4_k_m.json",
    "qa_gemini": f"qa_dev_gemini-{GEMINI_MODEL}.json",
}


def run_assemble(args: argparse.Namespace) -> dict:
    files = {key: load_bench(args.bench, name) for key, name in BENCH_FILES.items()}
    check_error_rates(files)
    return build_inputs(files)


def check_error_rates(files: dict) -> None:
    """Refuse runs with more than MAX_ERROR_RATE failed rows (broken runs, not measurements)."""
    for key in ("qa_vllm", "qa_gguf", "qa_gemini", "run_gemini_extract"):
        summary = files[key]["summary"]
        if summary["errors"] > MAX_ERROR_RATE * summary["requests"]:
            raise SystemExit(
                f"{files[key]['_file']}: {summary['errors']} of {summary['requests']} rows failed"
                f" (> {MAX_ERROR_RATE:.0%}): rerun it before assembling"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["extract", "qa", "assemble"])
    parser.add_argument("--backend", choices=["gemini", "vllm", "gguf"], default="gemini")
    parser.add_argument("--model", default=None, help="Gemini model id or vLLM build name")
    parser.add_argument("--url", default=None, help="override the local server's URL")
    parser.add_argument("--rpm", type=int, default=None, help="free-tier RPM from AI Studio")
    parser.add_argument("--rpd", type=int, default=None, help="free-tier RPD from AI Studio")
    parser.add_argument("--force", action="store_true", help="overwrite an earlier run")
    parser.add_argument("--bench", type=Path, default=Path("bench"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    if args.command == "extract":
        if args.backend != "gemini" or args.rpm is None or args.rpd is None:
            raise SystemExit("extract is for gemini (local builds have fields runs): --rpm --rpd")
        args.model = args.model or GEMINI_MODEL
        body = run_extract(args)
        out = args.bench / f"fields_zero_dev_{args.model}_run.json"
    elif args.command == "qa":
        body = run_qa(args)
        out = args.bench / f"qa_dev_{qa_tag(args)}.json"
    else:
        body = run_assemble(args)
        out = args.bench / "router_inputs.json"
    result = {
        "command": " ".join(["python -m filinglens.serve.router_inputs", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": "dev",
        **body,
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("requests", "sources")}, indent=2)
    )
    return 0 if result.get("summary", {}).get("errors", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
