"""Dev measurements for the FL-M5 router: accuracy, latency and tokens per backend and class (T9).

    set -a; source .env; set +a
    uv run python -m filinglens.serve.router_inputs extract --rpm 15 --rpd 500
    uv run python -m filinglens.extract.fields score --variant zero --split dev \
        --tag gemini-3.1-flash-lite

`extract`: the 71 dev extraction rows through Gemini with exactly the local builds' inputs (FL-M2's
zero-shot prompt, the same JSON schema, 400 output tokens), one request at a time under the
free-tier limits. Records go through the same parser and file names as `extract.fields`, so
`fields score --tag <model>` scores them unchanged; the local builds' dev scores already exist
(bench/fields_zero_dev_<build>.json). The run file adds per-request latency, tokens (thinking
counted apart) and finish reasons, and `loadgen.percentiles` for latency, as the local builds'
bench/load_*.json use.

Dev only: the router's inputs never touch the test split, and Gemini never sees it. Before sending
anything, the run checks that today's remaining free-tier requests (serve.backends.DailyQuota)
cover every row, and it refuses to overwrite an earlier run's predictions unless --force, so a
repeated command can't silently spend the day's quota.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from filinglens.serve.backends import Backend, DailyQuota, GeminiBackend, Reply
from filinglens.serve.loadgen import percentiles

GEMINI_MODEL = "gemini-3.1-flash-lite"
EXTRACT_MAX_TOKENS = 400  # as extract.fields.generate


async def collect(
    backend: Backend, convs: list[list[dict]], schema: dict | None, max_tokens: int
) -> list[Reply]:
    """One request at a time, in order: latency is then the backend's, not a queue's."""
    return [await backend.chat(conv, schema, max_tokens) for conv in convs]


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["extract"])
    parser.add_argument("--model", default=GEMINI_MODEL)
    parser.add_argument("--rpm", type=int, required=True, help="free-tier RPM from AI Studio")
    parser.add_argument("--rpd", type=int, required=True, help="free-tier RPD from AI Studio")
    parser.add_argument("--force", action="store_true", help="overwrite an earlier run")
    parser.add_argument("--bench", type=Path, default=Path("bench"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    body = run_extract(args)
    result = {
        "command": " ".join(["python -m filinglens.serve.router_inputs", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": "dev",
        **body,
    }
    out = args.bench / f"fields_zero_dev_{args.model}_run.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "requests"}, indent=2))
    return 0 if result["summary"]["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
