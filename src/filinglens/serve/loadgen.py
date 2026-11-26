r"""Closed-loop load generator for OpenAI-compatible chat servers (vLLM, llama-server) (FL-M5 T5).

    uv run python -m filinglens.serve.loadgen --url http://127.0.0.1:58000 --tag bf16 \
        --levels 1 4 16 --requests 64 --note "AC power, Best performance" --server-cmd "..."

Closed loop: at concurrency k, k workers each send their next request as soon as their previous one
returns, until N requests have completed (after W warm-up requests at the same k, not counted).
That models k users waiting for answers; it can't overload the queue without bound, and it doesn't
show how latency grows as an open-loop arrival rate passes capacity.

Requests are the eval's real extraction prompts (dev split only; the test split is never sent here),
in order and cycled, with the eval's decoding: temperature 0, at most 400 tokens, the JSON schema,
thinking off; streamed, so the first token's arrival is observed.

Per request: TTFT (send -> first non-empty content delta), end-to-end latency, completion tokens
from the server's `usage` in the last chunk, decode rate (tokens - 1) / (latency - TTFT).
Per level: p50/p95/p99 of TTFT and latency, mean decode rate, aggregate output tokens/s (all
completion tokens / wall time), requests/s, errors, and the GPU monitor's summary for that level.
Writes bench/load_<tag>.json.

VRAM: vLLM claims its whole memory budget at start, before the first sample, so the monitor's own
baseline shows no growth. `--idle-vram-mib` (measured before the server started; WSL counts the
Windows desktop too) gives the server's real footprint: peak during the level minus idle.
`--server-log` copies the server's KV-cache, kernel and context lines into the record, so they
survive without the git-ignored logs/.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import re
import sys
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

DONE = "[DONE]"
SERVER_LOG_LINES = re.compile(
    r"GPU KV cache size|Maximum concurrency|Kernel for|Model loading took|n_ctx|n_slots|n_parallel"
    r"|file type|model size|system_info|kv_unified|CPU_REPACK",
    re.IGNORECASE,
)
MAX_TOKENS = 400  # as extract.fields.generate


def stream_request(conv: list[dict], schema: dict, model: str) -> dict:
    """The eval's decoding as a streamed OpenAI chat request (`name` is required by vLLM)."""
    return {
        "model": model,
        "messages": conv,
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "risk_fields", "schema": schema},
        },
        "chat_template_kwargs": {"enable_thinking": False},
    }


def parse_sse(line: str) -> dict | str | None:
    """One server-sent-events line -> the JSON event, DONE, or None (blank, comment, other)."""
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if payload == DONE:
        return DONE
    return json.loads(payload)


def content_delta(event: dict) -> str:
    choices = event.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("delta") or {}).get("content") or ""


def request_metrics(started: float, first: float | None, ended: float, tokens: int | None) -> dict:
    ttft = None if first is None else first - started
    latency = ended - started
    decode = None
    if tokens and tokens > 1 and ttft is not None and latency > ttft:
        decode = (tokens - 1) / (latency - ttft)
    return {
        "ttft_s": ttft,
        "latency_s": latency,
        "completion_tokens": tokens,
        "decode_tok_s": decode,
    }


def percentiles(values: list[float]) -> dict:
    import numpy as np

    if not values:
        return {"n": 0}
    a = np.asarray(values, dtype=float)
    p50, p95, p99 = np.percentile(a, [50, 95, 99])
    return {
        "n": len(values),
        "mean": round(float(a.mean()), 4),
        "p50": round(float(p50), 4),
        "p95": round(float(p95), 4),
        "p99": round(float(p99), 4),
    }


def level_summary(rows: list[dict], wall_s: float, concurrency: int) -> dict:
    ok = [r for r in rows if "error" not in r]
    tokens = [r["completion_tokens"] for r in ok if r["completion_tokens"]]
    decode = [r["decode_tok_s"] for r in ok if r["decode_tok_s"]]
    return {
        "concurrency": concurrency,
        "requests": len(rows),
        "errors": len(rows) - len(ok),
        "missing_usage": sum(1 for r in ok if not r["completion_tokens"]),
        "wall_s": round(wall_s, 3),
        "requests_per_s": round(len(ok) / wall_s, 4) if wall_s else None,
        "output_tok_per_s": round(sum(tokens) / wall_s, 2) if wall_s else None,
        "decode_tok_per_s_per_request_mean": round(sum(decode) / len(decode), 2)
        if decode
        else None,
        "completion_tokens_mean": round(sum(tokens) / len(tokens), 1) if tokens else None,
        "ttft_s": percentiles([r["ttft_s"] for r in ok if r["ttft_s"] is not None]),
        "latency_s": percentiles([r["latency_s"] for r in ok]),
    }


async def closed_loop(
    send: Callable[[dict], Awaitable[dict]],
    bodies: list[dict],
    concurrency: int,
    n: int,
    start: int,
) -> tuple[list[dict], float]:
    """Exactly `concurrency` requests in flight until n have finished; bodies cycle from `start`."""
    if concurrency < 1 or n < 1:
        raise ValueError("concurrency and n must be >= 1")
    results: list[dict | None] = [None] * n
    counter = itertools.count()

    async def worker() -> None:
        while (i := next(counter)) < n:
            results[i] = await send(bodies[(start + i) % len(bodies)])

    began = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(min(concurrency, n))))
    return [r for r in results if r is not None], time.perf_counter() - began


def make_sender(client, url: str) -> Callable[[dict], Awaitable[dict]]:
    async def send(body: dict) -> dict:
        started = time.perf_counter()
        first, tokens = None, None
        try:
            async with client.stream("POST", f"{url}/v1/chat/completions", json=body) as response:
                if response.status_code != 200:
                    text = (await response.aread()).decode(errors="replace")[:300]
                    return {"error": f"HTTP {response.status_code}: {text}"}
                async for line in response.aiter_lines():
                    event = parse_sse(line)
                    if event is None:
                        continue
                    if event == DONE:
                        break
                    if first is None and content_delta(event):
                        first = time.perf_counter()
                    usage = event.get("usage")
                    if usage and usage.get("completion_tokens") is not None:
                        tokens = usage["completion_tokens"]
        except Exception as error:  # a failed request is a row, not a crash of the benchmark
            return {"error": f"{type(error).__name__}: {str(error)[:300]}"}
        return request_metrics(started, first, time.perf_counter(), tokens)

    return send


def server_log_facts(text: str, limit: int = 40) -> list[str]:
    """The lines of a server log that say how it was configured (deduplicated, prefixes cut)."""
    seen, out = set(), []
    for line in text.splitlines():
        if not SERVER_LOG_LINES.search(line):
            continue
        fact = re.sub(r"^.*?\] ", "", line).strip()
        if fact and fact not in seen:
            seen.add(fact)
            out.append(fact)
    return out[:limit]


def add_server_vram(gpu: dict, idle_mib: float | None) -> dict:
    """The server's VRAM: peak during the level minus the idle reading taken before it started."""
    if idle_mib is None or not gpu.get("samples"):
        return gpu
    return {
        **gpu,
        "idle_mib_before_server": idle_mib,
        "server_vram_mib_peak": gpu["memory_used_mib_peak"] - idle_mib,
    }


def dev_bodies(model: str) -> list[dict]:
    from filinglens.extract.fields import PROMPT, TASKS, conversations, load_rows
    from filinglens.extract.silver_label import schema

    rows = load_rows("dev", allow_test=False)
    tasks = {t["para_id"]: t for t in json.loads(TASKS.read_text())}
    convs = conversations(rows, tasks, PROMPT.read_text(), [])
    return [stream_request(conv, schema(), model) for conv in convs]


async def run(args: argparse.Namespace) -> dict:
    import httpx

    from filinglens.serve.gpu_monitor import GpuMonitor

    timeout = httpx.Timeout(args.timeout_s, connect=10)
    limits = httpx.Limits(max_connections=max(args.levels) + 4)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        models = (await client.get(f"{args.url}/v1/models")).json()
        model = args.model or models["data"][0]["id"]
        bodies = dev_bodies(model)
        send = make_sender(client, args.url)
        levels, offset = [], 0
        for k in args.levels:
            await closed_loop(send, bodies, k, args.warmup, offset)  # warm-up, not counted
            offset += args.warmup
            with GpuMonitor() as gpu:
                rows, wall = await closed_loop(send, bodies, k, args.requests, offset)
            offset += args.requests
            summary = level_summary(rows, wall, k)
            summary["gpu"] = add_server_vram(gpu.summary(), args.idle_vram_mib)
            summary["error_examples"] = [r["error"] for r in rows if "error" in r][:3]
            levels.append({**summary, "rows": rows})
            print(
                f"c={k}: {summary['requests']} req, {summary['errors']} errors, "
                f"TTFT p50 {summary['ttft_s'].get('p50')} s, latency p95 "
                f"{summary['latency_s'].get('p95')} s, {summary['output_tok_per_s']} tok/s",
                flush=True,
            )
    return {"model": model, "models_endpoint": models, "prompts": len(bodies), "levels": levels}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--tag", required=True, help="output: bench/load_<tag>.json")
    parser.add_argument("--model", default=None, help="default: the first /v1/models entry")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--requests", type=int, default=64, help="measured requests per level")
    parser.add_argument("--warmup", type=int, default=4, help="uncounted requests per level")
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--note", required=True, help="power mode, e.g. 'AC power, Best perf'")
    parser.add_argument("--server-cmd", default="", help="how the server was started (recorded)")
    parser.add_argument("--server-log", type=Path, default=None, help="server log to excerpt")
    parser.add_argument("--idle-vram-mib", type=float, default=None, help="before the server")
    parser.add_argument("--bench", type=Path, default=Path("bench"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    started = datetime.now(UTC).isoformat(timespec="seconds")
    body = asyncio.run(run(args))
    result = {
        "command": " ".join(["python -m filinglens.serve.loadgen", *argv]),
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "power_note": args.note,
        "server_cmd": args.server_cmd,
        "server_log_facts": server_log_facts(args.server_log.read_text(errors="replace"))
        if args.server_log
        else [],
        "url": args.url,
        "requests_per_level": args.requests,
        "warmup_per_level": args.warmup,
        "split": "dev",
        **body,
    }
    out = args.bench / f"load_{args.tag}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    errors = sum(level["errors"] for level in body["levels"])
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
