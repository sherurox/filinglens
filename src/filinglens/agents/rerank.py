r"""The reranker as a local sidecar for the agent tools, and its device measurement (FL-M6 T4).

The tools run in the uv environment; FL-M3's reranker (bge-reranker-base@2cfc18c9, PyTorch) runs in
the fl-train venv. Rather than mixing the two environments, the reranker serves on localhost, the
same pattern as the vLLM server:

    ~/venvs/fl-train/bin/python -m filinglens.agents.rerank serve --device cuda   # or cpu
    uv run python -m filinglens.agents.rerank bench --split dev --note "..."

`serve`: FL-M3's own `retrieval.rerank.Reranker` (fp16 on CUDA, fp32 on CPU; chunk truncated,
never the question; 512 tokens) behind a standard-library HTTP server on 127.0.0.1:RERANK_PORT.
POST /rerank {"question", "texts"} -> {"scores", "ms"}; GET /health -> model and device. One
request is scored at a time (a lock), so GPU memory use is one batch of at most MAX_TEXTS pairs.

`bench` (dev only): the 182 ret-dev questions with FL-M3's dense top-50 candidates
(data/interim/retrieval/candidates.dev.jsonl) through the running sidecar, scored with FL-M3's
rerank_order and metrics, and the latency the tools see (HTTP included). The device is chosen by
bench/m6_rerank_rule.json, committed before the measurement: CUDA if the sidecar runs next to the
vLLM student with no error, reproduces FL-M3's reranked NDCG@10 within the rule's tolerance, and
keeps p95 under the rule's limit; otherwise CPU (the same model in fp32, so the ranking quality is
FL-M3's; only latency changes).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

RERANK_HOST = "127.0.0.1"
RERANK_PORT = 58090
RERANK_URL = f"http://{RERANK_HOST}:{RERANK_PORT}"
MAX_TEXTS = 64
MAX_QUESTION_CHARS = 2000
MAX_BODY_BYTES = 4_000_000
RULE = Path("bench/m6_rerank_rule.json")
FL_M3_DEV = Path("bench/reranker_dev.json")


# ---------- server (fl-train venv) ----------


def validate(payload: Any) -> tuple[str, list[str]]:
    """The question and texts of a /rerank body, or ValueError with a short reason."""
    if not isinstance(payload, dict):
        raise ValueError("body must be a JSON object")
    question, texts = payload.get("question"), payload.get("texts")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"question longer than {MAX_QUESTION_CHARS} characters")
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise ValueError("texts must be a list of strings")
    if not 1 <= len(texts) <= MAX_TEXTS:
        raise ValueError(f"between 1 and {MAX_TEXTS} texts")
    return question, texts


def make_handler(score, info: dict):
    """A request handler around `score(question, texts) -> sequence of floats`."""
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 (http.server's method name)
            if self.path == "/health":
                self._send(200, {"status": "ok", **info})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/rerank":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY_BYTES:
                self._send(413, {"error": "missing or oversized body"})
                return
            try:
                question, texts = validate(json.loads(self.rfile.read(length)))
            except (ValueError, json.JSONDecodeError) as error:
                self._send(400, {"error": str(error)})
                return
            with lock:
                started = time.perf_counter()
                scores = [float(s) for s in score(question, texts)]
                ms = (time.perf_counter() - started) * 1000
            self._send(200, {"scores": scores, "ms": round(ms, 2)})

        def log_message(self, format: str, *args: Any) -> None:  # quiet: no request lines
            pass

    return Handler


def run_serve(args: argparse.Namespace) -> None:
    from filinglens.retrieval.config import RERANK_MODEL, RERANK_REVISION
    from filinglens.retrieval.rerank import Reranker

    model = Reranker(device=args.device, fp16=args.device == "cuda")
    info = {
        "model": f"{RERANK_MODEL}@{RERANK_REVISION}",
        "device": args.device,
        "dtype": "fp16" if args.device == "cuda" else "fp32",
    }
    model("warm up", ["one short text"])  # kernels and allocator before the first real request
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model, info))
    print(json.dumps({"listening": f"http://{args.host}:{args.port}", **info}), flush=True)
    server.serve_forever()


# ---------- client (uv env) ----------


class RerankError(RuntimeError):
    pass


class HttpReranker:
    """`await reranker(question, texts)` -> list of scores, from the sidecar."""

    def __init__(self, url: str = RERANK_URL, timeout_s: float = 60.0, client=None) -> None:
        import httpx

        self.url, self.timeout_s = url.rstrip("/"), timeout_s
        self._client = client or httpx.AsyncClient()

    async def __call__(self, question: str, texts: list[str]) -> list[float]:
        import httpx

        try:
            r = await self._client.post(
                self.url + "/rerank",
                json={"question": question, "texts": texts},
                timeout=self.timeout_s,
            )
        except httpx.HTTPError as error:
            raise RerankError(
                f"reranker unreachable at {self.url}: {type(error).__name__}"
            ) from None
        if r.status_code != 200:
            raise RerankError(f"reranker HTTP {r.status_code}: {r.text[:200]}")
        scores = r.json()["scores"]
        if len(scores) != len(texts):
            raise RerankError(f"{len(scores)} scores for {len(texts)} texts")
        return scores

    async def health(self) -> dict:
        r = await self._client.get(self.url + "/health", timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------- device measurement (uv env, sidecar running) ----------


def decide(rule: dict, result: dict) -> dict:
    """The rule's decision for one measured run of the sidecar on `result["device"]`."""
    cond = rule["cuda_if"]
    checks = {
        "device_is_cuda": result["device"] == "cuda",
        "no_errors": result["errors"] == 0,
        "ndcg_reproduces_fl_m3": abs(result["metrics"]["ndcg_after"] - rule["fl_m3_ndcg_after"])
        <= cond["max_abs_ndcg_diff"],
        "p95_within_limit": result["latency_ms"]["p95_ms"] <= cond["max_p95_ms"],
    }
    return {"checks": checks, "device": "cuda" if all(checks.values()) else "cpu"}


def gpu_memory() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or out.stderr.strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"nvidia-smi unavailable: {type(error).__name__}"


async def bench(args: argparse.Namespace) -> dict:
    import pandas as pd

    from filinglens.retrieval.config import CHUNKS, RET_DIR
    from filinglens.retrieval.queries import load_queries
    from filinglens.retrieval.rerank import (
        latency_summary,
        mean_metrics,
        per_query_metrics,
        rerank_order,
    )

    rule = json.loads(args.rule.read_text())
    queries = {q["query_id"]: q for q in load_queries("dev")}
    cands = [
        json.loads(line)
        for line in (RET_DIR / "candidates.dev.jsonl").read_text().splitlines()
        if line.strip()
    ]
    texts = pd.read_parquet(CHUNKS, columns=["chunk_id", "embed_text"])
    text_of = dict(zip(texts["chunk_id"], texts["embed_text"], strict=True))
    client = HttpReranker(args.url)
    health = await client.health()
    gpu_before = gpu_memory()
    rows, ms, errors = [], [], []
    for c in cands:
        q = queries[c["query_id"]]
        started = time.perf_counter()
        try:
            scores = await client(q["question"], [text_of[i] for i in c["candidates"]])
        except RerankError as error:
            errors.append(str(error))
            continue
        ms.append((time.perf_counter() - started) * 1000)
        order = rerank_order(c["candidates"], scores)
        rows.append(
            {"query_id": q["query_id"], **per_query_metrics(c["candidates"], order, q["grades"])}
        )
    await client.aclose()
    result = {
        "device": health["device"],
        "sidecar": health,
        "queries": len(cands),
        "scored": len(rows),
        "errors": len(errors),
        "error_examples": errors[:3],
        "metrics": mean_metrics(rows) if rows else {},
        "latency_ms": latency_summary(ms) if ms else {},
        "latency_note": "client-observed per query: HTTP + tokenize + forward pass over 50 pairs",
        "gpu_memory_used_total": {"before": gpu_before, "after": gpu_memory()},
        "vllm_running": args.vllm_running,
        "power_note": args.note,
    }
    if rows and ms:
        result["decision"] = decide(rule, result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--device", choices=["cuda", "cpu"], required=True)
    s.add_argument("--host", default=RERANK_HOST)
    s.add_argument("--port", type=int, default=RERANK_PORT)
    b = sub.add_parser("bench")
    b.add_argument("--split", choices=["dev"], required=True)
    b.add_argument("--url", default=RERANK_URL)
    b.add_argument("--rule", type=Path, default=RULE)
    b.add_argument("--vllm-running", action="store_true")
    b.add_argument("--note", required=True, help="power source and Windows power mode")
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    if args.step == "serve":
        run_serve(args)
        return
    import asyncio

    result = asyncio.run(bench(args))
    record = {
        "command": " ".join(["uv run python -m filinglens.agents.rerank", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **result,
    }
    out = Path(f"bench/agent_rerank_{result['device']}.json")
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
