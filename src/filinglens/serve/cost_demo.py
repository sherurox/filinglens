r"""Cost meter demo: a labeled replay through the metered backends, then the report and chart (T11).

    # vLLM (GPU) and llama-server (CPU) both up, compose postgres up, .env sourced
    uv run python -m filinglens.serve.cost_demo replay --tag demo-20261126 --rpm 15 --rpd 500
    uv run python -m filinglens.serve.cost_demo report --tag demo-20261126

`replay` sends dev prompts (extraction: FL-M2's 71 rows; short Q&A: ret-dev questions with the
paragraph, as T9) through `MeteredBackend`, in three lanes that run at the same time, each one
request at a time (as T9, so latencies are comparable):
- vllm (GPTQ): all extraction rows + the first --vllm-qa short-Q&A questions (S1's routing);
- gguf: the next --gguf-qa questions;
- gemini: the next --gemini-qa questions (a fallback demo; within today's free-tier quota).
Users alternate between "<tag>/analyst-a" and "<tag>/analyst-b". Rows go to Postgres
(serve.requests); spans go to logs/cost_meter_spans_<tag>.jsonl through a real TracerProvider
(without one, OpenTelemetry is a no-op and every trace id would be zero).

`report` reads this tag's rows back from Postgres, aggregates per route x backend and per user
(requests, errors, tokens, p50/p95 latency, list-price cost, cost per 1k requests), checks that
the span file has one span per row with the same trace ids, and writes bench/cost_meter_<tag>.json
and bench/cost_meter_<tag>.png. Actual spend is $0; the costs are list prices (cost_meter.py).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from filinglens.serve.backends import GeminiBackend, OpenAICompatBackend
from filinglens.serve.cost_meter import MeteredBackend, PostgresSink, Prices
from filinglens.serve.loadgen import percentiles
from filinglens.serve.router_inputs import (
    EXTRACT_MAX_TOKENS,
    GEMINI_MODEL,
    LOCAL,
    PARAGRAPHS,
    QA_MAX_TOKENS,
    QA_PROMPT,
    QUERIES,
    REFERENCES,
    VLLM_BUILD,
    qa_conversation,
    qa_items,
    read_jsonl,
    require_budget,
)

USERS = ("analyst-a", "analyst-b")
INPUTS = Path("bench/router_inputs.json")


def user_for(tag: str, i: int) -> str:
    return f"{tag}/{USERS[i % len(USERS)]}"


def plan(
    n_extraction: int, n_qa: int, vllm_qa: int, gguf_qa: int, gemini_qa: int
) -> dict[str, list[tuple[str, int]]]:
    """Lane -> [(route, item index)]: which prompt each backend gets, with no question reused."""
    if vllm_qa + gguf_qa + gemini_qa > n_qa:
        raise SystemExit(f"{vllm_qa + gguf_qa + gemini_qa} Q&A requests asked, {n_qa} questions")
    qa = list(range(n_qa))
    return {
        "vllm": [("extraction", i) for i in range(n_extraction)]
        + [("short_qa", i) for i in qa[:vllm_qa]],
        "gguf": [("short_qa", i) for i in qa[vllm_qa : vllm_qa + gguf_qa]],
        "gemini": [("short_qa", i) for i in qa[vllm_qa + gguf_qa : vllm_qa + gguf_qa + gemini_qa]],
    }


HEALTH = {"vllm": "/v1/models", "gguf": "/health"}


async def preflight(urls: dict[str, str]) -> None:
    """Stop before anything is sent or written if a local server isn't answering."""
    import httpx

    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in urls.items():
            try:
                ok = (await client.get(url + HEALTH[name])).status_code == 200
            except httpx.HTTPError:
                ok = False
            if not ok:
                raise SystemExit(f"{name} at {url} isn't answering {HEALTH[name]}: start it first")


def span_tracer(path: Path):
    """A real TracerProvider whose spans are written one JSON object per line to `path`."""
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    path.parent.mkdir(parents=True, exist_ok=True)
    out = path.open("w")
    provider = TracerProvider(resource=Resource.create({"service.name": "filinglens-cost-demo"}))
    exporter = ConsoleSpanExporter(out=out, formatter=lambda span: span.to_json(indent=None) + "\n")
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, provider.get_tracer("filinglens.serve.cost_demo"), out


async def run_lane(meter, items: list[tuple[str, list[dict], dict | None, int]], tag: str) -> int:
    failed = 0
    for i, (route, conv, schema, max_tokens) in enumerate(items):
        reply = await meter.chat(conv, schema, max_tokens, route=route, user=user_for(tag, i))
        failed += reply.error is not None
        if (i + 1) % 20 == 0 or i + 1 == len(items):
            print(f"{meter.name}: {i + 1}/{len(items)} ({failed} failed)", file=sys.stderr)
    return failed


def run_replay(args: argparse.Namespace) -> dict:
    from filinglens.extract.fields import PROMPT, TASKS, conversations, load_rows
    from filinglens.extract.silver_label import schema

    rows = load_rows("dev", allow_test=False)
    tasks = {t["para_id"]: t for t in json.loads(TASKS.read_text())}
    extraction = conversations(rows, tasks, PROMPT.read_text(), [])
    items = qa_items(read_jsonl(QUERIES), read_jsonl(REFERENCES), read_jsonl(PARAGRAPHS))
    system = QA_PROMPT.read_text()
    qa = [qa_conversation(item, system) for item in items]
    lanes = plan(len(extraction), len(qa), args.vllm_qa, args.gguf_qa, args.gemini_qa)

    def requests_for(lane: str) -> list:
        return [
            (route, extraction[i], schema(), EXTRACT_MAX_TOKENS)
            if route == "extraction"
            else (route, qa[i], None, QA_MAX_TOKENS)
            for route, i in lanes[lane]
        ]

    prices = Prices.from_inputs(json.loads(INPUTS.read_text()))
    if tag_rows(args.tag):
        raise SystemExit(f"serve.requests already has rows for {args.tag}: use a new --tag")
    span_path = Path(f"logs/cost_meter_spans_{args.tag}.jsonl")
    provider, tracer, span_file = span_tracer(span_path)
    sink = PostgresSink()

    async def go() -> dict:
        await preflight({k: LOCAL[k] for k in ("vllm", "gguf") if lanes[k]})
        inner = {
            "vllm": OpenAICompatBackend("vllm", LOCAL["vllm"], VLLM_BUILD),
            "gguf": OpenAICompatBackend("gguf", LOCAL["gguf"], "gguf"),
            "gemini": GeminiBackend.from_env(GEMINI_MODEL, args.rpm, args.rpd),
        }
        require_budget(inner["gemini"].quota, len(lanes["gemini"]))
        meters = {k: MeteredBackend(b, prices, [sink], tracer) for k, b in inner.items()}
        try:
            failed = await asyncio.gather(
                *(run_lane(meters[k], requests_for(k), args.tag) for k in meters)
            )
        finally:
            for b in inner.values():
                await b.transport.aclose()
        return {
            "requests": {k: len(lanes[k]) for k in lanes},
            "failed": dict(zip(meters, failed, strict=True)),
            "sink_errors": {k: m.sink_errors for k, m in meters.items()},
        }

    started = datetime.now(UTC)
    try:
        body = asyncio.run(go())
    finally:
        provider.shutdown()
        span_file.close()
        sink.close()
    return {
        "tag": args.tag,
        "started_utc": started.isoformat(timespec="seconds"),
        "wall_s": round((datetime.now(UTC) - started).total_seconds(), 1),
        "lanes": {"vllm": VLLM_BUILD, "gguf": "student-q4_k_m.gguf", "gemini": GEMINI_MODEL},
        **body,
        "spans": str(span_path),
    }


# ---------- report ----------

FIELDS = (
    "route",
    "user_id",
    "backend",
    "model",
    "prompt_tokens",
    "output_tokens",
    "thinking_tokens",
    "latency_s",
    "cost_usd",
    "error",
    "trace_id",
)
SELECT = f"SELECT {', '.join(FIELDS)} FROM serve.requests WHERE user_id LIKE %(prefix)s ORDER BY id"


def aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    """Per group: requests, errors, tokens, latency percentiles (successful calls), cost."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, rs in sorted(groups.items()):
        ok = [r for r in rs if r["error"] is None]
        cost = sum(r["cost_usd"] for r in rs)
        out.append(
            {
                **dict(zip(keys, key, strict=True)),
                "requests": len(rs),
                "errors": len(rs) - len(ok),
                "prompt_tokens": sum(r["prompt_tokens"] for r in rs),
                "output_tokens": sum(r["output_tokens"] + r["thinking_tokens"] for r in rs),
                "latency_s": percentiles([r["latency_s"] for r in ok]),
                "cost_usd": cost,
                "usd_per_1k_requests": 1000 * cost / len(rs),
            }
        )
    return out


def check_spans(rows: list[dict], span_path: Path) -> dict:
    """One span per row, same trace ids (the row stores the span's trace id)."""
    span_ids = []
    if span_path.is_file():
        for line in span_path.read_text().splitlines():
            if line.strip():
                span_ids.append(json.loads(line)["context"]["trace_id"].removeprefix("0x"))
    row_ids = [r["trace_id"] for r in rows]
    return {
        "rows": len(rows),
        "spans": len(span_ids),
        "same_trace_ids": sorted(row_ids) == sorted(span_ids),
        "zero_trace_ids": sum(t == "0" * 32 for t in row_ids),
    }


def chart(by_route: list[dict], path: Path, tag: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"vllm": "#257de0", "gguf": "#f47230", "gemini": "#00b894"}  # validated slots 1-3
    ink, grid = "#3d3d3d", "#e6e6e6"
    labels = [f"{g['route']} / {g['backend']}" for g in by_route]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    panels = (
        ("requests", "requests in the demo", False),
        ("usd_per_1k_requests", "list-price $ per 1,000 requests (log scale)", True),
    )
    for ax, (field, title, log) in zip(axes, panels, strict=True):
        values = [g[field] for g in by_route]
        bars = ax.barh(labels, values, color=[colors[g["backend"]] for g in by_route], height=0.6)
        if log:
            ax.set_xscale("log")
        for bar, v in zip(bars, values, strict=True):
            text = f"{v:,.0f}" if field == "requests" else f"${v:.4g}"
            ax.annotate(
                text,
                (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=ink,
            )
        ax.set_title(title, fontsize=10, color=ink, loc="left")
        ax.invert_yaxis()
        ax.grid(axis="x", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.margins(x=0.25)
    axes[1].set_yticklabels([])
    fig.suptitle(
        f"Cost meter demo ({tag}): serve.requests by route / backend; actual spend $0",
        fontsize=11,
        color=ink,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def tag_rows(tag: str) -> list[dict]:
    """This tag's rows in serve.requests (creates the table if it doesn't exist yet)."""
    sink = PostgresSink()
    try:
        cur = sink._connection().execute(SELECT, {"prefix": f"{tag}/%"})
        return [dict(zip(FIELDS, r, strict=True)) for r in cur.fetchall()]
    finally:
        sink.close()


def run_report(args: argparse.Namespace) -> dict:
    rows = tag_rows(args.tag)
    if not rows:
        raise SystemExit(f"no serve.requests rows for tag {args.tag}: run replay first")
    by_route = aggregate(rows, ("route", "backend"))
    png = args.bench / f"cost_meter_{args.tag}.png"
    chart(by_route, png, args.tag)
    return {
        "tag": args.tag,
        "query": SELECT,
        "rows": len(rows),
        "spend": {"actual_usd": 0, "modelled": "list prices (see serve/cost_meter.py)"},
        "total_cost_usd": sum(r["cost_usd"] for r in rows),
        "by_route_backend": by_route,
        "by_user": aggregate(rows, ("user_id",)),
        "span_check": check_spans(rows, Path(f"logs/cost_meter_spans_{args.tag}.jsonl")),
        "chart": png.as_posix(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["replay", "report"])
    parser.add_argument("--tag", required=True, help="run tag; users are <tag>/analyst-a|b")
    parser.add_argument("--rpm", type=int, default=None, help="Gemini free-tier RPM (replay)")
    parser.add_argument("--rpd", type=int, default=None, help="Gemini free-tier RPD (replay)")
    parser.add_argument("--vllm-qa", type=int, default=100)
    parser.add_argument("--gguf-qa", type=int, default=40)
    parser.add_argument("--gemini-qa", type=int, default=40)
    parser.add_argument("--bench", type=Path, default=Path("bench"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    if "/" in args.tag or "%" in args.tag or "_" in args.tag:
        raise SystemExit("--tag may not contain '/', '%' or '_' (it's a LIKE prefix)")
    if args.command == "replay":
        if args.rpm is None or args.rpd is None:
            raise SystemExit("replay needs --rpm and --rpd (AI Studio free-tier limits)")
        body = run_replay(args)
        out = args.bench / f"cost_meter_{args.tag}_replay.json"
    else:
        body = run_report(args)
        out = args.bench / f"cost_meter_{args.tag}.json"
    result = {
        "command": " ".join(["python -m filinglens.serve.cost_demo", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **body,
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
