r"""FilingLens API (FastAPI): per-company search and cited answers (FL-M5 T12).

    # local: compose postgres (demo schema loaded by api.demo_db), .env sourced
    uv run --group api uvicorn filinglens.api.app:app --port 8080
    curl -s localhost:8080/health

Routes:
- GET  /health      process up, encoder loaded, database reachable
- GET  /companies   the tickers the demo database holds
- POST /search      {ticker, question, k}: int8 bge-base query vector, then FL-M3's per-company
                    exact inner-product search (api.demo_db.SEARCH_SQL); each hit links to its
                    SEC filing
- POST /answer      {ticker, question}: /search, then Gemini answers from the top 5 excerpts with
                    numbered citations; metered (serve.cost_meter, route "answer") per user
                    (header X-User, default "anonymous"; a demo label, not authentication)
- GET  /costs       requests, tokens, p95 latency and list-price cost per route and backend from
                    serve.requests; actual spend is $0 (Gemini free tier)

Only Gemini runs in the cloud; the local GPU / CPU backends are demo-only on the laptop (spec).
Settings, all from the environment: DATABASE_URL (or POSTGRES_PASSWORD for the local compose
database), GEMINI_API_KEY, GEMINI_RPM / GEMINI_RPD (AI Studio free-tier limits), GEMINI_QUOTA_FILE,
EMBED_MODEL / EMBED_TOKENIZER (ONNX graph + tokenizer.json), ROUTER_INPUTS (list prices),
OTEL_CONSOLE=1 to print spans as JSON lines (Cloud Run sends stdout to Cloud Logging).
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from filinglens.api.logic import (
    ANSWER_CONTEXT_CHUNKS,
    ANSWER_MAX_TOKENS,
    Hit,
    answer_messages,
    citations,
    hit_json,
    http_status_for,
)

ONNX_DIR = Path("data/interim/serve/onnx")
COSTS_SQL = (
    "SELECT route, backend, count(*), sum(prompt_tokens), sum(output_tokens + thinking_tokens),"
    " percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_s), sum(cost_usd),"
    " count(*) FILTER (WHERE error IS NOT NULL)"
    " FROM serve.requests WHERE ts > now() - interval '30 days' GROUP BY 1, 2 ORDER BY 1, 2"
)


class Store(Protocol):
    def ping(self) -> bool: ...
    def companies(self) -> list[dict]: ...
    def company(self, ticker: str) -> dict | None: ...
    def search(self, cik: int, vector: Any, k: int) -> list[Hit]: ...
    def costs(self) -> list[dict]: ...


class PgStore:
    """The demo database over one psycopg connection (its lock serializes the threads)."""

    def __init__(self) -> None:
        from filinglens.api.demo_db import connect

        self.con = connect()
        self.lock = threading.Lock()

    def _all(self, sql: str, params: dict | None = None) -> list[tuple]:
        with self.lock:
            return self.con.execute(sql, params).fetchall()

    def ping(self) -> bool:
        return self._all("SELECT 1")[0][0] == 1

    def companies(self) -> list[dict]:
        rows = self._all("SELECT ticker, sector, chunks FROM demo.companies ORDER BY ticker")
        return [{"ticker": t, "sector": s, "chunks": n} for t, s, n in rows]

    def company(self, ticker: str) -> dict | None:
        from filinglens.api.demo_db import COMPANY_SQL

        rows = self._all(COMPANY_SQL, {"ticker": ticker.upper()})
        if not rows:
            return None
        cik, t, sector, n = rows[0]
        return {"cik": cik, "ticker": t, "sector": sector, "chunks": n}

    def search(self, cik: int, vector: Any, k: int) -> list[Hit]:
        from filinglens.api.demo_db import SEARCH_SQL

        rows = self._all(SEARCH_SQL, {"q": vector, "cik": cik, "k": k})
        return [Hit(*r) for r in rows]

    def costs(self) -> list[dict]:
        rows = self._all(COSTS_SQL)
        keys = ("route", "backend", "requests", "prompt_tokens", "output_tokens",
                "latency_p95_s", "cost_usd", "errors")  # fmt: skip
        return [dict(zip(keys, r, strict=True)) for r in rows]


@dataclass
class Deps:
    store: Store
    embedder: Any  # .embed_queries(list[str]) -> (n, 768) array
    llm: Any  # serve.cost_meter.MeteredBackend (or a fake with the same chat())
    prices: Any  # serve.cost_meter.Prices


def tracer_from_env():
    """A real TracerProvider (so trace ids are real); spans printed only if OTEL_CONSOLE=1."""
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider(resource=Resource.create({"service.name": "filinglens-api"}))
    if os.environ.get("OTEL_CONSOLE") == "1":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

        exporter = ConsoleSpanExporter(formatter=lambda s: s.to_json(indent=None) + "\n")
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("filinglens.api")


def default_deps() -> Deps:
    from filinglens.api.embedder import OnnxEmbedder
    from filinglens.serve.backends import DailyQuota, GeminiBackend
    from filinglens.serve.cost_meter import MeteredBackend, PostgresSink, Prices
    from filinglens.serve.router_inputs import GEMINI_MODEL

    env = os.environ
    embedder = OnnxEmbedder(
        Path(env.get("EMBED_MODEL", ONNX_DIR / "bge-base.eager-opt-int8.onnx")),
        Path(env.get("EMBED_TOKENIZER", ONNX_DIR / "bge-base.tokenizer.json")),
        threads=int(env.get("EMBED_THREADS", "1")),
    )
    key = env.get("GEMINI_API_KEY", "")
    quota = DailyQuota(
        int(env.get("GEMINI_RPD", "500")),
        Path(env.get("GEMINI_QUOTA_FILE", "data/interim/serve/gemini_quota.json")),
    )
    gemini = GeminiBackend(GEMINI_MODEL, key, int(env.get("GEMINI_RPM", "15")), quota)
    prices = Prices.from_inputs(
        json.loads(Path(env.get("ROUTER_INPUTS", "bench/router_inputs.json")).read_text())
    )
    llm = MeteredBackend(gemini, prices, [PostgresSink()], tracer_from_env())
    return Deps(PgStore(), embedder, llm, prices)


class SearchIn(BaseModel):
    ticker: str = Field(min_length=1, max_length=10)
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=10, ge=1, le=20)


class AnswerIn(BaseModel):
    ticker: str = Field(min_length=1, max_length=10)
    question: str = Field(min_length=3, max_length=500)


def create_app(deps: Deps | None = None) -> FastAPI:
    """`deps` for tests; None = build the real ones at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.deps = deps or default_deps()
        yield

    app = FastAPI(title="FilingLens", version="0.1.0", lifespan=lifespan)

    def company_or_404(d: Deps, ticker: str) -> dict:
        company = d.store.company(ticker)
        if company is None:
            raise HTTPException(404, f"{ticker!r} isn't in the demo database: see /companies")
        return company

    def search(d: Deps, body: SearchIn | AnswerIn, k: int) -> tuple[dict, list[Hit]]:
        company = company_or_404(d, body.ticker)
        vector = d.embedder.embed_queries([body.question])[0]
        return company, d.store.search(company["cik"], vector, k)

    @app.get("/health")
    def health() -> dict:
        d: Deps = app.state.deps
        return {"status": "ok", "database": d.store.ping(), "spend_usd": 0}

    @app.get("/companies")
    def companies() -> dict:
        rows = app.state.deps.store.companies()
        return {"companies": rows, "count": len(rows)}

    @app.post("/search")
    def search_route(body: SearchIn) -> dict:
        company, hits = search(app.state.deps, body, body.k)
        return {
            "ticker": company["ticker"],
            "question": body.question,
            "hits": [hit_json(h, company["cik"]) for h in hits],
            "method": "bge-base int8 query vector; exact inner product within the company",
        }

    @app.post("/answer")
    async def answer_route(body: AnswerIn, x_user: str = Header("anonymous")) -> dict:
        d: Deps = app.state.deps
        company, hits = await run_in_threadpool(search, d, body, ANSWER_CONTEXT_CHUNKS)
        if not hits:
            raise HTTPException(404, f"no excerpts for {company['ticker']}")
        conv = answer_messages(company["ticker"], body.question, hits)
        reply = await d.llm.chat(
            conv, None, ANSWER_MAX_TOKENS, route="answer", user=x_user[:64] or "anonymous"
        )
        if reply.error:
            raise HTTPException(
                http_status_for(reply.error),
                {"error": reply.error[:200], "hits": [hit_json(h, company["cik"]) for h in hits]},
            )
        from filinglens.serve.cost_meter import request_cost

        usd, basis = request_cost(reply, d.prices)
        return {
            "ticker": company["ticker"],
            "question": body.question,
            "answer": reply.text,
            "citations": citations(reply.text, hits, company["cik"]),
            "model": reply.model,
            "latency_s": round(reply.latency_s, 3),
            "tokens": {"prompt": reply.prompt_tokens, "output": reply.output_tokens},
            "cost": {"list_price_usd": usd, "basis": basis, "actual_spend_usd": 0},
        }

    @app.get("/costs")
    def costs() -> dict:
        rows = app.state.deps.store.costs()
        return {
            "window": "last 30 days",
            "by_route_backend": rows,
            "list_price_usd_total": sum(r["cost_usd"] or 0 for r in rows),
            "actual_spend_usd": 0,
        }

    return app


app = create_app()
