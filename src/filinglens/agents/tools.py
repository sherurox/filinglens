"""Read-only tools for the analyst crew and the MCP server (FL-M6 T4).

One implementation, two front ends: the crew (agents.crew) calls these methods directly, and the
MCP server (mcp_server) exposes the same four as MCP tools, so the demo shows exactly what the crew
uses. Every input is validated here and every output is plain JSON-able data.

- search_filings(company, query, fiscal_years, k): FL-M3's retrieval for one company, optionally
  restricted to some fiscal years: the int8 bge-base query vector (the API's encoder), exact
  inner-product search over that company's chunks (a MATERIALIZED CTE, so no approximate index can
  be used), FL-M3's reranker over the top DEPTH (the sidecar, agents.rerank), the top k returned
  with their filing URL.
- get_financials(company, fiscal_years): revenue, operating income and net income from
  marts.fct_financials with the same fiscal-year rule as the analyst questions
  (agents.questions.annual_figures), plus margins and revenue growth computed from those exact
  values. A figure the data doesn't have is listed as missing, never estimated.
- compare_peers(company, k, fiscal_year): FL-M3's peer list (agents.peers), optionally with each
  company's figures for one fiscal year.
- extract_risks(company, fiscal_year, query, k): the top k matching paragraphs of that 10-K, each
  labeled by the extraction backend (the FL-M4 student through the router) with FL-M2's prompt and
  schema: risk category, severity, first-seen year.

"Fiscal year" is the year of the 10-K's report date, as in agents.questions; every result carries
the report date, so a reader can tell a January year end from a December one.

The database sits behind the small `Db` protocol (PgDb is the real one), so the tests run with an
in-memory fake.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from filinglens.agents.questions import (
    FACT_TAGS,
    FACTS_SQL,
    annual_figures,
    filings_table,
)
from filinglens.api.logic import sec_filing_url

DEPTH = 50  # FL-M3: rerank the dense top 50
MAX_K = 20
MAX_PEERS = 20
MAX_YEARS = 8
MAX_QUERY_CHARS = 500
FIRST_YEAR, LAST_YEAR = 2018, 2025  # the 10-K corpus (FL-M1)
METRICS = ("revenue", "operating_income", "net_income")
COMPANY_KEY = re.compile(r"^[A-Za-z0-9.\-]{1,12}$")
EXTRACT_MAX_TOKENS = 400

COMPANY_SQL = (
    "SELECT cik, ticker, company_name, sector FROM marts.dim_company"
    " WHERE upper(ticker) = upper(%(key)s) OR cik::text = %(key)s"
)
SEARCH_SQL = """
WITH company AS MATERIALIZED (
    SELECT chunk_id, embedding FROM retrieval.chunk_vectors WHERE cik = %(cik)s
)
SELECT t.chunk_id, t.period_year, t.accession, t.report_date, t.heading, t.text, t.embed_text,
       -(company.embedding <#> %(q)s) AS dense_score
FROM company JOIN agents.chunks AS t USING (chunk_id)
WHERE %(years)s::int[] IS NULL OR t.period_year = ANY(%(years)s::int[])
ORDER BY company.embedding <#> %(q)s, t.chunk_id
LIMIT %(depth)s
"""
FILINGS_SQL = (
    "SELECT DISTINCT cik, period_year, accession, report_date FROM agents.chunks"
    " WHERE cik = %(cik)s"
)
PEERS_SQL = (
    "SELECT p.rank, p.peer_cik, p.score, p.z_text, p.z_ratio, c.ticker, c.company_name, c.sector"
    " FROM agents.peers p JOIN marts.dim_company c ON c.cik = p.peer_cik"
    " WHERE p.cik = %(cik)s ORDER BY p.rank LIMIT %(k)s"
)
EARLIER_SQL = (
    "SELECT chunk_id AS para_id, period_year, heading, text FROM agents.chunks"
    " WHERE cik = %(cik)s AND period_year < %(year)s ORDER BY period_year, chunk_id"
)


class ToolError(ValueError):
    """A bad request. The message is safe to show to a model or a user (no internals)."""


@dataclass(frozen=True)
class Company:
    cik: int
    ticker: str
    name: str
    sector: str


class Db(Protocol):
    def company(self, key: str) -> dict | None: ...
    def search(self, cik: int, vector: Any, years: list[int] | None, depth: int) -> list[dict]: ...
    def filings(self, cik: int) -> list[dict]: ...
    def facts(self, cik: int) -> list[dict]: ...
    def peers(self, cik: int, k: int) -> list[dict]: ...
    def earlier_chunks(self, cik: int, year: int) -> list[dict]: ...


Reranker = Callable[[str, list[str]], Awaitable[Sequence[float]]]
Extractor = Callable[[list[dict], dict, int], Awaitable[Any]]  # (conv, schema, max_tokens) -> Reply


class PgDb:
    """The local compose Postgres: marts (FL-M1), retrieval.chunk_vectors (FL-M3), agents.*."""

    def __init__(self, con) -> None:
        self.con = con

    @classmethod
    def connect(cls) -> PgDb:
        import os

        import psycopg
        from pgvector.psycopg import register_vector

        from filinglens.serve.cost_meter import pg_conninfo

        if os.environ.get("DATABASE_URL"):
            raise SystemExit("unset DATABASE_URL: the tools read the local compose Postgres")
        con = psycopg.connect(pg_conninfo(), autocommit=True)
        register_vector(con)
        return cls(con)

    def _rows(self, sql: str, params: dict) -> list[dict]:
        with self.con.cursor() as cur:
            cur.execute(sql, params)
            names = [d.name for d in cur.description]
            return [dict(zip(names, row, strict=True)) for row in cur.fetchall()]

    def company(self, key: str) -> dict | None:
        rows = self._rows(COMPANY_SQL, {"key": key})
        return rows[0] if len(rows) == 1 else None

    def search(self, cik: int, vector: Any, years: list[int] | None, depth: int) -> list[dict]:
        return self._rows(SEARCH_SQL, {"cik": cik, "q": vector, "years": years, "depth": depth})

    def filings(self, cik: int) -> list[dict]:
        return self._rows(FILINGS_SQL, {"cik": cik})

    def facts(self, cik: int) -> list[dict]:
        return self._rows(FACTS_SQL, {"tags": list(FACT_TAGS), "ciks": [cik]})

    def peers(self, cik: int, k: int) -> list[dict]:
        return self._rows(PEERS_SQL, {"cik": cik, "k": k})

    def earlier_chunks(self, cik: int, year: int) -> list[dict]:
        return self._rows(EARLIER_SQL, {"cik": cik, "year": year})


# ---------------------------------------------------------------- input checks


def check_k(k: Any, limit: int, name: str = "k") -> int:
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= limit:
        raise ToolError(f"{name} must be an integer from 1 to {limit}")
    return k


def check_years(years: Any, required: bool) -> list[int] | None:
    if years is None or years == []:
        if required:
            raise ToolError("give at least one fiscal year")
        return None
    if not isinstance(years, list | tuple) or len(years) > MAX_YEARS:
        raise ToolError(f"fiscal_years must be a list of at most {MAX_YEARS} years")
    for y in years:
        if isinstance(y, bool) or not isinstance(y, int) or not FIRST_YEAR <= y <= LAST_YEAR:
            raise ToolError(f"fiscal years must be integers from {FIRST_YEAR} to {LAST_YEAR}")
    return sorted(set(years))


def check_query(query: Any) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ToolError("query must be a non-empty string")
    query = " ".join(query.split())
    if len(query) > MAX_QUERY_CHARS:
        raise ToolError(f"query must be at most {MAX_QUERY_CHARS} characters")
    return query


# ---------------------------------------------------------------- financial figures (pure)


def year_figures(cik: int, years: list[int], filings_rows: list[dict], facts: list[dict]) -> list:
    """Per requested year: the filing, the three figures (or why they're missing), and margins."""
    filings = {k: v for k, v in filings_table(filings_rows).items() if k[0] == cik}
    figures, no_dates = annual_figures(facts, filings)
    out = []
    for fy in years:
        filing = filings.get((cik, fy))
        if filing is None:
            out.append(
                {
                    "fiscal_year": fy,
                    "available": False,
                    "figures": {},
                    "missing": list(METRICS),
                    "derived": {},
                    "reason": "no 10-K for that fiscal year in the corpus",
                }
            )
            continue
        entry = figures.get((cik, fy), {})
        figs = {m: _public(entry[m]) for m in METRICS if m in entry}
        missing = [m for m in METRICS if m not in figs]
        if (cik, fy) in no_dates:
            reason = "no annual facts dated within 20 days of the report date"
        elif not entry:
            reason = "two fact dates near the report date: ambiguous year, no figures given"
        else:
            reason = None
        out.append(
            {
                "fiscal_year": fy,
                "available": bool(figs),
                "report_date": filing["report_date"].isoformat(),
                "accession": filing["accession"],
                "filing_url": sec_filing_url(cik, filing["accession"]),
                "figures": figs,
                "missing": missing,
                "derived": margins(figs),
                "reason": reason,
            }
        )
    return out


def _public(fact: dict) -> dict:
    keep = ("tag", "value_date", "unit", "value", "first_reported_value", "was_restated")
    return {k: fact[k] for k in keep}


def margins(figs: dict) -> dict:
    rev = figs.get("revenue", {}).get("value")
    out = {}
    if rev is not None and rev > 0:
        for name, metric in (
            ("operating_margin", "operating_income"),
            ("net_margin", "net_income"),
        ):
            if metric in figs:
                out[name] = {"value": figs[metric]["value"] / rev, "of": [metric, "revenue"]}
    return out


def growth(rows: list[dict]) -> list[dict]:
    """Revenue growth between consecutive requested years that both have revenue > 0."""
    by_year = {r["fiscal_year"]: r for r in rows}
    out = []
    for fy in sorted(by_year):
        old, new = by_year.get(fy - 1), by_year[fy]
        r_old = (old or {}).get("figures", {}).get("revenue", {}).get("value")
        r_new = new.get("figures", {}).get("revenue", {}).get("value")
        if r_old is not None and r_new is not None and r_old > 0:
            out.append({"from": fy - 1, "to": fy, "revenue_growth": r_new / r_old - 1.0})
    return out


# ---------------------------------------------------------------- the tools


class Tools:
    def __init__(
        self,
        db: Db,
        embedder: Any,
        reranker: Reranker | None,
        extractor: Extractor | None = None,
        extract_prompt: str | None = None,
        depth: int = DEPTH,
    ) -> None:
        self.db, self.embedder, self.reranker = db, embedder, reranker
        self.extractor, self.extract_prompt, self.depth = extractor, extract_prompt, depth

    def company(self, key: Any) -> Company:
        if not isinstance(key, str | int) or not COMPANY_KEY.match(str(key).strip()):
            raise ToolError("company must be a ticker (e.g. DAL) or a CIK number")
        row = self.db.company(str(key).strip())
        if row is None:
            raise ToolError(
                f"unknown company {str(key).strip()!r}: not in the 300-company universe"
            )
        return Company(int(row["cik"]), row["ticker"], row["company_name"], row["sector"])

    async def search_filings(
        self, company: Any, query: Any, fiscal_years: Any = None, k: Any = 10, rerank: bool = True
    ) -> dict:
        c = self.company(company)
        query = check_query(query)
        years = check_years(fiscal_years, required=False)
        k = check_k(k, MAX_K)
        vector = self.embedder.embed_queries([query])[0]
        cands = self.db.search(c.cik, vector, years, self.depth)
        order = list(range(len(cands)))
        scores: list[float | None] = [None] * len(cands)
        reranked = bool(rerank and self.reranker is not None and cands)
        if reranked:
            from filinglens.retrieval.rerank import rerank_order

            got = list(await self.reranker(query, [x["embed_text"] for x in cands]))
            ids = [str(i) for i in order]
            order = [int(i) for i in rerank_order(ids, got)]
            scores = got
        hits = []
        for rank, i in enumerate(order[:k], start=1):
            x = cands[i]
            hits.append(
                {
                    "rank": rank,
                    "chunk_id": x["chunk_id"],
                    "fiscal_year": int(x["period_year"]),
                    "report_date": x["report_date"].isoformat(),
                    "accession": x["accession"],
                    "filing_url": sec_filing_url(c.cik, x["accession"]),
                    "heading": x["heading"],
                    "text": x["text"],
                    "dense_rank": i + 1,
                    "dense_score": round(float(x["dense_score"]), 6),
                    "rerank_score": None if scores[i] is None else round(float(scores[i]), 6),
                }
            )
        return {
            "company": asdict(c),
            "query": query,
            "fiscal_years": years,
            "candidates": len(cands),
            "reranked": reranked,
            "hits": hits,
        }

    def get_financials(self, company: Any, fiscal_years: Any) -> dict:
        c = self.company(company)
        years = check_years(fiscal_years, required=True)
        rows = year_figures(c.cik, years, self.db.filings(c.cik), self.db.facts(c.cik))
        return {
            "company": asdict(c),
            "source": "marts.fct_financials (SEC Financial Statement Data Sets), USD, annual",
            "fiscal_year_rule": "the year of the 10-K's report date",
            "years": rows,
            "growth": growth(rows),
        }

    def compare_peers(self, company: Any, k: Any = 5, fiscal_year: Any = None) -> dict:
        c = self.company(company)
        k = check_k(k, MAX_PEERS)
        year = check_years([fiscal_year], required=True)[0] if fiscal_year is not None else None
        peers = []
        for p in self.db.peers(c.cik, k):
            entry = {
                "rank": int(p["rank"]),
                "cik": int(p["peer_cik"]),
                "ticker": p["ticker"],
                "name": p["company_name"],
                "sector": p["sector"],
                "score": round(float(p["score"]), 4),
                "text_similarity_z": round(float(p["z_text"]), 4),
                "ratio_similarity_z": round(float(p["z_ratio"]), 4),
            }
            if year is not None:
                entry["financials"] = self.get_financials(p["ticker"], [year])["years"][0]
            peers.append(entry)
        out = {
            "company": asdict(c),
            "method": "FL-M3 recommender: alpha * z(latest-10-K text cosine)"
            " + (1 - alpha) * z(ratio-rank similarity)",
            "peers": peers,
        }
        if year is not None:
            out["company_financials"] = self.get_financials(c.ticker, [year])["years"][0]
        return out

    async def extract_risks(self, company: Any, fiscal_year: Any, query: Any, k: Any = 3) -> dict:
        if self.extractor is None or self.extract_prompt is None:
            raise ToolError("risk extraction is not available: no extraction backend configured")
        from filinglens.extract.silver_label import messages, parse, schema
        from filinglens.train.teacher_inputs import candidate_index, first_seen

        c = self.company(company)
        (year,) = check_years([fiscal_year], required=True)
        k = check_k(k, 5)
        found = await self.search_filings(c.ticker, query, [year], k)
        import pandas as pd

        earlier = pd.DataFrame(
            self.db.earlier_chunks(c.cik, year),
            columns=["para_id", "period_year", "heading", "text"],
        )
        index = candidate_index(earlier)
        risks = []
        for hit in found["hits"]:
            task = {
                "para_id": hit["chunk_id"],
                "ticker": c.ticker,
                "sector": c.sector,
                "period_year": year,
                "heading": hit["heading"],
                "text": hit["text"],
            }
            task["first_seen"] = first_seen(task, index)
            reply = await self.extractor(
                messages(task, self.extract_prompt), schema(), EXTRACT_MAX_TOKENS
            )
            base = {
                "chunk_id": hit["chunk_id"],
                "filing_url": hit["filing_url"],
                "heading": hit["heading"],
                "backend": getattr(reply, "backend", None),
            }
            if getattr(reply, "error", None):
                risks.append(base | {"valid": False, "problems": [f"backend: {reply.error}"]})
                continue
            rec = parse(reply.text, task)
            risks.append(
                base
                | {
                    k2: rec.get(k2)
                    for k2 in ("category", "severity", "first_seen_year", "valid", "problems")
                }
            )
        return {
            "company": asdict(c),
            "fiscal_year": year,
            "query": found["query"],
            "labels": "FL-M2 taxonomy (docs/LABELING_GUIDE.md); model-made, not checked",
            "risks": risks,
        }


# ---------------------------------------------------------------- local wiring and the T4 check

ONNX_DIR = "data/interim/serve/onnx"
ENCODER = f"{ONNX_DIR}/bge-base.eager-opt-int8.onnx"  # the API's encoder (bench/m5_embed_rule.json)
TOKENIZER = f"{ONNX_DIR}/bge-base.tokenizer.json"
FL_M5_INT8_RECALL = 0.6648  # bench/demo_db_check_local.json: per-company recall@10, int8 encoder


def local_tools(
    rerank_url: str | None = None,
    extractor: Extractor | None = None,
    extract_prompt: str | None = None,
) -> Tools:
    """The tools on this laptop: compose Postgres, int8 encoder, reranker sidecar (if a URL)."""
    from pathlib import Path

    from filinglens.agents.rerank import HttpReranker
    from filinglens.api.embedder import OnnxEmbedder

    embedder = OnnxEmbedder(Path(ENCODER), Path(TOKENIZER), threads=4)
    reranker = HttpReranker(rerank_url) if rerank_url else None
    return Tools(PgDb.connect(), embedder, reranker, extractor, extract_prompt)


async def run_check(rerank_url: str | None) -> dict:
    """ret-dev through search_filings: dense recall@10 must reproduce FL-M5's int8 per-company
    recall within one query; with the sidecar, the reranked metrics and end-to-end latency too."""
    import time

    import numpy as np

    from filinglens.api.demo_db import within_one_query
    from filinglens.retrieval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
    from filinglens.retrieval.queries import load_queries

    tools = local_tools(rerank_url)
    queries = load_queries("dev")
    out: dict = {"queries": len(queries)}
    variants = [("dense", False)] + ([("reranked", True)] if rerank_url else [])
    for name, rerank in variants:
        recall, mrr, ndcg, ms = [], [], [], []
        for q in queries:
            started = time.perf_counter()
            res = await tools.search_filings(str(q["cik"]), q["question"], None, 10, rerank)
            ms.append((time.perf_counter() - started) * 1000)
            ids = [h["chunk_id"] for h in res["hits"]]
            recall.append(recall_at_k(ids, q["grades"], 10))
            mrr.append(mrr_at_k(ids, q["grades"], 10))
            ndcg.append(ndcg_at_k(ids, q["grades"], 10))
        p50, p95 = np.percentile(ms, [50, 95])
        out[name] = {
            "recall_at_10": round(float(np.mean(recall)), 4),
            "mrr_at_10": round(float(np.mean(mrr)), 4),
            "ndcg_at_10": round(float(np.mean(ndcg)), 4),
            "latency_ms": {"p50": round(float(p50), 1), "p95": round(float(p95), 1)},
        }
    dense = out["dense"]["recall_at_10"]
    out["reproduces_fl_m5_int8"] = {
        "expected": FL_M5_INT8_RECALL,
        "got": dense,
        "within_one_query": within_one_query(dense, FL_M5_INT8_RECALL, len(queries)),
    }
    out["ok"] = out["reproduces_fl_m5_int8"]["within_one_query"]
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse
    import asyncio
    import json
    import sys
    from datetime import UTC, datetime
    from pathlib import Path

    parser = argparse.ArgumentParser(description="FL-M6 T4: check the tools against FL-M3/M5")
    parser.add_argument("step", choices=["check"])
    parser.add_argument("--split", choices=["dev"], required=True)
    parser.add_argument(
        "--rerank-url", default=None, help="the sidecar, e.g. http://127.0.0.1:58090"
    )
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    record = {
        "command": " ".join(["uv run --group api python -m filinglens.agents.tools", *argv]),
        **asyncio.run(run_check(args.rerank_url)),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    Path("bench/agent_tools_check.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if not record["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
