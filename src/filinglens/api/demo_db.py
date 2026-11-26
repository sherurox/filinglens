r"""The API's demo database: the 77 ret-dev companies' chunks in Postgres + pgvector (FL-M5 T12).

    set -a; source .env; set +a          # POSTGRES_PASSWORD (local) or DATABASE_URL (Neon, T13)
    uv run python -m filinglens.api.demo_db build
    uv run --group api python -m filinglens.api.demo_db check

Search is per company, as FL-M3 measured it: the questions name a ticker, and for those filtered
queries pgvector's planner used the B-tree on cik and an exact inner-product sort (100 / 100 plans,
bench/vector_stores.json), never the HNSW graph. So the demo has no HNSW index: a B-tree on cik and
the exact sort over one company's chunks (~770). Same results as FL-M3's filtered exact search, and
the vector index (about the size of the vectors) stays out of Neon's 1 GB.

`build`: schema `demo` with `demo.companies` (cik, ticker, sector, chunks) and `demo.chunks`
(chunk_id, cik, ticker, period_year, accession, heading, text, embedding vector(768)): FL-M3's chunk
rows and its stored bge-base vectors (no re-embedding), loaded with binary COPY; then the B-tree,
ANALYZE, and the measured sizes (pg_total_relation_size: table + TOAST + indexes).

`check` (dev only): the 182 ret-dev questions through the API's SQL (`SEARCH_SQL`), (1) with FL-M3's
stored query vectors: recall@10 must reproduce FL-M3's filtered exact recall, 0.6758, within one
query (0.0055: ties at the 10th place may order differently); (2) with the int8 ONNX encoder the API
ships (bench/m5_embed_rule.json): recall@10 reported, plus the SQL latency per query.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

QUERIES = Path("gold/retrieval/queries.dev.jsonl")
CHUNKS = Path("data/interim/retrieval/chunks.parquet")
EMB_DIR = Path("data/interim/retrieval/bge-base")
FL_M3_FILTERED_RECALL = 0.6758  # bench/vector_stores.json, reproduce_t4b (faiss-flat, filtered)
ONE_QUERY = 1 / 182
DIM = 768

DDL = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS demo;
DROP TABLE IF EXISTS demo.chunks;
DROP TABLE IF EXISTS demo.companies;
CREATE TABLE demo.companies (
    cik integer PRIMARY KEY,
    ticker text NOT NULL UNIQUE,
    sector text NOT NULL,
    chunks integer NOT NULL
);
CREATE TABLE demo.chunks (
    chunk_id text PRIMARY KEY,
    cik integer NOT NULL REFERENCES demo.companies (cik),
    ticker text NOT NULL,
    period_year integer NOT NULL,
    accession text NOT NULL,
    heading text,
    text text NOT NULL,
    embedding vector({DIM}) NOT NULL
);
"""
SEARCH_SQL = (
    "SELECT chunk_id, ticker, period_year, accession, heading, text,"
    " -(embedding <#> %(q)s) AS score"
    " FROM demo.chunks WHERE cik = %(cik)s ORDER BY embedding <#> %(q)s LIMIT %(k)s"
)
COMPANY_SQL = "SELECT cik, ticker, sector, chunks FROM demo.companies WHERE ticker = %(ticker)s"
SIZES_SQL = (
    "SELECT c.relname, pg_total_relation_size(c.oid) FROM pg_class c"
    " JOIN pg_namespace n ON n.oid = c.relnamespace"
    " WHERE n.nspname = 'demo' AND c.relkind = 'r' ORDER BY 1"
)


def connect():
    import psycopg
    from pgvector.psycopg import register_vector

    from filinglens.serve.cost_meter import pg_conninfo

    con = psycopg.connect(pg_conninfo(), autocommit=True)
    con.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(con)
    # a plan per parameter set, as FL-M3 (a generic plan could skip the cik B-tree)
    con.execute("SET plan_cache_mode = force_custom_plan")
    return con


def dev_queries() -> list[dict]:
    return [json.loads(line) for line in QUERIES.read_text().splitlines() if line.strip()]


def subset_rows(chunks, ids: list[str], vectors: np.ndarray, ciks: set[int]):
    """Chunks of the given companies with their stored vectors, in chunk_ids order."""
    pos = {cid: i for i, cid in enumerate(ids)}
    sub = chunks[chunks["cik"].isin(ciks)].copy()
    missing = [c for c in sub["chunk_id"] if c not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} subset chunks have no stored vector: {missing[:3]}")
    sub["row"] = [pos[c] for c in sub["chunk_id"]]
    sub = sub.sort_values("row")
    return sub, vectors[sub["row"].to_numpy()]


VECTOR_LITERAL = re.compile(r"'\[[^']*\]'::vector")


def short_plan(lines: list[str]) -> list[str]:
    """EXPLAIN lines with the 768-number query vector replaced by a placeholder."""
    return [VECTOR_LITERAL.sub("<query vector>", line) for line in lines]


def recall_hit(top_ids: list[str], grades: dict[str, int]) -> bool:
    return any(grades.get(c, 0) >= 1 for c in top_ids)


def run_build(args: argparse.Namespace) -> dict:
    import pandas as pd

    queries = dev_queries()
    ciks = {q["cik"] for q in queries}
    cols = ["chunk_id", "cik", "ticker", "sector", "period_year", "accession", "heading", "text"]
    chunks = pd.read_parquet(CHUNKS, columns=cols)
    ids = (EMB_DIR / "chunk_ids.txt").read_text().splitlines()
    vectors = np.load(EMB_DIR / "chunks.npy", mmap_mode="r")
    sub, vecs = subset_rows(chunks, ids, vectors, ciks)
    companies = (
        sub.groupby(["cik", "ticker", "sector"])
        .size()
        .reset_index(name="chunks")
        .sort_values("cik")
    )
    if companies["cik"].duplicated().any():
        raise SystemExit("a cik maps to more than one ticker or sector in the subset")

    con = connect()
    started = time.perf_counter()
    con.execute(DDL)
    with con.cursor() as cur:
        with cur.copy("COPY demo.companies (cik, ticker, sector, chunks) FROM STDIN") as copy:
            for r in companies.itertuples(index=False):
                copy.write_row([int(r.cik), r.ticker, r.sector, int(r.chunks)])
        copy_sql = (
            "COPY demo.chunks (chunk_id, cik, ticker, period_year, accession, heading, text,"
            " embedding) FROM STDIN WITH (FORMAT BINARY)"
        )
        with cur.copy(copy_sql) as copy:
            copy.set_types(["text", "int4", "text", "int4", "text", "text", "text", "vector"])
            for r, v in zip(sub.itertuples(index=False), vecs, strict=True):
                copy.write_row(
                    [r.chunk_id, int(r.cik), r.ticker, int(r.period_year), r.accession,
                     r.heading if isinstance(r.heading, str) else None, r.text,
                     np.asarray(v, dtype=np.float32)]
                )  # fmt: skip
    load_s = time.perf_counter() - started
    con.execute("CREATE INDEX chunks_cik_idx ON demo.chunks (cik)")
    con.execute("ANALYZE demo.companies")
    con.execute("ANALYZE demo.chunks")
    sizes = dict(con.execute(SIZES_SQL).fetchall())
    server = con.execute("SELECT version()").fetchone()[0]
    con.close()
    return {
        "target": "DATABASE_URL" if os.environ.get("DATABASE_URL") else "local",
        "server": server,
        "companies": len(companies),
        "rows": len(sub),
        "load_s": round(load_s, 1),
        "bytes": sizes,
        "bytes_total": sum(sizes.values()),
        "indexes": [
            "demo.chunks primary key",
            "chunks_cik_idx (B-tree)",
            "no HNSW (see docstring)",
        ],
    }


def run_check(args: argparse.Namespace) -> dict:
    from filinglens.api.embedder import OnnxEmbedder

    queries = dev_queries()
    stored = np.load(EMB_DIR / "queries.dev.npy")
    stored_ids = (EMB_DIR / "query_ids.dev.txt").read_text().splitlines()
    if stored_ids != [q["query_id"] for q in queries]:
        raise SystemExit("FL-M3's stored dev query vectors are not in the query order")
    onnx_dir = Path("data/interim/serve/onnx")
    encoder = OnnxEmbedder(onnx_dir / "bge-base.eager-opt-int8.onnx",
                           onnx_dir / "bge-base.tokenizer.json", threads=1)  # fmt: skip
    con = connect()
    out = {}
    for name in ("fl_m3_stored_vectors", "int8_onnx_encoder"):
        hits, ms = 0, []
        for i, q in enumerate(queries):
            vec = stored[i] if name == "fl_m3_stored_vectors" else encoder.embed_queries(
                [q["question"]])[0]  # fmt: skip
            started = time.perf_counter()
            rows = con.execute(SEARCH_SQL, {"q": vec, "cik": q["cik"], "k": 10}).fetchall()
            ms.append((time.perf_counter() - started) * 1000)
            hits += recall_hit([r[0] for r in rows], q["grades"])
        a = np.asarray(ms)
        out[name] = {
            "recall_at_10": round(hits / len(queries), 4),
            "sql_ms": {
                "p50": round(float(np.percentile(a, 50)), 2),
                "p95": round(float(np.percentile(a, 95)), 2),
            },
        }
    plan = con.execute(
        "EXPLAIN " + SEARCH_SQL, {"q": stored[0], "cik": queries[0]["cik"], "k": 10}
    ).fetchall()
    con.close()
    gap = abs(out["fl_m3_stored_vectors"]["recall_at_10"] - FL_M3_FILTERED_RECALL)
    return {
        "queries": len(queries),
        "fl_m3_filtered_exact_recall_at_10": FL_M3_FILTERED_RECALL,
        "results": out,
        "reproduces_fl_m3": gap <= ONE_QUERY + 1e-9,
        "reproduce_gap": round(gap, 4),
        "plan_first_query": short_plan([r[0] for r in plan]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["build", "check"])
    parser.add_argument("--bench", type=Path, default=Path("bench"))
    parser.add_argument("--tag", default="local", help="output suffix: local or neon")
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    body = run_build(args) if args.command == "build" else run_check(args)
    result = {
        "command": " ".join(["python -m filinglens.api.demo_db", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **body,
    }
    out = args.bench / f"demo_db_{args.command}_{args.tag}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result.get("reproduces_fl_m3", True) else 1


if __name__ == "__main__":
    sys.exit(main())
