"""Tables the agent tools read from the local Postgres (FL-M6 T4).

`retrieval.chunk_vectors` (FL-M3) holds only chunk_id, cik and the bge-base vector, so the tools
also need the chunks' text and filing metadata, and the peer lists. Both are built once here:

- agents.chunks: one row per FL-M3 chunk (chunk_id, cik, ticker, sector, period_year, accession,
  report_date, heading, text, embed_text), from data/interim/retrieval/chunks.parquet plus each
  filing's report date from the FL-M2 corpus; B-tree on (cik, period_year). `embed_text` is the
  heading-prefixed text the reranker scored in FL-M3, kept so the tools rerank the same input.
- agents.peers: each company's top TOP_PEERS peers from FL-M3's recommender, recomputed with its own
  functions (retrieval.peers.load_inputs: latest-10-K mean embedding + ratio percentile ranks,
  each z-scored) and the alpha it selected (bench/peers_select.json), ranking against all other
  companies, as FL-M3's test step did.

`check` proves the tables match their sources: the same chunk IDs as retrieval.chunk_vectors and
the parquet file (none missing on either side), TOP_PEERS peers per company, none of them itself.

    uv run python -m filinglens.agents.tool_tables build
    uv run python -m filinglens.agents.tool_tables check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from filinglens.retrieval.config import CHUNKS, CORPUS, EMB_DIR

TOP_PEERS = 20
PEERS_SELECT = Path("bench/peers_select.json")
STATS = Path("bench/agent_tables.json")
CHUNK_COLUMNS = [
    "chunk_id",
    "cik",
    "ticker",
    "sector",
    "period_year",
    "accession",
    "heading",
    "text",
    "embed_text",
]

DDL = """
CREATE SCHEMA IF NOT EXISTS agents;
DROP TABLE IF EXISTS agents.chunks;
DROP TABLE IF EXISTS agents.peers;
CREATE TABLE agents.chunks (
    chunk_id text PRIMARY KEY,
    cik integer NOT NULL,
    ticker text NOT NULL,
    sector text NOT NULL,
    period_year integer NOT NULL,
    accession text NOT NULL,
    report_date date NOT NULL,
    heading text,
    text text NOT NULL,
    embed_text text NOT NULL
);
CREATE TABLE agents.peers (
    cik integer NOT NULL,
    rank integer NOT NULL,
    peer_cik integer NOT NULL,
    score double precision NOT NULL,
    z_text double precision NOT NULL,
    z_ratio double precision NOT NULL,
    alpha double precision NOT NULL,
    PRIMARY KEY (cik, rank)
);
"""
INDEX_SQL = "CREATE INDEX agents_chunks_cik_year ON agents.chunks (cik, period_year)"
CHECK_SQL = {
    "chunks": "SELECT count(*) FROM agents.chunks",
    "chunk_vectors": "SELECT count(*) FROM retrieval.chunk_vectors",
    "only_in_chunks": "SELECT count(*) FROM agents.chunks c"
    " LEFT JOIN retrieval.chunk_vectors v USING (chunk_id) WHERE v.chunk_id IS NULL",
    "only_in_vectors": "SELECT count(*) FROM retrieval.chunk_vectors v"
    " LEFT JOIN agents.chunks c USING (chunk_id) WHERE c.chunk_id IS NULL",
    "cik_mismatch": "SELECT count(*) FROM agents.chunks c"
    " JOIN retrieval.chunk_vectors v USING (chunk_id) WHERE c.cik <> v.cik",
    "companies": "SELECT count(DISTINCT cik) FROM agents.chunks",
    "filings": "SELECT count(DISTINCT accession) FROM agents.chunks",
    "peer_rows": "SELECT count(*) FROM agents.peers",
    "peer_companies": "SELECT count(DISTINCT cik) FROM agents.peers",
    "self_peers": "SELECT count(*) FROM agents.peers WHERE cik = peer_cik",
    "short_peer_lists": "SELECT count(*) FROM (SELECT cik FROM agents.peers GROUP BY cik"
    f" HAVING count(*) <> {TOP_PEERS}) s",
    "table_bytes": "SELECT pg_total_relation_size('agents.chunks')"
    " + pg_total_relation_size('agents.peers')",
}


def find_alpha(record: Any) -> float:
    """The selected alpha in bench/peers_select.json, wherever the record nests it."""
    if isinstance(record, dict):
        if isinstance(record.get("alpha"), int | float) and not isinstance(record["alpha"], bool):
            return float(record["alpha"])
        for value in record.values():
            try:
                return find_alpha(value)
            except KeyError:
                continue
    raise KeyError("no numeric 'alpha' in the peer selection record")


def peer_rows(
    order: list[int], z_text: np.ndarray, z_ratio: np.ndarray, alpha: float, top: int = TOP_PEERS
) -> list[tuple]:
    """(cik, rank, peer_cik, score, z_text, z_ratio, alpha) for each company's `top` peers, with
    FL-M3's top_k (ties keep index order; a company never recommends itself)."""
    from filinglens.retrieval.peers import top_k

    score = alpha * z_text + (1 - alpha) * z_ratio
    best = top_k(score, k=min(top, len(order) - 1))
    rows = []
    for i, cik in enumerate(order):
        for rank, j in enumerate(best[i], start=1):
            rows.append(
                (
                    int(cik),
                    rank,
                    int(order[j]),
                    float(score[i, j]),
                    float(z_text[i, j]),
                    float(z_ratio[i, j]),
                    float(alpha),
                )
            )
    return rows


def connect():
    import psycopg

    from filinglens.serve.cost_meter import pg_conninfo

    if os.environ.get("DATABASE_URL"):
        raise SystemExit("unset DATABASE_URL: the agent tables live in the local compose Postgres")
    return psycopg.connect(pg_conninfo(), autocommit=True)


def run_build(args: argparse.Namespace) -> dict:
    import pandas as pd

    from filinglens.retrieval import peers

    chunks = pd.read_parquet(args.chunks, columns=CHUNK_COLUMNS)
    dates = pd.read_parquet(args.corpus, columns=["accession", "report_date"])
    dates = dates.drop_duplicates("accession")
    chunks = chunks.merge(dates, on="accession", how="left", validate="many_to_one")
    if chunks["report_date"].isna().any():
        raise SystemExit(f"{int(chunks['report_date'].isna().sum())} chunks have no report date")
    chunks["report_date"] = pd.to_datetime(chunks["report_date"]).dt.date

    alpha = find_alpha(json.loads(args.peers_select.read_text()))
    ns = SimpleNamespace(
        chunks=args.chunks,
        emb_dir=args.emb_dir,
        universe=peers.UNIVERSE,
        splits=peers.SPLITS,
        ratios=peers.RATIOS,
    )
    data = peers.load_inputs(ns)
    rows = peer_rows(data["order"], data["z_text"], data["z_ratio"], alpha)

    con = connect()
    started = time.perf_counter()
    con.execute(DDL)
    with con.cursor() as cur:
        copy_sql = (
            "COPY agents.chunks (chunk_id, cik, ticker, sector, period_year, accession,"
            " report_date, heading, text, embed_text) FROM STDIN"
        )
        with cur.copy(copy_sql) as copy:
            for r in chunks.itertuples(index=False):
                heading = r.heading if isinstance(r.heading, str) else None
                copy.write_row(
                    [
                        r.chunk_id,
                        int(r.cik),
                        r.ticker,
                        r.sector,
                        int(r.period_year),
                        r.accession,
                        r.report_date,
                        heading,
                        r.text,
                        r.embed_text,
                    ]
                )
        with cur.copy("COPY agents.peers FROM STDIN") as copy:
            for row in rows:
                copy.write_row(row)
    con.execute(INDEX_SQL)
    con.execute("ANALYZE agents.chunks")
    con.execute("ANALYZE agents.peers")
    load_s = time.perf_counter() - started
    return {
        "chunks_loaded": len(chunks),
        "peer_rows_loaded": len(rows),
        "alpha": alpha,
        "peer_info": data["info"],
        "load_s": round(load_s, 1),
    }


def run_check(args: argparse.Namespace) -> dict:
    import pandas as pd

    con = connect()
    counts = {name: int(con.execute(sql).fetchone()[0]) for name, sql in CHECK_SQL.items()}
    parquet_rows = len(pd.read_parquet(args.chunks, columns=["chunk_id"]))
    ok = (
        counts["chunks"] == counts["chunk_vectors"] == parquet_rows
        and counts["only_in_chunks"] == counts["only_in_vectors"] == counts["cik_mismatch"] == 0
        and counts["self_peers"] == counts["short_peer_lists"] == 0
        and counts["peer_rows"] == counts["peer_companies"] * TOP_PEERS
    )
    return {"parquet_rows": parquet_rows, **counts, "ok": ok}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["build", "check"])
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--emb-dir", type=Path, default=EMB_DIR)
    parser.add_argument("--peers-select", type=Path, default=PEERS_SELECT)
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    result = run_build(args) if args.step == "build" else run_check(args)
    record = {
        "command": " ".join(["uv run python -m filinglens.agents.tool_tables", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **result,
    }
    previous = json.loads(STATS.read_text()) if STATS.exists() else {}
    previous[args.step] = record
    STATS.parent.mkdir(parents=True, exist_ok=True)
    STATS.write_text(json.dumps(previous, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if record.get("ok") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
