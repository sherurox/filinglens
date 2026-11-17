"""Copy the dbt marts from DuckDB into Postgres, with no indexes (the "before" state for sql/).

DuckDB's postgres extension attaches the database, so each table is one CREATE TABLE AS SELECT:
no CSV files and no extra Python driver. No primary keys or indexes are created on purpose:
every query in sql/ adds the one index it needs, so each before/after pair isolates one change.

Run from the repo root (Postgres from compose.yaml must be up):
    set -a; source .env; set +a
    uv run python -m filinglens.sql.load_postgres
"""

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

MART_TABLES = ("dim_company", "dim_period", "dim_tag", "fct_financials")
SCHEMA = "marts"


def libpq_dsn(password: str, host: str = "127.0.0.1", port: int = 55432) -> str:
    """Connection string for the compose.yaml database (localhost only)."""
    return f"host={host} port={port} dbname=filinglens user=filinglens password={password}"


def copy_statements(tables: tuple[str, ...] = MART_TABLES, schema: str = SCHEMA) -> list[str]:
    """DuckDB statements that replace each Postgres table with the DuckDB mart of the same name.

    `src` is the dbt DuckDB file (attached read-only), `pg` the Postgres database.
    """
    statements = [f"CREATE SCHEMA IF NOT EXISTS pg.{schema}"]
    for table in tables:
        statements.append(f"DROP TABLE IF EXISTS pg.{schema}.{table}")
        statements.append(f"CREATE TABLE pg.{schema}.{table} AS SELECT * FROM src.main.{table}")
    return statements


def sql_literal(text: str) -> str:
    """A SQL string literal: single quotes, with any ' inside doubled."""
    return "'" + text.replace("'", "''") + "'"


def postgres_query_sql(sql: str, database: str = "pg") -> str:
    """DuckDB query that runs `sql` inside the attached Postgres and returns its rows."""
    return f"SELECT * FROM postgres_query({sql_literal(database)}, {sql_literal(sql)})"


def _one(con: Any, sql: str) -> Any:
    return con.execute(sql).fetchall()[0][0]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duckdb", type=Path, default=Path("data/filinglens.duckdb"))
    parser.add_argument("--port", type=int, default=55432)
    parser.add_argument("--stats", type=Path, default=Path("bench/postgres_load.json"))
    args = parser.parse_args(argv)
    password = os.environ.get("POSTGRES_PASSWORD")
    if not password:
        parser.error("set POSTGRES_PASSWORD (source .env)")

    # In-memory DuckDB: opening the dbt file itself read-only would make every attached
    # database read-only too, Postgres included. Attach the file read-only instead.
    con = duckdb.connect()
    con.execute(f"ATTACH {sql_literal(str(args.duckdb))} AS src (READ_ONLY)")
    con.execute("INSTALL postgres")
    con.execute("LOAD postgres")
    con.execute(f"ATTACH {sql_literal(libpq_dsn(password, port=args.port))} AS pg (TYPE postgres)")

    started = time.perf_counter()
    for statement in copy_statements():
        con.execute(statement)
    copy_seconds = round(time.perf_counter() - started, 1)
    # Fresh planner statistics, as after any bulk load; EXPLAIN plans depend on them.
    con.execute("CALL postgres_execute('pg', 'ANALYZE')")

    tables = {}
    for table in MART_TABLES:
        tables[table] = {
            "duckdb_rows": _one(con, f"SELECT count(*) FROM src.main.{table}"),
            "postgres_rows": _one(con, f"SELECT count(*) FROM pg.{SCHEMA}.{table}"),
        }
    version = _one(con, postgres_query_sql("SELECT version()"))
    total_bytes = _one(
        con,
        postgres_query_sql(
            "SELECT sum(pg_total_relation_size(c.oid))::bigint FROM pg_class c "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            f"WHERE n.nspname = '{SCHEMA}' AND c.relkind = 'r'"
        ),
    )
    stats = {
        "command": "uv run python -m filinglens.sql.load_postgres",
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "postgres_version": version,
        "schema": SCHEMA,
        "copy_seconds": copy_seconds,
        "tables": tables,
        "postgres_total_bytes": total_bytes,
        "rows_match": all(t["duckdb_rows"] == t["postgres_rows"] for t in tables.values()),
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    if not stats["rows_match"]:
        raise SystemExit("row counts differ between DuckDB and Postgres")


if __name__ == "__main__":
    main()
