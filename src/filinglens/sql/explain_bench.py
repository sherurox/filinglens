"""EXPLAIN ANALYZE before/after one index for every query in sql/ (FL-M1 D4).

Layout: sql/NN_name.sql is one SELECT; sql/NN_name.index.sql holds the change it is measured
against (usually one CREATE INDEX). For each query:
  1. drop every index in the marts schema (clean baseline, no index left by another query);
  2. warm the cache once, then time `runs` executions;
  3. apply the index file and ANALYZE; warm up and time again;
  4. check the result is identical before and after (sha256 of the sorted rows).
Saved plans use EXPLAIN (ANALYZE, BUFFERS): per-node times and buffer counts. Timed runs use
EXPLAIN (ANALYZE, TIMING OFF): server-side Execution Time without per-row clock calls.

Run from the repo root (Postgres from compose.yaml up, marts loaded):
    set -a; source .env; set +a
    uv run python -m filinglens.sql.explain_bench            # all queries
    uv run python -m filinglens.sql.explain_bench 03_rolling  # names containing "03_rolling"
"""

import argparse
import csv
import hashlib
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from filinglens.sql.load_postgres import SCHEMA, libpq_dsn

TIME_RE = re.compile(r"^(Planning|Execution) Time: ([0-9.]+) ms$")
CSV_FIELDS = [
    "query",
    "result_rows",
    "same_result",
    "before_execution_ms",
    "after_execution_ms",
    "speedup",
    "before_planning_ms",
    "after_planning_ms",
    "runs",
    "index_change",
]


class Database(Protocol):
    def rows(self, sql: str) -> list[tuple]: ...

    def run(self, sql: str) -> None: ...


@dataclass(frozen=True)
class QuerySpec:
    name: str
    query: str
    index_change: str


def strip_sql(text: str) -> str:
    """Drop `--` comment lines and the trailing semicolon, so the query can be wrapped."""
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("--")]
    return "\n".join(lines).strip().rstrip(";").strip()


def discover(sql_dir: Path, pattern: str = "") -> list[QuerySpec]:
    """Every NN_name.sql (not *.index.sql) with its NN_name.index.sql partner, in name order."""
    specs = []
    for path in sorted(sql_dir.glob("[0-9][0-9]_*.sql")):
        if path.name.endswith(".index.sql") or pattern not in path.stem:
            continue
        index_path = path.with_name(path.stem + ".index.sql")
        if not index_path.exists():
            raise FileNotFoundError(f"{index_path} is missing: every query needs its index change")
        specs.append(QuerySpec(path.stem, strip_sql(path.read_text()), index_path.read_text()))
    return specs


def parse_times(plan_lines: list[str]) -> tuple[float, float]:
    """(planning_ms, execution_ms) from the summary lines of an EXPLAIN ANALYZE text plan."""
    found = {}
    for line in plan_lines:
        match = TIME_RE.match(line.strip())
        if match:
            found[match.group(1)] = float(match.group(2))
    if set(found) != {"Planning", "Execution"}:
        raise ValueError("plan has no Planning/Execution Time lines")
    return found["Planning"], found["Execution"]


def result_digest(rows: list[tuple]) -> str:
    """Order-independent fingerprint of a result: sha256 over the sorted row reprs."""
    digest = hashlib.sha256()
    for row in sorted(repr(r) for r in rows):
        digest.update(row.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def drop_schema_indexes(db: Database, schema: str = SCHEMA) -> list[str]:
    names = [
        r[0] for r in db.rows(f"SELECT indexname FROM pg_indexes WHERE schemaname = '{schema}'")
    ]
    for name in names:
        db.run(f'DROP INDEX "{schema}"."{name}"')
    return names


def _plan(db: Database, query: str, options: str) -> list[str]:
    return [r[0] for r in db.rows(f"EXPLAIN ({options}) {query}")]


def measure_state(db: Database, query: str, runs: int) -> dict:
    """Warm up once, keep one full plan, then time `runs` executions (medians in ms)."""
    _plan(db, query, "ANALYZE, TIMING OFF")  # warm-up: loads the pages into shared_buffers
    plan = _plan(db, query, "ANALYZE, BUFFERS")
    timings = [parse_times(_plan(db, query, "ANALYZE, TIMING OFF")) for _ in range(runs)]
    return {
        "plan": plan,
        "planning_ms": round(statistics.median(t[0] for t in timings), 3),
        "execution_ms": round(statistics.median(t[1] for t in timings), 3),
    }


def bench_query(db: Database, spec: QuerySpec, runs: int, tables: list[str]) -> dict:
    drop_schema_indexes(db)
    for table in tables:
        db.run(f"ANALYZE {SCHEMA}.{table}")
    before_rows = db.rows(spec.query)
    before = measure_state(db, spec.query, runs)

    db.run(spec.index_change)
    for table in tables:
        db.run(f"ANALYZE {SCHEMA}.{table}")
    after_rows = db.rows(spec.query)
    after = measure_state(db, spec.query, runs)
    drop_schema_indexes(db)

    return {
        "query": spec.name,
        "result_rows": len(before_rows),
        "same_result": result_digest(before_rows) == result_digest(after_rows),
        "before_execution_ms": before["execution_ms"],
        "after_execution_ms": after["execution_ms"],
        "speedup": round(before["execution_ms"] / after["execution_ms"], 2)
        if after["execution_ms"]
        else None,
        "before_planning_ms": before["planning_ms"],
        "after_planning_ms": after["planning_ms"],
        "runs": runs,
        "index_change": " ".join(strip_sql(spec.index_change).split()),
        "_before_plan": before["plan"],
        "_after_plan": after["plan"],
    }


def write_outputs(results: list[dict], explain_dir: Path, csv_path: Path) -> None:
    """Plans to sql/explain/, one CSV row per query (rows for other queries are kept)."""
    explain_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        header = f"-- {r['query']}: median of {r['runs']} runs"
        for state in ("before", "after"):
            body = "\n".join(r[f"_{state}_plan"])
            ms = r[f"{state}_execution_ms"]
            text = f"{header}, execution {ms} ms ({state} the index change)\n{body}\n"
            (explain_dir / f"{r['query']}.{state}.txt").write_text(text)
    rows = {}
    if csv_path.exists():
        with csv_path.open() as f:
            rows = {row["query"]: row for row in csv.DictReader(f)}
    rows.update({r["query"]: {k: r[k] for k in CSV_FIELDS} for r in results})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows[name] for name in sorted(rows))


class PsycopgDatabase:
    """Autocommit connection: DDL applies immediately, as each step needs."""

    def __init__(self, dsn: str):
        import psycopg  # imported here so the unit tests need no driver

        self._conn = psycopg.connect(dsn, autocommit=True)

    def rows(self, sql: str) -> list[tuple]:
        with self._conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()

    def run(self, sql: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(sql)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pattern", nargs="?", default="", help="only queries whose name contains this"
    )
    parser.add_argument("--sql-dir", type=Path, default=Path("sql"))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--port", type=int, default=55432)
    parser.add_argument("--csv", type=Path, default=Path("bench/sql_explain.csv"))
    args = parser.parse_args(argv)
    password = os.environ.get("POSTGRES_PASSWORD")
    if not password:
        parser.error("set POSTGRES_PASSWORD (source .env)")

    specs = discover(args.sql_dir, args.pattern)
    if not specs:
        parser.error(f"no queries in {args.sql_dir} matching {args.pattern!r}")
    db = PsycopgDatabase(libpq_dsn(password, port=args.port))
    tables = [
        r[0] for r in db.rows(f"SELECT tablename FROM pg_tables WHERE schemaname = '{SCHEMA}'")
    ]
    for table in tables:  # visibility map set, so index-only scans need no heap visits
        db.run(f"VACUUM (ANALYZE) {SCHEMA}.{table}")
    results = []
    for spec in specs:
        r = bench_query(db, spec, args.runs, tables)
        results.append(r)
        print(
            f"{r['query']}: {r['result_rows']} rows, same_result={r['same_result']}, "
            f"{r['before_execution_ms']} ms -> {r['after_execution_ms']} ms (x{r['speedup']})"
        )
    write_outputs(results, args.sql_dir / "explain", args.csv)
    if not all(r["same_result"] for r in results):
        raise SystemExit("an index change altered a query's result")


if __name__ == "__main__":
    main()
