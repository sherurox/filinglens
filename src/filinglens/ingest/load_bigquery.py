"""Export the universe's FSDS rows from the Iceberg tables to Parquet, then load them into BigQuery.

The BigQuery sandbox allows load jobs (they are not DML) but caps storage at 10 GB, so only the
300 universe companies' filings go to BigQuery; DuckDB keeps the full copy locally.

Run from the repo root:
    set -a; source .env; set +a
    uv run python -m filinglens.ingest.load_bigquery --skip-load   # export + sizes only
    uv run python -m filinglens.ingest.load_bigquery               # export + load
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

TABLES = ("sub", "num", "pre", "tag")
SANDBOX_STORAGE_BYTES = 10 * 1000**3
DUCKDB_SETTINGS = (
    "SET memory_limit = '6GB'",
    "SET threads = 8",
    "SET preserve_insertion_order = false",
    "SET temp_directory = 'data/duckdb-tmp'",
    "SET max_temp_directory_size = '40GB'",
)


def iceberg_sources(warehouse: Path) -> dict[str, str]:
    """DuckDB relations for the four Iceberg tables (read through metadata: current snapshot)."""
    return {t: f"iceberg_scan('{(warehouse / 'fsds' / t).as_posix()}')" for t in TABLES}


def _copy(con: Any, sql: str, path: Path) -> dict[str, Any]:
    """Write `sql` to Parquet via a .part file, then rename; return rows, bytes and seconds."""
    started = time.perf_counter()
    tmp = path.with_name(path.name + ".part")
    con.execute(f"COPY ({sql}) TO '{tmp.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
    os.replace(tmp, path)
    rows = con.execute(f"SELECT count(*) FROM read_parquet('{path.as_posix()}')").fetchall()[0][0]
    return {
        "rows": rows,
        "parquet_bytes": path.stat().st_size,
        "seconds": round(time.perf_counter() - started, 1),
    }


def export_universe(
    con: Any, sources: dict[str, str], universe_csv: Path, out_dir: Path
) -> dict[str, dict[str, Any]]:
    """Rows of every universe filing (all forms) and the tag versions those numbers use."""
    out_dir.mkdir(parents=True, exist_ok=True)
    con.execute(
        "CREATE OR REPLACE TEMP TABLE universe_cik AS "
        f"SELECT cik FROM read_csv('{universe_csv.as_posix()}')"
    )
    con.execute(
        "CREATE OR REPLACE TEMP TABLE universe_adsh AS "
        f"SELECT adsh FROM {sources['sub']} WHERE cik IN (SELECT cik FROM universe_cik)"
    )
    stats = {}
    for name in ("sub", "num", "pre"):
        sql = f"SELECT * FROM {sources[name]} WHERE adsh IN (SELECT adsh FROM universe_adsh)"
        stats[name] = _copy(con, sql, out_dir / f"{name}.parquet")
    used = (
        f"SELECT DISTINCT tag, version FROM read_parquet('{(out_dir / 'num.parquet').as_posix()}')"
    )
    tag_sql = (
        f"SELECT t.* FROM {sources['tag']} AS t "
        f"JOIN ({used}) AS u ON t.tag = u.tag AND t.version = u.version"
    )
    stats["tag"] = _copy(con, tag_sql, out_dir / "tag.parquet")
    return stats


def bq_load_command(project: str, dataset: str, table: str, path: Path) -> list[str]:
    """`bq load` for one Parquet file. A load job replaces the table; it is not DML."""
    return [
        "bq",
        f"--project_id={project}",
        "--location=US",
        "load",
        "--source_format=PARQUET",
        "--replace",
        "--decimal_target_types=NUMERIC",  # repeat the flag, one value each
        "--decimal_target_types=BIGNUMERIC",
        f"{dataset}.{table}",
        str(path),
    ]


def _bq(args: list[str]) -> str:
    env = {**os.environ, "PYTHONWARNINGS": "ignore"}  # bq's bundled libraries warn noisily
    done = subprocess.run(args, capture_output=True, text=True, env=env)
    if done.returncode != 0:  # bq prints its error to stdout, so show both streams
        raise RuntimeError(f"{' '.join(args)}\n{done.stdout}{done.stderr}")
    return done.stdout


def bigquery_table_sizes(project: str, dataset: str) -> dict[str, dict[str, int]]:
    """Rows and stored bytes per table, from BigQuery's own metadata (no table scan)."""
    sql = f"SELECT table_id, row_count, size_bytes FROM `{project}.{dataset}.__TABLES__`"
    out = _bq(
        [
            "bq",
            f"--project_id={project}",
            "--location=US",
            "query",
            "--nouse_legacy_sql",
            "--format=json",
            sql,
        ]
    )
    return {
        r["table_id"]: {"rows": int(r["row_count"]), "bytes": int(r["size_bytes"])}
        for r in json.loads(out)
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.environ.get("GCP_PROJECT_ID"))
    parser.add_argument("--dataset", default="filinglens_raw")
    parser.add_argument("--universe", type=Path, default=Path("dbt/seeds/universe.csv"))
    parser.add_argument("--warehouse", type=Path, default=Path("data/warehouse"))
    parser.add_argument("--out", type=Path, default=Path("data/export/bigquery"))
    parser.add_argument("--stats", type=Path, default=Path("bench/bigquery_load.json"))
    parser.add_argument("--skip-load", action="store_true", help="export and report sizes only")
    args = parser.parse_args(argv)
    if not args.skip_load and not args.project:
        parser.error("set GCP_PROJECT_ID (source .env) or pass --project")

    con = duckdb.connect()
    con.execute("LOAD iceberg")
    for setting in DUCKDB_SETTINGS:
        con.execute(setting)
    exported = export_universe(con, iceberg_sources(args.warehouse), args.universe, args.out)
    print(json.dumps({"exported": exported}, indent=2))
    if args.skip_load:
        return

    loaded = {}
    for name in TABLES:
        started = time.perf_counter()
        _bq(bq_load_command(args.project, args.dataset, name, args.out / f"{name}.parquet"))
        loaded[name] = {"seconds": round(time.perf_counter() - started, 1)}
    sizes = bigquery_table_sizes(args.project, args.dataset)
    total = sum(s["bytes"] for s in sizes.values())
    stats = {
        "command": " ".join(["uv run python -m filinglens.ingest.load_bigquery", *sys.argv[1:]]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "exported": exported,
        "load_seconds": loaded,
        "bigquery_tables": sizes,
        "bigquery_total_bytes": total,
        "sandbox_storage_limit_bytes": SANDBOX_STORAGE_BYTES,
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "exported"}, indent=2))
    if total > 0.8 * SANDBOX_STORAGE_BYTES:
        print("WARNING: over 80% of the sandbox's 10 GB storage", file=sys.stderr)


if __name__ == "__main__":
    main()
