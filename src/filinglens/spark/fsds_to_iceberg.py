"""FSDS quarterly text files -> typed, deduplicated Iceberg tables, partitioned by fiscal year.

Run: uv run python -m filinglens.spark.fsds_to_iceberg
"""

import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from datetime import UTC, datetime
from functools import reduce
from pathlib import Path
from typing import Any

ICEBERG_RUNTIME = "org.apache.iceberg:iceberg-spark-runtime-4.1_2.13:1.12.0"
CATALOG = "local"
NAMESPACE = f"{CATALOG}.fsds"
MEMBERS = ("sub", "num", "tag", "pre")

# Columns converted from text. Everything else stays a string (codes with leading zeros, text).
TYPES: dict[str, dict[str, str]] = {
    "sub": {
        "cik": "bigint",
        "sic": "int",
        "fy": "int",
        "period": "yyyymmdd",
        "filed": "yyyymmdd",
        "accepted": "timestamp",
        "prevrpt": "boolean",
        "detail": "boolean",
        "wksi": "boolean",
        "nciks": "int",
    },
    "num": {"ddate": "yyyymmdd", "qtrs": "int", "value": "decimal(28,4)"},
    "tag": {"custom": "boolean", "abstract": "boolean"},
    "pre": {"report": "int", "line": "int", "inpth": "boolean", "negating": "boolean"},
}
# Natural keys. One row per key survives: the one from the newest quarter.
# Keys a quarter doesn't have (e.g. num.segments in older sets) are skipped.
KEYS: dict[str, list[str]] = {
    "sub": ["adsh"],
    "num": ["adsh", "tag", "version", "ddate", "qtrs", "uom", "segments", "coreg"],
    "tag": ["tag", "version"],
    "pre": ["adsh", "report", "line"],
}
PARTITIONED = {"sub", "num", "pre"}  # by fiscal year; tag is a small dimension


def extract_quarter(zip_path: Path, out_dir: Path) -> Path:
    """Unzip sub/num/tag/pre of one <quarter>.zip into <out_dir>/<quarter>/; return that folder.

    Spark can't read inside zip files. Files already extracted in full are skipped, and
    each file is written to .part first, so an interrupted run never leaves half a file.
    """
    with zipfile.ZipFile(zip_path) as zf:
        for member in MEMBERS:
            info = zf.getinfo(f"{member}.txt")
            target = out_dir / zip_path.stem / info.filename
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            with zf.open(info) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            os.replace(tmp, target)
    return out_dir / zip_path.stem


def extract_quarters(zip_dir: Path, out_dir: Path) -> list[str]:
    """extract_quarter for every <quarter>.zip in zip_dir; returns the quarters, in order."""
    quarters = []
    for zip_path in sorted(zip_dir.glob("*.zip")):
        extract_quarter(zip_path, out_dir)
        quarters.append(zip_path.stem)
    return quarters


def build_spark(
    warehouse: Path,
    driver_memory: str = "10g",
    master: str = "local[*]",
    local_dir: Path | None = None,
) -> Any:
    """Local Spark session with an Iceberg Hadoop catalog.

    `local_dir` is where Spark spills shuffle data. It must be a real disk: when /tmp is
    a RAM-backed tmpfs, a large shuffle fills it ("No space left on device").
    """
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName("fsds_to_iceberg")
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.jars.packages", ICEBERG_RUNTIME)
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config(f"spark.sql.catalog.{CATALOG}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{CATALOG}.type", "hadoop")
        .config(f"spark.sql.catalog.{CATALOG}.warehouse", str(warehouse.resolve()))
        .config("spark.ui.showConsoleProgress", "false")
    )
    if local_dir is not None:
        local_dir.mkdir(parents=True, exist_ok=True)
        builder = builder.config("spark.local.dir", str(local_dir.resolve()))
    spark = builder.getOrCreate()
    # Iceberg logs a harmless warning with a stack trace when a table has no
    # version-hint.text yet (every first create). Show errors only.
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def read_member(spark: Any, extracted: Path, member: str) -> Any:
    """One FSDS file (e.g. num.txt) across all quarters, all columns as strings.

    Quarters are read one by one and combined by column name, so a column added by SEC
    in later quarters becomes null in earlier ones instead of shifting columns.
    """
    from pyspark.sql import functions as F

    frames = [
        spark.read.option("sep", "\t")
        .option("header", True)
        .option("quote", "")  # FSDS text is not quoted; an empty string turns quoting off
        .csv(str(quarter_dir / f"{member}.txt"))
        .withColumn("source_quarter", F.lit(quarter_dir.name))
        for quarter_dir in sorted(extracted.iterdir())
        if (quarter_dir / f"{member}.txt").exists()
    ]
    return reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), frames)


def typed(df: Any, types: dict[str, str]) -> Any:
    """Cast text columns. try_cast gives null for a bad value instead of failing the job
    (Spark 4 runs in ANSI mode, where a plain cast of bad text raises)."""
    from pyspark.sql import functions as F

    for column, sql_type in types.items():
        if column not in df.columns:
            continue
        if sql_type == "yyyymmdd":
            expr = f"to_date(try_to_timestamp(`{column}`, 'yyyyMMdd'))"
        else:
            expr = f"try_cast(`{column}` AS {sql_type})"
        df = df.withColumn(column, F.expr(expr))
    return df


def latest_per_key(df: Any, keys: list[str]) -> Any:
    """Keep one row per key, from the newest source quarter."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    present = [k for k in keys if k in df.columns]
    window = Window.partitionBy(*present).orderBy(F.col("source_quarter").desc())
    return df.withColumn("_rn", F.row_number().over(window)).where("_rn = 1").drop("_rn")


def partition_summary(spark: Any, table: str, partitioned: bool) -> list[dict[str, Any]]:
    """Rows and data files per fiscal-year partition, from Iceberg's own metadata tables."""
    if partitioned:
        query = (
            f"SELECT partition.fy AS fy, record_count, file_count "
            f"FROM {table}.partitions ORDER BY fy"
        )
    else:
        query = (
            f"SELECT count(*) AS file_count, sum(record_count) AS record_count FROM {table}.files"
        )
    return [row.asDict() for row in spark.sql(query).collect()]


def build_tables(spark: Any, extracted: Path) -> dict[str, dict[str, Any]]:
    """Write sub, tag, num and pre. num and pre take their fiscal year from sub."""
    from pyspark.sql import functions as F

    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {NAMESPACE}")
    stats: dict[str, dict[str, Any]] = {}
    sub_fy = None
    for name in ("sub", "tag", "num", "pre"):  # sub first: num and pre join to it
        started = time.perf_counter()
        table = f"{NAMESPACE}.{name}"
        raw = read_member(spark, extracted, name)
        raw_rows = raw.count()
        df = latest_per_key(typed(raw, TYPES[name]), KEYS[name])
        if name in ("num", "pre"):
            df = df.join(F.broadcast(sub_fy), "adsh", "left")  # sub is small: no big shuffle
        writer = df.writeTo(table).using("iceberg")
        if name in PARTITIONED:
            writer = writer.partitionedBy(F.col("fy"))
        writer.createOrReplace()
        written = spark.table(table)
        rows = written.count()
        stats[name] = {
            "raw_rows": raw_rows,
            "rows": rows,
            "duplicates_removed": raw_rows - rows,
            "rows_without_fy": written.where("fy IS NULL").count() if name in PARTITIONED else 0,
            "seconds": round(time.perf_counter() - started, 1),
            "partitions": partition_summary(spark, table, name in PARTITIONED),
        }
        if name == "sub":
            sub_fy = written.select("adsh", "fy")
    return stats


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zips", type=Path, default=Path("data/raw/fsds"))
    parser.add_argument("--extracted", type=Path, default=Path("data/interim/fsds"))
    parser.add_argument("--warehouse", type=Path, default=Path("data/warehouse"))
    parser.add_argument("--stats", type=Path, default=Path("bench/fsds_to_iceberg.json"))
    parser.add_argument("--driver-memory", default="10g")
    parser.add_argument("--local-dir", type=Path, default=Path("data/spark-tmp"))
    args = parser.parse_args(argv)

    started = time.perf_counter()
    quarters = extract_quarters(args.zips, args.extracted)
    extract_seconds = time.perf_counter() - started
    spark = build_spark(args.warehouse, args.driver_memory, local_dir=args.local_dir)
    try:
        spark_version = spark.version
        tables = build_tables(spark, args.extracted)
    finally:
        spark.stop()
    stats = {
        "command": " ".join(["uv run python -m filinglens.spark.fsds_to_iceberg", *sys.argv[1:]]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "spark_version": spark_version,
        "iceberg_runtime": ICEBERG_RUNTIME,
        "catalog": "Iceberg Hadoop catalog (local filesystem)",
        "master": "local[*]",
        "cpu_threads": os.cpu_count(),
        "driver_memory": args.driver_memory,
        "spark_local_dir": str(args.local_dir),
        "quarters": len(quarters),
        "first_quarter": quarters[0] if quarters else "",
        "last_quarter": quarters[-1] if quarters else "",
        "extract_seconds": round(extract_seconds, 1),
        "extracted_bytes": dir_bytes(args.extracted),
        "warehouse_bytes": dir_bytes(args.warehouse),
        "total_seconds": round(time.perf_counter() - started, 1),
        "tables": tables,
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2, default=str) + "\n")
    summary = {k: v for k, v in stats.items() if k != "tables"}
    summary["tables"] = {
        name: {k: v for k, v in t.items() if k != "partitions"} for name, t in tables.items()
    }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
