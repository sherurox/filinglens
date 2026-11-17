"""Evidence for the FL-M1 Dagster backfill -> bench/dagster_backfill.json.

Reads the Dagster instance in DAGSTER_HOME: the latest backfill, its runs (status, partition
range, seconds), and checks that the zips it re-downloaded have the same SHA-256 as the T5
manifest (data/raw/fsds/manifest.csv), i.e. the backfill reproduced the original bytes.

Run from the repo root after the backfill finished:
    export DAGSTER_HOME="$PWD/data/dagster"
    uv run python dagster_project/backfill_report.py
"""

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import dagster as dg

from filinglens.ingest.fsds import quarters

RANGE_START_TAG = "dagster/asset_partition_range_start"
RANGE_END_TAG = "dagster/asset_partition_range_end"
BACKFILL_TAG = "dagster/backfill"


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def compare_with_manifest(wanted: list[str], manifest_csv: Path, zip_dir: Path) -> list[dict]:
    """For each quarter: the SHA-256 recorded in T5 and the SHA-256 of the file on disk now."""
    with manifest_csv.open() as f:
        recorded = {row["quarter"]: row["sha256"] for row in csv.DictReader(f)}
    out = []
    for quarter in wanted:
        path = zip_dir / f"{quarter}.zip"
        now = sha256_file(path) if path.exists() else ""
        out.append(
            {
                "quarter": quarter,
                "sha256_t5": recorded.get(quarter, ""),
                "sha256_now": now,
                "same": bool(now) and now == recorded.get(quarter),
            }
        )
    return out


def _utc(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="seconds") if ts else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/fsds/manifest.csv"))
    parser.add_argument("--zips", type=Path, default=Path("data/raw/fsds"))
    parser.add_argument("--out", type=Path, default=Path("bench/dagster_backfill.json"))
    args = parser.parse_args(argv)

    instance = dg.DagsterInstance.get()
    backfills = instance.get_backfills(limit=1)
    if not backfills:
        raise SystemExit("no backfill found in DAGSTER_HOME")
    backfill = backfills[0]
    records = instance.get_run_records(
        filters=dg.RunsFilter(tags={BACKFILL_TAG: backfill.backfill_id})
    )
    runs, ranges = [], []
    for rec in sorted(records, key=lambda r: r.start_time or 0):
        run = rec.dagster_run
        start, end = run.tags.get(RANGE_START_TAG), run.tags.get(RANGE_END_TAG)
        if start and end:
            ranges.append((start, end))
        runs.append(
            {
                "run_id": run.run_id,
                "status": run.status.value,
                "partition_range": f"{start}...{end}" if start else None,
                "assets": sorted("/".join(k.path) for k in (run.asset_selection or [])),
                "started_utc": _utc(rec.start_time),
                "seconds": round(rec.end_time - rec.start_time, 1)
                if rec.start_time and rec.end_time
                else None,
            }
        )
    wanted = sorted({q for start, end in ranges for q in quarters(start, end)})
    shas = compare_with_manifest(wanted, args.manifest, args.zips)
    report = {
        "command": "uv run python dagster_project/backfill_report.py",
        "backfill_id": backfill.backfill_id,
        "status": backfill.status.value,
        "created_utc": _utc(backfill.backfill_timestamp),
        "runs": runs,
        "quarters_backfilled": wanted,
        "zip_sha256_vs_t5_manifest": shas,
        "all_zips_identical_to_t5": bool(shas) and all(s["same"] for s in shas),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
