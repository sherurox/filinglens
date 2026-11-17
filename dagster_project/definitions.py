"""Dagster assets for FL-M1: FSDS download -> Spark/Iceberg -> dbt (DuckDB).

    fsds_quarter_zip   35 quarterly partitions (2018q1-2026q3): download + verify one quarter's
                       zip, then extract it. Backfills run as ONE run over the whole range, so
                       all quarters share one EDGAR client and its 8 req/s token bucket.
    fsds/{sub,num,tag,pre}
                       the Spark -> Iceberg job, unpartitioned: it dedupes across quarters
                       ("newest quarter wins"), so it rebuilds from all quarters each time.
    dbt models         every model and test from the dbt project; source('fsds', 'num') maps to
                       the asset fsds/num, so the lineage needs no manual wiring.

Run from the repo root:
    export DAGSTER_HOME="$PWD/data/dagster" && mkdir -p "$DAGSTER_HOME"
    uv run dagster dev -f dagster_project/definitions.py
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import dagster as dg
from dagster_dbt import DbtCliResource, DbtProject, dbt_assets

from filinglens.ingest.edgar_client import EdgarClient, load_user_agent
from filinglens.ingest.fsds import quarters, run
from filinglens.spark.fsds_to_iceberg import MEMBERS, extract_quarter

REPO = Path(__file__).resolve().parents[1]
RAW_FSDS = REPO / "data" / "raw" / "fsds"
EXTRACTED = REPO / "data" / "interim" / "fsds"
SPARK_STATS = REPO / "data" / "dagster" / "fsds_to_iceberg.json"

# Fixed range: the quarters downloaded in T5. New quarters would be added here (or with a
# DynamicPartitionsDefinition and a sensor, see docs/LATER.md).
FSDS_QUARTERS = dg.StaticPartitionsDefinition(quarters("2018q1", "2026q3"))


async def _download(wanted: list[str]) -> list:
    async with EdgarClient(load_user_agent()) as client:
        return await run(client, wanted, RAW_FSDS, workers=4)


@dg.asset(
    partitions_def=FSDS_QUARTERS,
    backfill_policy=dg.BackfillPolicy.single_run(),
    group_name="ingest",
    kinds={"python"},
)
def fsds_quarter_zip(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """SEC Financial Statement Data Set zips, one partition per quarter (download + extract)."""
    wanted = list(context.partition_keys)
    results = asyncio.run(_download(wanted))
    failed = [r for r in results if r.status not in ("downloaded", "skipped_existing")]
    if failed:
        raise dg.Failure(
            f"FSDS download failed: {[(r.quarter, r.status, r.error) for r in failed]}"
        )
    for r in results:
        extract_quarter(Path(r.path), EXTRACTED)
        context.log.info(f"{r.quarter}: {r.status}, {r.bytes} bytes, sha256 {r.sha256}")
    return dg.MaterializeResult(
        metadata={
            "quarters": len(results),
            "downloaded": sum(r.status == "downloaded" for r in results),
            "skipped_existing": sum(r.status == "skipped_existing" for r in results),
            "bytes": sum(r.bytes for r in results),
            "sha256": dg.MetadataValue.json({r.quarter: r.sha256 for r in results}),
        }
    )


@dg.multi_asset(
    specs=[
        dg.AssetSpec(
            ["fsds", table],
            deps=[fsds_quarter_zip],
            group_name="lakehouse",
            kinds={"spark", "iceberg"},
            description=f"Iceberg table local.fsds.{table} (typed, deduplicated)",
        )
        for table in MEMBERS
    ]
)
def fsds_iceberg(context: dg.AssetExecutionContext):
    """The Spark job as a subprocess: its own JVM and memory, same CLI as a manual run.
    Stats go to data/dagster/ so the committed bench/fsds_to_iceberg.json stays untouched."""
    command = [
        sys.executable,
        "-m",
        "filinglens.spark.fsds_to_iceberg",
        "--stats",
        str(SPARK_STATS),
    ]
    context.log.info(" ".join(command))
    subprocess.run(command, cwd=REPO, check=True)
    stats = json.loads(SPARK_STATS.read_text())
    for table in MEMBERS:
        t = stats["tables"][table]
        yield dg.MaterializeResult(
            asset_key=["fsds", table],
            metadata={
                "rows": t["rows"],
                "duplicates_removed": t["duplicates_removed"],
                "seconds": t["seconds"],
                "spark_total_seconds": stats["total_seconds"],
                "quarters": stats["quarters"],
            },
        )


dbt_project = DbtProject(project_dir=REPO / "dbt", profiles_dir=REPO / "dbt", target="duckdb")
dbt_project.prepare_if_dev()  # `dagster dev` runs `dbt parse` so the manifest is current


@dbt_assets(manifest=dbt_project.manifest_path, project=dbt_project)
def filinglens_dbt(context: dg.AssetExecutionContext, dbt: DbtCliResource):
    """`dbt build` (models + tests) on DuckDB, with the resource caps from profiles.yml."""
    yield from dbt.cli(["build"], context=context).stream()


defs = dg.Definitions(
    assets=[fsds_quarter_zip, fsds_iceberg, filinglens_dbt],
    resources={"dbt": DbtCliResource(project_dir=dbt_project, target="duckdb")},
)
