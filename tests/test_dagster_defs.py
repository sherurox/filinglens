"""The Dagster definitions load, and the graph is wired as designed (no runs are executed)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import dagster as dg
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def defs_module():
    # dagster-dbt reads dbt's manifest; `dbt parse` writes it without touching any warehouse.
    dbt = Path(sys.executable).parent / "dbt"
    subprocess.run([str(dbt), "parse", "--quiet"], cwd=REPO / "dbt", check=True)
    return _load("fl_dagster_definitions", REPO / "dagster_project" / "definitions.py")


def test_definitions_load(defs_module):
    dg.Definitions.validate_loadable(defs_module.defs)


def test_quarter_partitions_and_single_run_backfills(defs_module):
    asset = defs_module.fsds_quarter_zip
    keys = asset.partitions_def.get_partition_keys()
    assert (len(keys), keys[0], keys[-1]) == (35, "2018q1", "2026q3")
    assert asset.backfill_policy == dg.BackfillPolicy.single_run()


def test_spark_asset_produces_the_four_tables_from_all_quarters(defs_module):
    spark = defs_module.fsds_iceberg
    assert set(spark.keys) == {dg.AssetKey(["fsds", t]) for t in ("sub", "num", "tag", "pre")}
    assert spark.partitions_def is None
    for key in spark.keys:
        assert dg.AssetKey(["fsds_quarter_zip"]) in spark.asset_deps[key]


def test_dbt_sources_are_wired_to_the_spark_tables(defs_module):
    deps = defs_module.filinglens_dbt.asset_deps
    assert dg.AssetKey(["fsds", "num"]) in deps[dg.AssetKey(["stg_fsds__numbers"])]
    assert dg.AssetKey(["fsds", "sub"]) in deps[dg.AssetKey(["stg_fsds__submissions"])]
    assert dg.AssetKey(["fct_financials"]) in deps


def test_backfill_report_compares_shas_with_the_manifest(tmp_path):
    report = _load("fl_backfill_report", REPO / "dagster_project" / "backfill_report.py")
    (tmp_path / "2025q1.zip").write_bytes(b"same bytes")
    (tmp_path / "2025q2.zip").write_bytes(b"changed")
    same = report.sha256_file(tmp_path / "2025q1.zip")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(f"quarter,sha256\n2025q1,{same}\n2025q2,{same}\n2025q3,{same}\n")
    rows = report.compare_with_manifest(["2025q1", "2025q2", "2025q3"], manifest, tmp_path)
    assert [r["same"] for r in rows] == [True, False, False]  # 2025q3 is missing on disk
