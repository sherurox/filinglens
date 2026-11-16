"""FSDS -> Iceberg: extraction (pure Python) and one Spark run on a tiny two-quarter fixture."""

import io
import shutil
import zipfile
from decimal import Decimal

import pytest

from filinglens.spark.fsds_to_iceberg import MEMBERS, extract_quarters

FIXTURE = {
    "2024q1": {
        "sub": [
            ["adsh", "cik", "name", "sic", "fy", "fp", "form", "period", "filed"],
            ["A", "1", "Acme", "3571", "2023", "FY", "10-K", "20231231", "20240215"],
            ["B", "2", "Beta", "6021", "2024", "Q1", "10-Q", "20240331", "20240501"],
        ],
        "num": [
            ["adsh", "tag", "version", "ddate", "qtrs", "uom", "coreg", "value", "footnote"],
            ["A", "Revenues", "us-gaap/2023", "20231231", "4", "USD", "", "100.5", ""],
            ["A", "Revenues", "us-gaap/2023", "20231231", "4", "USD", "", "100.5", ""],  # dup
            ["B", "Revenues", "us-gaap/2023", "20240331", "1", "USD", "", "not-a-number", ""],
        ],
        "tag": [
            ["tag", "version", "custom", "abstract", "datatype", "tlabel", "doc"],
            ["Revenues", "us-gaap/2023", "0", "0", "monetary", "Revenues", "old text"],
        ],
        "pre": [
            ["adsh", "report", "line", "stmt", "inpth", "tag", "version", "plabel", "negating"],
            ["A", "2", "1", "IS", "0", "Revenues", "us-gaap/2023", "Revenue", "0"],
        ],
    },
    "2024q2": {
        "sub": [
            ["adsh", "cik", "name", "sic", "fy", "fp", "form", "period", "filed"],
            ["C", "1", "Acme", "3571", "2024", "Q1", "10-Q", "20240330", "20240505"],
        ],
        "num": [
            ["adsh", "tag", "version", "ddate", "qtrs", "uom", "coreg", "value", "footnote"],
            ["C", "Revenues", "us-gaap/2023", "20240331", "1", "USD", "", "30", ""],
        ],
        "tag": [
            ["tag", "version", "custom", "abstract", "datatype", "tlabel", "doc"],
            ["Revenues", "us-gaap/2023", "0", "0", "monetary", "Revenues", "new text"],  # dup key
        ],
        "pre": [
            ["adsh", "report", "line", "stmt", "inpth", "tag", "version", "plabel", "negating"],
            ["C", "2", "1", "IS", "0", "Revenues", "us-gaap/2023", "Revenue", "0"],
        ],
    },
}


def write_zips(zip_dir):
    zip_dir.mkdir(parents=True, exist_ok=True)
    for quarter, files in FIXTURE.items():
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            for member, table in files.items():
                zf.writestr(f"{member}.txt", "\n".join("\t".join(r) for r in table) + "\n")
        (zip_dir / f"{quarter}.zip").write_bytes(buffer.getvalue())


def test_extract_quarters_is_complete_and_idempotent(tmp_path):
    write_zips(tmp_path / "zips")
    out = tmp_path / "extracted"
    assert extract_quarters(tmp_path / "zips", out) == ["2024q1", "2024q2"]
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.txt"))
    assert files == [f"{q}/{m}.txt" for q in ("2024q1", "2024q2") for m in sorted(MEMBERS)]
    before = {p: p.stat().st_mtime_ns for p in out.rglob("*.txt")}
    extract_quarters(tmp_path / "zips", out)
    assert {p: p.stat().st_mtime_ns for p in out.rglob("*.txt")} == before  # nothing rewritten


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Run the real Spark job once on the fixture (needs Java; downloads the Iceberg jar)."""
    pytest.importorskip("pyspark")
    if shutil.which("java") is None:
        pytest.skip("Java not installed")
    from filinglens.spark.fsds_to_iceberg import build_spark, build_tables

    root = tmp_path_factory.mktemp("fsds")
    write_zips(root / "zips")
    extract_quarters(root / "zips", root / "extracted")
    spark = build_spark(root / "warehouse", driver_memory="1g", master="local[2]")
    stats = build_tables(spark, root / "extracted")
    yield spark, stats
    spark.stop()


def test_row_counts_and_dedupe(built):
    _, stats = built
    assert {name: s["rows"] for name, s in stats.items()} == {
        "sub": 3,
        "tag": 1,
        "num": 3,
        "pre": 2,
    }
    assert stats["num"]["duplicates_removed"] == 1
    assert stats["tag"]["duplicates_removed"] == 1


def test_newest_quarter_wins_and_types_are_cast(built):
    spark, _ = built
    assert spark.table("local.fsds.tag").first()["doc"] == "new text"
    num = {r["adsh"]: r for r in spark.table("local.fsds.num").collect()}
    assert num["A"]["value"] == Decimal("100.5000")
    assert num["B"]["value"] is None  # bad number -> null, not a failed job
    assert str(num["A"]["ddate"]) == "2023-12-31"
    assert num["A"]["fy"] == 2023  # fiscal year joined from sub


def test_partitioned_by_fiscal_year(built):
    _, stats = built
    assert [(p["fy"], p["record_count"]) for p in stats["num"]["partitions"]] == [
        (2023, 1),
        (2024, 2),
    ]
    assert stats["num"]["rows_without_fy"] == 0
