"""build_tables' table_format option and the per-format partition summary (fake Spark session)."""

from types import SimpleNamespace

import pytest

from filinglens.spark.fsds_to_iceberg import build_tables, partition_summary


class FakeRow(dict):
    def asDict(self):  # noqa: N802 (PySpark's name)
        return dict(self)


class FakeSpark:
    """Answers spark.sql(query) from a function of the query text; records every query."""

    def __init__(self, answer):
        self.answer, self.queries = answer, []

    def sql(self, query):
        self.queries.append(query)
        return SimpleNamespace(collect=lambda: [FakeRow(r) for r in self.answer(query)])


def test_iceberg_summary_reads_the_metadata_tables():
    spark = FakeSpark(lambda q: [{"fy": 2024, "record_count": 5, "file_count": 1}])
    assert partition_summary(spark, "local.fsds.num", True) == [
        {"fy": 2024, "record_count": 5, "file_count": 1}
    ]
    assert "local.fsds.num.partitions" in spark.queries[0]


def test_delta_summary_counts_rows_per_fiscal_year():
    spark = FakeSpark(lambda q: [{"fy": 2024, "record_count": 7}, {"fy": 2025, "record_count": 3}])
    rows = partition_summary(spark, "workspace.filinglens.num", True, "delta")
    assert rows == [{"fy": 2024, "record_count": 7}, {"fy": 2025, "record_count": 3}]
    assert "GROUP BY fy" in spark.queries[0] and ".partitions" not in spark.queries[0]


def test_delta_summary_of_an_unpartitioned_table_uses_describe_detail():
    def answer(query):
        if query.startswith("DESCRIBE DETAIL"):
            return [{"numFiles": 4, "format": "delta"}]
        return [{"record_count": 120}]

    spark = FakeSpark(answer)
    rows = partition_summary(spark, "workspace.filinglens.tag", False, "delta")
    assert rows == [{"file_count": 4, "record_count": 120}]


def test_unknown_table_format_is_rejected_before_any_work(tmp_path):
    spark = FakeSpark(lambda q: [])
    with pytest.raises(ValueError, match="table_format"):
        build_tables(spark, tmp_path, table_format="parquet")
    assert spark.queries == []
