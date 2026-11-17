"""EXPLAIN runner against a fake database: discovery, parsing, before/after flow, outputs."""

import csv
import re
from pathlib import Path

import pytest

from filinglens.sql.explain_bench import (
    QuerySpec,
    bench_query,
    discover,
    parse_times,
    result_digest,
    strip_sql,
    table_access,
    write_outputs,
    write_results_md,
)


class FakeDatabase:
    """Indexes appear on CREATE INDEX and vanish on DROP INDEX; plans are faster with one.

    `slow_runs` gives the before-state execution times in order, to test the median.
    """

    def __init__(self, result, slow_runs=None, leftover_index=None):
        self.result = result
        self.indexes = {leftover_index} if leftover_index else set()
        self.log = []
        self._slow = list(slow_runs or [])

    def run(self, sql):
        self.log.append(sql)
        if m := re.match(r"CREATE INDEX (\w+)", sql):
            self.indexes.add(m.group(1))
        elif m := re.match(r'DROP INDEX "marts"\."(\w+)"', sql):
            self.indexes.discard(m.group(1))

    def rows(self, sql):
        self.log.append(sql)
        if sql.startswith("SELECT indexname FROM pg_indexes"):
            return [(name,) for name in sorted(self.indexes)]
        if sql.startswith("EXPLAIN"):
            if self.indexes:
                ms = 2.0
            elif "TIMING OFF" in sql and self._slow:
                ms = self._slow.pop(0)
            else:
                ms = 10.0
            node = "Index Scan" if self.indexes else "Seq Scan"
            return [
                (f"{node} on fct_financials",),
                ("Planning Time: 0.5 ms",),
                (f"Execution Time: {ms} ms",),
            ]
        return list(self.result)


SPEC = QuerySpec(
    "01_demo",
    "SELECT cik FROM marts.fct_financials",
    "CREATE INDEX ix_demo ON marts.fct_financials (cik);",
)


def test_strip_sql_drops_comment_lines_and_trailing_semicolon():
    assert strip_sql("-- title\nSELECT 1\nFROM t;\n") == "SELECT 1\nFROM t"


def test_discover_pairs_each_query_with_its_index_file(tmp_path: Path):
    (tmp_path / "01_a.sql").write_text("SELECT 1;")
    (tmp_path / "01_a.index.sql").write_text("CREATE INDEX ix_a ON t (x);")
    (tmp_path / "02_b.sql").write_text("SELECT 2;")
    (tmp_path / "02_b.index.sql").write_text("CREATE INDEX ix_b ON t (y);")
    (tmp_path / "notes.sql").write_text("SELECT 3;")  # no NN_ prefix: ignored
    assert [s.name for s in discover(tmp_path)] == ["01_a", "02_b"]
    assert [s.name for s in discover(tmp_path, "02")] == ["02_b"]
    assert discover(tmp_path)[0].query == "SELECT 1"


def test_discover_requires_the_index_file(tmp_path: Path):
    (tmp_path / "01_a.sql").write_text("SELECT 1;")
    with pytest.raises(FileNotFoundError, match="01_a.index.sql"):
        discover(tmp_path)


def test_parse_times_reads_the_summary_lines():
    lines = [
        "Seq Scan on t  (cost=0.00..1.00 rows=1 width=4)",
        "Planning Time: 0.123 ms",
        "Execution Time: 45.6 ms",
    ]
    assert parse_times(lines) == (0.123, 45.6)
    with pytest.raises(ValueError):
        parse_times(["Seq Scan on t"])


def test_result_digest_ignores_row_order_but_not_values():
    assert result_digest([(1, "a"), (2, "b")]) == result_digest([(2, "b"), (1, "a")])
    assert result_digest([(1, "a")]) != result_digest([(1, "b")])


def test_bench_query_measures_before_and_after_and_cleans_up():
    db = FakeDatabase(result=[(1,), (2,)], leftover_index="ix_other")
    r = bench_query(db, SPEC, runs=3, tables=["fct_financials"])
    assert (r["before_execution_ms"], r["after_execution_ms"], r["speedup"]) == (10.0, 2.0, 5.0)
    assert r["same_result"] is True and r["result_rows"] == 2
    assert "Seq Scan" in r["_before_plan"][0] and "Index Scan" in r["_after_plan"][0]
    assert db.indexes == set()  # the leftover and its own index are both dropped
    assert db.log.index('DROP INDEX "marts"."ix_other"') < db.log.index(SPEC.index_change)
    assert sum(sql.startswith("EXPLAIN (ANALYZE, BUFFERS)") for sql in db.log) == 2  # one plan each


def test_bench_query_takes_the_median_so_one_slow_run_does_not_count():
    db = FakeDatabase(result=[(1,)], slow_runs=[10.0, 9.0, 100.0, 11.0, 10.0, 10.0])
    r = bench_query(db, SPEC, runs=5, tables=[])
    assert r["before_execution_ms"] == 10.0  # warm-up took 10.0; timed runs 9, 100, 11, 10, 10


def test_write_outputs_saves_plans_and_merges_csv_rows(tmp_path: Path):
    db = FakeDatabase(result=[(1,)])
    r = bench_query(db, SPEC, runs=1, tables=[])
    csv_path = tmp_path / "bench" / "sql_explain.csv"
    csv_path.parent.mkdir()
    csv_path.write_text("query,result_rows\n00_old,7\n")
    write_outputs([r], tmp_path / "explain", csv_path)
    before = (tmp_path / "explain" / "01_demo.before.txt").read_text()
    assert before.startswith("-- 01_demo: median of 1 runs, execution 10.0 ms (before")
    assert "Seq Scan" in before
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert [row["query"] for row in rows] == ["00_old", "01_demo"]
    assert rows[1]["index_change"] == "CREATE INDEX ix_demo ON marts.fct_financials (cik)"


def test_table_access_skips_wrapper_scans_and_strips_costs():
    plan = (
        "-- 01_demo: header\n"
        "Sort  (cost=1.00..2.00 rows=1 width=4)\n"
        "  ->  Subquery Scan on ranked  (cost=1.00..2.00 rows=1 width=4)\n"
        "        ->  Index Only Scan using ix_a on fct_financials f  (cost=0.43..8.45 rows=1)\n"
        "              Heap Fetches: 0\n"
    )
    assert table_access(plan) == "Index Only Scan using ix_a on fct_financials f"
    assert table_access("Seq Scan on dim_company c  (cost=0.00..8.00 rows=300 width=8)") == ""


def test_write_results_md_builds_one_row_per_query(tmp_path: Path):
    db = FakeDatabase(result=[(1,)])
    r = bench_query(db, SPEC, runs=1, tables=[])
    csv_path = tmp_path / "sql_explain.csv"
    write_outputs([r], tmp_path / "explain", csv_path)
    out = tmp_path / "RESULTS.md"
    write_results_md(csv_path, tmp_path / "explain", out)
    text = out.read_text()
    assert "do not edit" in text.splitlines()[0]
    row = text.splitlines()[-1]
    assert row.startswith("| `01_demo` | 1 | True | 10.0 | 2.0 | x5.0 | Seq Scan on fct_financials")
    assert "| Index Scan on fct_financials |" in row
