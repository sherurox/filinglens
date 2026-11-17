"""Profile the FSDS staging data to decide how the marts treat edge cases (FL-M1 T8).

Analysis only: the decisions it informs live in the dbt models and docs/DATA.md.
Run from the repo root (needs the staging views from `cd dbt && uv run dbt run`):
    uv run python notebooks/profile_fsds_t8.py | tee bench/fsds_profile_t8.txt
"""

import os
import time
from pathlib import Path
from typing import Any

import duckdb

DBT_DIR = Path(__file__).resolve().parents[1] / "dbt"
UNIVERSE = "(select cik from read_csv('seeds/universe.csv'))"
NUM_TXT = "../data/interim/fsds/2024q1/num.txt"

QUERIES: list[tuple[str, str]] = [
    (
        "Q0 Filings in FSDS, and how many belong to the 300-company universe",
        f"""select count(*) as filings,
                  count(*) filter (where cik in {UNIVERSE}) as universe_filings,
                  count(distinct cik) filter (where cik in {UNIVERSE}) as universe_ciks_found
           from stg_fsds__submissions""",
    ),
    (
        "Q1 Forms (top 15), with blank fiscal_year per form",
        """select form, count(*) as filings,
                  count(*) filter (where fiscal_year is null) as blank_fy
           from stg_fsds__submissions group by form order by filings desc limit 15""",
    ),
    (
        "Q2 Blank fiscal_year: form x fiscal_period, period range, universe share",
        f"""select form, fiscal_period, count(*) as filings,
                  count(*) filter (where cik in {UNIVERSE}) as in_universe,
                  min(period_end_date) as first_period, max(period_end_date) as last_period
           from stg_fsds__submissions where fiscal_year is null
           group by all order by filings desc limit 15""",
    ),
    (
        "Q3 fiscal_year minus year(period_end_date), where fiscal_year is known",
        """select fiscal_year - year(period_end_date) as fy_minus_period_year, count(*) as filings
           from stg_fsds__submissions where fiscal_year is not null
           group by 1 order by filings desc limit 10""",
    ),
    (
        "Q4 duration_quarters distribution (all num rows)",
        """select duration_quarters, count(*) as num_rows
           from stg_fsds__numbers group by 1 order by 1""",
    ),
    (
        "Q5 Segments / coregistrant (null = whole-company value), with null values",
        """select segments is null as whole_company_segments,
                  coregistrant is null as no_coregistrant,
                  count(*) as num_rows,
                  count(*) filter (where value is null) as null_values
           from stg_fsds__numbers group by all order by num_rows desc""",
    ),
    (
        "Q6 One quarter's raw num.txt (2024q1): empty value vs did not parse",
        f"""select count(*) as raw_rows,
                  count(*) filter (where value is null) as empty_value,
                  count(*) filter (
                      where value is not null and try_cast(value as decimal(28, 4)) is null
                  ) as unparseable_value
           from read_csv('{NUM_TXT}', delim = '\t', header = true, all_varchar = true)""",
    ),
    (
        "Q7 Examples of unparseable values in 2024q1 (top 10)",
        f"""select value, count(*) as n
           from read_csv('{NUM_TXT}', delim = '\t', header = true, all_varchar = true)
           where value is not null and try_cast(value as decimal(28, 4)) is null
           group by 1 order by n desc limit 10""",
    ),
    (
        "Q8 Universe, 10-K/10-Q, whole-company, non-null facts: repeats across filings",
        f"""with f as (
               select s.cik, n.tag, n.value_date, n.duration_quarters, n.unit, n.value
               from stg_fsds__numbers n
               join stg_fsds__submissions s using (accession_number)
               where s.cik in {UNIVERSE}
                 and s.form in ('10-K', '10-Q')
                 and n.segments is null and n.coregistrant is null and n.value is not null
           ),
           k as (
               select cik, tag, value_date, duration_quarters, unit,
                      count(*) as n_reports, count(distinct value) as n_values
               from f group by all
           )
           select sum(n_reports) as fact_rows,
                  count(*) as distinct_facts,
                  count(*) filter (where n_reports > 1) as reported_more_than_once,
                  count(*) filter (where n_values > 1) as value_changed_later
           from k""",
    ),
]


def show(con: Any, title: str, sql: str) -> None:
    print(f"\n=== {title} ===")
    started = time.perf_counter()
    try:
        cur = con.execute(sql)
        names = [d[0] for d in cur.description]
        rows = cur.fetchall()
    except Exception as exc:  # report and continue: one bad query shouldn't hide the rest
        print(f"ERROR: {exc}")
        return
    print(" | ".join(names))
    for row in rows:
        print(" | ".join("" if v is None else str(v) for v in row))
    print(f"({len(rows)} rows, {time.perf_counter() - started:.1f} s)")


def main() -> None:
    os.chdir(DBT_DIR)  # the staging views use paths relative to dbt/
    con = duckdb.connect("../data/filinglens.duckdb", read_only=True)
    for setting in (
        "LOAD iceberg",
        "SET memory_limit = '6GB'",
        "SET threads = 8",
        "SET preserve_insertion_order = false",
        "SET temp_directory = '../data/duckdb-tmp'",
        "SET max_temp_directory_size = '40GB'",
    ):
        con.execute(setting)
    for title, sql in QUERIES:
        show(con, title, sql)


if __name__ == "__main__":
    main()
