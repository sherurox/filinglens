"""Benchmark helpers: the annual pivot SQL (DuckDB), synthetic data, timing, summary."""

import duckdb
import numpy as np
import pandas as pd

from filinglens.features.bench_ratios import (
    annual_sql,
    assert_identical,
    load_annual,
    parse_power_scheme,
    summarise,
    synthetic_annual,
    time_calls,
)

FCT_ROWS = [
    # cik, value_date, tag, duration_quarters, unit, value
    (1, "2023-12-31", "RevenueFromContractWithCustomerExcludingAssessedTax", 4, "USD", 90.0),
    (1, "2023-12-31", "Revenues", 4, "USD", 100.0),  # first in the priority list: wins
    (1, "2023-12-31", "NetIncomeLoss", 4, "USD", 10.0),
    (1, "2023-12-31", "Assets", 0, "USD", 200.0),
    (1, "2023-12-31", "StockholdersEquity", 0, "USD", 80.0),
    (1, "2023-09-30", "Revenues", 1, "USD", 25.0),  # quarterly: not an annual row
    (1, "2022-12-31", "SalesRevenueNet", 4, "USD", 95.0),
    (2, "2023-06-30", "Assets", 0, "USD", 50.0),  # no annual revenue: no row
    (3, "2023-12-31", "Revenues", 4, "EUR", 70.0),  # not USD: no row
]


def fct_connection():
    con = duckdb.connect()
    con.execute(
        "create table fct (cik bigint, value_date date, tag varchar, duration_quarters int, "
        "unit varchar, value decimal(28, 4))"
    )
    con.executemany("insert into fct values (?, ?, ?, ?, ?, ?)", FCT_ROWS)
    return con


def test_annual_pivot_picks_revenue_by_priority_and_keeps_year_end_balances():
    df = load_annual(fct_connection(), "fct")
    assert list(df["cik"]) == [1, 1]
    assert [str(d.date()) for d in df["value_date"]] == ["2022-12-31", "2023-12-31"]
    last = df.iloc[1]
    assert (last["revenue"], last["net_income"], last["assets"], last["equity"]) == (
        100.0,
        10.0,
        200.0,
        80.0,
    )
    assert np.isnan(last["liabilities"]) and np.isnan(last["operating_income"])
    assert df.iloc[0]["revenue"] == 95.0
    assert "order by cik, value_date" in annual_sql("fct")


def test_synthetic_data_has_the_expected_shape_and_passes_all_three():
    df = synthetic_annual(1_000, seed=1)
    assert len(df) == 1_000
    assert df.equals(df.sort_values(["cik", "value_date"]))
    assert not df.duplicated(["cik", "value_date"]).any()
    assert df["revenue"].isna().mean() < 0.1
    assert_identical(df)  # raises if the three implementations differ


def test_time_calls_warms_up_once_then_times_each_repeat():
    calls = []
    times = time_calls(lambda df: calls.append(1), pd.DataFrame(), repeats=4)
    assert len(times) == 4 and len(calls) == 5 and all(t >= 0 for t in times)


def test_summarise_uses_the_pandas_apply_median_as_baseline():
    rows = summarise("demo", 1_000, {"pandas_apply": [2.0, 4.0, 3.0], "numpy": [0.1, 0.2, 0.3]})
    by_impl = {r["implementation"]: r for r in rows}
    assert by_impl["pandas_apply"]["speedup_vs_pandas_apply"] == 1.0
    assert by_impl["numpy"]["median_s"] == 0.2 and by_impl["numpy"]["min_s"] == 0.1
    assert by_impl["numpy"]["speedup_vs_pandas_apply"] == 15.0
    assert by_impl["numpy"]["rows_per_s"] == 5_000


def test_parse_power_scheme():
    text = "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)"
    assert parse_power_scheme(text) == "Balanced"
    assert parse_power_scheme("unknown") == "unknown"
