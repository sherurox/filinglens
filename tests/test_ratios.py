"""The three ratio implementations agree bit for bit, and follow the documented rules."""

import math

import numpy as np
import pandas as pd
import pytest

from filinglens.features.ratios import FEATURES, IMPLEMENTATIONS, INPUTS


def frame(rows):
    """Rows of (cik, date, revenue, net_income, operating_income, assets, liabilities, equity)."""
    df = pd.DataFrame(rows, columns=["cik", "value_date", *INPUTS])
    df["value_date"] = pd.to_datetime(df["value_date"])
    return df


EDGE_CASES = frame(
    [
        (1, "2022-12-31", 100.0, 10.0, 15.0, 200.0, 120.0, 80.0),
        (1, "2023-12-31", 120.0, 12.0, 18.0, 220.0, 130.0, 90.0),  # growth vs 2022
        (1, "2025-12-31", 150.0, 15.0, 20.0, 250.0, 150.0, 100.0),  # 2-year gap: no growth
        (2, "2023-01-28", 0.0, -5.0, -3.0, 50.0, 60.0, -10.0),  # zero revenue, negative equity
        (2, "2024-02-03", 80.0, math.nan, 4.0, math.nan, 40.0, 0.0),  # 53-week year, NaNs
        (3, "2024-06-30", -10.0, 1.0, 1.0, 10.0, 5.0, 5.0),  # negative revenue
    ]
)


def run_all(df):
    return {name: fn(df) for name, fn in IMPLEMENTATIONS.items()}


def test_documented_rules_on_edge_cases():
    out = IMPLEMENTATIONS["numpy"](EDGE_CASES)
    assert list(out.columns) == list(FEATURES)
    assert out.loc[0, "net_margin"] == 0.1
    assert out.loc[0, "liabilities_to_equity"] == 1.5
    assert math.isnan(out.loc[0, "revenue_growth"])  # first year of the company
    assert out.loc[1, "revenue_growth"] == 120.0 / 100.0 - 1.0
    assert out.loc[1, "assets_growth"] == 220.0 / 200.0 - 1.0
    assert math.isnan(out.loc[2, "revenue_growth"])  # 731 days after the previous row
    assert math.isnan(out.loc[3, "net_margin"])  # revenue 0
    assert math.isnan(out.loc[3, "equity_multiplier"])  # equity negative
    assert math.isnan(out.loc[3, "revenue_growth"])  # different company than row 2
    assert math.isnan(out.loc[4, "revenue_growth"])  # previous revenue 0
    assert math.isnan(out.loc[4, "assets_growth"])  # assets NaN
    assert math.isnan(out.loc[4, "liabilities_to_equity"])  # equity 0
    assert math.isnan(out.loc[5, "operating_margin"])  # negative revenue


def test_all_three_agree_on_edge_cases():
    results = run_all(EDGE_CASES)
    for name, out in results.items():
        np.testing.assert_array_equal(out.to_numpy(), results["numpy"].to_numpy(), err_msg=name)


def test_all_three_agree_bit_for_bit_on_random_data():
    rng = np.random.default_rng(7)
    n = 2_000
    df = pd.DataFrame({"cik": np.repeat(np.arange(n // 10), 10)})
    df["value_date"] = pd.to_datetime("2010-12-31") + pd.to_timedelta(
        np.tile(np.arange(10) * 365, n // 10) + rng.integers(-40, 40, n), unit="D"
    )
    for name in INPUTS:
        values = rng.lognormal(10, 2, n) * rng.choice([1.0, -1.0, 0.0], n, p=[0.9, 0.07, 0.03])
        values[rng.random(n) < 0.05] = np.nan
        df[name] = values
    df = df.sort_values(["cik", "value_date"]).drop_duplicates(["cik", "value_date"])
    df = df.reset_index(drop=True)
    results = run_all(df)
    for name, out in results.items():
        np.testing.assert_array_equal(out.to_numpy(), results["numpy"].to_numpy(), err_msg=name)
    assert results["numpy"]["revenue_growth"].notna().sum() > 0


def test_unsorted_input_is_rejected():
    shuffled = EDGE_CASES.iloc[[1, 0, 2, 3, 4, 5]]
    for fn in IMPLEMENTATIONS.values():
        with pytest.raises(ValueError, match="sorted"):
            fn(shuffled)
