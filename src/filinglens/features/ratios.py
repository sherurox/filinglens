"""Margin, leverage and growth ratios, computed three ways (FL-M1 step 8, D5).

Input: one row per company and fiscal year end, sorted by (cik, value_date), with the annual
flows (revenue, net_income, operating_income) and the year-end balances (assets, liabilities,
equity). Output: the six FEATURES as float64 columns, row for row.

Rules, identical in all three implementations (so the results are bit-for-bit equal):
- a ratio is NaN unless its denominator is > 0 (NaN, zero and negative denominators give NaN;
  negative equity makes leverage meaningless, so it is NaN too);
- growth compares with the previous row only if it is the same company and 330–400 days
  earlier (one fiscal year, allowing 52/53-week years); otherwise NaN.

Implementations:
- ratios_pandas_apply: a Python function per row via DataFrame.apply (the baseline);
- ratios_numpy: whole-column array operations;
- ratios_numba: one compiled loop over the rows (@njit).
"""

import math

import numpy as np
import pandas as pd
from numba import njit

INPUTS = ("revenue", "net_income", "operating_income", "assets", "liabilities", "equity")
FEATURES = (
    "net_margin",
    "operating_margin",
    "liabilities_to_equity",
    "equity_multiplier",
    "revenue_growth",
    "assets_growth",
)
MIN_GAP_DAYS, MAX_GAP_DAYS = 330, 400


def _prepare(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(cik, day) as int64 arrays; checks the (cik, value_date) sort order the methods rely on."""
    cik = df["cik"].to_numpy(dtype=np.int64)
    day = df["value_date"].to_numpy(dtype="datetime64[D]").astype(np.int64)
    if len(cik) > 1:
        later = (cik[1:] > cik[:-1]) | ((cik[1:] == cik[:-1]) & (day[1:] > day[:-1]))
        if not later.all():
            raise ValueError("rows must be sorted by (cik, value_date) with no duplicates")
    return cik, day


# ---------------------------------------------------------------- 1. pandas apply (baseline)


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else math.nan  # den > 0 is False for NaN


def _row_features(row: pd.Series) -> tuple[float, ...]:
    revenue, equity = row["revenue"], row["equity"]
    has_prev = row["same_company"] and MIN_GAP_DAYS <= row["gap_days"] <= MAX_GAP_DAYS
    revenue_growth = assets_growth = math.nan
    if has_prev and row["prev_revenue"] > 0:
        revenue_growth = revenue / row["prev_revenue"] - 1.0
    if has_prev and row["prev_assets"] > 0:
        assets_growth = row["assets"] / row["prev_assets"] - 1.0
    return (
        _ratio(row["net_income"], revenue),
        _ratio(row["operating_income"], revenue),
        _ratio(row["liabilities"], equity),
        _ratio(row["assets"], equity),
        revenue_growth,
        assets_growth,
    )


def ratios_pandas_apply(df: pd.DataFrame) -> pd.DataFrame:
    cik, day = _prepare(df)
    work = df[list(INPUTS)].astype("float64")
    work["same_company"] = np.r_[False, cik[1:] == cik[:-1]]
    work["gap_days"] = np.r_[0, day[1:] - day[:-1]]
    work["prev_revenue"] = work["revenue"].shift(1)
    work["prev_assets"] = work["assets"].shift(1)
    out = work.apply(_row_features, axis=1, result_type="expand")
    out.columns = list(FEATURES)
    return out


# ---------------------------------------------------------------- 2. vectorized NumPy


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(num.shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)  # NaN > 0 is False: stays NaN
    return out


def ratios_numpy(df: pd.DataFrame) -> pd.DataFrame:
    cik, day = _prepare(df)
    v = {name: df[name].to_numpy(dtype=np.float64) for name in INPUTS}
    has_prev = np.zeros(len(cik), dtype=bool)
    gap = day[1:] - day[:-1]
    has_prev[1:] = (cik[1:] == cik[:-1]) & (gap >= MIN_GAP_DAYS) & (gap <= MAX_GAP_DAYS)

    def growth(x: np.ndarray) -> np.ndarray:
        prev = np.r_[np.nan, x[:-1]]
        g = _safe_div(x, prev) - 1.0
        g[~has_prev] = np.nan
        return g

    columns = {
        "net_margin": _safe_div(v["net_income"], v["revenue"]),
        "operating_margin": _safe_div(v["operating_income"], v["revenue"]),
        "liabilities_to_equity": _safe_div(v["liabilities"], v["equity"]),
        "equity_multiplier": _safe_div(v["assets"], v["equity"]),
        "revenue_growth": growth(v["revenue"]),
        "assets_growth": growth(v["assets"]),
    }
    return pd.DataFrame(columns, index=df.index)


# ---------------------------------------------------------------- 3. Numba (compiled loop)


@njit(cache=True)
def _ratios_kernel(cik, day, revenue, net_income, operating_income, assets, liabilities, equity):
    n = cik.shape[0]
    out = np.full((n, 6), np.nan)
    for i in range(n):
        r, e = revenue[i], equity[i]
        if r > 0:
            out[i, 0] = net_income[i] / r
            out[i, 1] = operating_income[i] / r
        if e > 0:
            out[i, 2] = liabilities[i] / e
            out[i, 3] = assets[i] / e
        if i > 0 and cik[i] == cik[i - 1]:
            gap = day[i] - day[i - 1]
            if gap >= MIN_GAP_DAYS and gap <= MAX_GAP_DAYS:
                if revenue[i - 1] > 0:
                    out[i, 4] = r / revenue[i - 1] - 1.0
                if assets[i - 1] > 0:
                    out[i, 5] = assets[i] / assets[i - 1] - 1.0
    return out


def ratios_numba(df: pd.DataFrame) -> pd.DataFrame:
    cik, day = _prepare(df)
    arrays = [df[name].to_numpy(dtype=np.float64) for name in INPUTS]
    out = _ratios_kernel(cik, day, *arrays)
    return pd.DataFrame(out, columns=list(FEATURES), index=df.index)


IMPLEMENTATIONS = {
    "pandas_apply": ratios_pandas_apply,
    "numpy": ratios_numpy,
    "numba": ratios_numba,
}
