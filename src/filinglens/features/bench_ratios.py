"""Benchmark the three ratio implementations (FL-M1 D5): real annual data + synthetic sizes.

For each dataset: check that all three give identical output (stop otherwise), warm each up
once, then time `repeats` end-to-end calls (DataFrame in, DataFrame out) and keep the median
and the minimum. Numba's compile time is measured separately, on a fresh uncached copy of the
kernel. Writes bench/ratios.csv, bench/ratios.json and data/features/annual_ratios.parquet.

Run from the repo root (plugged in; note the Windows power mode from Settings):
    uv run python -m filinglens.features.bench_ratios --note "Best performance, plugged in"
"""

import argparse
import csv
import json
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from numba import njit

from filinglens.features.ratios import (  # the kernel itself, to time a fresh compile
    FEATURES,
    IMPLEMENTATIONS,
    INPUTS,
    _ratios_kernel,
)

REVENUE_TAGS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
)
CSV_FIELDS = [
    "dataset",
    "rows",
    "implementation",
    "repeats",
    "median_s",
    "min_s",
    "rows_per_s",
    "speedup_vs_pandas_apply",
]


def annual_sql(fct: str) -> str:
    """One row per company and fiscal year end that has annual revenue: the annual flows plus
    the balances at that same date. Revenue takes the first tag present in REVENUE_TAGS."""
    revenue_columns = ",\n".join(
        f"        max(case when tag = '{tag}' then value end) as rev_{i}"
        for i, tag in enumerate(REVENUE_TAGS)
    )
    revenue = "coalesce(" + ", ".join(f"rev_{i}" for i in range(len(REVENUE_TAGS))) + ")"
    flow_tags = ", ".join(f"'{t}'" for t in (*REVENUE_TAGS, "NetIncomeLoss", "OperatingIncomeLoss"))
    return f"""
with facts as (
    select cik, value_date, tag, cast(value as double) as value
    from {fct}
    where unit = 'USD'
        and (
            (duration_quarters = 4 and tag in ({flow_tags}))
            or (duration_quarters = 0
                and tag in ('Assets', 'Liabilities', 'StockholdersEquity'))
        )
),
wide as (
    select
        cik,
        value_date,
{revenue_columns},
        max(case when tag = 'NetIncomeLoss' then value end) as net_income,
        max(case when tag = 'OperatingIncomeLoss' then value end) as operating_income,
        max(case when tag = 'Assets' then value end) as assets,
        max(case when tag = 'Liabilities' then value end) as liabilities,
        max(case when tag = 'StockholdersEquity' then value end) as equity
    from facts
    group by cik, value_date
)
select cik, value_date, {revenue} as revenue, net_income, operating_income, assets,
    liabilities, equity
from wide
where {revenue} is not null
order by cik, value_date
"""


def load_annual(con: Any, fct: str) -> pd.DataFrame:
    df = con.execute(annual_sql(fct)).df()
    df["cik"] = df["cik"].astype(np.int64)
    df["value_date"] = pd.to_datetime(df["value_date"])
    return df


def synthetic_annual(rows: int, seed: int = 0, years: int = 15) -> pd.DataFrame:
    """Same shape as the real data: `years` fiscal years per company, ~5% missing values,
    a few zero or negative values, small date jitter around one year apart."""
    rng = np.random.default_rng(seed)
    companies = -(-rows // years)
    cik = np.repeat(np.arange(companies, dtype=np.int64), years)[:rows]
    step = np.tile(np.arange(years) * 365, companies)[:rows] + rng.integers(-3, 4, rows)
    df = pd.DataFrame({"cik": cik})
    df["value_date"] = pd.to_datetime("2010-12-31") + pd.to_timedelta(step, unit="D")
    for name in INPUTS:
        values = rng.lognormal(18, 2, rows) * rng.choice(
            [1.0, -1.0, 0.0], rows, p=[0.95, 0.04, 0.01]
        )
        values[rng.random(rows) < 0.05] = np.nan
        df[name] = values
    return df


def assert_identical(df: pd.DataFrame) -> None:
    results = {name: fn(df).to_numpy() for name, fn in IMPLEMENTATIONS.items()}
    for name, out in results.items():
        np.testing.assert_array_equal(out, results["numpy"], err_msg=f"{name} differs")


def time_calls(fn: Any, df: pd.DataFrame, repeats: int) -> list[float]:
    fn(df)  # warm-up: caches, page faults, first-call costs
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn(df)
        times.append(time.perf_counter() - started)
    return times


def summarise(dataset: str, rows: int, timings: dict[str, list[float]]) -> list[dict]:
    base = statistics.median(timings["pandas_apply"])
    out = []
    for name, times in timings.items():
        median = statistics.median(times)
        out.append(
            {
                "dataset": dataset,
                "rows": rows,
                "implementation": name,
                "repeats": len(times),
                "median_s": round(median, 6),
                "min_s": round(min(times), 6),
                "rows_per_s": round(rows / median),
                "speedup_vs_pandas_apply": round(base / median, 2),
            }
        )
    return out


def numba_compile_seconds() -> float:
    """Compile a fresh, uncached copy of the kernel and time its first call on tiny input."""
    fresh = njit(cache=False)(_ratios_kernel.py_func)
    tiny_int = np.zeros(2, dtype=np.int64)
    tiny = np.ones(2)
    started = time.perf_counter()
    fresh(tiny_int, tiny_int, tiny, tiny, tiny, tiny, tiny, tiny)
    return round(time.perf_counter() - started, 3)


def parse_power_scheme(text: str) -> str:
    """'Power Scheme GUID: 381b...  (Balanced)' -> 'Balanced'."""
    if "(" in text and ")" in text:
        return text[text.rindex("(") + 1 : text.rindex(")")].strip()
    return text.strip() or "unknown"


def _windows(args: list[str]) -> str:
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return done.stdout.strip() if done.returncode == 0 else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def environment(note: str) -> dict[str, Any]:
    cpu = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    on_ac = _windows(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus).PowerOnline",
        ]
    )
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "numba": version("numba"),
        "windows_power_scheme": parse_power_scheme(_windows(["powercfg.exe", "/getactivescheme"])),
        "on_ac_power": on_ac,
        "note": note,
    }


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.register("features", df)
    tmp = path.with_name(path.name + ".part")
    con.execute(f"COPY features TO '{tmp.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duckdb", type=Path, default=Path("data/filinglens.duckdb"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--apply-repeats-1m", type=int, default=3)
    parser.add_argument("--note", default="", help="power mode / plugged in, from Settings")
    parser.add_argument("--csv", type=Path, default=Path("bench/ratios.csv"))
    parser.add_argument("--json", type=Path, default=Path("bench/ratios.json"))
    parser.add_argument(
        "--features", type=Path, default=Path("data/features/annual_ratios.parquet")
    )
    args = parser.parse_args(argv)

    con = duckdb.connect()
    con.execute(f"ATTACH '{args.duckdb.as_posix()}' AS src (READ_ONLY)")
    real = load_annual(con, "src.main.fct_financials")
    datasets = [("real_annual", real)]
    datasets += [(f"synthetic_{n}", synthetic_annual(n, seed=n)) for n in args.sizes]

    compile_s = numba_compile_seconds()
    print(f"numba compile (fresh, uncached): {compile_s} s")
    rows_out = []
    for name, df in datasets:
        assert_identical(df)
        timings = {}
        for impl, fn in IMPLEMENTATIONS.items():
            repeats = args.repeats
            if impl == "pandas_apply" and len(df) >= 1_000_000:
                repeats = args.apply_repeats_1m
            timings[impl] = time_calls(fn, df, repeats)
        summary = summarise(name, len(df), timings)
        rows_out += summary
        for s in summary:
            print(
                f"{name:>18} {s['rows']:>9} {s['implementation']:>12}: median {s['median_s']} s, "
                f"x{s['speedup_vs_pandas_apply']} vs apply"
            )

    features = pd.concat([real, IMPLEMENTATIONS["numba"](real)], axis=1)
    write_parquet(features, args.features)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)
    report = {
        "command": shlex.join(  # quotes arguments with spaces, e.g. --note
            [
                "uv run python -m filinglens.features.bench_ratios",
                *(sys.argv[1:] if argv is None else argv),
            ]
        ).replace(
            "'uv run python -m filinglens.features.bench_ratios'",
            "uv run python -m filinglens.features.bench_ratios",
        ),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": environment(args.note),
        "numba_compile_seconds": compile_s,
        "real_annual": {
            "rows": len(real),
            "companies": int(real["cik"].nunique()),
            "first_value_date": str(real["value_date"].min().date()),
            "last_value_date": str(real["value_date"].max().date()),
            "non_null": {c: int(features[c].notna().sum()) for c in (*INPUTS, *FEATURES)},
        },
        "results": rows_out,
    }
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
