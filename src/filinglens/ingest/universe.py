"""Pick the company universe: about 300 companies across 6 sectors, from SEC's ticker list."""

import asyncio
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from filinglens.ingest.filings import SubmissionsSource, is_annual_10k, iter_filings

# SIC divisions (the SEC's standard industry codes), mapped to the six sectors we sample.
SECTOR_BY_SIC_RANGE = [
    (1000, 1499, "mining_energy"),
    (2000, 3999, "manufacturing"),
    (4000, 4999, "transport_utilities"),
    (5200, 5999, "retail"),
    (6000, 6799, "finance"),
    (7000, 8999, "services"),
]
BLANK_CHECK_SIC = 6770  # SPACs: shell companies with no operating business
DEFAULT_QUOTAS = {sector: 50 for *_, sector in SECTOR_BY_SIC_RANGE}
YEARS = range(2018, 2026)  # fiscal years 2018-2025
FIELDS = ["rank", "cik", "ticker", "name", "sic", "sic_description", "sector"]


def sic_to_sector(sic: str | int | None) -> str | None:
    """Sector for a SIC code, or None for codes we don't sample (and for SPACs)."""
    if sic in (None, "") or not str(sic).isdigit():
        return None
    code = int(sic)
    if code == BLANK_CHECK_SIC:
        return None
    for low, high, sector in SECTOR_BY_SIC_RANGE:
        if low <= code <= high:
            return sector
    return None


def ranked_companies(tickers: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """SEC's ticker file in its own order (largest companies first, roughly), one row per CIK."""
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    for key in sorted(tickers, key=int):
        entry = tickers[key]
        cik = int(entry["cik_str"])
        if cik in seen:  # e.g. GOOGL and GOOG share one CIK
            continue
        seen.add(cik)
        out.append(
            {"rank": len(out) + 1, "cik": cik, "ticker": entry["ticker"], "name": entry["title"]}
        )
    return out


async def classify(source: SubmissionsSource, company: dict[str, Any]) -> dict[str, Any] | None:
    """Add SIC and sector to `company`, or return None if it's out of scope or has no 10-K."""
    subs = await source.submissions(company["cik"])
    sector = sic_to_sector(subs.get("sic"))
    if sector is None:
        return None
    async for filing in iter_filings(source, subs):
        if is_annual_10k(filing, YEARS):
            return {
                **company,
                "sic": subs.get("sic"),
                "sic_description": subs.get("sicDescription", ""),
                "sector": sector,
            }
    return None


async def build_universe(
    source: SubmissionsSource,
    quotas: dict[str, int] | None = None,
    batch_size: int = 16,
    max_candidates: int = 2000,
) -> list[dict[str, Any]]:
    """Walk companies in rank order and fill each sector's quota.

    Requests run concurrently in batches, but results are taken in rank order, so the
    universe is the same no matter which response arrives first.
    """
    quotas = dict(quotas or DEFAULT_QUOTAS)
    counts = dict.fromkeys(quotas, 0)
    chosen: list[dict[str, Any]] = []
    candidates = ranked_companies(await source.company_tickers())[:max_candidates]
    for start in range(0, len(candidates), batch_size):
        if all(counts[s] >= quotas[s] for s in quotas):
            break
        batch = candidates[start : start + batch_size]
        for company in await asyncio.gather(*(classify(source, c) for c in batch)):
            if company and counts.get(company["sector"], 0) < quotas.get(company["sector"], 0):
                counts[company["sector"]] += 1
                chosen.append(company)
    return chosen


def write_csv(companies: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(companies)


async def main(out: Path = Path("dbt/seeds/universe.csv")) -> None:
    from filinglens.ingest.edgar_client import EdgarClient, load_user_agent

    async with EdgarClient(load_user_agent()) as client:
        universe = await build_universe(client)
    write_csv(universe, out)
    print(f"{len(universe)} companies -> {out}")
    for sector, n in sorted(Counter(c["sector"] for c in universe).items()):
        print(f"  {sector}: {n}")


if __name__ == "__main__":
    asyncio.run(main())
