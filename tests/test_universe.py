"""Universe selection with a fake SEC source: quotas, rank order, dedupe, older pages."""

import asyncio
from typing import Any

import pytest

from filinglens.ingest.filings import rows
from filinglens.ingest.universe import build_universe, ranked_companies, sic_to_sector


def tenk(year: int) -> dict[str, list[str]]:
    return {"form": ["10-K"], "reportDate": [f"{year}-12-31"], "filingDate": [f"{year + 1}-02-15"]}


NO_FILINGS: dict[str, list[str]] = {"form": [], "reportDate": [], "filingDate": []}


class FakeSource:
    """Stands in for EdgarClient: tickers, per-company submissions and older pages."""

    def __init__(self, tickers, subs_by_cik, pages=None):
        self.tickers = tickers
        self.subs_by_cik = subs_by_cik
        self.pages = pages or {}
        self.requested: list[int] = []

    async def company_tickers(self) -> dict[str, Any]:
        return self.tickers

    async def submissions(self, cik: int) -> dict[str, Any]:
        self.requested.append(cik)
        return self.subs_by_cik[cik]

    async def get_json(self, url: str) -> Any:
        return self.pages[url.rsplit("/", 1)[-1]]


def subs(sic: str, recent=None, files=None) -> dict[str, Any]:
    return {
        "sic": sic,
        "sicDescription": f"SIC {sic}",
        "filings": {"recent": recent or NO_FILINGS, "files": files or []},
    }


@pytest.mark.parametrize(
    ("sic", "sector"),
    [
        ("3571", "manufacturing"),  # computers (Apple)
        ("4512", "transport_utilities"),  # airlines (Delta)
        ("1311", "mining_energy"),  # oil and gas
        ("6021", "finance"),  # national banks
        ("7372", "services"),  # software
        ("5331", "retail"),  # variety stores
        ("6770", None),  # SPAC
        ("5000", None),  # wholesale: not sampled
        ("", None),
        (None, None),
    ],
)
def test_sic_to_sector(sic, sector):
    assert sic_to_sector(sic) == sector


def test_rows_turns_columns_into_records():
    assert rows({"form": ["10-K", "8-K"], "filingDate": ["2024-02-01", "2024-03-01"]}) == [
        {"form": "10-K", "filingDate": "2024-02-01"},
        {"form": "8-K", "filingDate": "2024-03-01"},
    ]


def test_ranked_companies_keeps_first_row_per_cik():
    tickers = {
        "0": {"cik_str": 1, "ticker": "GOOGL", "title": "Alphabet"},
        "1": {"cik_str": 2, "ticker": "MSFT", "title": "Microsoft"},
        "2": {"cik_str": 1, "ticker": "GOOG", "title": "Alphabet"},
    }
    assert [(c["rank"], c["ticker"]) for c in ranked_companies(tickers)] == [
        (1, "GOOGL"),
        (2, "MSFT"),
    ]


def test_build_universe_fills_quotas_in_rank_order():
    names = ["A", "B", "C", "D", "E", "F"]
    tickers = {str(i): {"cik_str": i + 1, "ticker": t, "title": t} for i, t in enumerate(names)}
    source = FakeSource(
        tickers,
        {
            1: subs("3571", tenk(2024)),  # manufacturing: chosen
            2: subs("6770", tenk(2024)),  # SPAC: skipped
            3: subs("6021", tenk(2017)),  # finance, but only a 2017 10-K: skipped
            4: subs("3711", tenk(2023)),  # manufacturing: chosen (quota now full)
            5: subs("2834", tenk(2022)),  # manufacturing: quota full, skipped
            6: subs("6021", tenk(2020)),  # finance: chosen
        },
    )
    universe = asyncio.run(build_universe(source, quotas={"manufacturing": 2, "finance": 1}))
    assert [c["ticker"] for c in universe] == ["A", "D", "F"]
    assert [c["sector"] for c in universe] == ["manufacturing", "manufacturing", "finance"]


def test_10k_in_older_page_is_found():
    """A heavy filer whose recent page has no 10-K, but an older page does."""
    tickers = {"0": {"cik_str": 7, "ticker": "BANK", "title": "Big Bank"}}
    older = {
        "name": "CIK0000000007-submissions-001.json",
        "filingFrom": "2019-01-01",
        "filingTo": "2021-12-31",
    }
    source = FakeSource(
        tickers,
        {
            7: subs(
                "6021",
                {"form": ["424B2"], "reportDate": [""], "filingDate": ["2025-06-01"]},
                [older],
            )
        },
        pages={older["name"]: tenk(2020)},
    )
    universe = asyncio.run(build_universe(source, quotas={"finance": 1}))
    assert [c["ticker"] for c in universe] == ["BANK"]


def test_stops_requesting_once_quotas_are_full():
    tickers = {str(i): {"cik_str": i + 1, "ticker": f"T{i}", "title": f"T{i}"} for i in range(40)}
    source = FakeSource(tickers, {i + 1: subs("3571", tenk(2024)) for i in range(40)})
    universe = asyncio.run(build_universe(source, quotas={"manufacturing": 3}, batch_size=8))
    assert len(universe) == 3
    assert len(source.requested) == 8  # one batch, then stop
