"""Read a company's filing list from the SEC submissions API, including older pages."""

from collections.abc import AsyncIterator
from typing import Any, Protocol

SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"


class SubmissionsSource(Protocol):
    """The part of EdgarClient this module needs. Tests pass a fake."""

    async def company_tickers(self) -> dict[str, Any]: ...

    async def submissions(self, cik: int) -> dict[str, Any]: ...

    async def get_json(self, url: str) -> Any: ...


def rows(columns: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """SEC lists filings column by column ({"form": [...], "filingDate": [...]}): make rows."""
    n = len(next(iter(columns.values()), []))
    return [{key: values[i] for key, values in columns.items()} for i in range(n)]


async def iter_filings(
    source: SubmissionsSource, subs: dict[str, Any], since: str = "2018-01-01"
) -> AsyncIterator[dict[str, Any]]:
    """Yield every filing from `subs`, newest first, fetching older pages only when needed.

    "recent" holds at least the last year or 1,000 filings. Heavy filers (big banks file
    thousands of notes a year) can push their 10-Ks into older pages, listed in "files".
    """
    for row in rows(subs["filings"]["recent"]):
        yield row
    for page in subs["filings"].get("files", []):
        if page["filingTo"] >= since:  # ISO dates compare correctly as strings
            for row in rows(await source.get_json(SUBMISSIONS_PAGE_URL.format(name=page["name"]))):
                yield row


def is_annual_10k(filing: dict[str, Any], years: range) -> bool:
    """An original 10-K (not an amendment) whose fiscal period ends in one of `years`."""
    report_date = filing.get("reportDate") or ""
    return (
        filing.get("form") == "10-K" and report_date[:4].isdigit() and int(report_date[:4]) in years
    )
