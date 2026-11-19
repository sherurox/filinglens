import asyncio

import httpx

from filinglens.kg.exhibits import exhibit_links, fetch_company, latest_10k

FOLDER = "/Archives/edgar/data/320193/000032019325000079/"


def row(seq: int, doc: str, kind: str, href: str | None = None) -> str:
    href = href or FOLDER + doc
    return f'<tr><td>{seq}</td><td></td><td><a href="{href}">{doc}</a></td><td>{kind}</td></tr>'


INDEX = (
    '<html><body><table class="tableFile" summary="Document Format Files">'
    "<tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th></tr>"
    + row(1, "aapl-20250927.htm", "10-K", "/ix?doc=" + FOLDER + "aapl-20250927.htm")
    + row(2, "a10-kexhibit2112025.htm", "EX-21.1")
    + row(3, "ex23.htm", "EX-23.1")
    + row(4, "ex21b.htm", "EX-21")
    + '</table><table class="tableFile" summary="Data Files">'
    + row(5, "x.xml", "EX-101.SCH")
    + "</table></body></html>"
).encode()


def test_exhibit_links_finds_ex21_only():
    base = "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/"
    assert exhibit_links(INDEX) == [
        ("EX-21.1", base + "a10-kexhibit2112025.htm"),
        ("EX-21", base + "ex21b.htm"),
    ]


def test_latest_10k_per_company_on_disk(tmp_path):
    files = {}
    for name in ("a2024", "a2025", "b2023"):
        files[name] = tmp_path / name
        files[name].write_text("x")
    rows = [
        {
            "cik": "1",
            "report_date": "2024-12-31",
            "status": "downloaded",
            "path": str(files["a2024"]),
        },
        {
            "cik": "1",
            "report_date": "2025-12-31",
            "status": "skipped_existing",
            "path": str(files["a2025"]),
        },
        {
            "cik": "1",
            "report_date": "2026-06-30",
            "status": "failed",
            "path": str(tmp_path / "gone"),
        },
        {
            "cik": "2",
            "report_date": "2023-12-31",
            "status": "downloaded",
            "path": str(files["b2023"]),
        },
    ]
    assert [r["report_date"] for r in latest_10k(rows)] == ["2025-12-31", "2023-12-31"]


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeClient:
    def __init__(self, pages, fail=False):
        self.pages, self.fail, self.urls = pages, fail, []

    async def get(self, url):
        self.urls.append(url)
        if self.fail:
            raise httpx.HTTPError("boom")
        return FakeResponse(self.pages.get(url, b"<html><body>Subsidiaries</body></html>"))


ROW = {
    "cik": "320193",
    "ticker": "AAPL",
    "sector": "manufacturing",
    "report_date": "2025-09-27",
    "accession": "0000320193-25-000079",
}
INDEX_URL = (
    "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/"
    "0000320193-25-000079-index.htm"
)


def test_fetch_company_downloads_then_uses_cache(tmp_path):
    client = FakeClient({INDEX_URL: INDEX})
    record = asyncio.run(fetch_company(client, ROW, tmp_path, asyncio.Semaphore(1)))
    assert record["status"] == "ok" and len(record["exhibits"].split("|")) == 2
    assert len(client.urls) == 3  # index + two EX-21 documents
    again = FakeClient({})
    assert asyncio.run(fetch_company(again, ROW, tmp_path, asyncio.Semaphore(1))) == record
    assert again.urls == []  # everything came from the cache


def test_fetch_company_without_ex21_and_on_error(tmp_path):
    empty = FakeClient({INDEX_URL: b"<html><table class='tableFile'></table></html>"})
    record = asyncio.run(fetch_company(empty, ROW, tmp_path / "a", asyncio.Semaphore(1)))
    assert (record["status"], record["exhibits"]) == ("no_ex21", "")
    failed = asyncio.run(
        fetch_company(FakeClient({}, fail=True), ROW, tmp_path / "b", asyncio.Semaphore(1))
    )
    assert failed["status"] == "failed" and "boom" in failed["error"]
