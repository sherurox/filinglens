"""10-K downloader tests: real EdgarClient on a fake server and fake clock, files in tmp_path."""

import asyncio
import random

import httpx
from fakes import FakeClock

from filinglens.ingest.edgar_client import EdgarClient
from filinglens.ingest.filings import rows
from filinglens.ingest.rate_limit import max_in_window
from filinglens.ingest.tenk import plan_jobs, run, summarize

UA = "FilingLens Test test@example.org"
RECENT = {
    "accessionNumber": [
        "0000000001-25-000010",
        "0000000001-25-000011",
        "0000000001-25-000012",
        "0000000001-24-000020",
        "0000000001-18-000030",
    ],
    "form": ["10-K", "10-K/A", "8-K", "10-K", "10-K"],
    "reportDate": ["2024-12-31", "2024-12-31", "", "2023-12-31", "2017-12-31"],
    "filingDate": ["2025-02-14", "2025-03-01", "2025-04-01", "2024-02-15", "2018-02-20"],
    "primaryDocument": ["acme-2024.htm", "acme-2024a.htm", "8k.htm", "acme-2023.htm", "a.htm"],
}
SUBS = {"name": "Acme", "sic": "3571", "filings": {"recent": RECENT, "files": []}}


def company(cik: int) -> dict[str, str]:
    return {"cik": str(cik), "ticker": f"T{cik}", "sector": "manufacturing"}


def make_client(fail_urls=()):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        if "/submissions/" in url:
            return httpx.Response(200, json=SUBS)
        if url in fail_urls:
            return httpx.Response(404)
        return httpx.Response(200, content=b"<html>10-K</html>")

    client = EdgarClient(
        UA, clock=FakeClock(), transport=httpx.MockTransport(handler), rng=random.Random(0)
    )
    return client, calls


def run_once(client, companies, tmp_path):
    async def go():
        async with client:
            return await run(client, companies, tmp_path / "10k", tmp_path / "subs")

    return asyncio.run(go())


def test_plan_jobs_keeps_original_10ks_in_years(tmp_path):
    jobs = plan_jobs(company(1), rows(RECENT), tmp_path)
    assert [job.accession for job in jobs] == ["0000000001-25-000010", "0000000001-24-000020"]
    assert jobs[0].url == (
        "https://www.sec.gov/Archives/edgar/data/1/000000000125000010/acme-2024.htm"
    )
    assert jobs[0].path == str(tmp_path / "1" / "2024-12-31_0000000001-25-000010.htm")


def test_download_then_rerun_fetches_nothing(tmp_path):
    client, calls = make_client()
    jobs, errors = run_once(client, [company(1)], tmp_path)
    assert errors == []
    assert [job.status for job in jobs] == ["downloaded", "downloaded"]
    assert len(calls) == 3  # 1 submissions + 2 documents
    assert (tmp_path / "10k" / "1" / "2024-12-31_0000000001-25-000010.htm").read_bytes() == (
        b"<html>10-K</html>"
    )

    client, calls = make_client()
    jobs, _ = run_once(client, [company(1)], tmp_path)
    assert [job.status for job in jobs] == ["skipped_existing", "skipped_existing"]
    assert calls == []  # submissions come from the cache, documents are already on disk


def test_failed_download_is_recorded_and_leaves_no_file(tmp_path):
    bad = "https://www.sec.gov/Archives/edgar/data/1/000000000124000020/acme-2023.htm"
    client, _ = make_client(fail_urls={bad})
    jobs, _ = run_once(client, [company(1)], tmp_path)
    assert sorted(job.status for job in jobs) == ["downloaded", "failed"]
    failed = next(job for job in jobs if job.status == "failed")
    assert "HTTPStatusError" in failed.error
    assert list((tmp_path / "10k" / "1").glob("*2023*")) == []  # no file, no .part leftover


def test_real_downloader_never_exceeds_sec_limit(tmp_path):
    client, calls = make_client()
    jobs, _ = run_once(client, [company(cik) for cik in range(1, 21)], tmp_path)
    assert len(calls) == 60  # 20 submissions + 40 documents
    assert max_in_window(client.request_log, 1.0) <= 10
    stats = summarize(jobs, [], client, elapsed=client.clock.now())
    assert stats["requests"] == 60
    assert stats["max_requests_in_any_1s_window"] <= stats["sec_limit_per_s"]
