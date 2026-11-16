"""Download 10-K documents for the universe: resumable, rate-limited, with a manifest and stats."""

import argparse
import asyncio
import csv
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from filinglens.ingest.edgar_client import EdgarClient, load_user_agent
from filinglens.ingest.filings import is_annual_10k, iter_filings
from filinglens.ingest.rate_limit import max_in_window

ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
YEARS = range(2018, 2026)  # fiscal years 2018-2025
SEC_LIMIT_PER_SECOND = 10


@dataclass
class Job:
    """One 10-K document to fetch, and what happened to it."""

    cik: int
    ticker: str
    sector: str
    report_date: str
    filing_date: str
    accession: str
    url: str
    path: str
    status: str = "planned"  # planned | downloaded | skipped_existing | failed
    bytes: int = 0
    error: str = ""


def read_universe(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def plan_jobs(company: dict[str, Any], filings: list[dict[str, Any]], root: Path) -> list[Job]:
    """One job per original 10-K in YEARS, saved as <root>/<cik>/<report date>_<accession><ext>."""
    cik = int(company["cik"])
    jobs = []
    for filing in filings:
        if not is_annual_10k(filing, YEARS):
            continue
        accession, document = filing["accessionNumber"], filing["primaryDocument"]
        suffix = Path(document).suffix or ".htm"
        jobs.append(
            Job(
                cik=cik,
                ticker=company["ticker"],
                sector=company["sector"],
                report_date=filing["reportDate"],
                filing_date=filing["filingDate"],
                accession=accession,
                url=ARCHIVE_URL.format(
                    cik=cik, accession=accession.replace("-", ""), document=document
                ),
                path=str(root / str(cik) / f"{filing['reportDate']}_{accession}{suffix}"),
            )
        )
    return jobs


def write_atomic(path: Path, data: bytes) -> None:
    """Write to a temp file, then rename: a crash never leaves a half file under the real name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


async def load_submissions(client: EdgarClient, cik: int, cache_dir: Path) -> dict[str, Any]:
    """Submissions JSON from the local cache, or from SEC (then cached for reruns and M2)."""
    path = cache_dir / f"CIK{cik:010d}.json"
    if path.exists():
        return json.loads(path.read_text())
    subs = await client.submissions(cik)
    write_atomic(path, json.dumps(subs).encode())
    return subs


async def plan_company(
    client: EdgarClient,
    company: dict[str, Any],
    root: Path,
    cache_dir: Path,
    slots: asyncio.Semaphore,
) -> tuple[list[Job], str]:
    """Jobs for one company, or ([], error) if its filing list can't be fetched."""
    async with slots:
        try:
            subs = await load_submissions(client, int(company["cik"]), cache_dir)
            filings = [f async for f in iter_filings(client, subs)]
        except httpx.HTTPError as exc:
            return [], f"{company['ticker']}: {type(exc).__name__}: {exc}"[:200]
    return plan_jobs(company, filings, root), ""


async def fetch_job(client: EdgarClient, job: Job, slots: asyncio.Semaphore) -> Job:
    """Download one document unless it's already on disk. A failure is recorded, not raised."""
    path = Path(job.path)
    if path.exists():
        job.status, job.bytes = "skipped_existing", path.stat().st_size
        return job
    async with slots:
        try:
            response = await client.get(job.url)
        except httpx.HTTPError as exc:
            job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"[:200]
            return job
    write_atomic(path, response.content)
    job.status, job.bytes = "downloaded", len(response.content)
    return job


async def run(
    client: EdgarClient,
    companies: list[dict[str, Any]],
    root: Path,
    cache_dir: Path,
    workers: int = 8,
) -> tuple[list[Job], list[str]]:
    """Plan every company, then download every job. `workers` caps requests in flight."""
    slots = asyncio.Semaphore(workers)
    planned = await asyncio.gather(
        *(plan_company(client, c, root, cache_dir, slots) for c in companies)
    )
    errors = [error for _, error in planned if error]
    jobs = [job for company_jobs, _ in planned for job in company_jobs]
    done = await asyncio.gather(*(fetch_job(client, job, slots) for job in jobs))
    return list(done), errors


def summarize(jobs: list[Job], errors: list[str], client: EdgarClient, elapsed: float) -> dict:
    """Run statistics. The request-rate fields are the evidence for SEC fair access."""
    status = Counter(job.status for job in jobs)
    requests = len(client.request_log)
    return {
        "companies_with_10k": len({job.cik for job in jobs}),
        "filings_planned": len(jobs),
        "downloaded": status["downloaded"],
        "skipped_existing": status["skipped_existing"],
        "failed": status["failed"],
        "plan_errors": len(errors),
        "bytes_downloaded": sum(job.bytes for job in jobs if job.status == "downloaded"),
        "requests": requests,
        "elapsed_s": round(elapsed, 1),
        "mean_requests_per_s": round(requests / elapsed, 2) if elapsed > 0 else 0.0,
        "max_requests_in_any_1s_window": max_in_window(client.request_log, 1.0),
        "limiter": {"rate_per_s": client.bucket.rate, "capacity": client.bucket.capacity},
        "sec_limit_per_s": SEC_LIMIT_PER_SECOND,
    }


def write_manifest(jobs: list[Job], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in fields(Job)])
        writer.writeheader()
        writer.writerows(asdict(job) for job in jobs)


async def _main(args: argparse.Namespace) -> None:
    companies = read_universe(args.universe)[: args.limit]
    async with EdgarClient(load_user_agent()) as client:
        started = client.clock.now()
        jobs, errors = await run(client, companies, args.out, args.cache, args.workers)
        elapsed = client.clock.now() - started
        stats = summarize(jobs, errors, client, elapsed)
    write_manifest(jobs, args.out / "manifest.csv")
    stats = {
        "command": "uv run python -m filinglens.ingest.tenk " + " ".join(sys.argv[1:]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **stats,
        "plan_error_messages": errors,
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "plan_error_messages"}, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=Path("docs/universe.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/raw/10k"))
    parser.add_argument("--cache", type=Path, default=Path("data/raw/submissions"))
    parser.add_argument("--stats", type=Path, default=Path("bench/edgar_10k_download.json"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="first N companies only (trial)")
    asyncio.run(_main(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
