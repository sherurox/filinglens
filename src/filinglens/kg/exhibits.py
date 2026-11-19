"""Download Exhibit 21 (subsidiaries of the registrant) from each company's latest 10-K.

Per company, through the M1 client (rate limit and User-Agent from .env): the filing's index page,
which lists every document with its exhibit type, then each EX-21 document on it. Both are cached
under data/raw/ex21/<cik>/, so a rerun makes 0 requests.

    uv run python -m filinglens.kg.exhibits
"""

import argparse
import asyncio
import csv
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
from lxml import html

from filinglens.ingest.edgar_client import EdgarClient, load_user_agent
from filinglens.ingest.rate_limit import max_in_window
from filinglens.ingest.tenk import write_atomic

SEC = "https://www.sec.gov"
INDEX_URL = SEC + "/Archives/edgar/data/{cik}/{folder}/{accession}-index.htm"
EX21 = re.compile(r"^EX-21(\.\d+)?$")
FIELDS = ["cik", "ticker", "sector", "report_date", "accession", "status", "exhibits", "error"]


def latest_10k(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Each company's 10-K with the latest report date, among files that are on disk."""
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        if (
            row["status"] not in ("downloaded", "skipped_existing")
            or not Path(row["path"]).exists()
        ):
            continue
        if row["cik"] not in best or row["report_date"] > best[row["cik"]]["report_date"]:
            best[row["cik"]] = row
    return sorted(best.values(), key=lambda r: int(r["cik"]))


def exhibit_links(index_html: bytes) -> list[tuple[str, str]]:
    """(exhibit type, absolute URL) of every EX-21 document listed on a filing index page."""
    doc = html.document_fromstring(index_html)
    out = []
    for tr in doc.xpath('//table[contains(@class, "tableFile")]//tr'):
        cells = tr.xpath("./td")
        if len(cells) < 4:
            continue
        kind = " ".join(cells[3].text_content().split()).upper()
        hrefs = cells[2].xpath(".//a/@href")
        if EX21.match(kind) and hrefs:
            href = hrefs[0].replace("/ix?doc=", "")  # inline-XBRL viewer link -> the file itself
            out.append((kind, urljoin(SEC, href)))
    return out


async def cached_get(client: EdgarClient, url: str, path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    data = (await client.get(url)).content
    write_atomic(path, data)
    return data


async def fetch_company(
    client: EdgarClient, row: dict[str, str], root: Path, slots: asyncio.Semaphore
) -> dict[str, Any]:
    cik, accession = int(row["cik"]), row["accession"]
    record = {k: row[k] for k in ("cik", "ticker", "sector", "report_date", "accession")}
    folder = root / str(cik)
    url = INDEX_URL.format(cik=cik, folder=accession.replace("-", ""), accession=accession)
    async with slots:
        try:
            links = exhibit_links(await cached_get(client, url, folder / f"{accession}-index.htm"))
            paths = []
            for _, link in links:
                path = folder / f"{accession}_{link.rsplit('/', 1)[-1]}"
                await cached_get(client, link, path)
                paths.append(str(path))
        except httpx.HTTPError as exc:
            return record | {"status": "failed", "exhibits": "", "error": str(exc)[:200]}
    status = "ok" if paths else "no_ex21"
    return record | {"status": status, "exhibits": "|".join(paths), "error": ""}


async def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with args.manifest.open(newline="") as f:
        filings = latest_10k(list(csv.DictReader(f)))[: args.limit]
    slots = asyncio.Semaphore(args.workers)
    async with EdgarClient(load_user_agent()) as client:
        started = client.clock.now()
        records = await asyncio.gather(
            *(fetch_company(client, r, args.out, slots) for r in filings)
        )
        elapsed = client.clock.now() - started
        requests = len(client.request_log)
        stats = {
            "companies": len(records),
            "status": dict(Counter(r["status"] for r in records)),
            "exhibit_files": sum(len(r["exhibits"].split("|")) for r in records if r["exhibits"]),
            "report_years": dict(sorted(Counter(r["report_date"][:4] for r in records).items())),
            "requests": requests,
            "elapsed_s": round(elapsed, 1),
            "max_requests_in_any_1s_window": max_in_window(client.request_log, 1.0),
            "limiter": {"rate_per_s": client.bucket.rate, "capacity": client.bucket.capacity},
        }
    return list(records), stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/10k/manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/raw/ex21"))
    parser.add_argument("--stats", type=Path, default=Path("bench/ex21_download.json"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="first N companies (trial)")
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    records, stats = asyncio.run(run(args))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    stats = {
        "command": " ".join(["uv run python -m filinglens.kg.exhibits", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **stats,
        "failed": [f"{r['ticker']}: {r['error']}" for r in records if r["status"] == "failed"],
    }
    if args.limit is None:  # only full runs are evidence
        args.stats.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
