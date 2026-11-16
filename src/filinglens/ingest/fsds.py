"""Download SEC Financial Statement Data Sets (quarterly zips), verified and resumable."""

import argparse
import asyncio
import csv
import hashlib
import json
import os
import sys
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, fields
from datetime import UTC, date, datetime
from pathlib import Path

import httpx

from filinglens.ingest.edgar_client import EdgarClient, load_user_agent
from filinglens.ingest.rate_limit import max_in_window

FSDS_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{quarter}.zip"
REQUIRED_MEMBERS = frozenset({"sub.txt", "num.txt", "tag.txt", "pre.txt"})


@dataclass
class QuarterResult:
    quarter: str
    url: str
    path: str
    status: str  # downloaded | skipped_existing | not_published | failed
    bytes: int = 0
    sha256: str = ""
    error: str = ""


def quarters(start: str, end: str) -> list[str]:
    """All quarters from `start` to `end` inclusive, e.g. quarters("2025q3", "2026q1")."""
    year, q = int(start[:4]), int(start[5])
    end_year, end_q = int(end[:4]), int(end[5])
    out = []
    while (year, q) <= (end_year, end_q):
        out.append(f"{year}q{q}")
        year, q = (year + 1, 1) if q == 4 else (year, q + 1)
    return out


def last_complete_quarter(today: date) -> str:
    """The quarter before the one `today` is in. SEC publishes each set after quarter end."""
    q = (today.month - 1) // 3 + 1
    return f"{today.year - 1}q4" if q == 1 else f"{today.year}q{q - 1}"


def validate_zip(path: Path) -> str:
    """Empty string if `path` is a readable zip holding sub/num/tag/pre; else the problem."""
    try:
        with zipfile.ZipFile(path) as zf:
            missing = REQUIRED_MEMBERS - set(zf.namelist())
            if missing:
                return f"missing members {sorted(missing)}"
            bad = zf.testzip()  # reads every member and checks its CRC
            if bad:
                return f"corrupt member {bad}"
    except zipfile.BadZipFile as exc:
        return f"BadZipFile: {exc}"
    return ""


def sha256_of(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


async def fetch_quarter(
    client: EdgarClient, quarter: str, root: Path, slots: asyncio.Semaphore
) -> QuarterResult:
    """Download and verify one quarter, unless a verified copy is already on disk."""
    url, path = FSDS_URL.format(quarter=quarter), root / f"{quarter}.zip"
    result = QuarterResult(quarter=quarter, url=url, path=str(path), status="failed")
    if path.exists():
        result.status, result.bytes = "skipped_existing", path.stat().st_size
        result.sha256 = sha256_of(path)
        return result
    async with slots:
        try:
            response = await client.get(url)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                result.status = "not_published"
            result.error = f"HTTP {exc.response.status_code}"
            return result
        except httpx.HTTPError as exc:
            result.error = f"{type(exc).__name__}: {exc}"[:200]
            return result
    root.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(response.content)
    problem = validate_zip(tmp)  # verify before rename: "exists" always means "verified"
    if problem:
        tmp.unlink()
        result.error = problem
        return result
    os.replace(tmp, path)
    result.status, result.bytes, result.sha256 = "downloaded", path.stat().st_size, sha256_of(path)
    return result


async def run(
    client: EdgarClient, wanted: list[str], root: Path, workers: int = 4
) -> list[QuarterResult]:
    slots = asyncio.Semaphore(workers)
    return list(await asyncio.gather(*(fetch_quarter(client, q, root, slots) for q in wanted)))


def summarize(results: list[QuarterResult], client: EdgarClient, elapsed: float) -> dict:
    status = Counter(r.status for r in results)
    return {
        "quarters_requested": len(results),
        "first_quarter": results[0].quarter if results else "",
        "last_quarter": results[-1].quarter if results else "",
        "downloaded": status["downloaded"],
        "skipped_existing": status["skipped_existing"],
        "not_published": status["not_published"],
        "failed": status["failed"],
        "bytes_on_disk": sum(r.bytes for r in results),
        "requests": len(client.request_log),
        "elapsed_s": round(elapsed, 1),
        "max_requests_in_any_1s_window": max_in_window(client.request_log, 1.0),
    }


def write_manifest(results: list[QuarterResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in fields(QuarterResult)])
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)


async def _main(args: argparse.Namespace) -> None:
    wanted = quarters(args.start, args.end or last_complete_quarter(date.today()))
    async with EdgarClient(load_user_agent()) as client:
        started = client.clock.now()
        results = await run(client, wanted, args.out, args.workers)
        stats = summarize(results, client, client.clock.now() - started)
    write_manifest(results, args.out / "manifest.csv")
    stats = {
        "command": " ".join(["uv run python -m filinglens.ingest.fsds", *sys.argv[1:]]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **stats,
        "problems": {r.quarter: r.error for r in results if r.error},
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2018q1")
    parser.add_argument("--end", default=None, help="default: the last complete quarter")
    parser.add_argument("--out", type=Path, default=Path("data/raw/fsds"))
    parser.add_argument("--stats", type=Path, default=Path("bench/fsds_download.json"))
    parser.add_argument("--workers", type=int, default=4)
    asyncio.run(_main(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
