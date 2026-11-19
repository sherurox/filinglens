"""Split Item 1A of every downloaded 10-K into paragraphs (FL-M2 T1).

Input: the 10-K manifest written by filinglens.ingest.tenk (one row per document).
Output:
- data/interim/risk_paragraphs.parquet: one row per paragraph, keyed by para_id
  ("<accession>:<index>"), the ID every later label refers to;
- data/interim/risk_filings.csv: one row per filing with its status, for inspecting failures;
- bench/risk_corpus.json: counts and runtime (the evidence quoted in docs).

Each filing is parsed in a worker process (parsing is CPU-bound Python, so threads would be
serialized by the GIL). Workers start from a forkserver, not fork: forking a parent that already
runs threads (pandas/pyarrow, pytest) can copy a held lock into the child and deadlock.
A document lxml can't parse is recorded as parse_error; any other exception is a bug and stops
the run.
"""

import argparse
import csv
import json
import multiprocessing
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import lxml
import pandas as pd
from lxml import etree

from filinglens.extract.sections import split_risk_factors

FILING_FIELDS = ("cik", "ticker", "sector", "report_date", "accession", "path")


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Manifest rows whose document is on disk, sorted by path (a deterministic order)."""
    with path.open(newline="") as f:
        rows = [row for row in csv.DictReader(f) if Path(row["path"]).exists()]
    return sorted(rows, key=lambda row: row["path"])


def process_filing(row: dict[str, str]) -> tuple[dict, list[dict]]:
    """(filing record, paragraph records) for one 10-K. Runs in a worker process."""
    filing = {key: row[key] for key in FILING_FIELDS}
    try:
        section = split_risk_factors(Path(row["path"]).read_bytes())
    except (etree.LxmlError, UnicodeDecodeError, RecursionError) as exc:
        error = f"{type(exc).__name__}: {exc}"[:200]
        filing |= {"status": "parse_error", "method": "", "chars": 0, "paragraphs": 0}
        return filing | {"error": error}, []
    filing |= {
        "status": section.status,
        "method": section.method,
        "chars": section.chars,
        "paragraphs": len(section.paragraphs),
        "error": "",
    }
    paragraphs = [
        {
            "para_id": f"{row['accession']}:{p.index}",
            "cik": int(row["cik"]),
            "ticker": row["ticker"],
            "sector": row["sector"],
            "report_date": row["report_date"],
            "period_year": int(row["report_date"][:4]),
            "accession": row["accession"],
            "para_index": p.index,
            "heading": p.heading,
            "text": p.text,
            "n_words": len(p.text.split()),
        }
        for p in section.paragraphs
    ]
    return filing, paragraphs


def run(rows: list[dict[str, str]], workers: int) -> tuple[list[dict], list[dict]]:
    """All filings in manifest order; results keep that order whatever finishes first."""
    filings, paragraphs = [], []
    context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for filing, paras in pool.map(process_filing, rows, chunksize=4):
            filings.append(filing)
            paragraphs.extend(paras)
    return filings, paragraphs


def summarize(filings: list[dict], paragraphs: list[dict], elapsed: float, workers: int) -> dict:
    status = Counter(f["status"] for f in filings)
    methods = Counter(f["method"] for f in filings if f["status"] == "ok")
    years = Counter(p["period_year"] for p in paragraphs)
    words = [p["n_words"] for p in paragraphs]
    ok_by_sector = Counter(f["sector"] for f in filings if f["status"] == "ok")
    all_by_sector = Counter(f["sector"] for f in filings)
    return {
        "filings": len(filings),
        "status": dict(sorted(status.items())),
        "method_when_ok": dict(sorted(methods.items())),
        "companies": len({f["cik"] for f in filings}),
        "companies_with_ok_filing": len({f["cik"] for f in filings if f["status"] == "ok"}),
        "ok_filings_by_sector": {
            sector: f"{ok_by_sector[sector]}/{n}" for sector, n in sorted(all_by_sector.items())
        },
        "paragraphs": len(paragraphs),
        "paragraphs_by_period_year": dict(sorted(years.items())),
        "words_per_paragraph": {
            "median": statistics.median(words) if words else 0,
            "p95": statistics.quantiles(words, n=20)[-1] if len(words) >= 2 else 0,
            "max": max(words, default=0),
        },
        "elapsed_s": round(elapsed, 1),
        "workers": workers,
        "lxml": lxml.__version__,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/10k/manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/interim/risk_paragraphs.parquet"))
    parser.add_argument("--filings", type=Path, default=Path("data/interim/risk_filings.csv"))
    parser.add_argument("--stats", type=Path, default=Path("bench/risk_corpus.json"))
    parser.add_argument("--workers", type=int, default=8)
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    rows = read_manifest(args.manifest)
    started = time.perf_counter()
    filings, paragraphs = run(rows, args.workers)
    elapsed = time.perf_counter() - started

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(paragraphs).to_parquet(args.out, index=False)
    pd.DataFrame(filings).to_csv(args.filings, index=False)
    stats = {
        "command": " ".join(["uv run python -m filinglens.extract.risk_corpus", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **summarize(filings, paragraphs, elapsed, args.workers),
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
