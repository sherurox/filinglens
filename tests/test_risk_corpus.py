"""Corpus runner: statuses per filing, stable paragraph IDs, parse errors recorded, end to end."""

import csv
import json

import pandas as pd
import pytest

from filinglens.extract.risk_corpus import main, process_filing, read_manifest, summarize

BODY = " ".join(["word"] * 500)  # 500 words: above min_chars=2000 in two paragraphs
GOOD = (
    "<html><body><p>Item 1A. Risk Factors</p><p><b>Demand may fall.</b></p>"
    f"<p>{BODY}</p><p>{BODY} more</p><p>Item 1B. Unresolved Staff Comments</p></body></html>"
)
SHORT = "<html><body><p>Item 1A. Risk Factors</p><p>Not applicable.</p><p>Item 2.</p></body></html>"


def write_filing(tmp_path, cik, accession, html):
    path = tmp_path / "10k" / str(cik) / f"2024-12-31_{accession}.htm"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(html.encode())
    return {
        "cik": str(cik),
        "ticker": f"T{cik}",
        "sector": "services",
        "report_date": "2024-12-31",
        "accession": accession,
        "path": str(path),
    }


def write_manifest(tmp_path, rows):
    path = tmp_path / "10k" / "manifest.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*rows[0], "status"])
        writer.writeheader()
        writer.writerows({**row, "status": "downloaded"} for row in rows)
    return path


def test_process_filing_gives_stable_ids_and_metadata(tmp_path):
    row = write_filing(tmp_path, 1, "0000000001-25-000001", GOOD)
    filing, paragraphs = process_filing(row)
    assert (filing["status"], filing["method"], filing["paragraphs"]) == ("ok", "item", 2)
    assert [p["para_id"] for p in paragraphs] == [
        "0000000001-25-000001:0",
        "0000000001-25-000001:1",
    ]
    first = paragraphs[0]
    assert (first["cik"], first["period_year"], first["heading"], first["n_words"]) == (
        1,
        2024,
        "Demand may fall.",
        500,
    )


def test_unparseable_document_is_recorded_not_raised(tmp_path):
    row = write_filing(tmp_path, 2, "0000000002-25-000001", "   ")
    filing, paragraphs = process_filing(row)
    assert (filing["status"], paragraphs) == ("parse_error", [])
    assert filing["error"].startswith("ParserError")


def test_manifest_rows_without_a_file_are_skipped(tmp_path):
    rows = [write_filing(tmp_path, 3, "0000000003-25-000001", GOOD)]
    rows.append({**rows[0], "accession": "missing", "path": str(tmp_path / "nope.htm")})
    assert [r["accession"] for r in read_manifest(write_manifest(tmp_path, rows))] == [
        "0000000003-25-000001"
    ]


def test_summarize_counts():
    filings = [
        {"cik": "1", "sector": "retail", "status": "ok", "method": "item"},
        {"cik": "1", "sector": "retail", "status": "too_short", "method": "item"},
        {"cik": "2", "sector": "finance", "status": "no_heading", "method": ""},
    ]
    paragraphs = [{"period_year": 2024, "n_words": n} for n in (40, 60, 80)]
    stats = summarize(filings, paragraphs, elapsed=1.234, workers=2)
    assert stats["status"] == {"no_heading": 1, "ok": 1, "too_short": 1}
    assert stats["ok_filings_by_sector"] == {"finance": "0/1", "retail": "1/2"}
    assert (stats["companies"], stats["companies_with_ok_filing"]) == (2, 1)
    assert stats["words_per_paragraph"]["median"] == 60
    assert stats["elapsed_s"] == 1.2


def test_main_end_to_end_with_two_workers(tmp_path):
    pytest.importorskip("pyarrow")
    rows = [
        write_filing(tmp_path, 4, "0000000004-25-000001", GOOD),
        write_filing(tmp_path, 5, "0000000005-25-000001", SHORT),
    ]
    out, filings, stats = tmp_path / "p.parquet", tmp_path / "f.csv", tmp_path / "s.json"
    main(
        [
            *("--manifest", str(write_manifest(tmp_path, rows)), "--out", str(out)),
            *("--filings", str(filings), "--stats", str(stats), "--workers", "2"),
        ]
    )
    assert len(pd.read_parquet(out)) == 2
    assert list(pd.read_csv(filings)["status"]) == ["ok", "too_short"]
    assert json.loads(stats.read_text())["status"] == {"ok": 1, "too_short": 1}
