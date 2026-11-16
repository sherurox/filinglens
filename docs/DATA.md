# Data

Every number here comes from a command's output; the command is listed next to it.

## Sources and terms

| Dataset | Source | Terms |
|---|---|---|
| SEC EDGAR: 10-K documents, submissions JSON, `company_tickers.json` | sec.gov, data.sec.gov | SEC's webmaster FAQ states that government-created content and EDGAR public filings are free to access and reuse (https://www.sec.gov/files/about/webmaster-faq.htm). Access follows SEC fair access: at most 10 requests/s, with a declared User-Agent. |

## Company universe: `docs/universe.csv`

- Command: `uv run python -m filinglens.ingest.universe` (2026-11-16, 4m6.7s).
- 300 companies, 50 per sector: finance, manufacturing, mining_energy, retail, services, transport_utilities (SIC divisions).
- Selection: SEC ticker-file order (roughly largest first; SEC does not document the order), one row per CIK, SPACs (SIC 6770) excluded, at least one original 10-K for fiscal 2018–2025.
- Known bias: large companies.

## 10-K documents: `data/raw/10k/` (git-ignored)

- Command: `uv run python -m filinglens.ingest.tenk` (2026-11-16). Stats: `bench/edgar_10k_download.json`.
- 2,258 original 10-Ks (fiscal 2018–2025) for 300 companies; 0 failed; 11,703,535,657 bytes.
- 2,865 requests (300 submissions + 307 older filing pages + 2,258 documents) in 362.4 s: mean 7.91 requests/s, **max 8 in any 1-second window** (limiter 8/s, capacity 1; SEC limit 10).
- Rerun: 0 downloaded, 2,258 skipped, 307 requests (older filing pages are not cached).
- Layout: `data/raw/10k/<cik>/<report date>_<accession>.<ext>`; manifest `data/raw/10k/manifest.csv`; submissions cache `data/raw/submissions/CIK##########.json`.

## Financial Statement Data Sets and Iceberg tables

- Download command: `uv run python -m filinglens.ingest.fsds` (2026-11-16). Stats: `bench/fsds_download.json`; SHA-256 per zip in `data/raw/fsds/manifest.csv`.
- 35 quarterly zips, 2018q1–2026q3; 0 failed; 3,428,814,515 bytes; 35 requests in 88.0 s (max 4 in any 1-second window). Rerun: 0 requests.
- Each zip is verified (sub/num/tag/pre present, CRC check of every member) before it is kept.

[T6: Iceberg tables: row counts, partition layout, runtime]

## Databricks Free Edition rerun

[T15]
