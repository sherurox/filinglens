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

- Build command: `uv run python -m filinglens.spark.fsds_to_iceberg` (2026-11-16). Stats: `bench/fsds_to_iceberg.json`.
- Spark 4.1.4 (local[*], 24 threads, driver memory 10g), Iceberg runtime 1.12.0, Iceberg Hadoop catalog (local filesystem).
- Input: 35 quarters (2018q1–2026q3), 19,941,034,487 bytes of extracted text. Output: 2,825,014,937 bytes in `data/warehouse/`.
- Total runtime: 186.4 s (extraction 0.1 s: files were already extracted by the first, failed run).
- Laptop: [power mode], plugged in.

| Table | Raw rows | Rows | Duplicates removed | Rows without fy | Seconds | Partitioning |
|---|---|---|---|---|---|---|
| `local.fsds.sub` | 238,543 | 238,543 | 0 | 10,773 | 10.5 | identity(fy) |
| `local.fsds.tag` | 2,968,300 | 2,732,499 | 235,801 | 0 | 8.5 | none (small dimension) |
| `local.fsds.num` | 113,997,016 | 113,996,783 | 233 | 6,169,841 | 129.7 | identity(fy) |
| `local.fsds.pre` | 26,160,268 | 26,160,268 | 0 | 1,372,861 | 31.6 | identity(fy) |

`tag` (unpartitioned): 2,732,499 rows in 25 data files.

Rows and data files per fiscal-year partition (from Iceberg's `partitions` metadata table):

| fy | sub rows | num rows | num files | pre rows |
|---|---|---|---|---|
| 2004 | 1 | 100 | 1 | 46 |
| 2005 | 1 | 97 | 1 | 46 |
| 2006 | 1 | 97 | 1 | 46 |
| 2007 | 1 | 100 | 1 | 46 |
| 2008 | 1 | 98 | 1 | 45 |
| 2010 | 1 | 318 | 1 | 113 |
| 2011 | 5 | 1,056 | 1 | 410 |
| 2012 | 11 | 2,025 | 1 | 832 |
| 2013 | 18 | 2,796 | 1 | 1,171 |
| 2014 | 24 | 3,629 | 1 | 1,479 |
| 2015 | 68 | 12,902 | 1 | 4,662 |
| 2016 | 279 | 76,320 | 1 | 22,303 |
| 2017 | 7,005 | 3,663,226 | 1 | 794,737 |
| 2018 | 24,410 | 10,354,884 | 3 | 2,511,579 |
| 2019 | 24,470 | 11,419,511 | 3 | 2,646,184 |
| 2020 | 25,049 | 11,120,372 | 3 | 2,723,991 |
| 2021 | 28,271 | 11,718,306 | 3 | 3,032,012 |
| 2022 | 28,494 | 12,406,712 | 3 | 3,107,577 |
| 2023 | 26,739 | 13,167,975 | 3 | 2,951,997 |
| 2024 | 25,056 | 13,121,253 | 3 | 2,799,296 |
| 2025 | 24,249 | 13,596,417 | 3 | 2,722,191 |
| 2026 | 13,246 | 7,038,850 | 2 | 1,430,545 |
| 2027 | 370 | 119,898 | 1 | 36,099 |
| null | 10,773 | 6,169,841 | 2 | 1,372,861 |

Notes:
- `fy` is blank in `sub` for some filings; their `num` and `pre` rows sit in the null partition. dbt staging decides how to treat them (T7–T8).
- Dedupe keeps the row from the newest quarter per natural key. `tag` repeats standard taxonomy tags every quarter, which is where most duplicates come from.
- The first run failed with "No space left on device" while Spark spilled shuffle data to its default `/tmp`. Spill now goes to `data/spark-tmp` (`--local-dir`).

## Databricks Free Edition rerun

[T15]
