# Data

Every number here comes from a command's output; the command is listed next to it.

## Sources and terms

| Dataset | Source | Terms |
|---|---|---|
| SEC EDGAR: 10-K documents, submissions JSON, `company_tickers.json` | sec.gov, data.sec.gov | SEC's webmaster FAQ states that government-created content and EDGAR public filings are free to access and reuse (https://www.sec.gov/files/about/webmaster-faq.htm). Access follows SEC fair access: at most 10 requests/s, with a declared User-Agent. |

## Company universe: `dbt/seeds/universe.csv`

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

## Data quality: dbt staging tests (DuckDB)

- Commands (2026-11-16, from `dbt/`): `uv run dbt run`, then `uv run dbt test --select <model>` one model at a time.
- DuckDB reads the Iceberg tables through `iceberg_scan` (current snapshot only) and returned the same row counts as Spark for all four tables.
- 4 staging views; 17 tests: 0 errors, 14 passed, 3 warnings (warnings are known data issues, reported by count):

| Check | Result |
|---|---|
| `sub`: blank fiscal_year | 10,773 filings (warn) |
| `num`: duration_quarters outside 0–4 | 25,077 rows (warn) |
| `num`: value is null | 4,891,874 rows (warn) |
| Unique keys: `sub` accession_number, `tag` (tag, version): exact; `pre` and `num`: by 64-bit key hash | pass |
| Every `num` and `pre` row belongs to a `sub` filing | pass |

- Resource guards (after the first `dbt build` filled the disk with DuckDB spill): `max_temp_directory_size` 40GB, `memory_limit` 6GB, DuckDB threads 8, dbt threads 2.

## Marts: star schema (dbt on DuckDB, universe only)

- Profile that decided the rules: `uv run python notebooks/profile_fsds_t8.py` → `bench/fsds_profile_t8.txt` (2026-11-17).
- Build (from `dbt/`): `uv run dbt seed`, `uv run dbt run --select path:models/intermediate path:models/marts` (2 views + 4 tables, 6.48 s), then tests one model group at a time.
- Rules:
  - Universe only: all 300 companies are in FSDS (10,214 of 238,543 filings).
  - Forms 10-K, 10-K/A, 10-Q, 10-Q/A. Blank fiscal years are almost all other forms (10-K: 1, 10-Q: 1 of 10,773).
  - Whole-company values: segments and coregistrant null (54,848,871 of 113,996,783 num rows).
  - Non-null values: nulls are empty in SEC's file (2024q1 raw: 122,967 empty, 0 unparseable).
  - duration_quarters 0–4 (instant to annual); 25,077 rows with longer spans are dropped.
  - "As last reported": latest filing wins; first_reported_value and was_restated kept. Before collapsing (10-K/10-Q only), 827,658 of 1,410,743 facts were reported more than once and 52,448 changed value.
  - Periods are calendar-based: fiscal_year ≠ year(period end) for 17,058 + 3,990 filings.
- Results:

| Table | Rows |
|---|---|
| `fct_financials` | 1,410,590 (51,866 restated; value dates 2012-02-29 to 2026-08-31) |
| `dim_company` | 300 (299 have facts) |
| `dim_period` | 662 |
| `dim_tag` | 101,822 tag versions |

- Value dates before 2018 come from prior-period comparatives inside 2018+ filings (FSDS starts at 2018q1).
- Universe companies with no facts:
- BNS (BANK OF NOVA SCOTIA, finance, CIK 9631): 0 10-K/10-Q filings in FSDS; forms present: 6-K, 40-F.
- Tests: 40 dbt tests in the project; marts and seed: 28 pass, 0 warnings.

## BigQuery sandbox: raw tables (universe subset)

- Command (repo root): `set -a; source .env; set +a`, then `uv run python -m filinglens.ingest.load_bigquery` → `bench/bigquery_load.json` (2026-11-17).
- What it loads: DuckDB reads the Iceberg tables and writes the universe's filings (all forms) to Parquet (zstd): `sub`, `num` and `pre` rows of those filings, and the `tag` rows for the (tag, version) pairs those numbers use. Then one `bq load --replace` per table into `filinglens_raw`.
- Why this way:
  - Load jobs are not DML, so the sandbox allows them. `--replace` makes a rerun overwrite instead of duplicate.
  - Local upload, no Cloud Storage bucket (a bucket would need billing).
  - Universe only: the sandbox caps storage at 10 GB, and BigQuery counts logical (uncompressed) bytes.

| Table | Rows exported = rows in BigQuery | Parquet bytes | BigQuery bytes | Load (s) |
|---|---|---|---|---|
| `sub` | 10,214 | 423,789 | 3,190,211 | 14.4 |
| `num` | 6,681,857 | 76,077,106 | 1,139,576,502 | 654.1 |
| `pre` | 1,300,570 | 13,615,688 | 201,737,122 | 123.3 |
| `tag` | 118,766 | 1,706,298 | 33,196,727 | 20.3 |
| **Total** | | 91,822,881 | 1,377,700,562 (13.8% of 10 GB) | 812.1 |

- BigQuery stores 15.0× the Parquet size for this data (logical bytes, read from `__TABLES__`, which costs no query bytes).
- Export: 3.4 s. Load: 812.1 s, limited by the home upload link (one 10-second sample during the `num` upload: 127 KiB/s).
- Parquet byte sizes differ slightly between runs (parallel writes with `preserve_insertion_order = false`); row counts were identical in all three runs.
- Teardown: `bq rm -r -f -d filinglens-515005:filinglens_raw`, or `terraform destroy` (README). Tables also expire after 60 days.

## dbt on BigQuery (sandbox)

- Command (from `dbt/`, after `set -a; source ../.env; set +a`): `uv run dbt build --target bigquery`, then `uv run python -m filinglens.bench.dbt_run_summary --out bench/dbt_build_bigquery.json` from the repo root (2026-11-17).
- Same project, second target:
  - Sources read `filinglens_raw` on BigQuery and the Iceberg tables on DuckDB (schema chosen from `target.type`).
  - `unique_key_by_hash` hashes with `adapter.dispatch`: `hash(...)` on DuckDB, `farm_fingerprint(to_json_string(struct(...)))` on BigQuery (`dbt/macros/key_hash.sql`).
- Result: 1 seed, 10 models, 40 tests: PASS=48, WARN=3, ERROR=0, SKIP=0, in 29.93 s. BigQuery processed 4,478,061,878 bytes and billed 4,722,786,304 (0.47% of the free 1 TB/month; at least 10 MB is billed per query). Guard: `maximum_bytes_billed` 20 GB per query.
- Same marts on both engines:

| Check | DuckDB | BigQuery |
|---|---|---|
| `fct_financials` rows | 1,410,590 | 1,410,590 |
| restated facts | 51,866 | 51,866 |
| companies with facts | 299 | 299 |
| value dates | 2012-02-29 to 2026-08-31 | 2012-02-29 to 2026-08-31 |
| `dim_company` / `dim_period` / `dim_tag` | 300 / 662 / 101,822 | 300 / 662 / 101,822 |

- Staging warnings are smaller on BigQuery because it holds only the universe's filings:

| Warning | DuckDB (all FSDS) | BigQuery (universe) |
|---|---|---|
| `duration_quarters` outside 0–4 | 25,077 | 703 |
| null `value` | 4,891,874 | 28,048 |
| blank `fiscal_year` | 10,773 | 260 |

- Found on the way:
  - The first BigQuery build failed in `dim_tag`. In BigQuery an unaliased table name is also a value (the whole row), so `select tag from ...tag` returned a STRUCT. DuckDB returned the column.
  - Fix: every staging model aliases its source (`as src`). `INFORMATION_SCHEMA.COLUMNS` now shows `tag` as STRING in `stg_fsds__tags` and `dim_tag`.
  - Model contracts are listed in `docs/LATER.md`.
- Docs site: `uv run dbt docs generate --target bigquery --static` writes one self-contained file, copied to `docs/dbt/index.html` (2.7 MB; open it in a browser). It includes the BigQuery catalog: column types and row counts.
- Teardown: `bq rm -r -f -d filinglens-515005:filinglens` (models and seed), or `terraform destroy` (README).

## Ratio features (FL-M1 step 8, D5)

- Code: `src/filinglens/features/ratios.py` computes 6 ratios per company and fiscal year end, three ways: pandas `apply` (a Python function per row), vectorized NumPy, and Numba (one `@njit` loop).
  - Margins: net margin, operating margin.
  - Leverage: liabilities / equity, assets / equity.
  - Growth: revenue growth, assets growth.
- Rules, identical in all three implementations; the tests require bit-for-bit equal results:
  - A ratio is NaN unless its denominator is > 0.
  - Growth is computed only against the same company 330–400 days earlier.
- Input: annual facts pivoted from `fct_financials`. Revenue uses the tag priority list from `sql/`; balances are taken at the fiscal year end.
- Benchmark command: `uv run python -m filinglens.features.bench_ratios --note "Best performance, plugged in"` → `bench/ratios.csv` (table), `bench/ratios.json` (environment, coverage), `data/features/annual_ratios.parquet` (git-ignored).
  - Each dataset: equality check, one warm-up, then the median of 5 end-to-end runs (DataFrame in, DataFrame out). pandas apply at 1M rows uses 3 runs.
  - The recorded `command` in this run lost the quotes around `--note` (fixed since); the note is under `environment.note`.
- Environment (17 Nov 2026): AMD Ryzen 9 8940HX (24 logical CPUs), WSL2, Python 3.12.16, numpy 2.5.5, pandas 3.1.1, numba 0.68.1. On AC power; Windows power mode Best performance (an overlay on the Balanced scheme, which `powercfg` reports).

| Dataset | Rows | pandas apply (s) | NumPy (s) | Numba (s) | NumPy vs apply | Numba vs apply |
|---|---|---|---|---|---|---|
| real annual | 2,852 | 0.046293 | 0.00028 | 0.000163 | ×165.2 | ×283.68 |
| synthetic | 10,000 | 0.161278 | 0.000414 | 0.000249 | ×389.55 | ×648.03 |
| synthetic | 100,000 | 1.589656 | 0.002289 | 0.001334 | ×694.6 | ×1191.83 |
| synthetic | 1,000,000 | 16.650744 | 0.032678 | 0.043817 | ×509.54 | ×380.0 |

- Numba compile time (fresh, uncached kernel): 0.318 s.
  - At real-data size, Numba saves 0.000117 s per call over NumPy, so the compile pays off after about 2,700 calls (arithmetic on the numbers above).
- Numba is fastest up to 100k rows; NumPy is fastest at 1M.
  - Not yet measured: whether building the output DataFrame from Numba's (n × 6) array dominates at that size (see `docs/LATER.md`).
- Real data: 2,852 company-years from 283 of the 300 companies (value dates 2015-11-30 to 2026-07-31). The other 17 use none of the priority revenue tags (mostly banks).
- Non-null counts:
  - inputs: revenue 2,852, net income 2,559, operating income 2,284, assets 2,671, liabilities 1,809, equity 2,486;
  - features: net margin 2,556, operating margin 2,281, liabilities/equity 1,560, assets/equity 2,214, revenue growth 2,561, assets growth 2,382.

## Databricks Free Edition rerun

[T15]
