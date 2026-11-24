# Data

Every number here comes from a command's output; the command is listed next to it.

## Sources and terms

| Dataset | Source | Terms |
|---|---|---|
| SEC EDGAR: 10-K documents, Exhibit 21 documents and filing index pages, submissions JSON, `company_tickers.json` | sec.gov, data.sec.gov | SEC's webmaster FAQ states that government-created content and EDGAR public filings are free to access and reuse (https://www.sec.gov/files/about/webmaster-faq.htm). Access follows SEC fair access: at most 10 requests/s, with a declared User-Agent. |
| SEC Financial Statement Data Sets (quarterly zips) | sec.gov (published by SEC's Division of Economic and Risk Analysis) | Same SEC terms as the row above: EDGAR public data, free to access and reuse; downloaded at most 8 requests/s with a declared User-Agent. |

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
- Laptop power settings were not recorded for this run; they are recorded from the ratio benchmark (T13) on.

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

## Orchestration (Dagster)

- Code: `dagster_project/definitions.py` (Dagster 1.13.30, dagster-dbt 0.29.30). UI: `export DAGSTER_HOME="$PWD/data/dagster"`, then `uv run dagster dev -f dagster_project/definitions.py`.
- Assets:
  - `fsds_quarter_zip`: 35 quarterly partitions (2018q1–2026q3). Downloads and verifies one quarter's zip, then extracts it. `BackfillPolicy.single_run()` makes a backfill one run over the whole range, so all quarters share one EDGAR client and its 8 req/s token bucket. Separate runs per partition could exceed SEC's limit together.
  - `fsds/sub`, `fsds/num`, `fsds/tag`, `fsds/pre`: the Spark → Iceberg job as a subprocess, unpartitioned. It dedupes across quarters ("newest quarter wins"), so it rebuilds from all quarters. Its stats go to `data/dagster/` (the T6 evidence file stays untouched).
  - dbt: every model and test via dagster-dbt (`dbt build` on DuckDB). `source('fsds', …)` maps to the `fsds/*` assets, so the lineage needs no manual wiring.
- Asset graph: `docs/img/dagster_asset_graph.png`. Backfill: `docs/img/dagster_backfill.png`.
- Backfill (17 Nov 2026), evidence in `bench/dagster_backfill.json` (`uv run python dagster_project/backfill_report.py`):
  - The zips and extracted files of 2025q1–2025q4 were deleted first, so the backfill had real work to do.
  - Backfill `lpxnvhwm`: COMPLETED_SUCCESS, 3 runs:
    - download of 2025q1...2025q4 in one run: 24.7 s;
    - the `universe` seed: 12.9 s;
    - Spark + all dbt models and tests: 221.7 s (Spark: 183.4 s).
  - The re-downloaded zips have the same SHA-256 as the T5 manifest (4 of 4).
  - Spark rebuilt the same rows as T6: sub 238,543 · num 113,996,783 · tag 2,732,499 · pre 26,160,268.
- Limits:
  - Only the download is partitioned. A backfill of any range still rebuilds every Iceberg table and dbt model.
  - `backfill_policy` is marked beta in Dagster 1.13.
  - New quarters need a code change (fixed partition list). See `docs/LATER.md`.

## Databricks Free Edition rerun

- **What ran:** `notebooks/databricks_fsds_to_delta.py`, from a Databricks Git folder of this repo. It imports the same `build_tables` (typing, newest-quarter dedupe, fiscal-year join) and calls it with `table_format="delta"`.
- **Free Edition limits that shaped the run:**
  - It can't download from sec.gov, so the zips were **uploaded** to the Unity Catalog volume `/Volumes/workspace/filinglens/fsds/zips`.
  - It has no custom storage, so the output is **Delta** tables in `workspace.filinglens`, not the local Iceberg Hadoop catalog.
  - Compute is serverless only.
- **Input: a subset of 4 quarters,** 2025q1, 2025q2, 2025q3, 2025q4: 400,411,293 bytes, chosen to keep the upload and the daily compute quota small.
  - The notebook checked the uploaded zips' SHA-256 against the T5 manifest: {'2025q1': True, '2025q2': True, '2025q3': True, '2025q4': True}.
- **Local baseline on the same 4 zips** (`bench/fsds_to_iceberg_2025.json`): `uv run python -m filinglens.spark.fsds_to_iceberg --zips data/subset_2025/zips --extracted data/subset_2025/extracted --warehouse data/subset_2025/warehouse --stats bench/fsds_to_iceberg_2025.json`.

| Table | Rows, local Iceberg | Rows, Databricks Delta | Duplicates removed, local | Duplicates removed, Databricks | Rows without fy, local | Rows without fy, Databricks | Seconds, local | Seconds, Databricks |
|---|---|---|---|---|---|---|---|---|
| `sub` | 26,085 | 26,085 | 0 | 0 | 1,400 | 1,400 | 5.1 | 26.1 |
| `tag` | 329,861 | 329,861 | 18,909 | 18,909 | 0 | 0 | 2.1 | 10.8 |
| `num` | 14,621,401 | 14,621,401 | 138 | 138 | 902,986 | 902,986 | 22.1 | 33.0 |
| `pre` | 2,965,835 | 2,965,835 | 0 | 0 | 185,353 | 185,353 | 5.6 | 13.6 |

- **Rows identical on both platforms:** True.
- **Rows per fiscal-year partition identical** (`sub`, `num`, `pre`): {'sub': True, 'num': True, 'pre': True}. `num` has fiscal years 2018–2027, plus rows without a fiscal year.
- **Total time,** including unzipping:
  - local: 45.7 s (extract 5.2 s);
  - Databricks: 123.0 s (extract 24.0 s).
  - The local extraction writes to the laptop's disk. On Databricks it writes into the volume, a network file system.
- **Environments:**
  - Local: Spark 4.1.4, local[*], 24 threads, driver memory 10g, Iceberg runtime org.apache.iceberg:iceberg-spark-runtime-4.1_2.13:1.12.0.
  - Databricks: Free Edition serverless; runtime `client.6.5`, Spark 4.2.0, Python 3.12.3.
- **Differences, and why they don't change the data:**
  - The table format (Iceberg vs Delta) and the catalog (Hadoop catalog vs Unity Catalog) differ.
  - The compute differs (one laptop vs serverless).
  - The input bytes and the transform code are the same, and the outputs agree row for row and partition for partition.
- **Teardown:** `DROP SCHEMA workspace.filinglens CASCADE` (tables, volume and zips), or delete the schema in Catalog Explorer.

## FL-M2: labels, tables, entities, synthetic Q&A and the knowledge graph

Results and their discussion are in [EXPERIMENTS.md](EXPERIMENTS.md). This section describes the data.

### Deviation from the spec: silver labels, not hand labels

The spec asks for a hand-labeled gold set, a re-label a week later (kappa) and 100 hand-checked entity pairs. All three were replaced by automatic methods (decided 19 Nov 2026):
- **Risk paragraphs** are labeled by a local open-weight model following the committed guide ([LABELING_GUIDE.md](LABELING_GUIDE.md) v1.0, condensed into [labeling/silver_prompt.md](labeling/silver_prompt.md)). These are **silver labels**: no person labeled them, and every F1 against them measures agreement with the labeling model, not correctness.
- **Agreement** is Cohen's kappa between two different model labelers, not a human test-retest.
- **Entity-resolution precision** is checked against SEC's own identifier (CIK), not by hand.
- **Table cells** need no labels: the ground truth is the HTML each page image is rendered from.
- The files under `gold/` keep that directory name for the layout the spec describes; `gold/MANIFEST.json` records `"labels": "silver (model-labeled, not hand-labeled)"`.

### Risk-factor corpus: `data/interim/risk_paragraphs.parquet` (git-ignored)

- Command: `uv run python -m filinglens.extract.risk_corpus` (2026-11-17, 49.6 s, 8 workers). Stats: `bench/risk_corpus.json`.
- 2,258 10-Ks: Item 1A found in 2,207 (2,177 by an "Item 1A" heading, 30 by a bare "Risk Factors" heading), 43 too short, 8 without a heading; 295 of 300 companies have at least one.
- 201,516 paragraphs of at least 30 words (median 106 words, p95 271, max 835); from 21,504 (2018) to 27,833 (2024) per year.
- Known misses (LATER.md): annual-report layouts (GE, C, NEM, FDX before FY2021), Item 1A incorporated by reference from Exhibit 13 (BNY, USB, WFC), a 40-F filer (BNS).

### Silver risk-paragraph set: `gold/risk_paragraphs.{train,dev,test}.jsonl`

- **Sample** (`uv run python -m filinglens.extract.gold_sample`, seed 20261119; `bench/gold_sample.json`): 330 paragraphs from 330 filings and 204 companies: 270 random (9 per sector × year for 2019, 2020, 2022, 2024, 2025) plus 20 each boosted by keywords for people, health_pandemic and geopolitical_international (rare categories); at most one paragraph per filing and 3 per company; 40–400 words, no summary bullets.
- **Splits by company** (sha256 of seed and CIK, within sector): 294 companies, 176 train / 59 dev / 59 test (`gold/splits.csv`). A company's paragraphs are all in one split.
- **Labels:** category (11 risk categories + `not_a_risk`), severity (1 generic, 2 specific, 3 realized), first-seen year (earliest year the same risk appears in the company's earlier 10-Ks; candidates suggested by word overlap).
  - Primary labeler `ibm-granite/granite-4.1-3b` (FP8 at load): 327 of 330 answers follow the rules (`bench/silver_ibm-granite-granite-4-1-3b.json`, 91.1 s, peak 7,695 MiB).
  - Check labeler `Qwen/Qwen3-4B-AWQ`: 328 of 330 (`bench/silver_qwen-qwen3-4b-awq.json`, 310.1 s, peak 7,759 MiB).
  - Greedy decoding, constrained to a JSON schema; rule checks mark violations invalid instead of fixing them.
- **Kept** (`uv run python -m filinglens.extract.gold_set`; `gold/MANIFEST.json` with SHA-256 of every file): 327 rows: train 191, dev 71, test 65. The two labelers agree on the category for 139 / 52 / 48 of them.
- Categories (primary labeler): regulatory_legal 72, competition_demand 45, macro_market 41, operations_supply 40, financial_capital 30, health_pandemic 23, geopolitical_international 19, people 18, strategy_corporate 15, technology_cyber 13, climate_environment 11. Severity: 1 → 10, 2 → 268, 3 → 49.
- **Test guard:** `gold_set.load_split("test")` raises unless called with `allow_test=True`; every test-split run in `bench/` has `--allow-test` in its command.
- Limits: silver labels (see above); severity carries little signal (kappa 0.303); 65 test rows over 11 categories give wide intervals.

### Table pages: `gold/tables/`

- Command: `uv run python -m filinglens.extract.table_pages` (seed 20261119; `bench/table_pages.json`).
- 50 pages, one 10-K table each from 50 different companies (8–9 per sector, 2019–2025): 10 dev, 40 test. Rendered by Chromium (Playwright) at letter width and 2× scale; 0 pages wider than the page.
- Ground truth: each row's label and numbers, taken from the same HTML (716 rows and 2,012 numbers in all 50 pages; 1,708 cells on the 40 test pages). No hand labels.
- The page set was drawn twice: a fix to number parsing ("—%" as nil) changed which tables qualified. Only the second draw is used; EXPERIMENTS.md keeps the first as history.

### Entity resolution inputs and master table

- `company_tickers.json`: one request through the rate-limited client on 19 Nov 2026 (`uv run python -m filinglens.extract.entities fetch-tickers`); 7,993 CIKs, 1,433 of them with several tickers. A snapshot: names and tickers change.
- FSDS `sub.txt` names, read from the 35 zips already downloaded: 14,900 (CIK, name) pairs from 12,078 filers, 2018q1–2026q3 (`data/interim/entities/fsds_names.csv`).
- Master table `gold/company_master.csv` (`uv run python -m filinglens.extract.entities build`; `bench/entities.json`): the 300 companies with CIK, current name, tickers, exchanges, SIC, sector, EIN, state of incorporation, fiscal year end, former names and FSDS names; 181 have former names, 48 several tickers.

### Synthetic Q&A: `gold/synthetic/risk_qa.train.jsonl`

- **What:** 550 question-answer pairs about risk paragraphs, each with an exact evidence quote and the source paragraph ID. Every row has `synthetic: true` and `split: "train"`. They are never used for evaluation.
- **Generator:** `Qwen/Qwen3-4B-AWQ` (Apache-2.0), greedy, JSON schema, prompt [labeling/synthetic_qa_prompt.md](labeling/synthetic_qa_prompt.md) version 2. Commands: `synthetic_qa prepare`, then `generate` (GPU environment), then `finalize`; stats in `bench/synthetic_qa.json` and `bench/synthetic_qa_run.json` (156.0 s, peak 7,985 MiB).
- **Sources:** 176 train companies only. Of 66,329 eligible train paragraphs, 141 were dropped because the same text also appears at a dev or test company (copied boilerplate). 420 paragraphs drawn: 70 per sector, 84 per year, at most one per filing, from 159 companies.
- **Checks:** 840 pairs generated, 550 kept (0.6548) from 357 paragraphs and 152 companies. Dropped for: evidence longer than 40 words (160), evidence not a verbatim quote (89), answer too long (56), ticker missing from the question (40), question copies 8+ words from the source (39), duplicate question (3), answer not from the source (1).
- **Leakage test** (finalize writes nothing unless all are 0): rows from non-train companies 0; rows whose source text appears at a dev or test company 0; rows not marked synthetic/train 0.
- Types kept: factual 264, impact 233, cause 37, mitigation 16.
- Limits: the checks prove the evidence quote is real, not that the answer is fully correct; greedy decoding makes the questions formulaic; few cause and mitigation questions.

### Exhibit 21 and the knowledge graph

- **Exhibit 21** (`uv run python -m filinglens.kg.exhibits`; `bench/ex21_download.json`): each company's latest 10-K (299 for fiscal 2025, 1 for 2020). The filing index page gives each document's exhibit type. 290 companies have an EX-21 (300 files), 10 have none. 590 requests in 74.8 s, **max 8 in any 1-second window**; cached in `data/raw/ex21/`, so a rerun makes 0 requests.
- **Graph** (`uv run python -m filinglens.kg.build graph`; `bench/kg_stats.json`, 49.0 s):
  - Nodes 47,612: 300 companies, 44,407 subsidiaries, 2,905 people. Edges 47,757: has_subsidiary 44,407, has_director 2,463, has_officer 792, signed 94 (no recognized title), same_as 1 (a subsidiary that is itself one of the 300 companies, matched with the T13 resolver).
  - At least one subsidiary parsed for 282 companies; of the 290 with an EX-21, 8 gave no rows (one of them a PDF, which isn't parsed). Median 33.5 subsidiaries per company, max 2,772 (Ventas). Signers found for 293 companies (median 12); a CEO recognized for 257, a CFO for 253.
  - Top companies by degree: Ventas 2,787, HCA Healthcare 2,324, Comcast 1,489, Blackstone 1,438, Apollo Global Management 1,363.
  - 203 people sign the 10-K of 2 or more of the 300 companies (board interlocks); 215 company pairs share a person. The most connected sign for 3 companies each (for example Brad D. Smith, Marissa A. Mayer, Susan C. Schwab, Teri List).
  - 136 connected components; the largest has 28,813 nodes and 152 companies.
- **Postgres** (`uv run python -m filinglens.kg.build load`; `bench/kg_load.json`): tables `kg.nodes` and `kg.edges` in the compose Postgres 18.7, with primary keys, foreign keys and an index on `(dst, kind)`; loaded with COPY in 1.5 s. The SQL count of people on 2+ companies (203) equals NetworkX's.
- Limits: a person is identified by first and last name, so namesakes merge; attorney-in-fact signatures are skipped (for example SO); some titles keep a name prefix; 43 companies have no recognized CEO; subsidiary lists are one snapshot, and their parsing has no ground truth (counts, not accuracy).

### Models and libraries added in FL-M2

| Name | Version | License | Used for |
|---|---|---|---|
| ibm-granite/granite-4.1-3b | Hugging Face | Apache-2.0 | primary silver labeler (FP8 at load) |
| Qwen/Qwen3-4B-AWQ | Hugging Face | Apache-2.0 | check labeler; synthetic Q&A generator |
| Qwen/Qwen3-1.7B | Hugging Face | Apache-2.0 | prompted extraction baseline (M4's student base) |
| Qwen/Qwen3-VL-2B-Instruct | Hugging Face | Apache-2.0 | table extraction (FP8) |
| vLLM | 0.34.0 | Apache-2.0 | local inference (separate GPU environment, `bench/silver_gpu_env.txt`) |
| PyTorch | 2.14.0+cu132 | BSD-3-Clause | vLLM backend |
| transformers | 5.22.2 | Apache-2.0 | model loading |
| xgrammar | 0.2.8 | Apache-2.0 | JSON-schema constrained decoding |
| nvidia-ml-py | 13.615.71 | BSD-3-Clause | peak GPU memory sampling |
| lxml | 6.1.3 | BSD-3-Clause | HTML parsing (Item 1A, tables, EX-21, signatures) |
| pyarrow | 26.0.1 | Apache-2.0 | Parquet |
| Playwright + Chromium | 1.65.2 (Playwright's bundled Chromium) | Apache-2.0 (Playwright), BSD-3-Clause (Chromium) | rendering table pages |
| Pillow | 13.0.0 | MIT-CMU | images |
| pytesseract + Tesseract OCR | 0.3.13; Tesseract 5.5.0 (apt) | Apache-2.0 | OCR baseline |
| scikit-learn | 1.9.1 | BSD-3-Clause | macro-F1, kappa cross-check |
| NetworkX | 3.7 | BSD-3-Clause | graph analysis |
| Label Studio | 1.23.2 (Docker) | Apache-2.0 | set up in T4; unused after the silver-label decision; torn down |

## FL-M4: teacher labels and SFT data

Results are in [EXPERIMENTS.md](EXPERIMENTS.md). Labels are still silver (see FL-M2 above); the teacher labels are model output too.

### Teacher inputs: `data/interim/teacher/` (git-ignored)

- Command: `uv run python -m filinglens.train.teacher_inputs` (`bench/teacher_inputs.json`, manifest `82ab17e6…`).
- Pool: train-split companies only; 65,996 eligible paragraphs → 8,274 drawn round-robin over sector × year cells (smallest cell 240 rows), so every prefix up to 7,200 rows is exactly balanced. At most 10 paragraphs per filing, no repeated text.
- Leakage guard (writes nothing unless all are 0): rows from dev/test companies 0; rows whose text also appears at a dev or test company 0; silver-sample rows 0.
- Labeled by the teacher: the first 7,200 pool rows (240 per cell × 30), plus dev 71 and test 65 (the test predictions are scored only in the final comparison).

### Teacher labels: `data/interim/teacher/kaggle_out/` (git-ignored), summary in `bench/teacher_run/`

- Produced on Kaggle (2× T4) by `notebooks/kaggle_teacher_driver.py` with `train/kaggle_teacher.py`; inputs uploaded as the private Kaggle dataset `shreyask007/filinglens-teacher-inputs`; outputs downloaded from the notebook's output and saved as the private Kaggle dataset `shreyask007/filinglens-teacher-labels` (23 Nov 2026; 12 files, 7,835,512 bytes as listed by Kaggle: the notebook log plus `teacher/` with the dev, test and two pool shards, their `_run.json` summaries, `driver.json` and both GPU logs).
- Each row: the teacher's JSON answer, rule checks, and the category's first-token log-probabilities (top 20), from which `train/confidence.py` computes the chosen category's confidence.

### Filtered teacher labels and SFT arms: `data/interim/teacher/filtered.jsonl`, `data/interim/sft/` (git-ignored)

- Filter: `uv run python -m filinglens.train.filter_teacher` → 6,259 of 7,200 kept (`bench/teacher_filter.json`; rule and drop counts in EXPERIMENTS.md).
- SFT examples: `uv run python -m filinglens.train.sft_data --teacher data/interim/teacher/filtered.jsonl` (`bench/sft_data.json`): silver 191, teacher 6,259, both 6,450; data version (SHA-256 of `data/interim/sft/MANIFEST.json`) `c22a1742…`, recorded on every MLflow run.
- Format: the exact prompt the evaluation uses (`fields.conversations`), completion = the full answer JSON (rationale first), compact separators, ending in `<|im_end|>`. `not_a_risk` teacher rows are left out.
- Teacher rows 1,500–1,799 are the held-out slice for the Optuna objective; no Optuna trial trains on them.

### Models, libraries and services added in FL-M4

| Name | Version | License / terms | Used for |
|---|---|---|---|
| Qwen/Qwen3-8B-AWQ | Hugging Face | Apache-2.0 | teacher (Kaggle) |
| Qwen/Qwen3-1.7B | Hugging Face | Apache-2.0 | student base (already used in FL-M2) |
| transformers | 5.23.0 | Apache-2.0 | `Trainer`, model loading |
| PEFT | 0.22.0 | Apache-2.0 | LoRA |
| accelerate, datasets | 1.16.0, 5.2.0 | Apache-2.0 | `Trainer` dependencies |
| TRL | 1.18.2 | Apache-2.0 | installed, not used (see EXPERIMENTS.md) |
| bitsandbytes | 0.50.2 | MIT | 4-bit NF4 base |
| Liger-Kernel | 0.8.5 | BSD-2-Clause | fused linear cross-entropy |
| PyTorch | 2.15.0+cu132 (training env; `bench/train_env.txt`) | BSD-3-Clause | training |
| MLflow (server and client) | 3.18.0 | Apache-2.0 | tracking and model registry (Docker, `compose.yaml`) |
| Optuna | 5.1.0 | MIT | hyperparameter search |
| vLLM | 0.34.0 | Apache-2.0 | teacher (Kaggle) and evaluation (local GPU environment) |
| Kaggle CLI | 2.2.5 | Apache-2.0 | dataset upload, notebook runs, output download |
| six | not recorded | MIT | missing dependency of the Kaggle CLI |
| Kaggle (notebooks, datasets) | – | Kaggle Terms of Use; $0, no card; account phone-verified for GPU | teacher run on 2× T4 |

## FL-M3: retrieval corpus, queries and indexes

Results are in [EXPERIMENTS.md](EXPERIMENTS.md). Labels for retrieval are synthetic and graded automatically; no person judged relevance.

### Chunks: `data/interim/retrieval/chunks.parquet` (git-ignored)

- Command: `uv run python -m filinglens.retrieval.chunks` (2026-11-24, 22.3 s; `bench/chunks.json`). Tokenizer: `BAAI/bge-base-en-v1.5` at revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`.
- Input: the FL-M2 risk-factor corpus `data/interim/risk_paragraphs.parquet` (sha256 `e7fb9314…`), Item 1A only.
- 201,516 paragraphs → 203,200 chunks. A paragraph over the 510-token budget (512 minus `[CLS]` and `[SEP]`) is split at sentence boundaries into balanced parts: 1,656 paragraphs, 1,628 into 2 parts and 28 into 3. 199,183 chunks start with their heading (first 40 words). Tokens per chunk: p50 157, p95 354, p99 446, max 510.
- Chunk ID: `<para_id>#<part>`; each row keeps CIK, ticker, sector, fiscal year and accession.

### Reconciliation with FL-M2: 2,198 10-Ks and 294 companies, not 2,207 and 295

- FL-M2's `bench/risk_corpus.json` counts 2,207 filings with status `ok` (Item 1A found) at 295 companies, and 201,516 paragraphs.
- The paragraph file itself holds those same 201,516 paragraphs from **2,198 filings and 294 companies** (counted on 24 Nov 2026 from `risk_paragraphs.parquet`). So 9 filings were found but kept no paragraph of at least 30 words, and for one company that was every filing it had.
- The 6 universe companies with no paragraph: WFC, BNY, USB (Item 1A incorporated by reference from Exhibit 13), BNS (40-F filer), NEM and GEV. `risk_corpus.json` keeps no per-filing status, so the 9 filings aren't named here (`docs/LATER.md`).
- FL-M3 uses the paragraph file, so 2,198 / 294 is the retrieval corpus.

### Retrieval queries: `gold/retrieval/queries.{dev,test}.jsonl`

- Command: `uv run python -m filinglens.retrieval.queries` (`bench/ret_queries.json`). One query per source paragraph (2,037 Q&A pairs → 1,319 queries); companies split by hash (seed 3) into ret-dev and ret-test; `load_queries` refuses the test split without `allow_test`.
- Grades: 2 = the source paragraph's chunk holding the evidence quote; 1 = a chunk of the same company's other filings whose content words overlap it by at least 0.8 and that is at least half its size.
- Sources: 550 FL-M2 pairs (`gold/synthetic/risk_qa.train.jsonl`, batch `fl_m2_t14`) and 1,487 T10a pairs (`gold/synthetic/risk_qa.ret_test_extra.jsonl`, batch `ret_test_extra_v1`); every evidence quote matched its chunk verbatim.
- ret-dev: 182 queries, 77 companies, 4.984 grade-1 chunks per query on average (12 queries with none); sha256 `1edc1608…`, unchanged by T10a.
- ret-test: 1,137 queries, 75 companies (175 from the FL-M2 pairs, unchanged by T10a, plus 962 new), 4.694 grade-1 chunks per query (65 with none); sha256 `d99ab192…`.
- No company is in both splits. All questions come from FL-M2 train-split companies.

### Extra ret-test questions (T10a): `gold/synthetic/risk_qa.ret_test_extra.jsonl`

- Commands: `retrieval.extra_queries prepare` → `generate` (vLLM, `Qwen/Qwen3-4B-AWQ`, FL-M2 prompt v2, greedy JSON-schema decoding) → `finalize` (`bench/synthetic_qa_ret_test_extra.json`, `..._run.json`). Seed 20261124.
- Pool: 30,190 paragraphs of the 75 ret-test companies, minus text that also appears at an FL-M2 dev or test company (39), text of a paragraph that already has a Q&A pair (295) and duplicate text (6,414) → 23,442. 1,163 paragraphs drawn, spread across each company's years, for about 15 queries per company.
- 2,326 pairs generated, 1,487 kept by the FL-M2 checks (0.6393): the main drops were evidence length (453) and evidence not verbatim (273). 962 of 1,163 paragraphs kept at least one pair (0.8272).
- Leakage counts, all 0: rows from non-train companies, rows whose text appears at a dev or test company, rows not marked `synthetic` / train, rows from outside the ret-test companies, rows reusing an existing Q&A paragraph.

### Embeddings: `data/interim/retrieval/bge-base/` (git-ignored)

- `embed corpus` (`bench/embed_corpus.json`): 203,200 × 768 float32 (`chunks.npy`, sha256 `56465072…`; row order in `chunk_ids.txt`), CLS pooling as the model's pooling config says, L2-normalized; fp16 weights, minimum cosine to fp32 0.9999995 on the checked rows; 0 chunks truncated; 424.5 s (478.7 rows/s); peak VRAM allocated 1,306 MiB.
- `embed queries`: questions get the model card's query instruction; passages get none. ret-dev 182 rows; ret-test 1,137 rows, embedded once, after the A/B rule was committed (`bench/embed_queries_test.json`).

### Indexes (all local, rebuildable from the embeddings)

- FAISS: `data/interim/retrieval/faiss/flat.index` and `hnsw.index` (git-ignored).
- pgvector: table `retrieval.chunk_vectors` in the compose Postgres (HNSW index and a B-tree on `cik`).
- Qdrant: one collection in the compose volume `qdrantdata` (payload index on `cik`).
- Build settings and sizes: `bench/index_build.json`; teardown in the README.

### Peer recommender inputs

- Text: the chunks of each company's latest 10-K, 30,220 chunks; fiscal year 2025 for 288 companies, 2024 for 4, 2019 for 1, 2018 for 1.
- Ratios: `data/features/annual_ratios.parquet` (FL-M1), latest fiscal year per company. 13 of 294 companies have none; missing share per ratio from 0.0476 (growth) to 0.4082 (liabilities / equity).
- SIC codes: `dbt/seeds/universe.csv` (from SEC submissions). Splits: `gold/splits.csv` (FL-M2).

### Models, libraries and services added in FL-M3

| Name | Version | License | Used for |
|---|---|---|---|
| BAAI/bge-base-en-v1.5 | revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` | MIT | chunk and query embeddings |
| BAAI/bge-reranker-base | revision `2cfc18c9415c912f9d8155881c133215df768a70` | MIT | cross-encoder reranking |
| Qwen/Qwen3-4B-AWQ | Hugging Face | Apache-2.0 | extra ret-test questions (already used in FL-M2) |
| faiss-cpu | 1.16.0 | MIT | exact and HNSW indexes |
| pgvector (PostgreSQL extension) | 0.8.7, image `pgvector/pgvector:0.8.7-pg18-trixie` (PostgreSQL 18.7) | PostgreSQL License | HNSW in Postgres |
| pgvector (Python) | 0.5.1 | MIT | psycopg vector adapter |
| Qdrant | 1.20.2 (Docker image `qdrant/qdrant:v1.20.2`, telemetry off) | Apache-2.0 | vector database |
| qdrant-client | 1.20.0 | Apache-2.0 | Qdrant gRPC client |
| grpcio | 1.85.1 | Apache-2.0 | qdrant-client dependency |
| protobuf | 6.33.6 | BSD-3-Clause | qdrant-client dependency |
| h2, hpack, hyperframe | 4.4.1, 4.2.0, 6.1.0 | MIT | HTTP/2 for httpx (qdrant-client dependency) |
| portalocker | 4.4.2 | BSD-3-Clause | qdrant-client dependency |
| SciPy | 1.18.1 | BSD-3-Clause (classifier "BSD License") | noncentral-t power, root finding (already installed with scikit-learn) |

Versions and licenses are from the installed package metadata (`importlib.metadata`, 24 Nov 2026) and `bench/retrieval_env.txt`; the model licenses are from the model cards at the pinned revisions.
