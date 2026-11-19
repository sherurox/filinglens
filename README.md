# FilingLens

Reads US public-company filings from SEC EDGAR and answers analyst questions about them with cited numbers. Full specification: [docs/SPEC.md](docs/SPEC.md).

**Status:** milestones **FL-M1 (data foundation)** and **FL-M2 (labels, OCR and extraction)** are complete. The later milestones (retrieval, fine-tuning, serving, agents, evaluation) are not built yet.

## FL-M1 results

Every number below comes from a committed output file, listed next to it. Details and commands are in [docs/DATA.md](docs/DATA.md).

| Part | Result | Evidence |
|---|---|---|
| Rate-limited EDGAR client | 2,258 original 10-Ks for 300 companies; 2,865 requests, **max 8 in any 1-second window** (SEC's limit: 10). A fake-clock test proves the limiter never exceeds it. | `bench/edgar_10k_download.json`, `tests/test_rate_limit.py` |
| FSDS → Iceberg (PySpark) | 35 quarters (2018q1–2026q3); `num` 113,996,783 rows, `pre` 26,160,268, `tag` 2,732,499, `sub` 238,543; partitioned by fiscal year; 186.4 s | `bench/fsds_to_iceberg.json` |
| Databricks Free Edition rerun | 2025q1–q4: identical rows and rows per fiscal year as the local run on the same zips; 123.0 s (serverless, Delta) vs 45.7 s (laptop, Iceberg) | `bench/fsds_to_delta_databricks.json`, `bench/fsds_to_iceberg_2025.json` |
| dbt star schema | `fct_financials` 1,410,590 rows + 3 dimensions; 40 tests. `dbt build` on BigQuery: PASS=48, WARN=3, ERROR=0; marts identical to DuckDB | `bench/dbt_build_bigquery.json`, [dbt docs site](docs/dbt/index.html) |
| BigQuery (sandbox) + Terraform | Universe subset loaded (1,377,700,562 bytes, 13.8% of the 10 GB cap); datasets created by Terraform, state in HCP Terraform | `bench/bigquery_load.json`, `infra/terraform/gcp/evidence/` |
| Hand-written SQL (Postgres 18) | 8 queries, each with EXPLAIN ANALYZE before/after one index: from ×2.99 to ×1873.18 | [sql/RESULTS.md](sql/RESULTS.md), [sql/README.md](sql/README.md) |
| Ratio features, three ways | 1M rows: pandas `apply` 16.65 s, NumPy 0.0327 s, Numba 0.0438 s; Numba fastest up to 100k rows; all three bit-for-bit equal | `bench/ratios.csv`, `bench/ratios.json` |
| Dagster orchestration | Download → Spark → dbt assets; backfill of 2025q1–q4 succeeded, re-downloaded zips byte-identical to the originals, Spark rows identical | `bench/dagster_backfill.json`, [asset graph](docs/img/dagster_asset_graph.png) |

**"Done when" (spec §4 M1):**
- [x] **D1:** the downloader test proves it never exceeds the rate limit (fake clock).
- [x] **D2:** the Iceberg tables exist, and DATA.md lists row counts, the partition layout and the runtime, both locally and on Databricks.
- [x] **D3:** `dbt build` passes in BigQuery, and the docs site is generated.
- [x] **D4:** 8 queries in `sql/`, each with EXPLAIN before/after and timing.
- [x] **D5:** the ratio benchmark table is in `bench/`.

## FL-M2 results

Labels in FL-M2 are **silver**: written by local open-weight models following the committed labeling guide, not by a person. F1 against them measures agreement with the labeling model. The spec asked for hand labels; the deviation and its consequences are in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) and [docs/DATA.md](docs/DATA.md).

| Part | Result | Evidence |
|---|---|---|
| Risk-factor corpus | Item 1A found in 2,207 of 2,258 10-Ks (295 of 300 companies); 201,516 paragraphs | `bench/risk_corpus.json` |
| Silver label set | 330 sampled paragraphs → 327 rows, split by company: train 191, dev 71, test 65; test split guarded in code | `gold/MANIFEST.json`, `bench/silver_*.json` |
| Labeler agreement | Cohen's kappa, Granite 4.1 3B vs Qwen3-4B (326 pairs): category 0.697 (95% CI 0.644–0.747), severity 0.303, first-seen year (quadratic) 0.781 | `bench/kappa.json` |
| Table extraction (40 test pages, 1,708 cells; truth from the HTML) | Cell accuracy: Tesseract OCR **0.7254**, Qwen3-VL-2B 0.5351, VLM with an OCR hint 0.4619 | `bench/tables_ocr_test_psm4.json`, `bench/tables_vlm_*_test_p2.json` |
| Prompted extraction baseline (what M4 must beat) | Qwen3-1.7B zero-shot on 65 test paragraphs: category macro-F1 **0.6295** (0.474–0.712), severity macro-F1 0.2832, first-seen year exact 0.4308 | `bench/fields_zero_test.json` |
| Entity resolution | Names filers used (FSDS) resolved to CIKs, checked against the filer's own CIK: precision 0.9861, recall 0.7884 on the held-out half; **97 of 100** sampled matches correct (Wilson 95% CI 0.916–0.990); master table of the 300 companies | `bench/entity_eval.json`, `bench/entity_precision.json`, `gold/company_master.csv` |
| Synthetic Q&A | 550 checked pairs from train-split companies only, all `synthetic=true`; leakage test 0 / 0 / 0 | `gold/synthetic/risk_qa.train.jsonl`, `bench/synthetic_qa.json` |
| Knowledge graph | 47,612 nodes (300 companies, 44,407 subsidiaries, 2,905 people), 47,757 edges, in Postgres edge tables; 203 people sign for 2+ of the 300 companies (SQL and NetworkX agree) | `bench/kg_stats.json`, `bench/kg_load.json` |

**"Done when" (spec §4 M2):**
- [x] **D1:** silver set exported with train/dev/test splits; the test split is never used for prompts or training (deviation: silver, not hand-labeled).
- [x] **D2:** kappa reported, between two model labelers rather than a human re-label (deviation); the labeling guide was committed before any label.
- [x] **D3:** table-extraction accuracy and baseline extraction F1 are in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).
- [x] **D4:** entity-resolution precision on 100 pairs, checked against SEC CIKs rather than by hand (deviation).

## Layout

```
src/filinglens/
  ingest/     EDGAR client (asyncio + httpx, token bucket), universe, 10-K and FSDS downloaders, BigQuery load
  spark/      FSDS -> Iceberg (local) or Delta (Databricks), partitioned by fiscal year
  sql/        copy of the marts to Postgres; EXPLAIN before/after runner
  features/   ratio features (pandas apply, NumPy, Numba) and their benchmark
  extract/    Item 1A splitter, sampling, silver labels, kappa, table pages, OCR and VLM table
              extraction, prompted baseline, entity resolution, synthetic Q&A
  kg/         Exhibit 21 download, EX-21 and signature parsers, graph build (NetworkX, Postgres)
  bench/      dbt run_results summary
dagster_project/  Dagster assets and the backfill report
dbt/              staging -> intermediate -> marts (DuckDB and BigQuery targets), tests, docs
gold/             silver risk-paragraph splits, table pages with truth, master company table,
                  synthetic Q&A (all committed; "gold" is the spec's directory name)
sql/              8 hand-written queries, their index changes, plans, RESULTS.md
infra/terraform/gcp/  BigQuery datasets (state in HCP Terraform)
notebooks/        data profiling; the Databricks notebook
bench/            committed benchmark outputs (each records the command that made it)
docs/             SPEC, DATA, EXPERIMENTS, LABELING_GUIDE, LATER, labeling prompts, dbt docs site
data/             git-ignored: downloads, Iceberg warehouse, DuckDB file
```

## Setup

- **Requirements:** Linux or WSL2, [uv](https://docs.astral.sh/uv/) (it installs Python 3.12), and JDK 21 for Spark. Optional, depending on the step: Docker (Postgres), the gcloud CLI and Terraform (BigQuery), a Databricks Free Edition account.
- **Install and configure:**
  - `uv sync`
  - Create `.env` from `.env.example`: `SEC_USER_AGENT` (your name and email, required by SEC), `POSTGRES_PASSWORD`, and `GCP_PROJECT_ID`, `TF_VAR_project_id`, `TF_CLOUD_ORGANIZATION` for the BigQuery steps.
  - Load it with `set -a; source .env; set +a`.
- **Checks, as in CI:** `uv run ruff format --check . && uv run ruff check . && uv run pytest`

## Running FL-M1

From the repo root. Downloads are resumable, and every step can be rerun.

```bash
uv run python -m filinglens.ingest.universe            # 300-company universe -> dbt/seeds/universe.csv
uv run python -m filinglens.ingest.tenk                # 10-K documents (8 requests/s max)
uv run python -m filinglens.ingest.fsds                # FSDS quarterly zips, verified
uv run python -m filinglens.spark.fsds_to_iceberg      # Iceberg tables in data/warehouse
cd dbt && uv run dbt build && cd ..                    # star schema on DuckDB

# BigQuery sandbox
(cd infra/terraform/gcp && terraform init && terraform apply)
uv run python -m filinglens.ingest.load_bigquery       # universe subset -> filinglens_raw
cd dbt && uv run dbt build --target bigquery && uv run dbt docs generate --target bigquery --static && cd ..

# Postgres + hand-written SQL
docker compose up -d --wait
uv run python -m filinglens.sql.load_postgres
uv run python -m filinglens.sql.explain_bench

# Ratio features benchmark
uv run python -m filinglens.features.bench_ratios --note "power mode, plugged in"

# Dagster (UI on http://localhost:3000; launch backfills from the UI)
export DAGSTER_HOME="$PWD/data/dagster" && mkdir -p "$DAGSTER_HOME"
uv run dagster dev -f dagster_project/definitions.py
uv run python dagster_project/backfill_report.py
```

Databricks: upload the zips to a Unity Catalog volume, add this repo as a Git folder, and run `notebooks/databricks_fsds_to_delta.py` on serverless compute (see docs/DATA.md).

## Running FL-M2

GPU steps run in a separate vLLM environment (`~/venvs/fl-gpu`, versions in `bench/silver_gpu_env.txt`; 8 GB laptop GPU, so `--gpu-memory-utilization 0.80 --enforce-eager`). Every test-split step needs `--allow-test` and was run once.

```bash
GPU=~/venvs/fl-gpu/bin/python
uv run python -m filinglens.extract.risk_corpus                  # Item 1A paragraphs
uv run python -m filinglens.extract.gold_sample                  # 330-paragraph sample, company splits
$GPU -m filinglens.extract.silver_label --model ibm-granite/granite-4.1-3b --quantization fp8 --gpu-memory-utilization 0.80 --enforce-eager
$GPU -m filinglens.extract.silver_label --model Qwen/Qwen3-4B-AWQ --quantization awq_marlin --gpu-memory-utilization 0.80 --enforce-eager
uv run python -m filinglens.extract.gold_set                     # silver splits + MANIFEST
uv run python -m filinglens.extract.agreement                    # kappa
uv run python -m filinglens.extract.table_pages                  # 50 table pages + truth
uv run python -m filinglens.extract.tables ocr --split test --psm 4 --allow-test
$GPU -m filinglens.extract.tables_vlm --variant image --split test --allow-test --quantization fp8
uv run python -m filinglens.extract.tables score --split test --allow-test --pred data/interim/tables/tables_vlm_image_test_p2.jsonl
$GPU -m filinglens.extract.fields generate --variant zero --split test --allow-test
uv run python -m filinglens.extract.fields score --variant zero --split test --allow-test
uv run python -m filinglens.extract.entities fetch-tickers        # 1 request
uv run python -m filinglens.extract.entities build
uv run python -m filinglens.extract.entity_eval
uv run python -m filinglens.extract.synthetic_qa prepare
$GPU -m filinglens.extract.synthetic_qa generate --model Qwen/Qwen3-4B-AWQ --gpu-memory-utilization 0.80 --enforce-eager
uv run python -m filinglens.extract.synthetic_qa finalize
uv run python -m filinglens.kg.exhibits                          # Exhibit 21 (8 requests/s max)
uv run python -m filinglens.kg.build graph
docker compose up -d --wait postgres && uv run python -m filinglens.kg.build load
```

## Resources and teardown ($0)

| Resource | Created by | Free because | Teardown |
|---|---|---|---|
| BigQuery datasets `filinglens`, `filinglens_raw` (project with the BigQuery sandbox) | `infra/terraform/gcp` (state in HCP Terraform, local execution) | Sandbox: no billing account; 10 GB storage, 1 TB queries/month; tables expire after 60 days | `cd infra/terraform/gcp && terraform destroy` |
| HCP Terraform workspace `filinglens-gcp` | `terraform init` | Free tier: up to 500 managed resources | Delete the workspace in HCP Terraform (Settings → Destruction and Deletion) |
| Google Cloud project | Google Cloud console | No billing attached | `gcloud projects delete <project-id>` |
| Databricks schema `workspace.filinglens` (Delta tables, volume with the zips) | Catalog Explorer + `notebooks/databricks_fsds_to_delta.py` | Databricks Free Edition: no card, serverless with a daily quota | `DROP SCHEMA workspace.filinglens CASCADE` (done 17 Nov 2026) |
| Local Postgres (`compose.yaml`: schemas `marts` and `kg`) and Dagster home (`data/dagster`) | `docker compose up`, `dagster dev` | Local | `docker compose rm -sf postgres && docker volume rm filinglens_pgdata`; `rm -rf data/dagster` |
| Local Label Studio (`compose.yaml`) | `docker compose up -d labelstudio` (FL-M2 T4) | Local | `docker compose rm -sf labelstudio && docker volume rm filinglens_lsdata` (done 19 Nov 2026) |
| Hugging Face model cache (4 models, FL-M2) | vLLM downloads | Local disk | Remove the four `models--*` folders under `~/.cache/huggingface/hub` |

## Provenance and license

- [PROVENANCE.md](PROVENANCE.md) lists every file as AI-assisted, hand-written or generated.
- Deferred ideas are in [docs/LATER.md](docs/LATER.md).
- Code: Apache-2.0 ([LICENSE](LICENSE)). Data: SEC EDGAR (terms in [docs/DATA.md](docs/DATA.md)).
