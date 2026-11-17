# FilingLens

Reads US public-company filings from SEC EDGAR and answers analyst questions about them with cited numbers. Full specification: [docs/SPEC.md](docs/SPEC.md).

**Status:** milestone **FL-M1 (data foundation) is complete.** The later milestones (extraction, retrieval, fine-tuning, serving, agents, evaluation) are not built yet.

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

## Layout

```
src/filinglens/
  ingest/     EDGAR client (asyncio + httpx, token bucket), universe, 10-K and FSDS downloaders, BigQuery load
  spark/      FSDS -> Iceberg (local) or Delta (Databricks), partitioned by fiscal year
  sql/        copy of the marts to Postgres; EXPLAIN before/after runner
  features/   ratio features (pandas apply, NumPy, Numba) and their benchmark
  bench/      dbt run_results summary
dagster_project/  Dagster assets and the backfill report
dbt/              staging -> intermediate -> marts (DuckDB and BigQuery targets), tests, docs
sql/              8 hand-written queries, their index changes, plans, RESULTS.md
infra/terraform/gcp/  BigQuery datasets (state in HCP Terraform)
notebooks/        data profiling; the Databricks notebook
bench/            committed benchmark outputs (each records the command that made it)
docs/             SPEC, DATA, LATER, dbt docs site, screenshots
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

## Resources and teardown ($0)

| Resource | Created by | Free because | Teardown |
|---|---|---|---|
| BigQuery datasets `filinglens`, `filinglens_raw` (project with the BigQuery sandbox) | `infra/terraform/gcp` (state in HCP Terraform, local execution) | Sandbox: no billing account; 10 GB storage, 1 TB queries/month; tables expire after 60 days | `cd infra/terraform/gcp && terraform destroy` |
| HCP Terraform workspace `filinglens-gcp` | `terraform init` | Free tier: up to 500 managed resources | Delete the workspace in HCP Terraform (Settings → Destruction and Deletion) |
| Google Cloud project | Google Cloud console | No billing attached | `gcloud projects delete <project-id>` |
| Databricks schema `workspace.filinglens` (Delta tables, volume with the zips) | Catalog Explorer + `notebooks/databricks_fsds_to_delta.py` | Databricks Free Edition: no card, serverless with a daily quota | `DROP SCHEMA workspace.filinglens CASCADE` (done 17 Nov 2026) |
| Local Postgres (`compose.yaml`) and Dagster home (`data/dagster`) | `docker compose up`, `dagster dev` | Local | `docker compose down -v`; `rm -rf data/dagster` |

## Provenance and license

- [PROVENANCE.md](PROVENANCE.md) lists every file as AI-assisted, hand-written or generated.
- Deferred ideas are in [docs/LATER.md](docs/LATER.md).
- Code: Apache-2.0 ([LICENSE](LICENSE)). Data: SEC EDGAR (terms in [docs/DATA.md](docs/DATA.md)).
