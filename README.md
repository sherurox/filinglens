# FilingLens

Reads US public-company filings from SEC EDGAR and answers analyst questions about them with cited numbers. Full specification: [docs/SPEC.md](docs/SPEC.md).

**Status:** milestones **FL-M1 (data foundation)**, **FL-M2 (labels, OCR and extraction)**, **FL-M3 (retrieval, ranking and offline A/B)** and **FL-M4 (fine-tune and distill)** are complete. The other milestones (serving, agents, evaluation) are not built yet.

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

## FL-M3 results

Retrieval over the Item 1A text of 2,198 10-Ks (294 companies; 203,200 chunks), with synthetic questions graded automatically: scores measure finding the paragraph a question was written from, not a human relevance judgment. Every search is filtered to the question's company. Details in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) and [docs/DATA.md](docs/DATA.md).

| Part | Result | Evidence |
|---|---|---|
| Embeddings | `bge-base-en-v1.5` (MIT), 203,200 × 768 in 424.5 s on the laptop GPU (fp16; min cosine to fp32 0.9999995) | `bench/embed_corpus.json` |
| Why filter by company | Exact recall@10 on ret-dev: 0.2967 over all chunks, 0.6758 within the question's company | `bench/retrieval_dev_diagnostics.json` |
| **Vector stores (D1)** | 100 ret-dev queries, filtered: FAISS Flat, pgvector 0.8.7 and Qdrant 1.20.2 all exact (tie-aware overlap 1.0; p95 0.488 / 3.44 / 1.154 ms); FAISS HNSW with a selector starves (47 of 100 searches < 10 results). Unfiltered HNSW tie-aware overlap: FAISS 0.875, pgvector 0.903, Qdrant 0.98 | `bench/vector_stores.csv`, `bench/index_build.json` |
| **Reranker (D1)** | `bge-reranker-base` (MIT) over the filtered dense top 50, 182 ret-dev queries: NDCG@10 0.4368 → 0.488, MRR@10 0.5264 → 0.5708; rerank p95 279.26 ms | `bench/reranker.csv`, `bench/reranker_dev.json` |
| Power analysis | Dev sd of the NDCG@10 difference 0.2752; 663 independent queries for a 0.03 gain at 80% power; company ICC 0.1434 (ANOVA) or 0.0297 (implied), so ret-test was grown to 1,137 queries over 75 companies | `bench/ab_power.json` |
| **Offline A/B (D2)** | Rule committed before the test run (`15b4381`). ret-test, 1,137 queries: NDCG@10 0.4238 → 0.5125, +0.0888 (company-cluster 95% CI 0.0669–0.1099, d_z 0.2854); rerank p95 246.63 ms ≤ 500 ms budget → **ship the reranker** | `bench/ab_rule.json`, `bench/ab_test.json`, MLflow `fl-m3-retrieval` |
| Peer recommender | Latest-10-K text + ratio ranks, α = 0.9 chosen on train + dev companies; 49 scored test companies, hit-rate@5 vs 4-digit SIC peers: combined 0.898 (0.8163–0.9796), text only 0.8571, ratios only 0.3061, random 0.1206 | `bench/peers_select.json`, `bench/peers.json` |

**"Done when" (spec §4 M3):**
- [x] **D1:** the vector-store comparison table and the reranker table are committed (`bench/vector_stores.csv`, `bench/reranker.csv`).
- [x] **D2:** the A/B write-up states the decision and why ([docs/EXPERIMENTS.md](docs/EXPERIMENTS.md), FL-M3).

## FL-M4 results

A 1.7B student (Qwen3-1.7B, QLoRA on the 8 GB laptop GPU) distilled from an 8B teacher (Qwen3-8B-AWQ, run on Kaggle's free 2× T4). Labels are still silver, so scores are agreement with the silver labeler. Every FL-M4 comparison uses deterministic decoding (`--max-num-seqs 1`); see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for why.

| Part | Result | Evidence |
|---|---|---|
| Teacher labels (Kaggle, 2× T4) | 7,200 pool rows labeled in 7,098.4 s; 6,871 valid risks; **6,259 kept** by a filter fixed in advance (rule checks, first-token consistency, confidence ≥ 0.9, keyword heuristic) | `bench/teacher_run/`, `bench/teacher_filter.json` |
| Training memory (D2) | QLoRA NF4 + Liger fused cross-entropy: 4,096-token examples train in ≤ 4,180 MiB reserved on the 8 GB card (all 6,259 rows: 783 steps, 3.7 h, 4,118 MiB); VRAM capped so WSL can't spill silently | `bench/sft_*.json`, MLflow `fl-m4-sft` |
| Dev ablations (71 rows) | Base 0.6437, silver arm 0.6404, teacher arm (1,500 rows) **0.7414** category macro-F1; both arms, rank 8 and lr 1e-4 all tie; training on all 6,259 teacher rows dropped to 0.6411 (the student copies the teacher more closely: teacher agreement 0.859 → 0.901), so the 1,500-row adapter ships | `bench/fields_zero_dev_*.json`, `bench/compare_dev_*.json` |
| Optuna (8 trials, held-out teacher loss) | Best: r16, lr 4.86e-4 (loss 0.32874 vs 0.33982 for the default); on dev a tie (+0.0109, CI −0.0476 to 0.0763), so the default r16 / lr 2e-4 is kept | `bench/optuna.csv`, `bench/optuna.json` |
| **Test comparison (D1)** | Category macro-F1 on 65 test rows: base 0.6525, teacher 0.783, fine-tuned student **0.7948**; JSON-valid 65 / 65 / 65 (rule-valid 64 / 64 / 65). Student vs base +0.1423 (95% CI 0.0276–0.2621); student vs teacher a tie | `bench/m4_test_comparison.json`, MLflow `fl-m4-compare` |
| Registry (D3) | `filinglens-risk-student` v1 (LoRA adapter for Qwen3-1.7B) with config, data version, git SHA and scores; registry copy verified by SHA-256 | MLflow registry |

**"Done when" (spec §4 M4):**
- [x] **D1:** one MLflow comparison of base prompted vs teacher vs fine-tuned student on the test split (macro-F1, JSON-valid rate).
- [x] **D2:** training curves and peak VRAM recorded.
- [x] **D3:** model and adapter in the registry; `docs/EXPERIMENTS.md` updated.

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
  train/      teacher inputs and Kaggle driver helpers, confidence, filter, SFT data, QLoRA trainer,
              MLflow helpers, dev evaluation, paired bootstrap, Optuna search
  retrieval/  chunks, graded queries, embeddings, FAISS / pgvector / Qdrant stores and benchmark,
              ranking metrics, reranker, offline A/B (power analysis, cluster bootstrap), peers
  bench/      dbt run_results summary
dagster_project/  Dagster assets and the backfill report
dbt/              staging -> intermediate -> marts (DuckDB and BigQuery targets), tests, docs
gold/             silver risk-paragraph splits, table pages with truth, master company table,
                  synthetic Q&A, retrieval queries (all committed; "gold" is the spec's directory name)
sql/              8 hand-written queries, their index changes, plans, RESULTS.md
infra/terraform/gcp/  BigQuery datasets (state in HCP Terraform)
notebooks/        data profiling; the Databricks notebook; the Kaggle teacher driver
scripts/          FL-M4 ablation runner, FL-M3 environment record
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

## Running FL-M3

`uv run` for the stores, the A/B statistics and the peers; `~/venvs/fl-train` (PyTorch, transformers) for embedding and reranking; `~/venvs/fl-gpu` (vLLM) to generate the extra test questions. Postgres (pgvector image), Qdrant and MLflow run in Docker.

```bash
TRAIN=~/venvs/fl-train/bin/python; GPU=~/venvs/fl-gpu/bin/python
docker compose up -d --wait mlflow postgres qdrant
bash scripts/fl_m3_env.sh "AC power, Best performance"           # bench/retrieval_env.txt
uv run python -m filinglens.retrieval.chunks                       # 203,200 chunks
uv run python -m filinglens.retrieval.queries                      # ret-dev / ret-test queries
$TRAIN -m filinglens.retrieval.embed corpus --note "..."
$TRAIN -m filinglens.retrieval.embed queries --split dev --note "..."
uv run python -m filinglens.retrieval.diagnose_dev                 # why the CIK filter
uv run python -m filinglens.retrieval.build_indexes --note "..."    # FAISS, pgvector, Qdrant
uv run python -m filinglens.retrieval.bench_stores --note "..."     # bench/vector_stores.csv (D1)
uv run python -m filinglens.retrieval.rerank candidates --split dev
$TRAIN -m filinglens.retrieval.rerank score --split dev --note "..." # bench/reranker.csv (D1)
# extra ret-test questions (T10a), then rebuild the query files
uv run python -m filinglens.retrieval.extra_queries prepare
$GPU -m filinglens.retrieval.extra_queries generate --model Qwen/Qwen3-4B-AWQ --gpu-memory-utilization 0.80 --enforce-eager
uv run python -m filinglens.retrieval.extra_queries finalize
uv run python -m filinglens.retrieval.queries
# A/B: rule from dev, commit it, then ret-test once
uv run python -m filinglens.retrieval.ab pilot --mde 0.03 --latency-budget-ms 500
git add bench/ab_rule.json bench/ab_power.json && git commit -m "pre-registered A/B rule"
$TRAIN -m filinglens.retrieval.embed queries --split test --allow-test --note "..."
uv run python -m filinglens.retrieval.rerank candidates --split test --allow-test
$TRAIN -m filinglens.retrieval.rerank score --split test --allow-test --note "..."
uv run python -m filinglens.retrieval.ab test --allow-test --note "..."
# peers: choose on train + dev, commit, then test once
uv run python -m filinglens.retrieval.peers select --note "..."
git add bench/peers_select.json && git commit -m "peers selection"
uv run python -m filinglens.retrieval.peers test --allow-test --note "..."
```

## Running FL-M4

Three environments: `uv run` (main), `~/venvs/fl-train` (training: PyTorch, PEFT, bitsandbytes, Liger, MLflow, Optuna; versions in `bench/train_env.txt`), `~/venvs/fl-gpu` (vLLM). MLflow runs in Docker. The teacher step needs a Kaggle account (phone-verified for GPU) and its API token in `~/.kaggle/access_token`.

```bash
TRAIN=~/venvs/fl-train/bin/python; GPU=~/venvs/fl-gpu/bin/python
EVAL="--gpu-memory-utilization 0.84 --max-model-len 4608 --max-num-seqs 1"
docker compose up -d --wait mlflow                                # http://127.0.0.1:55000
uv run python -m filinglens.train.teacher_inputs                  # balanced pool, leakage 0/0/0
# Kaggle: upload the inputs as a private dataset, run notebooks/kaggle_teacher_driver.py on 2x T4,
# download the notebook output to data/interim/teacher/kaggle_out/ (also saved as the private
# Kaggle dataset filinglens-teacher-labels)
uv run python -m filinglens.train.filter_teacher
uv run python -m filinglens.train.sft_data --teacher data/interim/teacher/filtered.jsonl
bash scripts/fl_m4_ablate.sh "AC power"                           # dev ablations
$TRAIN -m filinglens.train.optuna_search --arm teacher --baseline-rank 16 --baseline-lr 2e-4
bash scripts/fl_m4_final.sh "AC power"                            # dev check + all-rows retrain (a finding, not shipped)
# shipped student (already trained by the ablations): teacher arm, 1,500 rows, r16, lr 2e-4, 1 epoch
$TRAIN -m filinglens.train.sft --arm teacher --teacher-limit 1500 --epochs 1 --rank 16 --lr 2e-4
# test, once: base and student predictions, then comparison + registry
$GPU -m filinglens.extract.fields generate --variant zero --split test --allow-test $EVAL --tag base-det
$GPU -m filinglens.extract.fields generate --variant zero --split test --allow-test $EVAL --max-lora-rank 16 \
  --lora data/interim/adapters/teacher-n1500-r16-lr0.0002-e1/adapter
uv run python -m filinglens.train.final_comparison --allow-test
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
| Kaggle account `shreyask007`: private datasets `filinglens-teacher-inputs` and `filinglens-teacher-labels`, notebook `filinglens-teacher`, API token | Kaggle website + Kaggle CLI (FL-M4 T4) | Kaggle free tier: no card, about 30 GPU hours a week | Expire the API token (Kaggle → Settings → API) [done: date]; optionally delete both datasets and the notebook |
| Local MLflow server (`compose.yaml`, volume `filinglens_mlflowdata`: runs, registry, adapters) | `docker compose up -d mlflow` (FL-M4 T2) | Local | After exporting what the docs need: `docker compose rm -sf mlflow && docker volume rm filinglens_mlflowdata` |
| Local Qdrant (`compose.yaml`, volume `qdrantdata`) and the pgvector table `retrieval.chunk_vectors` in the local Postgres (FL-M3) | `docker compose up -d qdrant postgres`, `retrieval.build_indexes` | Local | `docker compose rm -sf qdrant && docker volume rm filinglens_qdrantdata`; `DROP SCHEMA retrieval CASCADE` in Postgres |
| FL-M3 embeddings and FAISS indexes (`data/interim/retrieval/`, git-ignored) and the Hugging Face cache for `bge-base-en-v1.5` and `bge-reranker-base` | `retrieval.embed`, `retrieval.build_indexes` | Local disk | `rm -rf data/interim/retrieval`; remove the two `models--BAAI--*` folders under `~/.cache/huggingface/hub` |

## Provenance and license

- [PROVENANCE.md](PROVENANCE.md) lists every file as AI-assisted, hand-written or generated.
- Deferred ideas are in [docs/LATER.md](docs/LATER.md).
- Code: Apache-2.0 ([LICENSE](LICENSE)). Data: SEC EDGAR (terms in [docs/DATA.md](docs/DATA.md)).
