# Later

Ideas that came up during the build but are not in docs/SPEC.md. They stay here until a spec change pulls them in.

| Added | Idea | Why not now | Came up in |
|---|---|---|---|
| 2026-11-16 | Secret scanning (gitleaks pre-commit hook) | Not in the FilingLens spec; `.gitignore` + `.env.example` cover M1 | FL-M1 T1 (.gitignore) |
| 2026-11-16 | Version the raw data (DVC) instead of re-downloading | `data/` is fully re-downloadable from SEC; DVC adds tooling and storage | FL-M1 T1 (.gitignore) |
| 2026-11-16 | Upgrade to Spark 4.2 | No Iceberg Spark 4.2 runtime yet (Iceberg 1.12 supports up to Spark 4.1.3) | FL-M1 env check |
| 2026-11-16 | Cache older submissions pages too (rerun made 307 requests for them) | Spec check "re-run downloads 0" passes; costs about 40 s per rerun | FL-M1 T4 rerun |
| 2026-11-16 | Try dbt v2 (Fusion engine) | dbt Core v1.12 works with dbt-duckdb, dbt-bigquery and Dagster today; v2 reads v1 manifests, so moving later is cheap | FL-M1 T7 |
| 2026-11-17 | Segment-level facts (segments / coregistrant not null) | Spec asks for company financials; 57.5M segment rows would multiply the marts | FL-M1 T8 |
| 2026-11-17 | Fiscal-year mapping in dim_period (per company fiscal year end) | Calendar periods are comparable across companies; fiscal mapping needs per-company rules | FL-M1 T8 |
| 2026-11-17 | Include 40-F / 20-F filers (IFRS taxonomy) in the marts | Universe company BNS files 40-F/6-K only; IFRS tags don't map 1:1 to US GAAP tags | FL-M1 T8 |
| 2026-11-17 | terraform fmt/validate (and tflint) in CI | Planned for SS-M6 (platform milestone); M1 has 2 resources | FL-M1 T9 |
| 2026-11-17 | dbt model contracts (enforced column data types) on staging and marts | Would have failed the BigQuery `stg_fsds__tags` view at build time when `tag` resolved to the whole row (a STRUCT); tests check values, not types | FL-M1 T10 |
| 2026-11-17 | Check why 84 companies first report OperatingLeaseRightOfUseAsset in 2018 (and 3 in 2017) | Earlier than most ASC 842 effective dates: adoption-date balances or early adopters? Not verified | FL-M1 T12 |
| 2026-11-17 | Revenue concept mapping beyond a tag priority list | Banks/insurers use other tags; a tag switch between years can look like a decline (sql/ 02, 05–08) | FL-M1 T12 |
| 2026-11-17 | Ratio benchmark: time the Numba kernel alone vs the end-to-end call | At 1M rows NumPy (0.032678 s) beat Numba (0.043817 s) end to end; check whether building the output DataFrame dominates | FL-M1 T13 |
| 2026-11-17 | Incremental Spark: per-quarter staging table with dynamic partition overwrite, dedupe moved downstream | Today a backfill of any quarter range rebuilds all Iceberg tables (183.4 s) | FL-M1 T14 |
| 2026-11-17 | Dagster: DynamicPartitionsDefinition + a sensor for newly published FSDS quarters | The partition list is fixed at 2018q1–2026q3 | FL-M1 T14 |
| 2026-11-17 | Dagster `backfill_policy` is beta (1.13); fallback is per-partition runs with a concurrency limit of 1 for the SEC pool | Keeps SEC's 10 req/s safe if the beta API changes | FL-M1 T14 |
| 2026-11-17 | Risk-factor sections in annual-report-style 10-Ks (GE, C, NEM, FDX before FY2021): bare "RISK FACTORS" heading with no item heading after it | 4 companies; each needs its own end rule; 300-paragraph gold sample doesn't need them | FL-M2 T1 |
| 2026-11-17 | Fetch Exhibit 13 (annual report) for filers that incorporate Item 1A by reference (BNY, USB, WFC) | New downloads; the text isn't in the 10-K document | FL-M2 T1 |
| 2026-11-19 | Date-aware entity resolution (a name resolves to the CIK that was filing at the mention's date; predecessor/successor CIK links) | Most D4 errors are successor registrants with the same name; names alone can't separate them | FL-M2 T16 |
| 2026-11-19 | Keep SEC state tags (`/PA`, `/OH`) as a tie-breaker when both names carry one | Moves 1–2 eval matches; precision-only change, not needed for D4 | FL-M2 T16 |
| 2026-11-19 | Freeze evaluation sets by ID (table pages, silver test rows) so later parser fixes recompute values but never change membership | The "—%" fix redrew all 50 table pages; acceptable once, but every later fix would do it again | FL-M2 T10 |
| 2026-11-19 | Hand-check the 65 silver test rows (or a sample) to turn the test split into real gold | User decision 19 Nov 2026: fully automated labels; needed before any claim of correctness rather than agreement | FL-M2 T5 |
| 2026-11-19 | M4 ablation "gold only" has no gold data: rename to "silver only" or drop (needs a spec change) | **Resolved in FL-M4:** renamed "silver only" (EXPERIMENTS.md, FL-M4 deviations) | FL-M2 T5 |
| 2026-11-19 | Person identity by SEC reporting-owner CIK (Forms 3/4) instead of first + last name | Namesakes merge today; needs new downloads per company | FL-M2 T15 |
| 2026-11-19 | Signatures given by an attorney-in-fact ("*" next to directors' names), and titles that keep a name prefix | SO lists 1 signer; WEC titles start with the name; counts are still right | FL-M2 T15 |
| 2026-11-19 | Exhibit 21 history (every year, not the latest only) and OCR for the 1 PDF exhibit | A snapshot is enough for the graph; 8× the requests | FL-M2 T15 |
| 2026-11-19 | Synthetic Q&A with sampling (temperature > 0) and per-type quotas | Greedy gives few cause (37) and mitigation (16) questions | FL-M2 T14 |
| 2026-11-20 | DPO on ~200 preference pairs (good vs bad extraction) | Optional spec item (M4 step 5); time went to the teacher pipeline and the ablations | FL-M4 plan |
| 2026-11-20 | Train with `not_a_risk` teacher rows (`sft_data.py --keep-not-a-risk`) so the student can reject non-risk text | All eval rows are risks, so the class can only cost points on this eval | FL-M4 T7 |
| 2026-11-20 | Rename `gradient_checkpointing_kwargs` → `activation_checkpointing_kwargs` in `prepare_model_for_kbit_training` | PEFT deprecation warning (removed in PEFT 1.2); the new name exists in the installed version | FL-M4 T8 |
| 2026-11-21 | Calibrate the teacher's confidence (temperature scaling on a labeled slice) before filtering by it | Greedy + long rule prompt makes confidence saturated (4,994 of 6,871 ≥ 0.9999); no human labels to calibrate against | FL-M4 T6 |
| 2026-11-21 | Several seeds per setting | 71 dev rows already make most differences ties; GPU time | FL-M4 T10 |
| 2026-11-23 | A backfill command for runs that finished with `mlflow_synced: false` (replay `steps.jsonl`, log the adapter, close the run) | Not needed so far: the outage test synced at the end; everything stays in `steps.jsonl` and `bench/` | FL-M4 T10 |
| 2026-11-23 | Find out why Docker Desktop stopped the MLflow container on 22 Nov (Docker Desktop logs) and set it to start at sign-in | Training no longer depends on MLflow staying up | FL-M4 T10 |
| 2026-11-23 | Widen the Optuna learning-rate range above 5e-4 and search on longer runs (more rows / steps) | The best trial sat at the range edge after 50-step trials; no time before the milestone ended, and dev couldn't separate it from the default | FL-M4 T11 |
| 2026-11-23 | `scripts/fl_m4_final.sh`: Optuna trial records live in `data/interim/optuna/bench/` but `train.evaluate` reads `bench/` (copied by hand on 23 Nov); copy them in the script or pass `--bench` | Worked around by hand; not needed again in FL-M4 | FL-M4 T11 |
| 2026-11-24 | Score the teacher with the deterministic setting too (the 8B AWQ model on Kaggle with `--max-num-seqs 1`) | The teacher's dev and test predictions are batched; student and base are deterministic | FL-M4 T12 |
| 2026-11-24 | Counter the student drifting toward the teacher's errors with more data: mix silver rows, filter teacher rows that disagree with a second labeler, or stop on dev agreement | All-rows training lowered dev agreement (0.7414 → 0.6411) while teacher agreement rose (0.859 → 0.901) | FL-M4 final retrain |
