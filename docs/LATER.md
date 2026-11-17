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
