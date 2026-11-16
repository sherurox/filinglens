# Later

Ideas that came up during the build but are not in docs/SPEC.md. They stay here until a spec change pulls them in.

| Added | Idea | Why not now | Came up in |
|---|---|---|---|
| 2026-11-16 | Secret scanning (gitleaks pre-commit hook) | Not in the FilingLens spec; `.gitignore` + `.env.example` cover M1 | FL-M1 T1 (.gitignore) |
| 2026-11-16 | Version the raw data (DVC) instead of re-downloading | `data/` is fully re-downloadable from SEC; DVC adds tooling and storage | FL-M1 T1 (.gitignore) |
| 2026-11-16 | Upgrade to Spark 4.2 | No Iceberg Spark 4.2 runtime yet (Iceberg 1.12 supports up to Spark 4.1.3) | FL-M1 env check |
| 2026-11-16 | Cache older submissions pages too (rerun made 307 requests for them) | Spec check "re-run downloads 0" passes; costs about 40 s per rerun | FL-M1 T4 rerun |
| 2026-11-16 | Try dbt v2 (Fusion engine) | dbt Core v1.12 works with dbt-duckdb, dbt-bigquery and Dagster today; v2 reads v1 manifests, so moving later is cheap | FL-M1 T7 |
