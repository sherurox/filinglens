# Databricks notebook source
# MAGIC %md
# MAGIC # FL-M1 T15: FSDS → Delta on Databricks Free Edition
# MAGIC
# MAGIC Runs the **same transforms** as `src/filinglens/spark/fsds_to_iceberg.py` (typed columns,
# MAGIC newest-quarter dedupe, fiscal year joined from `sub`), imported from this Git folder.
# MAGIC Differences from the local run: serverless compute, Delta tables in `workspace.filinglens`
# MAGIC (Free Edition has no custom storage, so no Iceberg Hadoop catalog), and the input zips
# MAGIC were uploaded to a Unity Catalog volume (Free Edition can't download from sec.gov).
# MAGIC
# MAGIC Input: `/Volumes/workspace/filinglens/fsds/zips/2025q1.zip` … `2025q4.zip`.
# MAGIC Output: 4 Delta tables + `/Volumes/workspace/filinglens/fsds/fsds_to_delta_databricks.json`.
# MAGIC Teardown: `DROP SCHEMA workspace.filinglens CASCADE`.

# COMMAND ----------

import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# The repo code needs Python 3.11+ (datetime.UTC, hashlib.file_digest). If this fails, pick
# a newer serverless environment version in the notebook's Environment panel.
assert sys.version_info >= (3, 11), (
    f"Python {platform.python_version()}: choose a newer environment"
)

# Import the package from <repo>/src of this Git folder (no install): walk up to the repo root.
here = Path(os.getcwd())
REPO = next(p for p in [here, *here.parents] if (p / "src" / "filinglens").is_dir())
sys.path.insert(0, str(REPO / "src"))
from filinglens.spark.fsds_to_iceberg import build_tables, extract_quarters  # noqa: E402

VOLUME = Path("/Volumes/workspace/filinglens/fsds")
ZIPS, EXTRACTED = VOLUME / "zips", VOLUME / "extracted"
NAMESPACE = "workspace.filinglens"
# SHA-256 recorded when the zips were downloaded locally (data/raw/fsds/manifest.csv, T5).
EXPECTED_SHA256 = {
    "2025q1": "4386e7057cc216b6fc729d8f957998acb831d29be7b16f6d8ed0f6b1ab74eba5",
    "2025q2": "81bf19bdb5606d5ee0164a8de8b7e96d5546170e42e59bcbf95b652ca83ac5d7",
    "2025q3": "ebe25fa74c3655f4738ed9eb144af3953c3981c80f57d74829fb9ad68b114710",
    "2025q4": "2b36ac3850c022cf19edd882e31c3c453c7666677b9fdd2e1f7748fdb5768c6e",
}

# COMMAND ----------

# 1. The uploaded bytes are exactly the bytes downloaded from SEC.
uploaded = {}
for path in sorted(ZIPS.glob("*.zip")):
    with path.open("rb") as f:
        uploaded[path.stem] = hashlib.file_digest(f, "sha256").hexdigest()
sha_check = {q: uploaded.get(q) == sha for q, sha in EXPECTED_SHA256.items()}
print(sha_check)
assert sorted(uploaded) == sorted(EXPECTED_SHA256) and all(sha_check.values()), "zip mismatch"

# COMMAND ----------

# 2. Extract into the volume (Spark can't read inside zips), then build the Delta tables.
started = time.perf_counter()
quarters = extract_quarters(ZIPS, EXTRACTED)
extract_seconds = round(time.perf_counter() - started, 1)
tables = build_tables(spark, EXTRACTED, namespace=NAMESPACE, table_format="delta")  # noqa: F821
total_seconds = round(time.perf_counter() - started, 1)

# COMMAND ----------

# 3. Stats, in the same shape as the local bench JSON, written into the volume for download.
stats = {
    "notebook": "notebooks/databricks_fsds_to_delta.py",
    "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017
    "platform": "Databricks Free Edition, serverless compute",
    "databricks_runtime": os.environ.get("DATABRICKS_RUNTIME_VERSION", "unknown"),
    "spark_version": spark.version,  # noqa: F821
    "python": platform.python_version(),
    "table_format": "delta",
    "namespace": NAMESPACE,
    "quarters": quarters,
    "zip_sha256_matches_t5": sha_check,
    "extract_seconds": extract_seconds,
    "total_seconds": total_seconds,
    "tables": tables,
}
out = VOLUME / "fsds_to_delta_databricks.json"
out.write_text(json.dumps(stats, indent=2, default=str) + "\n")
print(
    json.dumps(
        {k: v for k, v in stats.items() if k != "tables"}
        | {
            "rows": {t: s["rows"] for t, s in tables.items()},
            "seconds": {t: s["seconds"] for t, s in tables.items()},
        },
        indent=2,
        default=str,
    )
)
