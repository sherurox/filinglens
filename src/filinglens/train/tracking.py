"""MLflow helpers for FL-M4: one place for the tracking URI, experiment names and run tags.

Every training, evaluation and filter run carries the same provenance tags: the git commit
(and whether tracked files had uncommitted changes), the data version (SHA-256 of the data
manifest), a free-text power note, and a GPU snapshot from nvidia-smi.

mlflow is imported inside `start_run` only, so the pure helpers stay cheap to import.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DEFAULT_TRACKING_URI = "http://127.0.0.1:55000"  # the `mlflow` service in compose.yaml
EXPERIMENTS = {
    "sft": "fl-m4-sft",
    "teacher": "fl-m4-teacher",
    "optuna": "fl-m4-optuna",
    "compare": "fl-m4-compare",
}
GPU_FIELDS = ("name", "temperature.gpu", "memory.used", "memory.total", "power.draw", "clocks.sm")


def tracking_uri() -> str:
    """MLFLOW_TRACKING_URI from the environment (.env), else the local Compose service."""
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(repo: Path | None = None) -> dict[str, str]:
    """Commit SHA and whether tracked files differ from it; 'unknown' outside a git repo."""
    cwd = repo or Path.cwd()

    def git(*args: str) -> str:
        done = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        )
        return done.stdout.strip()

    try:
        sha = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no") != ""
    except (OSError, subprocess.SubprocessError):
        return {"git_sha": "unknown", "git_dirty": "unknown"}
    return {"git_sha": sha, "git_dirty": str(dirty).lower()}


def parse_gpu_csv(line: str) -> dict[str, str]:
    """One `nvidia-smi --format=csv,noheader,nounits` line → {'gpu.<field>': value}."""
    values = [v.strip() for v in line.split(",")]
    if len(values) != len(GPU_FIELDS):
        raise ValueError(f"expected {len(GPU_FIELDS)} fields, got {len(values)}: {line!r}")
    return {f"gpu.{field}": value for field, value in zip(GPU_FIELDS, values, strict=True)}


def gpu_snapshot() -> dict[str, str]:
    """GPU 0 name, temperature, memory, power and clock right now; {} if nvidia-smi is absent."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {}
    query = [exe, f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits", "-i", "0"]
    try:
        out = subprocess.run(query, capture_output=True, text=True, check=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    lines = out.strip().splitlines()
    return parse_gpu_csv(lines[0]) if lines else {}


def run_tags(
    *, data_manifest: Path | None = None, note: str = "", repo: Path | None = None
) -> dict[str, str]:
    """Provenance tags for one run. The data version is the manifest's SHA-256."""
    tags = {**git_state(repo), "power_note": note}
    if data_manifest is not None:
        tags["data_manifest"] = data_manifest.as_posix()
        tags["data_version"] = file_sha256(data_manifest)
    tags.update(gpu_snapshot())
    return tags


@contextmanager
def start_run(
    kind: str,
    run_name: str,
    *,
    tags: dict[str, str] | None = None,
    uri: str | None = None,
    nested: bool = False,
) -> Iterator[Any]:
    """Open an MLflow run in the experiment for `kind` (a key of EXPERIMENTS)."""
    import mlflow

    if kind not in EXPERIMENTS:
        raise KeyError(f"unknown experiment kind {kind!r}; expected one of {sorted(EXPERIMENTS)}")
    mlflow.set_tracking_uri(uri or tracking_uri())
    mlflow.set_experiment(EXPERIMENTS[kind])
    with mlflow.start_run(run_name=run_name, tags=tags or {}, nested=nested) as run:
        yield run
