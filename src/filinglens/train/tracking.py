"""MLflow helpers (FL-M4, FL-M3): one place for the tracking URI, experiment names and run tags.

Every training, evaluation and filter run carries the same provenance tags: the git commit
(and whether tracked files had uncommitted changes), the data version (SHA-256 of the data
manifest), a free-text power note, and a GPU snapshot from nvidia-smi.

mlflow is imported inside `start_run` only, so the pure helpers stay cheap to import.
`StepLog` keeps per-step training metrics on local disk so a tracking-server outage can't
kill a training run.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DEFAULT_TRACKING_URI = "http://127.0.0.1:55000"  # the `mlflow` service in compose.yaml
EXPERIMENTS = {
    "sft": "fl-m4-sft",
    "teacher": "fl-m4-teacher",
    "optuna": "fl-m4-optuna",
    "compare": "fl-m4-compare",
    "retrieval": "fl-m3-retrieval",  # embeddings, vector stores, reranker, A/B, peers
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


class StepLog:
    """Per-step metrics: always appended to a local JSONL, sent to MLflow while it answers.

    Why (FL-M4, 22 Nov): the MLflow container stopped at step 8 of 212 and the Trainer died in
    its metrics callback after 8 min of HTTP retries. Now the first failed send switches to
    offline mode: no more sends during training (each failed call can spend minutes retrying),
    and `sync()` replays the rows MLflow missed, in order, from the JSONL. Rows are counted by
    position, not by step, because the Trainer logs its final summary at the last step again.
    """

    def __init__(self, path: Path, send: Callable[[dict[str, float], int], None]) -> None:
        self.path = path
        self.send = send
        self.rows = 0  # rows written to the JSONL
        self.pending_from: int | None = None  # first row MLflow hasn't got; None = in sync
        self.offline_at_step: int | None = None  # for the bench summary
        self.errors: list[str] = []
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")  # a rerun starts a fresh log

    def log(self, metrics: dict[str, float], step: int) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
        row, self.rows = self.rows, self.rows + 1
        if self.pending_from is not None:
            return
        try:
            self.send(metrics, step)
        except Exception as e:  # any transport error: keep training, catch up later
            self.pending_from, self.offline_at_step = row, step
            self.record_error("log", e)

    def sync(self) -> bool:
        """Replay the rows MLflow missed; True when MLflow has every row. Resumable."""
        if self.pending_from is None:
            return True
        lines = [line for line in self.path.read_text().splitlines() if line.strip()]
        for i in range(self.pending_from, len(lines)):
            row = json.loads(lines[i])
            step = row.pop("step")
            try:
                self.send(row, step)
            except Exception as e:
                self.record_error("sync", e)
                return False
            self.pending_from = i + 1
        self.pending_from = None
        return True

    def record_error(self, where: str, error: BaseException) -> None:
        message = f"{where}: {type(error).__name__}: {error}"[:500]
        self.errors.append(message)
        print(f"[StepLog] MLflow {message} (metrics continue in {self.path})", file=sys.stderr)
