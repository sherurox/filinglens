"""Tests for filinglens.train.tracking (no MLflow server needed: SQLite store in tmp_path)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from filinglens.train import tracking

REPO = Path(__file__).resolve().parents[1]


def test_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    payload = b'{"rows": 3}\n' * 200_000  # > 1 MiB, so more than one chunk is read
    path.write_bytes(payload)
    assert tracking.file_sha256(path) == hashlib.sha256(payload).hexdigest()


def test_parse_gpu_csv() -> None:
    line = "NVIDIA GeForce RTX 5070 Laptop GPU, 48, 895, 8151, 12.40, 1500"
    assert tracking.parse_gpu_csv(line) == {
        "gpu.name": "NVIDIA GeForce RTX 5070 Laptop GPU",
        "gpu.temperature.gpu": "48",
        "gpu.memory.used": "895",
        "gpu.memory.total": "8151",
        "gpu.power.draw": "12.40",
        "gpu.clocks.sm": "1500",
    }


def test_parse_gpu_csv_rejects_wrong_field_count() -> None:
    with pytest.raises(ValueError, match="expected 6 fields"):
        tracking.parse_gpu_csv("NVIDIA GeForce RTX 5070 Laptop GPU, 48")


def test_git_state_in_repo() -> None:
    state = tracking.git_state(REPO)
    assert re.fullmatch(r"[0-9a-f]{40}", state["git_sha"])
    assert state["git_dirty"] in {"true", "false"}


def test_git_state_outside_repo(tmp_path: Path) -> None:
    assert tracking.git_state(tmp_path) == {"git_sha": "unknown", "git_dirty": "unknown"}


def test_run_tags_data_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracking, "gpu_snapshot", lambda: {"gpu.name": "fake"})
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text('{"train": 191}\n')
    tags = tracking.run_tags(data_manifest=manifest, note="AC power", repo=REPO)
    assert tags["data_version"] == hashlib.sha256(b'{"train": 191}\n').hexdigest()
    assert tags["power_note"] == "AC power"
    assert tags["gpu.name"] == "fake"
    assert "git_sha" in tags


def test_tracking_uri_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert tracking.tracking_uri() == tracking.DEFAULT_TRACKING_URI
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://example:1")
    assert tracking.tracking_uri() == "http://example:1"


def test_start_run_unknown_kind(tmp_path: Path) -> None:
    pytest.importorskip("mlflow")
    with pytest.raises(KeyError, match="unknown experiment kind"):
        with tracking.start_run("nope", "x", uri=f"sqlite:///{tmp_path / 'm.db'}"):
            pass


def test_start_run_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)  # the default artifact root (./mlruns) lands in tmp_path
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    with tracking.start_run("sft", "unit-test", tags={"data_version": "abc"}, uri=uri) as run:
        mlflow.log_metric("loss", 1.5, step=0)
        mlflow.log_metric("loss", 1.0, step=1)
        run_id = run.info.run_id

    client = mlflow.MlflowClient(tracking_uri=uri)
    stored = client.get_run(run_id)
    assert stored.data.tags["data_version"] == "abc"
    assert stored.info.run_name == "unit-test"
    assert [m.value for m in client.get_metric_history(run_id, "loss")] == [1.5, 1.0]
    experiment = client.get_experiment(stored.info.experiment_id)
    assert experiment.name == tracking.EXPERIMENTS["sft"]
