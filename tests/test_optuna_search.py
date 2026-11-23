"""Optuna search helpers: trial command, resume count, CSV rows, best-vs-baseline summary.

Fake trials stand in for optuna's FrozenTrial, so these run in CI without optuna or a GPU.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from filinglens.train import optuna_search as search


def fake_trial(number: int, state: str, rank=None, lr=None, value=None, **attrs) -> SimpleNamespace:
    params = {} if rank is None else {"rank": rank, "lr": lr}
    return SimpleNamespace(
        number=number,
        state=SimpleNamespace(name=state),
        params=params,
        value=value,
        user_attrs=attrs,
    )


TRIALS = [
    fake_trial(0, "COMPLETE", 16, 2e-4, 0.41, run_name="t0", sec_per_optimizer_step=20.1),
    fake_trial(1, "FAIL", 8, 3e-4),
    fake_trial(2, "COMPLETE", 8, 1.2e-4, 0.39, run_name="t2"),
    fake_trial(3, "RUNNING", 16, 4e-4),
]


def test_names() -> None:
    assert search.study_name("teacher", 400) == "fl-m4-teacher-n400"
    assert search.trial_name("both", 400, 3, 8, 0.000123456) == "optuna-both-n400-t03-r8-lr0.000123"


def test_sft_command_keeps_heldout_rows_out_of_training() -> None:
    args = argparse.Namespace(
        arm="teacher",
        teacher_limit=400,
        heldout_from=1500,
        heldout_n=300,
        out=Path("data/interim/optuna"),
        study="fl-m4-teacher-n400",
        note="AC",
    )
    cmd = search.sft_command(args, "py", "optuna-x", 8, 1e-4, 2)
    flags = dict(zip(cmd[3::2], cmd[4::2], strict=True))
    assert cmd[:3] == ["py", "-m", "filinglens.train.sft"]
    assert flags["--teacher-limit"] == "400" and flags["--heldout-from"] == "1500"
    assert int(flags["--teacher-limit"]) <= int(flags["--heldout-from"])
    assert flags["--heldout-n"] == "300" and flags["--epochs"] == "1"
    assert (flags["--rank"], float(flags["--lr"])) == ("8", 1e-4)
    assert flags["--bench"] == "data/interim/optuna/bench"
    assert flags["--note"] == "optuna fl-m4-teacher-n400 trial 2; AC"


def test_remaining_reruns_failed_and_unfinished_trials() -> None:
    assert search.remaining(TRIALS, 8) == 6  # only COMPLETE counts
    assert search.remaining(TRIALS, 2) == 0
    assert search.remaining([], 8) == 8


def test_trial_rows_and_csv(tmp_path: Path) -> None:
    rows = search.trial_rows(TRIALS)
    assert [r["state"] for r in rows] == ["COMPLETE", "FAIL", "COMPLETE", "RUNNING"]
    assert rows[0]["heldout_loss"] == 0.41 and rows[1]["heldout_loss"] is None
    assert rows[0]["sec_per_optimizer_step"] == 20.1 and rows[2]["run_id"] is None
    path = tmp_path / "bench" / "optuna.csv"
    search.write_csv(path, rows)
    with path.open() as f:
        read = list(csv.DictReader(f))
    assert tuple(read[0]) == search.CSV_FIELDS and len(read) == 4


def test_summarize_best_vs_baseline() -> None:
    summary = search.summarize(search.trial_rows(TRIALS), baseline_rank=16, baseline_lr=2e-4)
    assert summary["best"]["number"] == 2 and summary["baseline"]["number"] == 0
    assert summary["best_minus_baseline"] == pytest.approx(-0.02)
    assert summary["best_is_baseline"] is False
    assert (summary["trials_complete"], summary["trials_failed"]) == (2, 1)


def test_summarize_needs_a_finished_trial() -> None:
    with pytest.raises(ValueError, match="no finished trials"):
        search.summarize(search.trial_rows(TRIALS[1:2]), 16, 2e-4)


@pytest.mark.parametrize(
    "extra",
    [
        ["--teacher-limit", "1600"],  # would overlap the held-out rows
        ["--baseline-lr", "1e-3"],  # outside the search range: the baseline couldn't be enqueued
    ],
)
def test_cli_rejects_bad_settings(extra: list[str]) -> None:
    argv = ["--arm", "teacher", "--baseline-rank", "16", "--baseline-lr", "2e-4", *extra]
    with pytest.raises(SystemExit):
        search.main(argv)
