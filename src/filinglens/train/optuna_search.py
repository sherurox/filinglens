"""Optuna search over the SFT learning rate and LoRA rank (FL-M4 T11).

Objective: heldout_loss, the student's mean completion-token loss on teacher rows 1,500-1,799,
which no trial trains on (sft.py --heldout-from / --heldout-n). Dev macro-F1 is not the
objective: 71 rows can't rank configs a few points apart, and 8 trials picked on dev would overfit
it, leaving nothing to confirm with. Dev only confirms the result afterwards (best trial vs the
baseline trial, paired bootstrap; a tie keeps the ablation winner).

Each trial is a subprocess running filinglens.train.sft for 1 epoch on the first --teacher-limit
teacher rows (plus the 191 silver rows for the both arm); heldout_loss is read back from its bench
JSON. A subprocess per trial gives a clean GPU each time (the VRAM cap is measured fresh), and a
failed training run is recorded as a FAIL trial without stopping the study. Anything else (MLflow
down, a bug here) stops the study.

The study lives in SQLite, so rerunning the same command resumes it: finished trials are kept, a
trial left RUNNING by a crash is marked FAIL, and only the missing trials run. (A resumed study
isn't bit-reproducible: the sampler's random state isn't stored.) The ablation winner is enqueued
as the first trial: the baseline at this data size.

Sampler: TPE, seed 0, 4 random start-up trials. Optuna's default of 10 start-up trials would make
an 8-trial study pure random search.

    ~/venvs/fl-train/bin/python -m filinglens.train.optuna_search --arm teacher \\
        --baseline-rank 16 --baseline-lr 2e-4

Top-level imports are standard library only, so CI can test the pure helpers without optuna.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

OUT = Path("data/interim/optuna")
LR_RANGE = (5e-5, 5e-4)
RANKS = (8, 16)
ATTRS = (
    "run_name",
    "run_id",
    "optimizer_steps",
    "final_train_loss",
    "sec_per_optimizer_step",
    "train_runtime_s",
    "heldout_examples",
    "heldout_eval_s",
    "peak_reserved_mib",
    "mlflow_synced",
    "wall_s",
)
CSV_FIELDS = ("number", "state", "rank", "lr", "heldout_loss", *ATTRS)


class TrialFailed(RuntimeError):
    """A trial's training run failed: recorded as FAIL, the study goes on."""


class MlflowDown(RuntimeError):
    """The tracking server doesn't answer: the study stops (rerun to resume)."""


# ---------- pure helpers (tested in CI) ----------


def study_name(arm: str, teacher_limit: int) -> str:
    return f"fl-m4-{arm}-n{teacher_limit}"


def trial_name(arm: str, teacher_limit: int, number: int, rank: int, lr: float) -> str:
    return f"optuna-{arm}-n{teacher_limit}-t{number:02d}-r{rank}-lr{lr:.3g}"


def sft_command(
    args: argparse.Namespace, python: str, name: str, rank: int, lr: float, number: int
) -> list[str]:
    """The sft CLI call for one trial (held-out rows are never in its training rows)."""
    return [
        python,
        "-m",
        "filinglens.train.sft",
        "--arm",
        args.arm,
        "--teacher-limit",
        str(args.teacher_limit),
        "--epochs",
        "1",
        "--rank",
        str(rank),
        "--lr",
        str(lr),
        "--heldout-from",
        str(args.heldout_from),
        "--heldout-n",
        str(args.heldout_n),
        "--run-name",
        name,
        "--bench",
        str(args.out / "bench"),
        "--note",
        f"optuna {args.study} trial {number}; {args.note}",
    ]


def remaining(trials: list[Any], n_trials: int) -> int:
    """Trials still to run so that n_trials finish (failed ones are re-run)."""
    return max(0, n_trials - sum(t.state.name == "COMPLETE" for t in trials))


def trial_rows(trials: list[Any]) -> list[dict]:
    """One CSV row per trial (any object with optuna FrozenTrial's attributes)."""
    rows = []
    for t in trials:
        row = {
            "number": t.number,
            "state": t.state.name,
            "rank": t.params.get("rank"),
            "lr": t.params.get("lr"),
            "heldout_loss": t.value if t.state.name == "COMPLETE" else None,
        }
        rows.append(row | {k: t.user_attrs.get(k) for k in ATTRS})
    return rows


def summarize(rows: list[dict], baseline_rank: int, baseline_lr: float) -> dict:
    """Best finished trial vs the baseline (the enqueued ablation winner)."""
    done = [r for r in rows if r["state"] == "COMPLETE"]
    if not done:
        raise ValueError("no finished trials")
    best = min(done, key=lambda r: r["heldout_loss"])
    base = next((r for r in done if r["rank"] == baseline_rank and r["lr"] == baseline_lr), None)
    return {
        "trials_complete": len(done),
        "trials_failed": sum(r["state"] == "FAIL" for r in rows),
        "best": best,
        "baseline": base,
        "best_is_baseline": base is not None and best["number"] == base["number"],
        "best_minus_baseline": (
            None if base is None else round(best["heldout_loss"] - base["heldout_loss"], 5)
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def mlflow_up(uri: str) -> bool:
    try:
        with urllib.request.urlopen(f"{uri}/health", timeout=5) as response:
            return response.status == 200
    except OSError:
        return False


# ---------- the search (fl-train environment) ----------


def search(args: argparse.Namespace) -> dict:
    import mlflow
    import optuna
    from optuna.trial import TrialState

    from filinglens.train.tracking import run_tags, start_run, tracking_uri

    (args.out / "bench").mkdir(parents=True, exist_ok=True)  # sft also creates it
    study = optuna.create_study(
        study_name=args.study,
        storage=f"sqlite:///{(args.out / 'study.db').as_posix()}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=args.startup),
        load_if_exists=True,
    )
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study.tell(t.number, state=TrialState.FAIL)  # one worker: RUNNING means it crashed
    study.enqueue_trial({"lr": args.baseline_lr, "rank": args.baseline_rank}, skip_if_exists=True)

    def objective(trial: Any) -> float:
        lr = trial.suggest_float("lr", *LR_RANGE, log=True)
        rank = trial.suggest_categorical("rank", list(RANKS))
        name = trial_name(args.arm, args.teacher_limit, trial.number, rank, lr)
        if not mlflow_up(tracking_uri()):
            raise MlflowDown(f"MLflow not answering at {tracking_uri()}: start it, then rerun")
        log = args.out / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with log.open("a") as f:
            done = subprocess.run(
                sft_command(args, sys.executable, name, rank, lr, trial.number),
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if done.returncode != 0:
            raise TrialFailed(f"{name}: exit code {done.returncode}, see {log}")
        record = json.loads((args.out / "bench" / f"sft_{name}.json").read_text())
        record["wall_s"] = round(time.perf_counter() - started, 1)
        for key in ATTRS:
            trial.set_user_attr(key, record.get(key))
        print(
            f"trial {trial.number}: rank {rank} lr {lr:.3g} -> heldout_loss "
            f"{record['heldout_loss']} ({record['wall_s']} s)",
            flush=True,
        )
        return record["heldout_loss"]

    todo = remaining(study.trials, args.trials)
    print(f"study {args.study}: {todo} trial(s) to run", flush=True)
    if todo:
        study.optimize(
            objective,
            n_trials=todo,
            catch=(TrialFailed,),
            callbacks=[lambda s, _t: write_csv(args.csv, trial_rows(s.trials))],
        )
    rows = trial_rows(study.trials)
    write_csv(args.csv, rows)
    summary = {
        "study": args.study,
        "objective": f"heldout_loss on teacher rows {args.heldout_from}.."
        f"{args.heldout_from + args.heldout_n - 1} (minimize)",
        "space": {"lr": [*LR_RANGE, "log"], "rank": list(RANKS)},
        "sampler": {"name": "TPE", "seed": args.seed, "n_startup_trials": args.startup},
        "teacher_limit": args.teacher_limit,
        "baseline": {"rank": args.baseline_rank, "lr": args.baseline_lr},
    } | summarize(rows, args.baseline_rank, args.baseline_lr)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")

    with start_run("optuna", args.study, tags=run_tags(note=args.note)) as run:
        mlflow.log_params(
            {
                "arm": args.arm,
                "teacher_limit": args.teacher_limit,
                "heldout_from": args.heldout_from,
                "heldout_n": args.heldout_n,
                "trials": args.trials,
                "n_startup_trials": args.startup,
                "seed": args.seed,
                "lr_range": f"{LR_RANGE[0]}..{LR_RANGE[1]} log",
                "ranks": ",".join(map(str, RANKS)),
                "best_rank": summary["best"]["rank"],
                "best_lr": summary["best"]["lr"],
            }
        )
        metrics = {"best_heldout_loss": summary["best"]["heldout_loss"]}
        if summary["baseline"] is not None:
            metrics["baseline_heldout_loss"] = summary["baseline"]["heldout_loss"]
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(args.csv))
        mlflow.log_artifact(str(args.summary))
        summary["mlflow_run_id"] = run.info.run_id
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=["teacher", "both"], required=True)
    parser.add_argument("--baseline-rank", type=int, choices=RANKS, required=True)
    parser.add_argument("--baseline-lr", type=float, required=True)
    parser.add_argument("--teacher-limit", type=int, default=400)
    parser.add_argument("--heldout-from", type=int, default=1500)
    parser.add_argument("--heldout-n", type=int, default=300)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--startup", type=int, default=4, help="random trials before TPE")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--study", default=None, help="default: fl-m4-<arm>-n<teacher-limit>")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--csv", type=Path, default=Path("bench/optuna.csv"))
    parser.add_argument("--summary", type=Path, default=Path("bench/optuna.json"))
    parser.add_argument("--note", default="AC power, Best performance, VRAM capped to free")
    args = parser.parse_args(argv)
    if not 0 < args.teacher_limit <= args.heldout_from or args.heldout_n <= 0:
        parser.error("need 0 < --teacher-limit <= --heldout-from and --heldout-n > 0")
    if not LR_RANGE[0] <= args.baseline_lr <= LR_RANGE[1]:
        parser.error(f"--baseline-lr must be inside the search range {LR_RANGE}")
    args.study = args.study or study_name(args.arm, args.teacher_limit)
    print(json.dumps(search(args), indent=2))


if __name__ == "__main__":
    main()
