"""FL-M4 T12: the one test comparison (D1) and the student adapter in the MLflow registry (D3).

Base prompted vs teacher vs fine-tuned student on the 65 test rows, all scored by the FL-M2 scorer
against the same silver labels:
- base: Qwen3-1.7B, no adapter, deterministic decoding (`fields generate --tag base-det
  --max-num-seqs 1`); FL-M2's batched test score (bench/fields_zero_test.json) is reported next
  to it, labeled as batched;
- teacher: Qwen3-8B-AWQ's test answers from the Kaggle run (test_shard0of1.jsonl), re-parsed with
  the same `parse()` rules as the other two so validity is judged identically; they came from
  batched fp16 decoding on a T4 (not re-run: the 8B model isn't part of the local eval path);
- student: the dev-selected adapter (default teacher-n1500-r16-lr0.0002-e1), deterministic.

Reported per model: category macro-F1 (95% CI), accuracy, JSON-valid rate (the answer parses as
JSON), rule-valid rate (it also passes the labeling rules), severity macro-F1, first-seen exact and
within 1 year. Paired bootstraps: student vs base, student vs teacher, teacher vs base.

Guards (test is scored once): --allow-test is required; base and student predictions must have been
generated with --max-num-seqs 1; a complete bench/m4_test_comparison.json (with its registry
entry) stops the run. Scoring is deterministic, so a run that failed before registering can be
rerun.

    uv run python -m filinglens.train.final_comparison --allow-test
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from filinglens.extract import fields
from filinglens.train import compare, evaluate

STUDENT = "teacher-n1500-r16-lr0.0002-e1"
BASE_TAG = "base-det"
TEACHER_TAG = "teacher-qwen3-8b-awq"
TEACHER_RAW = Path("data/interim/teacher/kaggle_out/teacher/test_shard0of1.jsonl")
FL_M2_SCORE = Path("bench/fields_zero_test.json")
REGISTERED_NAME = "filinglens-risk-student"
EVAL_SETTINGS = "--gpu-memory-utilization 0.84 --max-model-len 4608 --max-num-seqs 1"
TABLE_KEYS = (
    "category_macro_f1",
    "category_macro_f1_ci95",
    "category_accuracy",
    "severity_macro_f1",
    "first_seen_exact",
    "first_seen_within_1y",
)


# ---------- pure helpers (tested) ----------


def json_valid_count(records: list[dict]) -> int:
    """Answers whose raw text parses as JSON (rule-valid is stricter: it also checks the rules)."""
    count = 0
    for r in records:
        try:
            json.loads(r.get("raw") or "")
        except (TypeError, ValueError):
            continue
        count += 1
    return count


def model_row(score: dict, records: list[dict] | None, decoding: str) -> dict:
    """One row of the comparison table from a `fields score` result (+ its prediction records)."""
    rows = score["all_rows"]
    out = {k: rows[k] for k in TABLE_KEYS}
    out["rows"] = rows["rows"]
    out["rule_valid"] = rows["rule_valid"]
    out["rule_valid_rate"] = round(rows["rule_valid"] / rows["rows"], 4)
    if records is not None:
        out["json_valid"] = json_valid_count(records)
        out["json_valid_rate"] = round(out["json_valid"] / rows["rows"], 4)
    out["agreed_rows_category_macro_f1"] = score["rows_where_silver_labelers_agree"][
        "category_macro_f1"
    ]
    out["decoding"] = decoding
    return out


def check_deterministic(run_record: dict, name: str) -> None:
    if run_record.get("max_num_seqs") != 1:
        raise SystemExit(
            f"{name}: predictions were generated with max_num_seqs={run_record.get('max_num_seqs')}"
            f", not 1; regenerate with {EVAL_SETTINGS}"
        )


def check_rows(records: dict[str, dict], rows: list[dict], name: str) -> None:
    missing = [r["para_id"] for r in rows if r["para_id"] not in records]
    if missing:
        raise SystemExit(f"{name}: {len(missing)} test rows have no prediction, e.g. {missing[:3]}")


def flat_metrics(table: dict[str, dict]) -> dict[str, float]:
    """MLflow metrics: <model>.<metric> for every number in the table (CIs as _low/_high)."""
    out = {}
    for model, row in table.items():
        for key, value in row.items():
            if key == "category_macro_f1_ci95":
                out[f"{model}.category_macro_f1_ci95_low"] = float(value[0])
                out[f"{model}.category_macro_f1_ci95_high"] = float(value[1])
            elif isinstance(value, int | float) and not isinstance(value, bool):
                out[f"{model}.{key}"] = float(value)
    return out


# ---------- steps (main env) ----------


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def teacher_predictions(rows: list[dict], raw_path: Path) -> tuple[Path, dict]:
    """Re-parse the teacher's raw answers with the same rules; write them like a fields file."""
    from filinglens.extract.silver_label import parse

    teacher = {r["para_id"]: r for r in read_jsonl(raw_path)}
    check_rows(teacher, rows, "teacher")
    tasks = {t["para_id"]: t for t in json.loads(fields.TASKS.read_text())}
    records = [parse(teacher[r["para_id"]]["raw"], fields.task_view(r, tasks)) for r in rows]
    flags_differ = sum(
        bool(rec["valid"]) != bool(teacher[r["para_id"]].get("valid"))
        for rec, r in zip(records, rows, strict=True)
    )
    out = fields.OUT_DIR / f"fields_zero_test_{TEACHER_TAG}.jsonl"
    out.write_text("".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records))
    return out, {"source": raw_path.as_posix(), "valid_flags_differ_from_kaggle": flags_differ}


def score(tag: str, bench: Path) -> dict:
    fields.main(
        ["score", "--variant", "zero", "--split", "test", "--allow-test"]
        + ["--tag", tag, "--bench", str(bench)]
    )
    return json.loads((bench / f"fields_zero_test_{tag}.json").read_text())


def paired(a: str, b: str, bench: Path) -> dict:
    compare.main(["--a", a, "--b", b, "--split", "test", "--allow-test", "--bench", str(bench)])
    result = json.loads((bench / f"compare_test_{a}_vs_{b}.json").read_text())
    keep = (
        "category_macro_f1",
        "category_accuracy",
        "rows_where_only_a_right",
        "rows_where_only_b_right",
    )
    return {"a": a, "b": b, **{k: result[k] for k in keep}}


def register(client, name: str, run_id: str, local_adapter: Path, tags: dict) -> dict:
    """Register the run's logged adapter; skip if this run is already a version of `name`."""
    from mlflow.artifacts import download_artifacts
    from mlflow.exceptions import MlflowException

    from filinglens.train.tracking import file_sha256

    logged = [a.path for a in client.list_artifacts(run_id, "adapter")]
    if not any(p.endswith("adapter_model.safetensors") for p in logged):
        raise SystemExit(f"run {run_id} has no logged adapter (artifacts under adapter/: {logged})")
    try:
        client.create_registered_model(
            name,
            description="QLoRA adapter for Qwen/Qwen3-1.7B (Apache-2.0): FilingLens risk-factor "
            "category, severity and first-seen year, distilled from Qwen3-8B-AWQ labels (FL-M4).",
        )
    except MlflowException as e:
        if "RESOURCE_ALREADY_EXISTS" not in str(e) and "already exists" not in str(e):
            raise
    existing = [v for v in client.search_model_versions(f"name='{name}'") if v.run_id == run_id]
    if existing:
        version = existing[0]
    else:
        source = f"{client.get_run(run_id).info.artifact_uri}/adapter"
        version = client.create_model_version(
            name, source, run_id=run_id, tags={k: str(v) for k, v in tags.items()}
        )
    with tempfile.TemporaryDirectory() as tmp:
        got = Path(
            download_artifacts(artifact_uri=f"models:/{name}/{version.version}", dst_path=tmp)
        )
        files = {p.name for p in got.rglob("*") if p.is_file()}
        weights = next(got.rglob("adapter_model.safetensors"))
        same = file_sha256(weights) == file_sha256(local_adapter / "adapter_model.safetensors")
    return {
        "name": name,
        "version": int(version.version),
        "source": version.source,
        "files": sorted(files),
        "weights_match_local_adapter": same,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--student", default=STUDENT, help="run name of the dev-selected adapter")
    parser.add_argument("--base-tag", default=BASE_TAG)
    parser.add_argument("--teacher-raw", type=Path, default=TEACHER_RAW)
    parser.add_argument("--register-name", default=REGISTERED_NAME)
    parser.add_argument("--bench", type=Path, default=Path("bench"))
    parser.add_argument("--note", default="")
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    if not args.allow_test:
        parser.error("this is the one test comparison: pass --allow-test")
    out_path = args.bench / "m4_test_comparison.json"
    if out_path.exists() and "registry" in json.loads(out_path.read_text()):
        parser.error(f"{out_path} is complete: test is scored once")

    rows = fields.load_rows("test", allow_test=True)
    preds = {}
    for name, tag in (("base", args.base_tag), ("student", args.student)):
        check_deterministic(
            json.loads((args.bench / f"fields_zero_test_{tag}_run.json").read_text()), name
        )
        path = fields.OUT_DIR / f"fields_zero_test_{tag}.jsonl"
        preds[name] = read_jsonl(path)
        check_rows({r["para_id"]: r for r in preds[name]}, rows, name)
    teacher_path, teacher_note = teacher_predictions(rows, args.teacher_raw)
    preds["teacher"] = read_jsonl(teacher_path)

    fl_m2 = json.loads(FL_M2_SCORE.read_text())
    table = {
        "base_fl_m2_batched": model_row(fl_m2, None, "batched (FL-M2, 19 Nov)"),
        "base": model_row(score(args.base_tag, args.bench), preds["base"], "max_num_seqs 1"),
        "teacher": model_row(
            score(TEACHER_TAG, args.bench), preds["teacher"], "batched fp16, Kaggle T4"
        ),
        "student": model_row(score(args.student, args.bench), preds["student"], "max_num_seqs 1"),
    }
    pairs = {
        "student_vs_base": paired(args.base_tag, args.student, args.bench),
        "student_vs_teacher": paired(TEACHER_TAG, args.student, args.bench),
        "teacher_vs_base": paired(args.base_tag, TEACHER_TAG, args.bench),
    }

    import mlflow

    from filinglens.train.tracking import run_tags, start_run, tracking_uri

    client = mlflow.MlflowClient(tracking_uri=tracking_uri())
    train_record = json.loads((args.bench / f"sft_{args.student}.json").read_text())
    student_run = client.get_run(train_record["run_id"])
    result = {
        "command": "python -m filinglens.train.final_comparison " + " ".join(argv),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "labels": "silver (agreement with ibm-granite/granite-4.1-3b, not correctness)",
        "split": "test",
        "rows": len(rows),
        "eval_settings": EVAL_SETTINGS,
        "student": args.student,
        "student_selection": "dev-selected (highest dev category macro-F1 among FL-M4 adapters; "
        "chosen before test was scored)",
        "teacher": {"model": "Qwen/Qwen3-8B-AWQ", **teacher_note},
        "models": table,
        "paired_bootstrap": pairs,
    }
    with start_run("compare", "m4-test-comparison", tags=run_tags(note=args.note)) as run:
        mlflow.log_params(
            {
                "split": "test",
                "student": args.student,
                "student_run_id": student_run.info.run_id,
                "base_tag": args.base_tag,
                "teacher": "Qwen/Qwen3-8B-AWQ",
                "eval": EVAL_SETTINGS,
            }
        )
        metrics = flat_metrics(table)
        for key, pair in pairs.items():
            metrics[f"{key}.macro_f1_diff"] = pair["category_macro_f1"]["diff_b_minus_a"]
            metrics[f"{key}.macro_f1_diff_ci95_low"] = pair["category_macro_f1"]["diff_ci95"][0]
            metrics[f"{key}.macro_f1_diff_ci95_high"] = pair["category_macro_f1"]["diff_ci95"][1]
        mlflow.log_metrics(metrics)
        result["mlflow_run_id"] = run.info.run_id
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        mlflow.log_artifact(str(out_path))

    for key, value in evaluate.run_metrics(
        json.loads((args.bench / f"fields_zero_test_{args.student}.json").read_text()), "test"
    ).items():
        client.log_metric(student_run.info.run_id, key, value)

    params = student_run.data.params
    run_tags_ = student_run.data.tags
    dev = json.loads((args.bench / f"fields_zero_dev_{args.student}.json").read_text())["all_rows"]
    version_tags = {
        "base_model": "Qwen/Qwen3-1.7B",
        "base_model_license": "Apache-2.0",
        "adapter": "LoRA (QLoRA training on an NF4 base; served on the bf16 base in vLLM)",
        "data_version": run_tags_.get("data_version", ""),
        "git_sha": run_tags_.get("git_sha", ""),
        "git_dirty": run_tags_.get("git_dirty", ""),
        "train_run_id": student_run.info.run_id,
        "dev.category_macro_f1": dev["category_macro_f1"],
        "test.category_macro_f1": table["student"]["category_macro_f1"],
        "test.rule_valid_rate": table["student"]["rule_valid_rate"],
        "test_comparison_run_id": result["mlflow_run_id"],
        "labels": "silver (Granite 4.1 3B); scores are agreement, not correctness",
        **{f"config.{k}": v for k, v in params.items() if not k.startswith("len.")},
    }
    adapter = Path(train_record["adapter_dir"])
    result["registry"] = register(
        client, args.register_name, student_run.info.run_id, adapter, version_tags
    )
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    client.log_artifact(result["mlflow_run_id"], str(out_path))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
