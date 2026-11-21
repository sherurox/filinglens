"""Score a fine-tuned adapter with the FL-M2 scorer and log the result into its training run (T9).

The predictions come from `fields generate --lora data/interim/adapters/<run>/adapter` (GPU env).
This step runs in the main env: it calls `fields score` with the run name as the output suffix
(the same scorer and the same silver rows as the FL-M2 baseline), then logs the scores as
`<split>.<metric>` metrics on the MLflow run recorded in bench/sft_<run>.json, with the score
file as an artifact. One MLflow run then holds the training curves and the dev result.

    uv run python -m filinglens.train.evaluate --run-name silver-r16-lr0.0002-e2 --split dev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from filinglens.extract import fields

METRICS = (
    "category_macro_f1",
    "category_accuracy",
    "severity_macro_f1",
    "first_seen_exact",
    "first_seen_within_1y",
)


def run_metrics(score: dict, split: str) -> dict[str, float]:
    """Flat MLflow metrics from a `fields score` result."""
    rows = score["all_rows"]
    agreed = score["rows_where_silver_labelers_agree"]
    metrics = {f"{split}.{k}": float(rows[k]) for k in METRICS}
    metrics[f"{split}.rule_valid_rate"] = rows["rule_valid"] / rows["rows"]
    low, high = rows["category_macro_f1_ci95"]
    metrics[f"{split}.category_macro_f1_ci95_low"] = float(low)
    metrics[f"{split}.category_macro_f1_ci95_high"] = float(high)
    metrics[f"{split}.agreed_rows.category_macro_f1"] = float(agreed["category_macro_f1"])
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", required=True, help="as in bench/sft_<run-name>.json")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--variant", default="zero")
    parser.add_argument("--bench", type=Path, default=Path("bench"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    train_record = json.loads((args.bench / f"sft_{args.run_name}.json").read_text())
    score_argv = ["score", "--variant", args.variant, "--split", args.split]
    score_argv += ["--tag", args.run_name, "--bench", str(args.bench)]
    if args.allow_test:
        score_argv.append("--allow-test")
    fields.main(score_argv)
    score_path = args.bench / f"fields_{args.variant}_{args.split}_{args.run_name}.json"
    score = json.loads(score_path.read_text())
    metrics = run_metrics(score, args.split)

    import mlflow

    from filinglens.train.tracking import tracking_uri

    client = mlflow.MlflowClient(tracking_uri=tracking_uri())
    run_id = train_record["run_id"]
    for key, value in metrics.items():
        client.log_metric(run_id, key, value)
    client.log_artifact(run_id, str(score_path), artifact_path="eval")
    print(json.dumps({"run_id": run_id, **metrics}, indent=2))


if __name__ == "__main__":
    main()
