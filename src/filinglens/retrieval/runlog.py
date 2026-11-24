"""One MLflow run per FL-M3 step, in the `fl-m3-retrieval` experiment.

Tracking is a side channel: if the server is down, the error goes into the bench record and the
step's own results are still written (the FL-M4 rule, QA Q126).
"""

from __future__ import annotations

from pathlib import Path


def log_run(
    run_name: str,
    record: dict,
    params: dict,
    metrics: dict[str, float],
    data_manifest: Path,
    note: str,
) -> str | None:
    """Log params, metrics and the whole record as JSON; the run ID, or None on failure."""
    try:
        import mlflow

        from filinglens.train.tracking import run_tags, start_run

        tags = run_tags(data_manifest=data_manifest, note=note)
        with start_run("retrieval", run_name, tags=tags):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            mlflow.log_dict(record, f"{run_name}.json")
            return mlflow.active_run().info.run_id
    except Exception as error:  # side channel: record it, keep the work
        record["mlflow_error"] = f"{type(error).__name__}: {error}"
        return None
