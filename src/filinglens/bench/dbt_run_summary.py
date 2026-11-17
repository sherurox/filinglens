"""Summarise dbt's target/run_results.json into a small JSON file for bench/.

dbt writes run_results.json after every build; it is overwritten by the next run and lives in
git-ignored target/, so the numbers that matter are copied out here. On BigQuery each node's
adapter_response carries bytes_processed / bytes_billed, the figures the 1 TB free tier counts.

Run from the repo root after a build:
    uv run python -m filinglens.bench.dbt_run_summary --out bench/dbt_build_bigquery.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def summarise(run_results: dict[str, Any]) -> dict[str, Any]:
    nodes = []
    for r in run_results["results"]:
        response = r.get("adapter_response") or {}
        nodes.append(
            {
                "node": r["unique_id"],
                "status": r["status"],
                "seconds": round(r["execution_time"], 2),
                "rows_affected": response.get("rows_affected"),
                "bytes_processed": response.get("bytes_processed"),
                "bytes_billed": response.get("bytes_billed"),
                "failures": r.get("failures"),
            }
        )
    args = run_results.get("args", {})
    by_status = Counter(f"{n['node'].split('.')[0]}:{n['status']}" for n in nodes)
    return {
        "dbt_version": run_results["metadata"]["dbt_version"],
        "generated_at": run_results["metadata"]["generated_at"],
        "command": args.get("invocation_command"),
        "target": args.get("target"),
        "elapsed_seconds": round(run_results["elapsed_time"], 2),
        "counts": dict(sorted(by_status.items())),
        "total_bytes_processed": sum(n["bytes_processed"] or 0 for n in nodes),
        "total_bytes_billed": sum(n["bytes_billed"] or 0 for n in nodes),
        "nodes": sorted(nodes, key=lambda n: n["node"]),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-results", type=Path, default=Path("dbt/target/run_results.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = summarise(json.loads(args.run_results.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "nodes"}, indent=2))


if __name__ == "__main__":
    main()
