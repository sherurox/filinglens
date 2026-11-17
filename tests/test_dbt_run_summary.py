"""dbt run_results.json → bench summary, on a hand-made two-node example."""

from filinglens.bench.dbt_run_summary import summarise

RUN_RESULTS = {
    "metadata": {"dbt_version": "1.12.7", "generated_at": "2026-11-17T06:30:00Z"},
    "args": {"invocation_command": "dbt build --target bigquery", "target": "bigquery"},
    "elapsed_time": 12.345,
    "results": [
        {
            "unique_id": "model.filinglens.fct_financials",
            "status": "success",
            "execution_time": 4.567,
            "adapter_response": {"rows_affected": 10, "bytes_processed": 300, "bytes_billed": 1000},
            "failures": None,
        },
        {
            "unique_id": "test.filinglens.not_null_fct_financials_value.abc",
            "status": "pass",
            "execution_time": 1.0,
            "adapter_response": {"bytes_processed": 50, "bytes_billed": 1000},
            "failures": 0,
        },
    ],
}


def test_summarise_totals_bytes_and_counts_by_type_and_status():
    s = summarise(RUN_RESULTS)
    assert s["counts"] == {"model:success": 1, "test:pass": 1}
    assert s["total_bytes_processed"] == 350
    assert s["total_bytes_billed"] == 2000
    assert s["elapsed_seconds"] == 12.35
    assert s["command"] == "dbt build --target bigquery"
    assert [n["node"] for n in s["nodes"]] == sorted(n["node"] for n in s["nodes"])


def test_summarise_handles_nodes_without_adapter_response():
    rr = {**RUN_RESULTS, "results": [{**RUN_RESULTS["results"][0], "adapter_response": {}}]}
    s = summarise(rr)
    assert s["total_bytes_processed"] == 0 and s["nodes"][0]["bytes_billed"] is None
