"""evaluate.run_metrics: flat MLflow metrics from a fields score result."""

import pytest

from filinglens.train.evaluate import run_metrics


def test_run_metrics() -> None:
    rows = {
        "rows": 71,
        "rule_valid": 70,
        "category_macro_f1": 0.6,
        "category_macro_f1_ci95": (0.45, 0.7),
        "category_accuracy": 0.65,
        "severity_macro_f1": 0.3,
        "first_seen_exact": 0.4,
        "first_seen_within_1y": 0.55,
    }
    score = {
        "all_rows": rows,
        "rows_where_silver_labelers_agree": rows | {"category_macro_f1": 0.7},
    }
    m = run_metrics(score, "dev")
    assert m["dev.category_macro_f1"] == 0.6
    assert m["dev.rule_valid_rate"] == pytest.approx(70 / 71)
    assert m["dev.category_macro_f1_ci95_low"] == 0.45
    assert m["dev.agreed_rows.category_macro_f1"] == 0.7
    assert all(k.startswith("dev.") for k in m)
