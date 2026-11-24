"""T12 helpers: JSON-valid count, table rows, determinism and row guards, MLflow metric names."""

from __future__ import annotations

import pytest

from filinglens.train import final_comparison as tc

SCORE = {
    "all_rows": {
        "rows": 4,
        "rule_valid": 3,
        "category_macro_f1": 0.5,
        "category_macro_f1_ci95": (0.3, 0.7),
        "category_accuracy": 0.75,
        "severity_macro_f1": 0.25,
        "first_seen_exact": 0.5,
        "first_seen_within_1y": 0.75,
    },
    "rows_where_silver_labelers_agree": {"category_macro_f1": 0.6},
}


def test_json_valid_count_is_looser_than_rule_valid() -> None:
    records = [{"raw": '{"category": "x"}'}, {"raw": "{broken"}, {"raw": None}, {}, {"raw": "[]"}]
    assert tc.json_valid_count(records) == 2


def test_model_row() -> None:
    records = [{"raw": "{}"}] * 4
    row = tc.model_row(SCORE, records, "max_num_seqs 1")
    assert row["rule_valid_rate"] == 0.75 and row["json_valid"] == 4
    assert row["json_valid_rate"] == 1.0 and row["agreed_rows_category_macro_f1"] == 0.6
    assert row["category_macro_f1_ci95"] == (0.3, 0.7) and row["decoding"] == "max_num_seqs 1"
    assert "json_valid" not in tc.model_row(SCORE, None, "batched")  # FL-M2 file: no records


def test_check_deterministic() -> None:
    tc.check_deterministic({"max_num_seqs": 1}, "base")
    for bad in ({"max_num_seqs": None}, {}):
        with pytest.raises(SystemExit, match="max_num_seqs"):
            tc.check_deterministic(bad, "base")


def test_check_rows() -> None:
    rows = [{"para_id": "a"}, {"para_id": "b"}]
    tc.check_rows({"a": {}, "b": {}, "c": {}}, rows, "teacher")
    with pytest.raises(SystemExit, match="1 test rows have no prediction"):
        tc.check_rows({"a": {}}, rows, "teacher")


def test_flat_metrics() -> None:
    table = {"student": tc.model_row(SCORE, [{"raw": "{}"}] * 4, "max_num_seqs 1")}
    m = tc.flat_metrics(table)
    assert m["student.category_macro_f1"] == 0.5
    assert (m["student.category_macro_f1_ci95_low"], m["student.category_macro_f1_ci95_high"]) == (
        0.3,
        0.7,
    )
    assert "student.decoding" not in m and m["student.rows"] == 4.0


def test_cli_needs_allow_test() -> None:
    with pytest.raises(SystemExit):
        tc.main([])
