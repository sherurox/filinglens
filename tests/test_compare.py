"""Paired bootstrap comparison: invalid handling, identical models, a clear winner."""

from __future__ import annotations

import pytest

from filinglens.train import compare


def test_predicted_categories_marks_invalid_and_missing() -> None:
    rows = [{"para_id": "a"}, {"para_id": "b"}, {"para_id": "c"}]
    preds = {"a": {"valid": True, "category": "people"}, "b": {"valid": False, "category": "x"}}
    assert compare.predicted_categories(rows, preds) == ["people", "invalid", "invalid"]


def test_identical_models_have_zero_difference() -> None:
    pytest.importorskip("sklearn")
    gold = ["people", "macro_market", "people", "regulatory_legal"] * 5
    pred = ["people", "people", "people", "regulatory_legal"] * 5
    result = compare.paired_bootstrap(gold, pred, pred, n=200)
    for metric in ("category_macro_f1", "category_accuracy"):
        assert result[metric]["diff_b_minus_a"] == 0
        assert result[metric]["diff_ci95"] == [0.0, 0.0]
        assert result[metric]["share_b_not_better"] == 1.0  # never strictly better
    assert result["rows_where_only_a_right"] == result["rows_where_only_b_right"] == 0


def test_clear_winner() -> None:
    pytest.importorskip("sklearn")
    gold = ["people", "macro_market"] * 20
    a = ["people", "people"] * 20  # half right
    b = list(gold)  # all right
    result = compare.paired_bootstrap(gold, a, b, n=300)
    acc = result["category_accuracy"]
    assert acc["a"] == 0.5 and acc["b"] == 1.0 and acc["diff_b_minus_a"] == 0.5
    assert acc["diff_ci95"][0] > 0
    assert acc["share_b_not_better"] == 0.0
    assert result["rows_where_only_b_right"] == 20 and result["rows_where_only_a_right"] == 0
