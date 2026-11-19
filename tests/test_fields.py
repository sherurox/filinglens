"""Prompted baseline: examples, prompt structure, scoring and the test guard (no GPU needed)."""

import json

import pytest

from filinglens.extract.fields import (
    RISK_CATEGORIES,
    conversations,
    example_turns,
    example_view,
    load_rows,
    pick_examples,
    scores,
)


def row(i, category, agree=True, severity=2, year=2019):
    base = {"para_id": f"p{i}", "task_no": i, "ticker": "T", "sector": "retail"}
    text = {"period_year": 2024, "heading": "H", "text": " ".join(["word"] * 120)}
    labels = {"category": category, "severity": severity, "first_seen_year": year}
    return base | text | labels | {"models_agree_category": agree}


def tasks_for(rows):
    candidate = {"score": 0.9, "para_id": "x", "heading": "h", "text": "t " * 50}
    candidates = [candidate | {"year": y} for y in (2018, 2019, 2020)]
    return {
        r["para_id"]: {"first_seen": {"suggested": 2019, "candidates": candidates}} for r in rows
    }


def test_pick_examples_one_agreed_row_per_category_lowest_task_first():
    train = [row(3, "people"), row(1, "people", agree=False), row(2, "macro_market")]
    train.append(row(4, "people"))
    picked = pick_examples(train)
    assert [r["para_id"] for r in picked] == ["p2", "p3"]  # category order follows RISK_CATEGORIES


def test_example_view_is_short_and_keeps_only_the_labeled_year_candidate():
    r = row(1, "people")
    view = example_view(r, tasks_for([r]))
    assert view["text"].endswith(" ...") and len(view["text"].split()) == 91
    assert [c["year"] for c in view["first_seen"]["candidates"]] == [2019]


def test_conversation_is_system_then_examples_then_the_question():
    rows = [row(1, "people"), row(2, "macro_market")]
    tasks = tasks_for(rows)
    shots = example_turns(rows[:1], tasks, {"p1": "talent risk"})
    assert [t["role"] for t in shots] == ["user", "assistant"]
    assert json.loads(shots[1]["content"])["rationale"] == "talent risk"
    convs = conversations(rows, tasks, "RULES", shots)
    assert [m["role"] for m in convs[0]] == ["system", "user", "assistant", "user"]
    assert convs[0][0]["content"] == "RULES"


def test_scores_count_invalid_answers_as_wrong():
    pytest.importorskip("sklearn")
    rows = [row(1, "people"), row(2, "macro_market", severity=3, year=2020)]
    preds = {
        "p1": {"valid": True, "category": "people", "severity": 2, "first_seen_year": 2019},
        "p2": {"valid": False, "category": "macro_market", "severity": 3, "first_seen_year": 2021},
    }
    s = scores(rows, preds)
    assert s["rule_valid"] == 1 and s["category_accuracy"] == 0.5
    assert s["first_seen_exact"] == 0.5 and s["first_seen_within_1y"] == 1.0
    assert 0 < s["category_macro_f1"] < 1 and len(s["category_macro_f1_ci95"]) == 2
    assert len(RISK_CATEGORIES) == 11


def test_not_a_risk_prediction_counts_as_wrong_severity():
    pytest.importorskip("sklearn")
    rows = [row(1, "people"), row(2, "people", severity=1)]
    preds = {
        "p1": {"valid": True, "category": "not_a_risk", "severity": None, "first_seen_year": None},
        "p2": {"valid": True, "category": "people", "severity": 1, "first_seen_year": 2019},
    }
    s = scores(rows, preds)
    assert s["category_accuracy"] == 0.5 and s["first_seen_exact"] == 0.5
    assert 0 < s["severity_macro_f1"] < 1


def test_test_split_guard(tmp_path):
    with pytest.raises(PermissionError):
        load_rows("test", allow_test=False, gold=tmp_path)
