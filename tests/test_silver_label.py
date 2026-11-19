"""Silver labeler: schema, prompt content, rule checks and parsing (no GPU needed)."""

import json

from filinglens.extract.silver_label import (
    CATEGORIES,
    check,
    messages,
    parse,
    schema,
    slug,
    summarize,
    user_message,
)

TASK = {
    "task_no": 3,
    "para_id": "acc:4",
    "ticker": "KLAC",
    "sector": "manufacturing",
    "period_year": 2020,
    "heading": "A change in accounting standards could affect results",
    "text": "New accounting standards and taxation rules have occurred and will continue.",
    "first_seen": {
        "suggested": 2018,
        "candidates": [{"year": 2018, "score": 1.0, "para_id": "x", "heading": "H", "text": "T"}],
    },
}


def answer(**kw):
    base = {"rationale": "r", "category": "regulatory_legal", "severity": 1}
    return base | {"first_seen_year": 2018, "unsure": False} | kw


def test_schema_lists_every_category_and_allows_null_for_not_a_risk():
    s = schema(last_year=2025)
    assert s["properties"]["category"]["enum"] == CATEGORIES and len(CATEGORIES) == 12
    assert s["properties"]["severity"]["enum"] == [1, 2, 3, None]
    assert s["properties"]["first_seen_year"]["enum"][0] == 2018
    assert s["properties"]["first_seen_year"]["enum"][-2:] == [2025, None]
    assert list(s["properties"])[0] == "rationale"


def test_user_message_has_paragraph_year_and_candidates():
    msg = user_message(TASK)
    assert "fiscal year: 2020" in msg and TASK["text"] in msg
    assert "- 2018 (1.00): H | T" in msg
    empty = user_message(TASK | {"first_seen": {"suggested": 2020, "candidates": []}})
    assert "no earlier 10-K" in empty
    assert [m["role"] for m in messages(TASK, "rules")] == ["system", "user"]


def test_check_rules():
    assert check(answer(), 2020) == []
    assert check(answer(first_seen_year=2021), 2020) == [
        "first_seen_year after the paragraph's year"
    ]
    assert check(answer(severity=None), 2020) == ["risk without severity"]
    na = answer(category="not_a_risk", severity=None, first_seen_year=None)
    assert check(na, 2020) == []
    assert check(na | {"severity": 2}, 2020) == ["not_a_risk with severity or year"]


def test_parse_valid_and_invalid():
    ok = parse(json.dumps(answer()), TASK)
    assert ok["valid"] and ok["category"] == "regulatory_legal" and ok["para_id"] == "acc:4"
    bad = parse('{"rationale": "cut off', TASK)
    assert not bad["valid"] and bad["problems"][0].startswith("invalid JSON")


def test_summarize_and_slug():
    records = [parse(json.dumps(answer()), TASK), parse("{", TASK)]
    stats = summarize(records)
    assert (stats["tasks"], stats["valid"]) == (2, 1)
    assert stats["categories"] == {"regulatory_legal": 1}
    assert slug("ibm-granite/granite-4.1-8b") == "ibm-granite-granite-4-1-8b"
