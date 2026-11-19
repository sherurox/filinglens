import json

import pandas as pd
import pytest

from filinglens.extract.synthetic_qa import (
    build_rows,
    check_pair,
    check_task,
    draw,
    leakage,
    mentions,
    prepare,
    schema,
    squash,
    text_key,
    user_message,
)

PARA = (
    "Our operations depend on a small number of suppliers for specialized semiconductor "
    "components. A disruption at any of these suppliers, including one caused by natural "
    "disasters or export restrictions, could delay our shipments and reduce our revenue. "
    "We are qualifying second sources for several components."
)
TASK = {
    "task_no": 1,
    "para_id": "0001-24-000001:7",
    "cik": 10,
    "ticker": "XYZ",
    "sector": "manufacturing",
    "period_year": 2024,
    "heading": "Supply chain",
    "text": PARA,
    "text_key": text_key(PARA),
}
GOOD = {
    "question": "According to its FY2024 10-K, what could interrupt XYZ's supply of chips?",
    "answer": "A disruption at its few specialized semiconductor suppliers.",
    "evidence": "A disruption at any of these suppliers, including one caused by natural disasters",
    "type": "cause",
}


def test_text_key_ignores_case_and_spacing():
    assert text_key("Risk  factors\nApply") == text_key("risk factors apply")
    assert text_key("a") != text_key("b")


def test_squash_straightens_quotes():
    assert squash("the company’s   costs") == "the company's costs"


def test_mentions_whole_ticker_only():
    assert mentions("What does brk-b say?", "BRK-B")
    assert mentions("How is XYZ's margin?", "XYZ")
    assert not mentions("What does XYZW say?", "XYZ")


def test_good_pair_passes():
    assert check_pair(GOOD, TASK, None) == []


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"question": "What could interrupt XYZ's supply of chips"}, "question_form"),
        (
            {"question": "According to its FY2024 10-K, what could interrupt the chips?"},
            "question_no_ticker",
        ),
        (
            {"question": "XYZ: a disruption at any of these suppliers, including one caused by?"},
            "question_copies_paragraph",
        ),
        ({"answer": ""}, "answer_length"),
        ({"evidence": "natural disasters"}, "evidence_length"),
        (
            {"evidence": "A disruption at any supplier, including ones caused by floods or fires"},
            "evidence_not_verbatim",
        ),
        (
            {"answer": "Tariffs imposed by foreign governments on imported steel goods."},
            "answer_not_from_paragraph",
        ),
        ({"type": "opinion"}, "type"),
    ],
)
def test_each_check_fires(change, problem):
    assert problem in check_pair(GOOD | change, TASK, None)


def test_evidence_with_curly_quote_and_newline_is_verbatim():
    task = TASK | {
        "text": "We rely on the company’s\nlargest   customer for most of our sales today."
    }
    pair = GOOD | {"evidence": "We rely on the company's largest customer for most of our sales"}
    assert "evidence_not_verbatim" not in check_pair(pair, task, None)


def test_evidence_may_quote_the_heading_in_any_case():
    task = TASK | {"heading": "We depend on a few suppliers for critical semiconductor parts"}
    pair = GOOD | {"evidence": "we depend on a few suppliers for critical semiconductor parts"}
    assert check_pair(pair, task, None) == []


def test_evidence_may_not_span_heading_and_text():
    task = TASK | {"heading": "Supply risk"}
    pair = GOOD | {"evidence": "Supply risk Our operations depend on a small number of suppliers"}
    assert "evidence_not_verbatim" in check_pair(pair, task, None)


def test_duplicate_question_is_flagged():
    assert check_pair(GOOD, TASK, GOOD["question"].upper()) == ["duplicate_question"]


def test_check_task_handles_bad_json():
    assert [r["problems"] for r in check_task(TASK, "{not json")] == [["unparseable"]] * 2


def test_build_rows_marks_synthetic_train_and_drops_failures():
    second = GOOD | {"question": "What is XYZ doing about it", "type": "mitigation"}  # no "?"
    raw = json.dumps({"pair_1": GOOD, "pair_2": second})
    kept, checked = build_rows([TASK], {1: raw}, "Qwen/Qwen3-4B-AWQ")
    assert len(checked) == 2 and len(kept) == 1
    row = kept[0]
    assert (row["synthetic"], row["split"], row["qa_id"]) == (True, "train", "0001-24-000001:7:q1")
    assert checked[1]["problems"] == ["question_form"]


def test_leakage_counts_each_kind():
    rows = [
        {"cik": 10, "para_id": "p1", "synthetic": True, "split": "train"},
        {"cik": 20, "para_id": "p2", "synthetic": True, "split": "train"},  # dev company
        {"cik": 10, "para_id": "p3", "synthetic": False, "split": "train"},
    ]
    tasks = {"p1": {"text_key": "k1"}, "p2": {"text_key": "k2"}, "p3": {"text_key": "shared"}}
    assert leakage(rows, {10}, {"shared"}, tasks) == {
        "rows_from_non_train_companies": 1,
        "rows_with_source_text_at_dev_or_test_company": 1,
        "rows_not_marked_synthetic_train": 1,
    }


def test_draw_one_per_filing_and_per_cell_cap():
    pool = [
        {"para_id": f"a{i}:{j}", "accession": f"a{i}", "sector": "retail", "period_year": 2020}
        for i in range(5)
        for j in range(3)
    ]
    picked = draw(pool, per_cell=3, seed=1)
    assert len(picked) == 3 and len({p["accession"] for p in picked}) == 3
    assert picked == draw(pool, per_cell=3, seed=1)


def test_prepare_uses_train_companies_and_drops_shared_boilerplate():
    words = " ".join(["word"] * 50)
    boiler = "Boilerplate shared text " + words
    corpus = pd.DataFrame(
        [
            # train company 1: one unique paragraph, one copied boilerplate paragraph
            ("t:1", 1, "AAA", "retail", 2020, "t", "h", "Unique train text " + words),
            ("t:2", 1, "AAA", "retail", 2020, "t", "h", boiler),
            # dev company 2 has the same boilerplate, plus its own text
            ("d:1", 2, "BBB", "retail", 2020, "d", "h", boiler),
            ("d:2", 2, "BBB", "retail", 2020, "d", "h", "Dev only text " + words),
        ],
        columns=[
            "para_id",
            "cik",
            "ticker",
            "sector",
            "period_year",
            "accession",
            "heading",
            "text",
        ],
    ).assign(n_words=lambda df: df["text"].str.split().str.len())
    splits = pd.DataFrame({"cik": [1, 2], "split": ["train", "dev"]})
    tasks, stats = prepare(corpus, splits, per_cell=5, seed=1)
    assert [t["para_id"] for t in tasks] == ["t:1"]
    assert stats["dropped_text_also_at_dev_or_test_company"] == 1
    assert tasks[0]["cik"] == 1 and tasks[0]["task_no"] == 1


def test_schema_and_prompt_message():
    s = schema()
    assert s["required"] == ["pair_1", "pair_2"]
    assert s["properties"]["pair_1"]["properties"]["type"]["enum"][0] == "factual"
    msg = user_message(TASK)
    assert "Ticker: XYZ" in msg and "FY2024" in msg and PARA in msg
