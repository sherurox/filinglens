import json
import random

import pandas as pd

from filinglens.extract import synthetic_qa as sq
from filinglens.retrieval import extra_queries as eq


def para(pid, cik, acc, year, text, n_words=60):
    return {"para_id": pid, "cik": cik, "ticker": f"T{cik}", "sector": "retail",
            "period_year": year, "accession": acc, "heading": "Risks", "text": text,
            "n_words": n_words}  # fmt: skip


def words(tag, n=60):
    return " ".join(f"{tag}{i}" for i in range(n))


def test_per_company_draws():
    draws = eq.per_company_draws({1: 2, 2: 15, 3: 20}, target=15)
    assert draws == {1: 16, 2: 0, 3: 0}  # ceil(13 / 0.85) = 16; already at or above the target


def test_spread_draw_covers_every_filing_before_repeating_one():
    rows = [para(f"a{i}", 1, f"acc{i % 4}", 2020, "x") for i in range(12)]  # 4 filings x 3
    picked = eq.spread_draw(rows, 6, random.Random(0))
    filings = [r["accession"] for r in picked]
    assert len(picked) == 6 and len(set(filings[:4])) == 4  # one per filing in the first round
    assert len({r["para_id"] for r in picked}) == 6
    assert eq.spread_draw(rows, 50, random.Random(0)).__len__() == 12  # no more than there are
    assert eq.spread_draw(rows, 6, random.Random(0)) == picked  # seeded


def test_build_pool_filters():
    corpus = pd.DataFrame([
        para("t1", 1, "a", 2020, words("keep")),
        para("t2", 1, "a", 2020, words("held")),  # same text as at a held-out company
        para("h1", 9, "h", 2020, words("held")),
        para("t3", 1, "b", 2022, words("used")),  # the existing Q&A paragraph
        para("t4", 1, "c", 2024, words("used")),  # its copy in another year
        para("t5", 1, "d", 2025, words("dup")),
        para("t6", 1, "e", 2019, words("dup")),  # duplicate text within the pool
        para("o1", 5, "o", 2020, words("other")),  # not a ret-test company
        para("s1", 1, "f", 2020, words("short", 10), n_words=10),  # not eligible (< 40 words)
    ])  # fmt: skip
    splits = pd.DataFrame({"cik": [1, 5, 9], "split": ["train", "train", "test"]})
    pool, stats = eq.build_pool(corpus, splits, test_ciks={1}, used_para_ids={"t3"})
    assert sorted(r["para_id"] for r in pool) == ["t1", "t5"]
    assert stats["dropped_text_also_at_m2_dev_or_test_company"] == 1
    assert stats["dropped_text_of_an_existing_qa_paragraph"] == 2
    assert stats["dropped_duplicate_text_within_pool"] == 1


def test_prepare_counts_and_numbers_tasks():
    corpus = pd.DataFrame(
        [
            para(f"p{c}_{i}", c, f"acc{c}_{i % 3}", 2020, words(f"c{c}_{i}"))
            for c in (1, 2)
            for i in range(30)
        ]
    )
    splits = pd.DataFrame({"cik": [1, 2], "split": ["train", "train"]})
    test_queries = [{"cik": 1}] * 14 + [{"cik": 2}] * 3
    tasks, stats = eq.prepare(corpus, splits, test_queries, existing_qa=[])
    by_cik = pd.Series([t["cik"] for t in tasks]).value_counts().to_dict()
    assert by_cik == {1: 2, 2: 15}  # ceil(1/0.85)=2, ceil(12/0.85)=15
    assert [t["task_no"] for t in tasks] == list(range(1, len(tasks) + 1))
    assert stats["existing_test_queries"] == 17 and stats["companies_short_of_paragraphs"] == 0
    assert all("text_key" in t for t in tasks)


def test_finalize_runs_fl_m2_checks_and_extra_leakage_counts():
    text = "Our stores depend on steady consumer demand, and a recession could cut sales sharply."
    task = {"task_no": 1, "para_id": "acc9:3", "cik": 7, "ticker": "SHOP", "sector": "retail",
            "period_year": 2024, "accession": "acc9", "heading": "Demand", "text": text,
            "text_key": sq.text_key(text)}  # fmt: skip
    good = {"question": "What does SHOP's FY2024 10-K say a recession could do to sales?",
            "answer": "A recession could cut sales sharply.", "type": "impact",
            "evidence": "a recession could cut sales sharply"}  # fmt: skip
    bad = {**good, "question": "No ticker here at all, is there?"}
    raws = {1: json.dumps({"pair_1": good, "pair_2": bad})}
    splits = pd.DataFrame({"cik": [7, 8], "split": ["train", "test"]})
    corpus = pd.DataFrame([{"para_id": "acc9:3", "cik": 7, "text": text}])
    kept, checked, leaks = eq.finalize_rows([task], raws, "gen", splits, corpus, {7}, [])
    assert [r["qa_id"] for r in kept] == ["acc9:3:q1"] and kept[0]["batch"] == eq.BATCH
    assert "question_no_ticker" in checked[1]["problems"]
    assert not any(leaks.values())
    _, _, leaks = eq.finalize_rows(
        [task], raws, "gen", splits, corpus, {99}, [{"para_id": "acc9:3"}]
    )
    assert leaks["rows_from_non_ret_test_companies"] == 1
    assert leaks["rows_reusing_an_existing_qa_paragraph"] == 1


def test_outputs_never_overwrite_fl_m2_files():
    assert eq.OUTPUT != sq.OUTPUT and eq.TASKS != sq.TASKS and eq.WORK != sq.WORK
    assert eq.BENCH.name != "synthetic_qa.json" and eq.BENCH_RUN.name != "synthetic_qa_run.json"
