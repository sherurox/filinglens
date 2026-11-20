"""SFT data: prompt identical to the eval prompt, completion format, arms, guards."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from filinglens.extract.fields import conversations
from filinglens.train import sft_data as sd

PROMPT = "You label paragraphs.\n"
FIRST_SEEN = {
    "suggested": 2018,
    "candidates": [
        {"year": 2018, "score": 0.8, "para_id": "p0", "heading": "H", "text": "earlier text"}
    ],
}


def silver_row(para_id: str, cik: int, category: str = "people") -> dict:
    return {
        "para_id": para_id,
        "task_no": 1,
        "cik": cik,
        "ticker": "AAA",
        "sector": "services",
        "period_year": 2020,
        "split": "train",
        "heading": "We depend on key staff",
        "text": "Losing our chief executive could hurt us.",
        "category": category,
        "severity": 2,
        "first_seen_year": 2018,
    }


def raw_answer(para_id: str, category: str = "people") -> dict:
    return {
        "para_id": para_id,
        "rationale": "Key personnel risk.",
        "category": category,
        "severity": 2,
        "first_seen_year": 2018,
        "unsure": False,
        "valid": True,
    }


def teacher_row(para_id: str, category: str = "macro_market", cik: int = 1) -> dict:
    task = {
        "para_id": para_id,
        "cik": cik,
        "ticker": "AAA",
        "sector": "services",
        "period_year": 2022,
        "heading": "Inflation",
        "text": "Inflation could raise our costs.",
        "first_seen": FIRST_SEEN,
    }
    answer = {
        "rationale": "Inflation is a macro risk.",
        "category": category,
        "severity": None if category == "not_a_risk" else 1,
        "first_seen_year": None if category == "not_a_risk" else 2019,
        "unsure": False,
    }
    return {"para_id": para_id, "task": task, "answer": answer, "confidence": 0.9}


def test_silver_prompt_is_the_eval_prompt() -> None:
    rows = [silver_row("s1", 1), silver_row("s2", 1)]
    tasks = {r["para_id"]: {"first_seen": FIRST_SEEN} for r in rows}
    raw = {r["para_id"]: raw_answer(r["para_id"]) for r in rows}
    examples = sd.silver_examples(rows, tasks, raw, PROMPT)
    expected = conversations(rows, tasks, PROMPT, shots=[])
    assert [e["prompt"] for e in examples] == expected
    assert [m["role"] for m in examples[0]["prompt"]] == ["system", "user"]


def test_completion_is_schema_json_in_order() -> None:
    rows = [silver_row("s1", 1)]
    tasks = {"s1": {"first_seen": FIRST_SEEN}}
    example = sd.silver_examples(rows, tasks, {"s1": raw_answer("s1")}, PROMPT)[0]
    (turn,) = example["completion"]
    assert turn["role"] == "assistant"
    assert list(json.loads(turn["content"])) == list(sd.ANSWER_KEYS)
    assert turn["content"].startswith('{"rationale":"')  # compact, no spaces
    assert json.loads(turn["content"]) == {
        "rationale": "Key personnel risk.",
        "category": "people",
        "severity": 2,
        "first_seen_year": 2018,
        "unsure": False,
    }


def test_silver_raw_disagreement_raises() -> None:
    rows = [silver_row("s1", 1, category="people")]
    tasks = {"s1": {"first_seen": FIRST_SEEN}}
    with pytest.raises(ValueError, match="differ on category"):
        sd.silver_examples(rows, tasks, {"s1": raw_answer("s1", "macro_market")}, PROMPT)


def test_teacher_examples_drop_not_a_risk_and_use_eval_prompt() -> None:
    rows = [teacher_row("t1"), teacher_row("t2", category="not_a_risk")]
    examples, dropped = sd.teacher_examples(rows, PROMPT)
    assert [e["id"] for e in examples] == ["t1"]
    assert dropped == {"not_a_risk": 1}
    task_view = rows[0]["task"]
    assert examples[0]["prompt"] == conversations([task_view], {"t1": task_view}, PROMPT, [])[0]
    kept, _ = sd.teacher_examples(rows, PROMPT, drop_not_a_risk=False)
    assert len(kept) == 2


def test_teacher_rule_violation_raises() -> None:
    row = teacher_row("t1")
    row["answer"]["first_seen_year"] = 2030  # after the paragraph's own year
    with pytest.raises(ValueError, match="breaks the rules"):
        sd.teacher_examples([row], PROMPT)


def test_combine_keeps_silver_on_overlap() -> None:
    silver = [{"id": "a", "source": "silver"}]
    teacher = [{"id": "a", "source": "teacher"}, {"id": "b", "source": "teacher"}]
    both, overlap = sd.combine(silver, teacher)
    assert [(e["id"], e["source"]) for e in both] == [("a", "silver"), ("b", "teacher")]
    assert overlap == 1


def write_fixture(tmp_path: Path, teacher_cik: int = 1) -> list[str]:
    gold = tmp_path / "gold"
    gold.mkdir()
    pd.DataFrame(
        [
            {"cik": 1, "ticker": "AAA", "sector": "services", "split": "train"},
            {"cik": 9, "ticker": "ZZZ", "sector": "services", "split": "test"},
        ]
    ).to_csv(gold / "splits.csv", index=False)
    (gold / "MANIFEST.json").write_text("{}\n")
    rows = [silver_row("s1", 1), silver_row("s2", 1, category="people")]
    (gold / "risk_paragraphs.train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    tasks = [{"para_id": r["para_id"], "first_seen": FIRST_SEEN} for r in rows]
    (tmp_path / "tasks.json").write_text(json.dumps(tasks))
    raw = [raw_answer(r["para_id"]) for r in rows]
    (tmp_path / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw))
    (tmp_path / "prompt.md").write_text(PROMPT)
    teacher = [
        teacher_row("t1", cik=teacher_cik),
        teacher_row("s1"),
        teacher_row("t3", "not_a_risk"),
    ]
    (tmp_path / "teacher.jsonl").write_text("".join(json.dumps(r) + "\n" for r in teacher))
    return [
        "--gold",
        str(gold),
        "--tasks",
        str(tmp_path / "tasks.json"),
        "--prompt",
        str(tmp_path / "prompt.md"),
        "--silver-raw",
        str(tmp_path / "raw.jsonl"),
        "--out",
        str(tmp_path / "out"),
        "--stats",
        str(tmp_path / "bench.json"),
    ]


def test_main_silver_only_then_all_arms(tmp_path: Path) -> None:
    argv = write_fixture(tmp_path)
    sd.main(argv)
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == ["MANIFEST.json", "silver.jsonl"]

    sd.main([*argv, "--teacher", str(tmp_path / "teacher.jsonl")])
    assert sorted(p.name for p in out.iterdir()) == [
        "MANIFEST.json",
        "both.jsonl",
        "silver.jsonl",
        "teacher.jsonl",
    ]
    stats = json.loads((tmp_path / "bench.json").read_text())
    assert stats["arms"]["silver"]["examples"] == 2
    assert stats["arms"]["teacher"]["examples"] == 2  # t1 and s1; t3 is not_a_risk
    assert stats["arms"]["both"]["examples"] == 3  # s1 from silver only once
    assert stats["teacher"]["dropped"] == {"not_a_risk": 1}
    assert stats["teacher"]["overlap_with_silver_dropped"] == 1
    assert stats["examples_from_non_train_companies"] == {"silver": 0, "teacher": 0, "both": 0}
    assert len(stats["data_version"]) == 64

    sd.main(argv)  # back to silver only: no stale teacher/both files
    assert sorted(p.name for p in out.iterdir()) == ["MANIFEST.json", "silver.jsonl"]


def test_main_refuses_non_train_company(tmp_path: Path) -> None:
    argv = write_fixture(tmp_path, teacher_cik=9)
    with pytest.raises(SystemExit, match="non-train companies"):
        sd.main([*argv, "--teacher", str(tmp_path / "teacher.jsonl")])
    assert not (tmp_path / "out").exists()
