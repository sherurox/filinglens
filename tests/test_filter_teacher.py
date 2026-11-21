"""Teacher filter: the keep rule in its documented order, the heuristic, bands, outputs, guards."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from filinglens.train import filter_teacher as ft


def record(
    para_id: str = "p1",
    category: str = "macro_market",
    confidence: float | None = 0.99,
    valid: bool = True,
    consistent: bool | None = True,
    heading: str = "Inflation could raise our costs",
    cik: int = 1,
    task_no: int = 1,
) -> dict:
    risk = category != "not_a_risk"
    return {
        "para_id": para_id,
        "valid": valid,
        "category": category,
        "rationale": "r",
        "severity": 2 if risk else None,
        "first_seen_year": 2019 if risk else None,
        "unsure": False,
        "first_token_consistent": consistent,
        "confidence": None
        if confidence is None
        else {"strategy_corporate": 0.0, category: confidence},
        "teacher": "Qwen/Qwen3-8B-AWQ",
        "task": {
            "task_no": task_no,
            "para_id": para_id,
            "cik": cik,
            "period_year": 2022,
            "heading": heading,
            "text": "Some paragraph text about the risk.",
        },
    }


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({}, None),
        ({"valid": False}, "rule_invalid"),
        ({"category": "not_a_risk"}, "not_a_risk"),
        ({"consistent": False}, "inconsistent_first_token"),
        ({"consistent": None}, "inconsistent_first_token"),
        ({"confidence": 0.89}, "low_confidence"),
        ({"confidence": None}, "low_confidence"),
        ({"heading": "The COVID-19 pandemic could hurt demand"}, "heuristic_disagrees"),
        ({"heading": "The COVID-19 pandemic", "category": "health_pandemic"}, None),
    ],
)
def test_drop_reason(kwargs: dict, reason: str | None) -> None:
    assert ft.drop_reason(record(**kwargs)) == reason


def test_rules_apply_in_order() -> None:
    # invalid and low confidence and not_a_risk at once: the first rule in the order wins
    r = record(valid=False, category="not_a_risk", confidence=0.1, consistent=False)
    assert ft.drop_reason(r) == "rule_invalid"


def test_tau_boundary_is_inclusive() -> None:
    assert ft.drop_reason(record(confidence=0.9)) is None


def test_heuristic() -> None:
    def task(heading: str | None, text: str = "x") -> dict:
        return {"heading": heading, "text": text}

    assert ft.heuristic(task("Tariffs and trade policy")) == "geopolitical_international"
    assert ft.heuristic(task("A ransomware attack could disrupt us")) == "technology_cyber"
    assert ft.heuristic(task("Climate change regulation")) == "climate_environment"
    assert ft.heuristic(task("Competition is intense")) is None
    # two patterns fire: ambiguous, so the heuristic abstains
    assert ft.heuristic(task("Pandemic and war in Ukraine")) is None
    # no heading: the first 40 words of the text are used instead
    assert ft.heuristic(task(None, "Cyber incidents could expose data")) == "technology_cyber"
    assert ft.heuristic(task(float("nan"), "The pandemic hurt us")) == "health_pandemic"
    assert ft.heuristic(task("Software awards and hardware")) is None  # 'war' only as a word


def test_filtered_row_format() -> None:
    row = ft.filtered_row(record())
    assert set(row) == {"para_id", "task", "answer", "confidence", "teacher"}
    assert list(row["answer"]) == list(ft.ANSWER_KEYS)
    assert row["confidence"] == 0.99


def test_bands() -> None:
    counts = ft.bands([0.1, 0.6, 0.95, 0.995, 0.9995, 0.99999, 1.0, None])
    assert counts == {
        "[0.0, 0.5)": 1,
        "[0.5, 0.9)": 1,
        "[0.9, 0.99)": 1,
        "[0.99, 0.999)": 1,
        "[0.999, 0.9999)": 1,
        "[0.9999, 1.0]": 2,
        "none": 1,
    }


def write_fixture(tmp_path: Path, pool_cik: int = 1) -> list[str]:
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    pool0 = [record("a", task_no=1, cik=pool_cik), record("b", task_no=3, valid=False)]
    pool1 = [record("c", task_no=2, confidence=0.5), record("d", task_no=4)]
    for name, rows in zip(ft.POOL_FILES, (pool0, pool1), strict=True):
        (teacher / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    dev_pred = record("dev1", category="people", confidence=0.95)
    (teacher / ft.DEV_FILE).write_text(json.dumps(dev_pred) + "\n")
    gold = tmp_path / "gold"
    gold.mkdir()
    pd.DataFrame([{"cik": 1, "split": "train"}, {"cik": 9, "split": "test"}]).to_csv(
        gold / "splits.csv", index=False
    )
    pd.DataFrame({"para_id": ["s1"]}).to_csv(gold / "risk_sample.csv", index=False)
    dev_row = {
        "para_id": "dev1",
        "heading": "Key employees",
        "text": "We depend on key staff.",
        "category": "people",
        "severity": 2,
        "first_seen_year": 2019,
        "models_agree_category": True,
    }
    (gold / "risk_paragraphs.dev.jsonl").write_text(json.dumps(dev_row) + "\n")
    return [
        "--teacher-dir",
        str(teacher),
        "--gold",
        str(gold),
        "--out",
        str(tmp_path / "filtered.jsonl"),
        "--stats",
        str(tmp_path / "bench.json"),
    ]


def test_main(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    ft.main(write_fixture(tmp_path))
    kept = [json.loads(line) for line in (tmp_path / "filtered.jsonl").read_text().splitlines()]
    assert [r["para_id"] for r in kept] == ["a", "d"]  # sorted by task_no across shards
    stats = json.loads((tmp_path / "bench.json").read_text())
    assert stats["pool_rows"] == 4 and stats["kept"] == 2
    assert stats["dropped"] == {"low_confidence": 1, "rule_invalid": 1}
    assert stats["leakage"] == {
        "kept_rows_not_from_train_company": 0,
        "kept_rows_in_gold_sample": 0,
    }
    dev = stats["dev"]
    assert dev["teacher_vs_silver_all_rows"]["category_accuracy"] == 1.0
    assert dev["kept_by_threshold"]["0.9"] == {"kept": 1, "category_agreement_with_silver": 1.0}
    assert dev["kept_by_threshold"]["0.99"]["kept"] == 0


def test_main_refuses_non_train_company(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="leakage"):
        ft.main(write_fixture(tmp_path, pool_cik=9))
    assert not (tmp_path / "filtered.jsonl").exists()
