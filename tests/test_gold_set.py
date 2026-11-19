"""Silver set: rows kept and dropped, text-drift check, split files, and the test-split guard."""

import hashlib
import json

import pandas as pd
import pytest

from filinglens.extract.gold_set import build_rows, load_split, main, write_split_files


def text(i):
    return f"Paragraph {i} about suppliers."


def sha(t):
    return hashlib.sha256(t.encode()).hexdigest()


SPLIT_ORDER = ["train", "dev", "test", "train"]
SAMPLE = pd.DataFrame(
    {
        "para_id": f"p{i}",
        "task_no": i,
        "cik": 100 + i,
        "ticker": f"T{i}",
        "sector": "retail",
        "period_year": 2024,
        "split": split,
        "source": "random",
        "text_sha256": sha(text(i)),
    }
    for i, split in enumerate(SPLIT_ORDER, start=1)
).set_index("para_id")
TASKS = [{"para_id": f"p{i}", "heading": "H", "text": text(i)} for i in range(1, 5)]


def answer(i, **kw):
    base = {"para_id": f"p{i}", "valid": True, "category": "operations_supply", "severity": 2}
    return base | {"first_seen_year": 2020, "unsure": False} | kw


PRIMARY = [
    answer(1),
    answer(2, category="people"),
    answer(3),
    answer(4, valid=False, problems=["risk without severity"]),
]
SECONDARY = [answer(1), answer(2), answer(3, valid=False), answer(4)]


def test_build_rows_keeps_valid_risks_and_flags_agreement():
    rows, dropped = build_rows(SAMPLE, TASKS, PRIMARY, SECONDARY)
    assert [r["para_id"] for r in rows] == ["p1", "p2", "p3"]
    assert dropped == {"primary answer broke the rules": 1}
    agree = {r["para_id"]: r["models_agree_category"] for r in rows}
    assert agree == {"p1": True, "p2": False, "p3": False}  # p3: second answer invalid
    assert rows[2]["check_category"] is None
    assert rows[0]["label_source"] == "silver:ibm-granite/granite-4.1-3b"


def test_not_a_risk_is_dropped():
    primary = [answer(1, category="not_a_risk", severity=None, first_seen_year=None)]
    rows, dropped = build_rows(SAMPLE, TASKS, primary, [])
    assert rows == [] and dropped == {"primary said not_a_risk": 1}


def test_changed_text_is_an_error():
    tasks = [dict(TASKS[0], text="edited")] + TASKS[1:]
    with pytest.raises(ValueError, match="text changed"):
        build_rows(SAMPLE, tasks, PRIMARY[:1], [])


def test_split_files_and_test_guard(tmp_path):
    rows, _ = build_rows(SAMPLE, TASKS, PRIMARY, SECONDARY)
    write_split_files(rows, tmp_path)
    assert [r["para_id"] for r in load_split("train", tmp_path)] == ["p1"]
    with pytest.raises(PermissionError):
        load_split("test", tmp_path)
    assert [r["para_id"] for r in load_split("test", tmp_path, allow_test=True)] == ["p3"]
    with pytest.raises(ValueError):
        load_split("holdout", tmp_path)


def test_main_writes_manifest_with_hashes(tmp_path):
    gold, silver = tmp_path / "gold", tmp_path / "silver"
    gold.mkdir()
    silver.mkdir()
    SAMPLE.reset_index().to_csv(gold / "risk_sample.csv", index=False)
    (gold / "splits.csv").write_text("cik,split\n")
    (tmp_path / "tasks.json").write_text(json.dumps(TASKS))
    for name, records in (
        ("ibm-granite-granite-4-1-3b", PRIMARY),
        ("qwen-qwen3-4b-awq", SECONDARY),
    ):
        (silver / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    main(
        ["--gold", str(gold), "--tasks", str(tmp_path / "tasks.json"), "--silver-dir", str(silver)]
    )
    manifest = json.loads((gold / "MANIFEST.json").read_text())
    assert (manifest["kept"], manifest["by_split"]) == (3, {"dev": 1, "test": 1, "train": 1})
    assert manifest["labels"].startswith("silver")
    assert len(manifest["files"]) == 7 and (gold / "silver_raw").is_dir()
