"""OCR rows from word boxes, the cell-accuracy metric, and the test-split guard."""

import json

import pytest

from filinglens.extract.table_metrics import aggregate, match_rows, normalize_label, score_table
from filinglens.extract.tables import group_rows, load_pages, row_to_cells, words_to_rows

GOLD = [
    {"label": "(In millions)", "values": ["2022", "2021"]},
    {"label": "Operating activities", "values": ["3879", "5051"]},
    {"label": "Investing activities", "values": ["-1540", "-2378"]},
    {"label": "Total", "values": ["10", "20"]},
    {"label": "Total", "values": ["30", "40"]},
    {"label": "Liabilities:", "values": []},
]


def word(text, left, top, width=40, height=20):
    return {"text": text, "left": left, "top": top, "width": width, "height": height, "conf": 90}


def test_words_become_rows_and_cells_by_position():
    words = [
        word("Operating", 10, 100, 90),
        word("activities", 105, 101, 90),
        word("$", 600, 99, 10),
        word("3,879", 640, 100, 60),
        word("$", 800, 100, 10),
        word("5,051", 840, 102, 60),
        word("Investing", 10, 140, 90),
        word("activities", 105, 140, 90),
        word("(1,540)", 630, 141, 70),
        word("(2,378)", 830, 139, 70),
    ]
    rows = group_rows(words)
    assert [len(r) for r in rows] == [6, 4]
    assert row_to_cells(rows[0], 20) == ["Operating activities", "$", "3,879", "$", "5,051"]
    assert words_to_rows(words) == GOLD[1:3]


def test_label_with_a_year_stays_one_cell():
    words = [word("Notes", 10, 10, 50), word("due", 65, 10, 30), word("2025", 100, 10, 40)]
    words.append(word("1,000", 600, 10, 50))
    assert words_to_rows(words) == [{"label": "Notes due 2025", "values": ["1000"]}]


def test_perfect_prediction_scores_one():
    score = score_table(GOLD, GOLD)
    assert score["correct"] == score["gold_cells"] == 10
    assert aggregate([score])["cell_accuracy_micro"] == 1.0


def test_missing_row_does_not_shift_later_rows():
    pred = [GOLD[0], GOLD[2], GOLD[3], GOLD[4]]  # "Operating activities" missed
    assert score_table(GOLD, pred)["correct"] == 8


def test_repeated_labels_match_in_order_and_values_are_canonicalized():
    pred = [
        {"label": "TOTAL:", "values": ["$10", "20"]},
        {"label": "total", "values": ["30", "(40)"]},
    ]
    score = score_table(GOLD, pred)
    assert score["correct"] == 3  # 10, 20, 30 right; "(40)" is -40, not 40


def test_fuzzy_label_match_for_ocr_typos():
    pred = [{"label": "Operatlng activitles", "values": ["3,879", "5,051"]}]
    assert match_rows(GOLD, pred) == {1: 0}
    assert normalize_label("Revenue (1):") == "revenue"


def test_missing_column_and_extra_cells():
    pred = [{"label": "Operating activities", "values": ["3879"]}]
    pred.append({"label": "Noise", "values": ["1", "2", "3"]})
    score = score_table(GOLD, pred)
    assert (score["correct"], score["pred_cells"]) == (1, 4)
    overall = aggregate([score])
    assert overall["cell_accuracy_micro"] == 0.1 and overall["cell_precision_micro"] == 0.25


def test_test_pages_need_allow_test(tmp_path):
    path = tmp_path / "tables.jsonl"
    path.write_text(json.dumps({"page_id": "a", "split": "test"}) + "\n")
    with pytest.raises(PermissionError):
        load_pages("test", path)
    assert len(load_pages("test", path, allow_test=True)) == 1
    assert load_pages("dev", path) == []


def test_score_subcommand_scores_a_predictions_file(tmp_path):
    from filinglens.extract.tables import main

    tables = tmp_path / "tables.jsonl"
    page = {"page_id": "p1", "split": "dev", "rows": GOLD}
    tables.write_text(json.dumps(page) + "\n" + json.dumps(page | {"page_id": "p2"}) + "\n")
    pred = tmp_path / "tables_vlm_image_dev.jsonl"
    pred.write_text(json.dumps({"page_id": "p1", "rows": GOLD}) + "\n")
    args = ["score", "--split", "dev", "--pred", str(pred), "--tables", str(tables)]
    main([*args, "--bench", str(tmp_path)])
    result = json.loads((tmp_path / "tables_vlm_image_dev.json").read_text())
    assert result["pages_missing"] == 1
    assert result["overall"]["cell_accuracy_micro"] == 0.5
