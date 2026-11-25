import json

import pytest

from filinglens.serve.parity import compare, load


def rec(pid, raw, cat="supply_chain", sev=2, year=2020, valid=True):
    return {
        "para_id": pid,
        "raw": raw,
        "category": cat,
        "severity": sev,
        "first_seen_year": year,
        "valid": valid,
    }


def test_compare_counts_raw_and_field_agreement():
    a = {"p1": rec("p1", "x"), "p2": rec("p2", "y"), "p3": rec("p3", "z")}
    b = {"p1": rec("p1", "x"), "p2": rec("p2", "y2", sev=3), "p3": rec("p3", "z2", cat="credit")}
    result = compare(a, b)
    assert result["rows_compared"] == 3 and result["raw_identical"] == 1
    assert result["raw_differs"] == ["p2", "p3"]
    assert result["agree"] == {"valid": 3, "category": 2, "severity": 2, "first_seen_year": 3}
    assert result["category_changes"] == [{"para_id": "p3", "a": "supply_chain", "b": "credit"}]


def test_compare_reports_one_sided_rows():
    result = compare({"p1": rec("p1", "x"), "p2": rec("p2", "y")}, {"p1": rec("p1", "x")})
    assert result["only_a"] == ["p2"] and result["rows_compared"] == 1


def test_compare_with_no_overlap_raises():
    with pytest.raises(ValueError):
        compare({"p1": rec("p1", "x")}, {"p2": rec("p2", "x")})


def test_load_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_text(json.dumps(rec("p1", "x")) + "\n" + json.dumps(rec("p1", "y")) + "\n")
    with pytest.raises(ValueError):
        load(path)
