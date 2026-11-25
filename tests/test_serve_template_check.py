import pytest

from filinglens.serve.template_check import compare_prompts, first_diff


def test_first_diff():
    assert first_diff([1, 2, 3], [1, 2, 3]) is None
    assert first_diff([1, 2, 3], [1, 9, 3]) == 1
    assert first_diff([1, 2], [1, 2, 3]) == 2
    assert first_diff("abc", "abd") == 2


def test_compare_prompts_identical():
    a = [{"text": "x", "ids": [1, 2]}, {"text": "y", "ids": [3]}]
    result = compare_prompts(["p1", "p2"], a, [dict(x) for x in a])
    assert result["ok"] and result["ids_identical"] == 2 and result["first_difference"] is None
    assert result["prompt_tokens_min_max"] == [1, 2]


def test_compare_prompts_reports_first_difference():
    a = [{"text": "x", "ids": [1, 2, 3, 4]}, {"text": "y", "ids": [5]}]
    b = [{"text": "x", "ids": [1, 2, 3, 4]}, {"text": "y!", "ids": [6]}]
    result = compare_prompts(["p1", "p2"], a, b)
    assert not result["ok"]
    assert result["ids_differ"] == ["p2"] and result["text_identical"] == 1
    assert result["first_difference"] == {
        "para_id": "p2",
        "first_diff_token": 0,
        "a_ids": [5],
        "b_ids": [6],
    }


def test_compare_prompts_length_mismatch_raises():
    with pytest.raises(ValueError):
        compare_prompts(["p1"], [], [])
