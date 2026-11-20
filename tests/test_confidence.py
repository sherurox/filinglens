"""Tests for filinglens.train.confidence.category_confidence."""

import math

import pytest

from filinglens.train.confidence import category_confidence


def test_equal_logprobs_split_evenly() -> None:
    result = category_confidence({10: -0.5, 20: -0.5}, {"a": 10, "b": 20})
    assert result == pytest.approx({"a": 0.5, "b": 0.5})


def test_sums_to_one_and_ignores_other_tokens() -> None:
    result = category_confidence(
        {10: -0.1, 20: -2.3, 30: -4.0, 99: -1.0},
        {"a": 10, "b": 20, "c": 30},
    )
    assert set(result) == {"a", "b", "c"}
    assert sum(result.values()) == pytest.approx(1.0)


def test_stable_for_very_negative_logprobs() -> None:
    assert math.exp(-1000.0) == 0.0
    result = category_confidence({10: -1000.0, 20: -1001.0}, {"a": 10, "b": 20})
    assert result == pytest.approx({"a": 0.7310585786, "b": 0.2689414214})


def test_missing_category_gets_zero() -> None:
    result = category_confidence({10: -0.2, 20: -1.9}, {"a": 10, "b": 20, "c": 77})
    assert result["c"] == 0.0
    assert result["a"] + result["b"] == pytest.approx(1.0)


def test_shared_first_token_raises() -> None:
    with pytest.raises(ValueError, match="share first token"):
        category_confidence({10: -0.1}, {"a": 10, "b": 10})


def test_none_present_gives_all_zeros() -> None:
    result = category_confidence({99: -0.1}, {"a": 10, "b": 20})
    assert result == {"a": 0.0, "b": 0.0}


def test_infinite_logprobs_count_as_absent() -> None:
    assert category_confidence({10: -math.inf, 20: -math.inf}, {"a": 10, "b": 20}) == {
        "a": 0.0,
        "b": 0.0,
    }
    assert category_confidence({10: -math.inf, 20: -0.5}, {"a": 10, "b": 20}) == {
        "a": 0.0,
        "b": 1.0,
    }
