"""figure_matches(): the numbers checker's comparison (hand-written by me, FL-M6 T6)."""

import pytest

from filinglens.agents.figures import figure_matches


def test_match_and_non_match_on_the_same_reference():
    # "$1.2 billion" vs 1,234,567,000 matches; "$1.3 billion" vs the same value does not
    assert figure_matches(1.2, 1, 1e9, 1_234_567_000)
    assert not figure_matches(1.3, 1, 1e9, 1_234_567_000)
    # more digits written, tighter window: "$1.23 billion" matches, "$1.24 billion" does not
    assert figure_matches(1.23, 2, 1e9, 1_234_567_000)
    assert not figure_matches(1.24, 2, 1e9, 1_234_567_000)


def test_half_a_unit_away_matches_and_just_past_it_does_not():
    # the boundary: exactly half a unit of the last digit away, and just past it
    # "$1.2 billion" covers 1.15e9 to 1.25e9, both ends included
    assert figure_matches(1.2, 1, 1e9, 1.25e9)
    assert not figure_matches(1.2, 1, 1e9, 1.25e9 + 1)
    assert figure_matches(1.2, 1, 1e9, 1.15e9)
    assert not figure_matches(1.2, 1, 1e9, 1.15e9 - 1)
    # "$812" (decimals=0, plain dollars) covers 811.5 to 812.5
    assert figure_matches(812, 0, 1.0, 812.5)
    assert figure_matches(812, 0, 1.0, 811.5)
    assert not figure_matches(812, 0, 1.0, 812.51)


def test_boundary_survives_float_error():
    # naive float math puts 0.155 and 0.145 a hair outside "15%"
    # (|0.155 - 0.15| -> 0.0050000000000000044);
    # on paper both are exactly half a unit away, so both must match
    assert figure_matches(15, 0, 0.01, 0.155)
    assert figure_matches(15, 0, 0.01, 0.145)
    assert not figure_matches(15, 0, 0.01, 0.1550001)


def test_percentages_with_and_without_decimals():
    # e.g. "15.0%" vs 0.1504, and a decimals=0 case such as "15%"
    assert figure_matches(15.0, 1, 0.01, 0.1504)
    assert not figure_matches(15.0, 1, 0.01, 0.1506)
    assert figure_matches(15, 0, 0.01, 0.154)
    assert not figure_matches(15, 0, 0.01, 0.156)
    # "12.5%" vs 0.125 exactly
    assert figure_matches(12.5, 1, 0.01, 0.125)


def test_negative_values_and_sign_mismatch():
    # e.g. "-$0.4 billion" vs -412,000,000 matches; vs +412,000,000 does not
    assert figure_matches(-0.4, 1, 1e9, -412_000_000)
    assert not figure_matches(-0.4, 1, 1e9, 412_000_000)
    assert not figure_matches(0.4, 1, 1e9, -412_000_000)
    # boundary is symmetric on the negative side: -0.35e9 and -0.45e9 are both in
    assert figure_matches(-0.4, 1, 1e9, -0.35e9)
    assert figure_matches(-0.4, 1, 1e9, -0.45e9)
    assert not figure_matches(-0.4, 1, 1e9, -0.45e9 - 1)


def test_bad_inputs_raise_value_error():
    # decimals < 0, scale == 0, scale < 0
    with pytest.raises(ValueError):
        figure_matches(1.2, -1, 1e9, 1.2e9)
    # decimals must be an int: not a float, and not a bool (True would act as 1)
    with pytest.raises(ValueError):
        figure_matches(1.2, 1.5, 1e9, 1.2e9)
    with pytest.raises(ValueError):
        figure_matches(1.2, True, 1e9, 1.2e9)
    with pytest.raises(ValueError):
        figure_matches(1.2, 1, 0, 1.2e9)
    with pytest.raises(ValueError):
        figure_matches(1.2, 1, -1e9, 1.2e9)
    # non-finite inputs are rejected rather than silently compared
    with pytest.raises(ValueError):
        figure_matches(1.2, 1, float("nan"), 1.2e9)
    with pytest.raises(ValueError):
        figure_matches(1.2, 1, 1e9, float("inf"))
