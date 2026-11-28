"""Numbers checker: finding figures, skipping non-figures, matching against the tools' values."""

import pytest

from filinglens.agents.numbers import (
    Allowed,
    allowed_values,
    check_answer,
    find_figures,
    margin_changes,
)


def figs(text):
    return [(f.stated, f.decimals, f.scale, f.kind, f.signed) for f in find_figures(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Revenue was $1.23 billion.", [(1.23, 2, 1e9, "money", False)]),
        ("a loss of -$0.4 billion", [(-0.4, 1, 1e9, "money", True)]),
        ("US$812 million in costs", [(812.0, 0, 1e6, "money", False)]),
        (
            "$5,512 and 1,234,567,000",
            [(5512.0, 0, 1.0, "money", False), (1234567000.0, 0, 1.0, "money", False)],
        ),
        ("a margin of 12.5% (12.5 percent)", [(12.5, 1, 0.01, "percent", False)] * 2),
        ("down 3.2 percentage points, or 3.2 pp", [(3.2, 1, 0.01, "points", False)] * 2),
        ("−2.1% growth", [(-2.1, 1, 0.01, "percent", True)]),
        ("it lists 3 risks", [(3.0, 0, 1.0, "bare", False)]),
    ],
)
def test_figures_and_how_they_are_written(text, expected):
    assert figs(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "in the fiscal year that ended in 2023 and FY2024",
        "as its 10-K and 10-Q say [1][2] in Item 1A",
        "on December 31, 2023 and 31 December 2023",
        "the 1st and 2nd risk factors",
        "see https://www.sec.gov/Archives/edgar/data/27904/000002790425000004/x-index.htm",
        "from 2020-2024",
    ],
)
def test_non_figures_are_skipped(text):
    assert find_figures(text) == []


ALLOWED = [
    Allowed("DAL FY2023 revenue (value)", 58_048_000_000.0, "usd"),
    Allowed("DAL FY2023 revenue (first_reported_value)", 57_500_000_000.0, "usd"),
    Allowed("DAL FY2023 net income (value)", -412_000_000.0, "usd"),
    Allowed("DAL FY2023 operating_margin", 0.0937, "ratio"),
    Allowed("DAL operating_margin change FY2020-FY2023", -0.0213, "points"),
]


def test_an_answer_with_only_given_figures_passes():
    text = (
        "Revenue was $58.05 billion [1] (first reported as $57.5 billion), a net loss of"
        " $0.4 billion, an operating margin of 9.4%, down 2.1 percentage points since 2020 [2]."
    )
    out = check_answer(text, ALLOWED)
    assert out.ok and out.problems == []
    assert [f["matched"] for f in out.figures] == [
        "DAL FY2023 revenue (value)",
        "DAL FY2023 revenue (first_reported_value)",
        "DAL FY2023 net income (value)",
        "DAL FY2023 operating_margin",
        "DAL operating_margin change FY2020-FY2023",
    ]


def test_a_wrong_figure_or_a_stray_number_blocks_the_answer():
    for text in [
        "Revenue was $58.2 billion.",
        "Revenue rose 7% to $58.05 billion.",
        "It names 4 risks.",
    ]:
        out = check_answer(text, ALLOWED)
        assert not out.ok and len(out.problems) == 1, text


def test_an_explicit_sign_must_match_and_kinds_must_be_compatible():
    assert not check_answer("net income of +$0.4 billion", ALLOWED).ok  # '+' is a sign too
    assert check_answer("net income of -$0.4 billion", ALLOWED).ok
    assert not check_answer("net income of -$58.05 billion", ALLOWED).ok  # revenue is positive
    assert not check_answer("a margin of 58.05 percentage points", ALLOWED).ok  # usd isn't points


def test_no_figures_passes_and_missing_reference_blocks():
    assert check_answer("The 10-K does not report revenue for that year [1].", ALLOWED).ok
    assert not check_answer("Revenue was $1.2 billion.", []).ok


def entry(fy, rev, op, first=None):
    return {
        "fiscal_year": fy,
        "figures": {
            "revenue": {"value": rev, "first_reported_value": first or rev},
            "operating_income": {"value": op, "first_reported_value": op},
        },
        "derived": {"operating_margin": {"value": op / rev, "of": ["operating_income", "revenue"]}},
    }


def test_allowed_values_cover_facts_restatements_margins_growth_and_changes():
    fin = {
        "company": {"ticker": "LOW"},
        "years": [entry(2020, 100.0, 10.0, first=99.0), entry(2024, 120.0, 15.0)],
        "growth": [{"from": 2023, "to": 2024, "revenue_growth": 0.05}],
    }
    peers = {
        "company": {"ticker": "LOW"},
        "company_financials": entry(2024, 120.0, 15.0),
        "peers": [{"ticker": "HD", "financials": entry(2024, 150.0, 20.0)}],
    }
    values = {a.label: (a.value, a.kind) for a in allowed_values([fin], [peers])}
    assert values["LOW FY2020 revenue (first_reported_value)"] == (99.0, "usd")
    assert values["LOW FY2024 operating_margin"] == (0.125, "ratio")
    assert values["LOW operating_margin change FY2020-FY2024"] == (pytest.approx(0.025), "points")
    assert values["LOW revenue growth FY2023-FY2024"] == (0.05, "ratio")
    assert values["HD FY2024 revenue (value)"] == (150.0, "usd")
    assert margin_changes("X", [entry(2020, 1.0, 0.1)]) == []  # one year: no change
