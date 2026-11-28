"""Scoring: Wilson intervals, what each template asks for (R1), year-specific citations (R2)."""

import pytest
from test_agents_questions import SMALL, universe

from filinglens.agents import questions as qs
from filinglens.agents.score import asked, cited_excerpts, r1, r2, wilson


def generated():
    data, split_of = universe(bank=1000)
    return {q["template"]: q for q in qs.generate(data, split_of, SMALL)["dev"]}


def money(v: float) -> str:
    return f"${v:,.0f}"


def cite_all(q: dict) -> dict:
    """A citation map that covers every (company, year) the reference needs."""
    out = {}
    for i, (cik, fy) in enumerate(
        sorted(
            {(e["cik"], e["fiscal_year"]) for e in q["reference"]["risk_evidence"]}
            | {(f["cik"], f["fiscal_year"]) for f in q["reference"]["figures"]}
        ),
        start=1,
    ):
        out[str(i)] = {str(fy): f"https://www.sec.gov/Archives/edgar/data/{cik}/x/y-index.htm"}
    return out


def rec(answer: str, q: dict, status: str = "answered", citations=None) -> dict:
    return {
        "status": status,
        "answer": answer if status == "answered" else None,
        "citations": cite_all(q) if citations is None else citations,
    }


def test_wilson_interval():
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0
    assert wilson(5, 10) == pytest.approx([0.2366, 0.7634], abs=1e-4)
    assert wilson(0, 0) is None


def test_metric_margin_needs_revenue_operating_income_and_the_margin():
    q = generated()["metric_margin_risk"]
    rev, op = q["reference"]["figures"]
    (margin,) = q["reference"]["derived"]
    good = (
        f"Revenue {money(rev['value'])}, operating income {money(op['value'])}, a margin of"
        f" {margin['value'] * 100:.1f}% [1]."
    )
    assert r1(q, rec(good, q)) == (True, [])
    ok, why = r1(q, rec(good.split(", a margin")[0] + " [1].", q))
    assert not ok and why == [margin["label"]]
    assert r1(q, rec(good, q, status="blocked"))[0] is False


def test_a_change_may_be_stated_as_the_change_or_as_both_years():
    q = generated()["risk_change_metric"]
    change = q["reference"]["derived"][-1]
    labels, alternatives = asked(q)[0]
    assert labels == change["label"] and len(alternatives) == 2
    parts = []
    for item in alternatives[1]:  # both years' values
        if "first_reported_value" in item:
            parts.append(money(item["value"]))
        else:
            parts.append(f"{item['value'] * 100:.2f}%")
    assert r1(q, rec(" and ".join(parts) + " [1].", q))[0]
    assert not r1(q, rec(parts[0] + " [1].", q))[0]  # one year alone is not the change


def test_missing_data_passes_without_a_dollar_amount_and_fails_with_one():
    q = generated()["missing_data"]
    assert r1(q, rec("Revenue for that year is not available [F1].", q))[0]
    assert not r1(q, rec("Revenue was $1,000 [F1].", q))[0]


def test_r2_needs_each_company_years_own_filing():
    q = generated()["peer_compare"]
    assert r2(q, rec("x [1].", q)) == (True, [])
    some = dict(list(cite_all(q).items())[:1])  # only one of the two companies' filings
    ok, gaps = r2(q, rec("x [1].", q, citations=some))
    assert not ok and len(gaps) == 1


def test_cited_excerpts_returns_the_full_text_of_each_cited_excerpt():
    record = {
        "excerpts": "[1] A 10-K. one\n\n[2] B 10-K. two\n\n[3] C 10-K. three",
        "citations": {"3": {}, "1": {}, "F1": {}},
    }
    assert cited_excerpts(record) == "[1] A 10-K. one\n\n[3] C 10-K. three"
