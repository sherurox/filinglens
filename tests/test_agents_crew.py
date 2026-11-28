"""The crew on fake backends and the in-memory tools: routing, plans, the check-and-revise gate."""

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("workflows")  # the `agents` group

from test_agents_tools import FakeDb, FakeEmbedder, reverse_reranker  # noqa: E402

from filinglens.agents.crew import (  # noqa: E402
    CallCap,
    Crew,
    answer,
    check_citations,
    fallback_plan,
    figure_lines,
    route_table,
    validate_plan,
)
from filinglens.agents.tools import Tools  # noqa: E402
from filinglens.serve.backends import Reply  # noqa: E402

QUESTION = (
    "What were DELTA AIR LINES, INC. (DAL)'s revenue and operating income for the fiscal year that"
    " ended in 2023, what operating margin does that imply, and what does its 10-K for that year"
    " say about cybersecurity risk?"
)
PLAN = {
    "companies": ["DAL"],
    "fiscal_years": [2023],
    "metrics": ["revenue", "operating_income", "operating_margin"],
    "searches": [{"company": "DAL", "fiscal_year": 2023, "query": "cybersecurity risk"}],
}
GOOD = (
    "Delta's revenue was $200 and its operating income $30 in the fiscal year that ended in"
    " 2023, an operating margin of 15.0% [F1]. Its 10-K describes cybersecurity risk [1]."
)


def count(texts):
    return [len(t.split()) for t in texts]


class FakeChat:
    def __init__(self, backend: str, texts: list[str | Exception]) -> None:
        self.backend, self.texts, self.calls = backend, list(texts), []

    async def __call__(self, conv, schema=None, max_tokens=400, *, route, user):
        self.calls.append({"conv": conv, "schema": schema, "route": route, "user": user})
        item = self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]
        if isinstance(item, Exception):
            return Reply(self.backend, "m", "", 0, 0, 0, 0.1, None, str(item))
        return Reply(self.backend, "m", item, 50, 20, 0, 0.1, "stop")


def crew(planner_texts, writer_texts, budget=500):
    tools = Tools(FakeDb(), FakeEmbedder(), reverse_reranker)
    planner = FakeChat("vllm", planner_texts)
    writer = FakeChat("gemini", writer_texts)
    return Crew(tools, planner, writer, budget, count, user="test"), planner, writer


def run(c, question=QUESTION):
    async def go():
        return await answer(c, question)

    return asyncio.run(go())


def test_route_table_takes_the_largest_lp_share():
    results = {
        "results": {
            "S": {
                "lp": {
                    "shares": {
                        "extraction/vllm": 1.0,
                        "extraction/gemini": 0.0,
                        "short_qa/vllm": 0.4,
                        "short_qa/gguf": 0.6,
                        "long_synthesis/gemini": 1.0,
                    }
                }
            }
        }
    }
    assert route_table(results, "S") == {
        "extraction": "vllm",
        "short_qa": "gguf",
        "long_synthesis": "gemini",
    }


def test_committed_router_results_send_the_writer_to_gemini():
    path = Path("bench/router_results.json")
    if not path.exists():
        pytest.skip("router results not in this checkout")
    routes = route_table(json.loads(path.read_text()))
    assert routes == {"extraction": "vllm", "short_qa": "vllm", "long_synthesis": "gemini"}


def test_validate_plan_and_the_fallback():
    def known(t):
        return t.upper() in {"DAL", "UAL"}

    plan, problems = validate_plan(PLAN, known)
    assert problems == [] and plan["companies"] == ["DAL"]
    bad, why = validate_plan(PLAN | {"companies": ["ZZZ"]}, known)
    assert bad is None and "unknown" in why[0]
    assert validate_plan({"companies": ["DAL"]}, known)[0] is None
    fb = fallback_plan(QUESTION)
    assert fb["companies"] == ["DAL"] and fb["fiscal_years"] == [2023]
    assert {"revenue", "operating_income", "operating_margin"} <= set(fb["metrics"])
    assert fb["searches"] == [{"company": "DAL", "fiscal_year": 2023, "query": QUESTION}]


def test_a_good_draft_is_answered_first_time_with_its_citations():
    c, planner, writer = crew([json.dumps(PLAN)], [GOOD])
    out = run(c)
    assert out["status"] == "answered" and out["answer"] == GOOD and out["attempts"] == 1
    assert out["plan_source"] == "model" and out["problems"] == []
    assert set(out["citations"]) == {"1", "F1"}
    assert out["citations"]["F1"]["2023"].startswith("https://www.sec.gov/")
    assert [f["matched"] for f in out["figures_checked"]] == [
        "DAL FY2023 revenue (value)",
        "DAL FY2023 operating_income (value)",
        "DAL FY2023 operating_margin",
    ]
    assert planner.calls[0]["route"] == "crew:planner" and planner.calls[0]["schema"]
    assert writer.calls[0]["route"] == "crew:writer" and writer.calls[0]["schema"] is None
    user_msg = writer.calls[0]["conv"][1]["content"]
    assert "[F1] DAL, fiscal year ended in 2023" in user_msg and "[1] DAL 10-K" in user_msg
    assert [t["step"] for t in out["trace"]] == ["plan", "retrieve", "write", "check"]


def test_a_wrong_figure_gets_one_revision_with_the_problem_listed():
    wrong = GOOD.replace("$200", "$210")
    c, _, writer = crew([json.dumps(PLAN)], [wrong, GOOD])
    out = run(c)
    assert out["status"] == "answered" and out["attempts"] == 2
    feedback = writer.calls[1]["conv"][-1]["content"]
    assert "rejected" in feedback and "'$210'" in feedback


def test_two_bad_drafts_block_the_answer():
    c, _, writer = crew([json.dumps(PLAN)], [GOOD.replace("[1]", ""), GOOD.replace("$30", "$31")])
    out = run(c)
    assert out["status"] == "blocked" and out["answer"] is None and len(writer.calls) == 2
    assert any("$31" in p for p in out["problems"]) and out["last_draft"]


def test_bad_planner_output_falls_back_and_writer_errors_are_not_blocks():
    c, _, _ = crew(["not json"], [GOOD])
    out = run(c)
    assert out["plan_source"] == "fallback" and out["status"] == "answered"
    assert out["plan_problems"][0].startswith("planner JSON")
    c2, _, _ = crew([json.dumps(PLAN)], [RuntimeError("HTTP 503: busy")])
    out2 = run(c2)
    assert out2["status"] == "error" and out2["problems"] == ["writer: HTTP 503: busy"]


def test_a_year_outside_the_data_becomes_an_unavailable_line():
    plan = PLAN | {"fiscal_years": [2010], "searches": []}
    c, _, writer = crew([json.dumps(plan)], ["Revenue for that year is not available [F1]."])
    out = run(c, QUESTION.replace("2023", "2010"))
    assert out["status"] == "answered"
    assert "outside the data" in out["figure_lines"][0]


def test_citation_check_and_call_cap():
    assert check_citations("A [1]. B [F2].", {1}, {"F2"}) == []
    problems = check_citations("A [1]. B without one. C [7].", {1}, set())
    assert len(problems) == 2

    async def go():
        cap = CallCap(FakeChat("gemini", ["x"]), 1, "gemini")
        first = await cap([], None, 10, route="r", user="u")
        second = await cap([], None, 10, route="r", user="u")
        return first, second, cap.used

    first, second, used = asyncio.run(go())
    assert first.error is None and second.error.startswith("cap:") and used == 1


def test_figure_lines_cover_restatements_growth_and_margin_changes():
    def year(fy, rev, op, first=None):
        return {
            "fiscal_year": fy,
            "available": True,
            "report_date": f"{fy}-12-31",
            "filing_url": f"u{fy}",
            "figures": {
                "revenue": {
                    "value": rev,
                    "first_reported_value": first or rev,
                    "was_restated": bool(first),
                },
                "operating_income": {
                    "value": op,
                    "first_reported_value": op,
                    "was_restated": False,
                },
            },
            "derived": {"operating_margin": {"value": op / rev}},
            "missing": ["net_income"],
        }

    fin = [
        {
            "company": {"ticker": "X"},
            "years": [year(2022, 100.0, 10.0, first=90.0), year(2023, 120.0, 18.0)],
            "growth": [{"from": 2022, "to": 2023, "revenue_growth": 0.2}],
        }
    ]
    lines, cites = figure_lines(fin, [])
    text = "\n".join(lines)
    assert "revenue first reported as $90" in text and "not available: net income" in text
    assert (
        "revenue growth from the fiscal year ended in 2022 to the one ended in 2023: 20.00%" in text
    )
    assert "operating margin change from 2022 to 2023: +5.00 percentage points" in text
    assert cites["F1"] == {2022: "u2022"} and set(cites["F3"]) == {2022, 2023}
