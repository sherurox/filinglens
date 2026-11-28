"""Agent tools on an in-memory database, a fake encoder, reranker and extraction backend."""

import asyncio
import json
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from filinglens.agents.tools import MAX_K, ToolError, Tools, growth, margins

DAL = {"cik": 27904, "ticker": "DAL", "company_name": "DELTA AIR LINES, INC.", "sector": "transport"}
UAL = {"cik": 100517, "ticker": "UAL", "company_name": "UNITED AIRLINES", "sector": "transport"}


def chunk(i: int, year: int, score: float) -> dict:
    return {"chunk_id": f"c{year}-{i}", "period_year": year, "accession": f"0000027904-{year}-1",
            "report_date": date(year, 12, 31), "heading": f"Heading {i}", "text": f"text {i}",
            "embed_text": f"Heading {i}\ntext {i}", "dense_score": score}  # fmt: skip


def fact(tag: str, year: int, value: float) -> dict:
    return {"cik": 27904, "tag": tag, "value_date": date(year, 12, 31), "value": value,
            "first_reported_value": value, "was_restated": False,
            "last_accession_number": f"0000027904-{year}-1"}  # fmt: skip


class FakeDb:
    def __init__(self) -> None:
        self.chunks = [chunk(i, y, 0.9 - i / 100) for y in (2022, 2023) for i in range(4)]
        self.searches: list[tuple] = []

    def company(self, key):
        for c in (DAL, UAL):
            if key.upper() == c["ticker"] or key == str(c["cik"]):
                return c
        return None

    def search(self, cik, vector, years, depth):
        self.searches.append((cik, years, depth))
        rows = [c for c in self.chunks if years is None or c["period_year"] in years]
        return sorted(rows, key=lambda c: -c["dense_score"])[:depth]

    def filings(self, cik):
        return [{"cik": cik, "period_year": y, "accession": f"0000027904-{y}-1",
                 "report_date": date(y, 12, 31)} for y in (2022, 2023)]  # fmt: skip

    def facts(self, cik):
        rows = [fact("Revenues", 2023, 200.0), fact("OperatingIncomeLoss", 2023, 30.0),
                fact("NetIncomeLoss", 2023, 20.0), fact("NetIncomeLoss", 2022, 10.0)]  # fmt: skip
        return [r | {"cik": cik} for r in rows]

    def peers(self, cik, k):
        return [{"rank": 1, "peer_cik": UAL["cik"], "score": 2.5, "z_text": 2.7, "z_ratio": 0.4,
                 "ticker": "UAL", "company_name": UAL["company_name"], "sector": "transport"}][:k]  # fmt: skip

    def earlier_chunks(self, cik, year):
        return [{"para_id": "c2022-0", "period_year": 2022, "heading": "Heading 0",
                 "text": "text 0"}]  # fmt: skip


class FakeEmbedder:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed_queries(self, questions):
        self.seen.extend(questions)
        return np.ones((len(questions), 768), dtype=np.float32)


async def reverse_reranker(question, texts):
    return [float(i) for i in range(len(texts))]  # the last dense candidate scores highest


def tools(reranker=reverse_reranker, extractor=None) -> tuple[Tools, FakeDb]:
    db = FakeDb()
    return Tools(db, FakeEmbedder(), reranker, extractor, "rules" if extractor else None), db


def test_company_by_ticker_or_cik_and_bad_keys():
    t, _ = tools()
    assert t.company("dal").ticker == "DAL" and t.company("27904").cik == 27904
    for bad in ["DROP TABLE x;", "", "a" * 13, None, 3.5]:
        with pytest.raises(ToolError):
            t.company(bad)
    with pytest.raises(ToolError, match="unknown company"):
        t.company("ZZZZ")


@pytest.mark.parametrize(
    "kwargs",
    [{"k": 0}, {"k": MAX_K + 1}, {"k": True}, {"fiscal_years": [2017]}, {"fiscal_years": [2026]},
     {"fiscal_years": list(range(2018, 2027))}, {"fiscal_years": ["2023"]}, {"query": "  "},
     {"query": "x" * 501}],
)  # fmt: skip
def test_search_inputs_are_validated(kwargs):
    t, _ = tools()
    args = {"company": "DAL", "query": "fuel prices", "fiscal_years": None, "k": 5} | kwargs
    with pytest.raises(ToolError):
        asyncio.run(t.search_filings(**args))


def test_search_reranks_keeps_dense_rank_and_links_the_filing():
    t, db = tools()
    out = asyncio.run(t.search_filings("DAL", "  fuel   prices ", [2023, 2023], k=3))
    assert out["query"] == "fuel prices" and out["fiscal_years"] == [2023]
    assert db.searches == [(27904, [2023], 50)]
    assert out["reranked"] and out["candidates"] == 4
    assert [h["chunk_id"] for h in out["hits"]] == ["c2023-3", "c2023-2", "c2023-1"]
    first = out["hits"][0]
    assert first["rank"] == 1 and first["dense_rank"] == 4 and first["rerank_score"] == 3.0
    assert first["report_date"] == "2023-12-31" and first["fiscal_year"] == 2023
    assert first["filing_url"].startswith("https://www.sec.gov/Archives/edgar/data/27904/")


def test_search_without_reranker_keeps_dense_order():
    t, _ = tools(reranker=None)
    out = asyncio.run(t.search_filings("DAL", "fuel", None, k=2))
    assert not out["reranked"]
    assert [h["dense_rank"] for h in out["hits"]] == [1, 2]
    assert all(h["rerank_score"] is None for h in out["hits"])


def test_get_financials_figures_margins_missing_and_no_filing():
    t, _ = tools()
    out = t.get_financials("DAL", [2021, 2022, 2023])
    y21, y22, y23 = out["years"]
    assert not y21["available"] and y21["reason"].startswith("no 10-K")
    assert y22["missing"] == ["revenue", "operating_income"] and y22["derived"] == {}
    assert y23["figures"]["revenue"]["value"] == 200.0 and y23["missing"] == []
    assert y23["derived"]["operating_margin"]["value"] == 30.0 / 200.0
    assert y23["derived"]["net_margin"]["of"] == ["net_income", "revenue"]
    assert out["growth"] == []  # FY2022 has no revenue
    with pytest.raises(ToolError):
        t.get_financials("DAL", [])


def test_margins_and_growth_need_positive_revenue():
    assert margins({"revenue": {"value": 0.0}, "net_income": {"value": 1.0}}) == {}
    rows = [{"fiscal_year": 2022, "figures": {"revenue": {"value": 100.0}}},
            {"fiscal_year": 2023, "figures": {"revenue": {"value": 110.0}}},
            {"fiscal_year": 2025, "figures": {"revenue": {"value": 90.0}}}]  # fmt: skip
    (g,) = growth(rows)  # 2023 vs 2022 only: 2024 is missing
    assert (g["from"], g["to"]) == (2022, 2023) and g["revenue_growth"] == pytest.approx(0.1)


def test_compare_peers_with_and_without_financials():
    t, _ = tools()
    plain = t.compare_peers("DAL", k=1)
    assert plain["peers"][0]["ticker"] == "UAL" and "financials" not in plain["peers"][0]
    with_year = t.compare_peers("DAL", k=1, fiscal_year=2023)
    assert with_year["peers"][0]["financials"]["fiscal_year"] == 2023
    assert with_year["company_financials"]["figures"]["revenue"]["value"] == 200.0
    with pytest.raises(ToolError):
        t.compare_peers("DAL", k=21)


def reply(text="", error=None):
    return SimpleNamespace(backend="vllm", text=text, error=error)


def test_extract_risks_labels_each_hit_and_reports_backend_errors():
    pytest.importorskip("pandas")
    answer = {"rationale": "r", "category": "regulatory_legal", "severity": 2,
              "first_seen_year": 2022, "unsure": False}  # fmt: skip
    calls = []

    async def extractor(conv, schema, max_tokens):
        calls.append((conv, schema, max_tokens))
        return reply(json.dumps(answer)) if len(calls) == 1 else reply(error="HTTP 500: boom")

    t, _ = tools(extractor=extractor)
    out = asyncio.run(t.extract_risks("DAL", 2023, "regulation", k=2))
    first, second = out["risks"]
    assert first["valid"] and first["category"] == "regulatory_legal" and first["severity"] == 2
    assert first["filing_url"].startswith("https://www.sec.gov/")
    assert not second["valid"] and second["problems"] == ["backend: HTTP 500: boom"]
    conv, schema, max_tokens = calls[0]
    assert conv[0] == {"role": "system", "content": "rules"} and max_tokens == 400
    assert "Paragraph's fiscal year: 2023" in conv[1]["content"]
    assert "category" in schema["properties"]


def test_extract_risks_needs_a_backend():
    t, _ = tools()
    with pytest.raises(ToolError, match="not available"):
        asyncio.run(t.extract_risks("DAL", 2023, "regulation"))
