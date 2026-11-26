"""The API's pure pieces: SEC links, the answer prompt, citations, error mapping."""

from filinglens.api.logic import (
    ANSWER_CONTEXT_CHUNKS,
    Hit,
    answer_messages,
    citations,
    cited_numbers,
    http_status_for,
    sec_filing_url,
)


def hit(i: int, year: int = 2024, heading: str | None = "Fuel costs") -> Hit:
    return Hit(f"c{i}", "DAL", year, "0000027904-25-000004", heading, f"text {i}", 0.9 - i / 100)


def test_sec_filing_url():
    assert sec_filing_url(4904, "0000004904-25-000027") == (
        "https://www.sec.gov/Archives/edgar/data/4904/000000490425000027/"
        "0000004904-25-000027-index.htm"
    )


def test_answer_messages_number_the_top_excerpts():
    hits = [hit(i) for i in range(8)]
    system, user = answer_messages("DAL", "How do fuel prices matter?", hits)
    assert system["role"] == "system" and "ONLY" in system["content"]
    text = user["content"]
    assert text.startswith("Company: DAL\nQuestion: How do fuel prices matter?")
    assert f"[{ANSWER_CONTEXT_CHUNKS}] FY2024. Fuel costs. text 4" in text
    assert f"[{ANSWER_CONTEXT_CHUNKS + 1}]" not in text  # only the top 5 go in


def test_answer_messages_without_heading():
    _, user = answer_messages("DAL", "q?", [hit(0, heading=None)])
    assert "[1] FY2024. text 0" in user["content"]


def test_cited_numbers_in_order_in_range_once():
    assert cited_numbers("A [2]. B [1][2]. C [7]. D [x] [3", 5) == [2, 1]
    assert cited_numbers("no citations", 5) == []


def test_citations_link_the_cited_excerpts():
    hits = [hit(i) for i in range(5)]
    out = citations("Fuel is volatile [2]; hedges help [1].", hits, 27904)
    assert [c["n"] for c in out] == [2, 1]
    assert out[0]["chunk_id"] == "c1" and "text" not in out[0]
    assert out[0]["filing_url"].endswith("0000027904-25-000004-index.htm")


def test_http_status_for():
    assert http_status_for("quota: 500 / 500 requests already sent") == 429
    assert http_status_for("HTTP 429: rate") == 429
    assert http_status_for("HTTP 503: busy") == 502
    assert http_status_for("transport: ConnectTimeout") == 502
