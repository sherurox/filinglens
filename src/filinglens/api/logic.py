"""The API's pure pieces: answer prompt, citations, SEC links, cost summary (no web framework).

Kept apart from app.py so they're tested without FastAPI, and so the prompt is one readable
place. Everything here is deterministic and has no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

ANSWER_SYSTEM = (
    "You answer an analyst's question about one company's 10-K Risk Factors (Item 1A), using ONLY "
    "the numbered excerpts you are given. Rules:\n"
    "1. Every claim cites the excerpt it comes from, like [1] or [2][3].\n"
    "2. No outside knowledge, no advice, no numbers that aren't in the excerpts.\n"
    "3. If the excerpts don't answer the question, say so in one sentence.\n"
    "4. At most 120 words."
)
ANSWER_MAX_TOKENS = 300
ANSWER_CONTEXT_CHUNKS = 5
EXCERPT_CHARS = 1500  # chunks are <= 510 tokens (FL-M3); this only guards a pathological row


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    ticker: str
    fiscal_year: int
    accession: str
    heading: str | None
    text: str
    score: float


def sec_filing_url(cik: int, accession: str) -> str:
    """The filing's EDGAR index page (accession 0000004904-25-000027 -> 000000490425000027/)."""
    folder = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{folder}/{accession}-index.htm"


def hit_json(hit: Hit, cik: int) -> dict:
    return {
        "chunk_id": hit.chunk_id,
        "ticker": hit.ticker,
        "fiscal_year": hit.fiscal_year,
        "heading": hit.heading,
        "text": hit.text,
        "score": round(float(hit.score), 6),
        "filing_url": sec_filing_url(cik, hit.accession),
    }


def answer_messages(ticker: str, question: str, hits: list[Hit]) -> list[dict]:
    """System rules + the question and the top excerpts, numbered from 1."""
    lines = [f"Company: {ticker}", f"Question: {question}", "", "Excerpts:"]
    for i, h in enumerate(hits[:ANSWER_CONTEXT_CHUNKS], start=1):
        heading = f" {h.heading}." if h.heading else ""
        lines.append(f"[{i}] FY{h.fiscal_year}.{heading} {h.text[:EXCERPT_CHARS]}")
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]


def cited_numbers(answer: str, n_excerpts: int) -> list[int]:
    """Excerpt numbers the answer cites ([1], [2][3]), in first-seen order, in range only."""
    seen: list[int] = []
    i = 0
    while (start := answer.find("[", i)) != -1:
        end = answer.find("]", start)
        if end == -1:
            break
        token = answer[start + 1 : end]
        if token.isdigit() and 1 <= int(token) <= n_excerpts and int(token) not in seen:
            seen.append(int(token))
        i = end + 1
    return seen


def citations(answer: str, hits: list[Hit], cik: int) -> list[dict]:
    used = hits[:ANSWER_CONTEXT_CHUNKS]
    return [
        {"n": n, **{k: v for k, v in hit_json(used[n - 1], cik).items() if k != "text"}}
        for n in cited_numbers(answer, len(used))
    ]


def http_status_for(error: str) -> int:
    """API status for a failed model call: the free-tier quota is the caller's 429, else 502."""
    if error.startswith("quota:") or error.startswith("HTTP 429"):
        return 429
    return 502
