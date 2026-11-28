"""Token-budgeted context builder for the writer (FL-M6 T5).

Input: the hits of one or more search_filings calls (each with chunk_id, ticker, fiscal_year,
report_date, filing_url, heading, text, rerank_score / dense_score). Output: numbered excerpts that
fit a token budget, the rendered text the writer sees, and a record of what was dropped and why.

1. Merge: a chunk found by several sub-queries is kept once, with its best score.
2. Rank: rerank score (dense score when not reranked), ties in first-seen order.
3. Near-duplicates: within one company, a chunk whose content words overlap a higher-ranked one's by
   at least DUP_OVERLAP (overlap coefficient, and at least DUP_MIN_SIZE_RATIO of its size: FL-M3's
   near-copy rule) is folded into it. The same risk paragraph repeated in several 10-Ks becomes ONE
   excerpt labeled with every fiscal year it appears in, and each year's filing URL is kept, so the
   writer can still say "unchanged from 2020 to 2024" and cite each year's own filing.
4. Budget: excerpts are taken round-robin across (company, fiscal year) groups, each group in rank
   order and the groups in the order of their best excerpt, so one 10-K can't use the whole budget
   of a two-year or two-company question. When the next excerpt doesn't fit, it is cut at a
   sentence end to the room left (if at least MIN_EXCERPT_TOKENS) and nothing after it is taken.
   The order never depends on the budget, so a larger budget keeps every excerpt a smaller one
   kept (tested).

Tokens are counted on each rendered excerpt (header included; the blank lines between excerpts are
not counted, one or two tokens each) by an injected counter; the crew
uses the student's tokenizer (Qwen3), so "4k" means 4,096 student tokens of context. The writer's
own prompt-token count (Gemini's) is recorded per call by the cost meter, next to it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from filinglens.extract.gold_sample import content_words, overlap
from filinglens.retrieval.chunks import split_sentences
from filinglens.retrieval.queries import DUP_MIN_SIZE_RATIO, DUP_OVERLAP

Counter = Callable[[Sequence[str]], list[int]]
BUDGETS = (2048, 4096, 8192, 16384)
MIN_EXCERPT_TOKENS = 40
ELLIPSIS = " [...]"


@dataclass
class Excerpt:
    n: int
    ticker: str
    years: dict[int, str]  # fiscal year -> that year's filing URL
    chunk_ids: list[str]  # the representative first, then the folded near-copies
    heading: str | None
    text: str
    score: float
    tokens: int = 0
    truncated: bool = False

    def header(self) -> str:
        years = ", ".join(str(y) for y in sorted(self.years))
        same = " (near-identical text in each)" if len(self.years) > 1 else ""
        return f"[{self.n}] {self.ticker} 10-K, fiscal year ended in {years}{same}."

    def render(self) -> str:
        heading = f" {self.heading}." if self.heading else ""
        return f"{self.header()}{heading} {self.text}"


@dataclass
class Context:
    budget: int
    excerpts: list[Excerpt]
    text: str
    tokens: int
    stats: dict = field(default_factory=dict)

    def citation_map(self) -> dict[int, dict[int, str]]:
        """Excerpt number -> {fiscal year: filing URL}: what each [n] may be cited for."""
        return {e.n: dict(e.years) for e in self.excerpts}


def score_of(hit: dict) -> float:
    s = hit.get("rerank_score")
    return float(hit["dense_score"] if s is None else s)


def merge(hits: Sequence[dict]) -> list[dict]:
    """One entry per chunk_id with its best score, in rank order (ties: first seen)."""
    best: dict[str, tuple[float, int, dict]] = {}
    for i, h in enumerate(hits):
        s = score_of(h)
        if h["chunk_id"] not in best or s > best[h["chunk_id"]][0]:
            first = best[h["chunk_id"]][1] if h["chunk_id"] in best else i
            best[h["chunk_id"]] = (s, first, h)
    ranked = sorted(best.values(), key=lambda t: (-t[0], t[1]))
    return [h | {"_score": s} for s, _, h in ranked]


def near_copy(a: frozenset[str], b: frozenset[str]) -> bool:
    """FL-M3's rule, symmetric here: overlap >= DUP_OVERLAP and sizes within DUP_MIN_SIZE_RATIO."""
    if not a or not b:
        return False
    small, large = sorted((len(a), len(b)))
    return small >= DUP_MIN_SIZE_RATIO * large and overlap(a, b) >= DUP_OVERLAP


def fold(ranked: Sequence[dict]) -> tuple[list[Excerpt], int]:
    """Near-copies (same company) folded into the higher-ranked excerpt; and how many were."""
    groups: list[tuple[Excerpt, frozenset[str]]] = []
    folded = 0
    for h in ranked:
        words = content_words(f"{h.get('heading') or ''} {h['text']}")
        for ex, ex_words in groups:
            if ex.ticker == h["ticker"] and near_copy(words, ex_words):
                ex.years.setdefault(int(h["fiscal_year"]), h["filing_url"])
                ex.chunk_ids.append(h["chunk_id"])
                folded += 1
                break
        else:
            ex = Excerpt(
                0,
                h["ticker"],
                {int(h["fiscal_year"]): h["filing_url"]},
                [h["chunk_id"]],
                h.get("heading"),
                h["text"],
                h["_score"],
            )
            groups.append((ex, words))
    return [ex for ex, _ in groups], folded


def round_robin(excerpts: Sequence[Excerpt]) -> list[Excerpt]:
    """Interleave (company, fiscal year of the representative chunk) groups, in the order of their
    best excerpt; each group keeps rank order."""
    groups: dict[tuple[str, int], list[Excerpt]] = {}
    for ex in excerpts:  # excerpts arrive in rank order, so each group is too
        groups.setdefault((ex.ticker, next(iter(ex.years))), []).append(ex)  # representative's year
    queues = list(groups.values())
    out: list[Excerpt] = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def cut_to(ex: Excerpt, room: int, count: Counter) -> Excerpt | None:
    """The excerpt cut at a sentence end (else between words) to fit `room` tokens, or None."""
    if room < MIN_EXCERPT_TOKENS:
        return None
    sentences = split_sentences(ex.text) or [ex.text]
    best = None
    for i in range(1, len(sentences) + 1):  # longest prefix of whole sentences that fits
        trial = Excerpt(**{**ex.__dict__, "text": " ".join(sentences[:i]) + ELLIPSIS})
        if count([trial.render()])[0] > room:
            break
        best = trial
    if best is None:  # even the first sentence is too long: longest word prefix that fits
        words = sentences[0].split()
        lo, hi = 0, len(words)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            trial = Excerpt(**{**ex.__dict__, "text": " ".join(words[:mid]) + ELLIPSIS})
            if count([trial.render()])[0] <= room:
                lo = mid
            else:
                hi = mid - 1
        if lo == 0:
            return None
        best = Excerpt(**{**ex.__dict__, "text": " ".join(words[:lo]) + ELLIPSIS})
    best.truncated = True
    best.tokens = count([best.render()])[0]
    return best


def build_context(hits: Sequence[dict], budget: int, count: Counter) -> Context:
    if budget < 1:
        raise ValueError("budget must be >= 1 token")
    ranked = merge(hits)
    excerpts, folded = fold(ranked)
    ordered = round_robin(excerpts)
    chosen: list[Excerpt] = []
    used = 0
    for ex in ordered:
        ex.n = len(chosen) + 1
        tokens = count([ex.render()])[0]
        if used + tokens <= budget:
            ex.tokens = tokens
            chosen.append(ex)
            used += tokens
            continue
        cut = cut_to(ex, budget - used, count)
        if cut is not None:
            chosen.append(cut)
            used += cut.tokens
        break  # nothing after a cut: the order never depends on the budget
    left_out = len(ordered) - len(chosen)  # a cut excerpt counts as used
    stats = {
        "hits_in": len(hits),
        "unique_chunks": len(ranked),
        "near_copies_folded": folded,
        "excerpts_available": len(ordered),
        "excerpts_used": len(chosen),
        "excerpts_left_out": left_out,
        "truncated": sum(e.truncated for e in chosen),
        "tokens_used": used,
    }
    text = "\n\n".join(e.render() for e in chosen)
    return Context(budget, chosen, text, used, stats)


STUDENT_TOKENIZER = "data/interim/serve/gptq-w4a16/tokenizer.json"  # the served FL-M4 student


def student_counter(path: str = STUDENT_TOKENIZER) -> Counter:
    """Token counts with the student's own tokenizer (no special tokens), for budgets."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(path)

    def count(texts: Sequence[str]) -> list[int]:
        return [len(e.ids) for e in tokenizer.encode_batch(list(texts), add_special_tokens=False)]

    return count
