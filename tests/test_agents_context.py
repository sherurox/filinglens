"""Context builder: merge, near-copy folding, round-robin, budget properties, truncation."""

import random

import pytest

from filinglens.agents.context import (
    ELLIPSIS,
    MIN_EXCERPT_TOKENS,
    build_context,
    fold,
    merge,
    round_robin,
)

WORDS = (
    "fuel prices airline capacity demand labor union pilots aircraft engines supply delays "
    "regulation emissions climate cyber breach systems outage pandemic travel revenue costs "
    "hedging currency interest debt liquidity credit rating weather storms airports slots"
).split()


def count(texts):
    return [len(t.split()) for t in texts]  # a whitespace "tokenizer": additive, easy to check


def hit(cid, ticker="DAL", year=2023, text=None, score=1.0, heading=None):
    return {
        "chunk_id": cid,
        "ticker": ticker,
        "fiscal_year": year,
        "heading": heading,
        "text": text or f"{cid} " + " ".join(WORDS[:10]) + ".",
        "filing_url": f"https://www.sec.gov/{ticker}/{year}",
        "rerank_score": score,
        "dense_score": 0.5,
    }


def random_hits(rng: random.Random, n: int) -> list[dict]:
    out = []
    for i in range(n):
        sentences = [
            " ".join(rng.choices(WORDS, k=rng.randint(4, 30))) + "."
            for _ in range(rng.randint(1, 6))
        ]
        out.append(
            hit(
                f"c{i}",
                rng.choice(["DAL", "UAL"]),
                rng.choice([2020, 2024]),
                " ".join(sentences).capitalize(),
                rng.random(),
            )
        )
    return out


def test_merge_keeps_each_chunk_once_with_its_best_score():
    hits = [hit("a", score=0.2), hit("b", score=0.9), hit("a", score=0.7), hit("c", score=0.9)]
    ranked = merge(hits)
    assert [h["chunk_id"] for h in ranked] == ["b", "c", "a"]  # tie b/c: first seen first
    assert ranked[2]["_score"] == 0.7
    assert merge([hit("x", score=None) | {"dense_score": 0.3}])[0]["_score"] == 0.3


def test_near_copies_fold_across_years_within_one_company_only():
    text = "Fuel prices are volatile and could raise our costs significantly next year."
    hits = [
        hit("d24", year=2024, text=text, score=0.9),
        hit("d20", year=2020, text=text, score=0.8),
        hit("u24", ticker="UAL", year=2024, text=text, score=0.7),
        hit("other", year=2020, text="Pilots may unionize and strike at airports.", score=0.1),
    ]
    excerpts, folded = fold(merge(hits))
    assert folded == 1 and len(excerpts) == 3
    dal = excerpts[0]
    assert dal.chunk_ids == ["d24", "d20"] and set(dal.years) == {2024, 2020}
    assert dal.years[2020] == "https://www.sec.gov/DAL/2020"  # each year keeps its own filing
    assert "fiscal year ended in 2020, 2024 (near-identical text in each)" in dal.render()
    assert excerpts[1].ticker == "UAL"  # same text, other company: not folded


def test_folding_is_idempotent_under_repeated_hits():
    hits = random_hits(random.Random(7), 30)
    once, _ = fold(merge(hits))
    twice, _ = fold(merge(hits + hits))
    assert [e.chunk_ids for e in once] == [e.chunk_ids for e in twice]


def test_round_robin_alternates_company_year_groups():
    hits = [
        hit("a1", year=2024, score=0.9, text="alpha one two three four five six."),
        hit("a2", year=2024, score=0.8, text="beta seven eight nine ten eleven."),
        hit("b1", year=2020, score=0.7, text="gamma cyber breach systems outage now."),
    ]
    excerpts, _ = fold(merge(hits))
    assert [e.chunk_ids[0] for e in round_robin(excerpts)] == ["a1", "b1", "a2"]


@pytest.mark.parametrize("seed", range(25))
def test_never_over_budget_and_larger_budgets_keep_everything(seed):
    rng = random.Random(seed)
    hits = random_hits(rng, rng.randint(1, 40))
    budgets = sorted(rng.sample(range(1, 1500), 5))
    previous_ids, previous_full = set(), []
    for budget in budgets:
        ctx = build_context(hits, budget, count)
        assert ctx.tokens == sum(e.tokens for e in ctx.excerpts) <= budget
        assert all(count([e.render()])[0] == e.tokens for e in ctx.excerpts)
        ids = {c for e in ctx.excerpts for c in e.chunk_ids}
        assert previous_ids <= ids  # monotone in the budget
        full = [e.chunk_ids for e in ctx.excerpts if not e.truncated]
        assert full[: len(previous_full)] == previous_full  # same order, a longer prefix
        assert sum(e.truncated for e in ctx.excerpts) <= 1
        assert [e.n for e in ctx.excerpts] == list(range(1, len(ctx.excerpts) + 1))
        previous_ids, previous_full = ids, full


def test_the_last_excerpt_is_cut_at_a_sentence_end():
    long = " ".join(f"Sentence {i} has exactly six words." for i in range(1, 30))
    ctx = build_context([hit("a", text=long)], budget=60, count=count)
    (ex,) = ctx.excerpts
    assert ex.truncated and ex.text.endswith("words." + ELLIPSIS) and ex.tokens <= 60
    assert ctx.stats["truncated"] == 1


def test_one_huge_sentence_is_cut_between_words_and_tiny_room_is_skipped():
    huge = " ".join(WORDS * 10) + "."
    first = hit("a", text="Short first excerpt here.", score=0.9)
    ctx = build_context([first, hit("b", text=huge, score=0.5)], budget=200, count=count)
    assert len(ctx.excerpts) == 2 and ctx.excerpts[1].truncated
    assert ctx.excerpts[1].text.endswith(ELLIPSIS) and ctx.tokens <= 200
    tiny = build_context([first], 1000, count).excerpts[0].tokens
    ctx2 = build_context(
        [first, hit("b", text=huge, score=0.5)], tiny + MIN_EXCERPT_TOKENS - 1, count
    )
    assert [e.chunk_ids for e in ctx2.excerpts] == [["a"]]  # no room for a useful cut


def test_citation_map_and_stats():
    text = "Fuel prices are volatile and could raise our costs significantly next year."
    hits = [hit("d24", year=2024, text=text), hit("d20", year=2020, text=text, score=0.5)]
    ctx = build_context(hits, 1000, count)
    assert ctx.citation_map() == {
        1: {2024: "https://www.sec.gov/DAL/2024", 2020: "https://www.sec.gov/DAL/2020"}
    }
    assert ctx.stats["near_copies_folded"] == 1 and ctx.stats["excerpts_left_out"] == 0
    assert ctx.text.startswith("[1] DAL 10-K")
    with pytest.raises(ValueError):
        build_context(hits, 0, count)


def test_student_counter_counts_tokens_when_the_tokenizer_is_present():
    pytest.importorskip("tokenizers")
    from pathlib import Path

    from filinglens.agents.context import STUDENT_TOKENIZER, student_counter

    if not Path(STUDENT_TOKENIZER).is_file():
        pytest.skip("the served student's tokenizer.json is not on this machine (CI)")
    n_short, n_long = student_counter()(["fuel prices", "fuel prices rose sharply in 2022"])
    assert 0 < n_short < n_long
