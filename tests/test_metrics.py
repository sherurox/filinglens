"""Tests for retrieval/metrics.py (FL-M3 T6). Written by hand: replace every pytest.fail."""

import pytest

from filinglens.retrieval.metrics import mrr_at_k, ndcg_at_k, recall_at_k


def test_perfect_ranking_scores_one():
    grades = {"a": 2, "b": 1, "c": 1}
    ranked = ["a", "b", "c", "junk"]  # every graded chunk first, highest grade first

    assert ndcg_at_k(ranked, grades, k=10) == 1.0


def test_nothing_relevant_scores_zero():
    # (a) grades exist, but nothing ranked is in them
    assert ndcg_at_k(["x", "y", "z"], {"a": 2, "b": 1}, k=3) == 0.0

    # (b) empty grades: IDCG is 0, must return 0.0 rather than divide by zero
    assert ndcg_at_k(["x", "y", "z"], {}, k=3) == 0.0


def test_relevant_chunk_beyond_k_does_not_count():
    grades = {"a": 2}
    ranked = ["x", "y", "a"]  # the only relevant chunk sits at rank 3

    assert ndcg_at_k(ranked, grades, k=2) == 0.0  # cut at k = 2 drops it
    assert ndcg_at_k(ranked, grades, k=3) > 0.0  # k = 3 reaches it


def test_hand_computed_example():
    ranked = ["m", "n", "o", "p"]
    grades = {"m": 1, "o": 2, "z": 2}  # n, p ungraded; z is graded but never retrieved

    # DCG  = rank 1: m -> grade 1 -> 1 / log2(2) = 1 / 1       = 1.0
    #        rank 2: n -> grade 0 -> 0 / log2(3)              = 0.0
    #        rank 3: o -> grade 2 -> 3 / log2(4) = 3 / 2       = 1.5
    #        rank 4: p -> grade 0 -> 0 / log2(5)              = 0.0
    #        DCG = 2.5
    #
    # IDCG = ideal grades [2, 2, 1]  (z counts even though it wasn't retrieved)
    #        3 / log2(2) + 3 / log2(3) + 1 / log2(4)
    #      = 3 + 3 / 1.5849625007 + 0.5
    #      = 3 + 1.8927892607 + 0.5
    #      = 5.3927892607
    #
    # NDCG = 2.5 / 5.3927892607 = 0.463581994240
    assert ndcg_at_k(ranked, grades, k=4) == pytest.approx(0.463581994240, abs=1e-9)


def test_order_matters():
    grades = {"a": 2, "b": 1}

    best_first = ndcg_at_k(["a", "b"], grades, k=2)
    worse_first = ndcg_at_k(["b", "a"], grades, k=2)

    assert best_first > worse_first


def test_ranking_shorter_than_k():
    grades = {"a": 2, "b": 1}
    score = ndcg_at_k(["a", "x"], grades, k=10)  # 2 IDs, k = 10: must not IndexError

    # DCG  = 3 / log2(2) + 0 / log2(3)                  = 3.0
    # IDCG = 3 / log2(2) + 1 / log2(3) = 3 + 0.6309297536 = 3.6309297536  (b counts, unretrieved)
    # NDCG = 3.0 / 3.6309297536                        = 0.826234657129
    assert score == pytest.approx(0.826234657129, abs=1e-9)


def test_k_below_one_is_rejected():
    with pytest.raises(ValueError, match="k=0"):
        ndcg_at_k(["a"], {"a": 2}, k=0)


def test_duplicate_id_in_top_k_is_rejected():
    # counted twice, ["a", "a"] would score 1.63: must fail loudly instead
    with pytest.raises(ValueError, match="duplicate chunk ID 'a'"):
        ndcg_at_k(["a", "a"], {"a": 2}, k=2)

    # a repeat after rank k is outside the scored window, so it's allowed
    assert ndcg_at_k(["a", "x", "a"], {"a": 2}, k=2) == 1.0


# ---- below: AI-assisted tests for recall_at_k and mrr_at_k ------------------------------------


def test_recall_at_k_any_graded_chunk_in_the_top_k():
    grades = {"a": 2, "b": 1}
    assert recall_at_k(["x", "b", "y"], grades, k=3) == 1.0  # grade 1 counts as found
    assert recall_at_k(["x", "y", "a"], grades, k=2) == 0.0  # a is at rank 3, beyond k
    assert recall_at_k(["x", "y", "a"], grades, k=3) == 1.0
    assert recall_at_k([], grades, k=10) == 0.0
    assert recall_at_k(["a"], {}, k=10) == 0.0


def test_mrr_at_k_first_hit_only():
    grades = {"a": 2, "b": 1}
    assert mrr_at_k(["a", "b"], grades, k=10) == 1.0
    assert mrr_at_k(["x", "y", "b", "a"], grades, k=10) == pytest.approx(1 / 3)  # b at rank 3
    assert mrr_at_k(["x", "y", "b"], grades, k=2) == 0.0
    assert mrr_at_k([], grades, k=10) == 0.0


def test_recall_and_mrr_reject_k_below_one():
    for metric in (recall_at_k, mrr_at_k):
        with pytest.raises(ValueError, match="k=0"):
            metric(["a"], {"a": 2}, k=0)
