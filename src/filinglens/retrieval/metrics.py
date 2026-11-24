"""Ranking metrics for FL-M3 retrieval: graded relevance, one query at a time (FL-M3 T6).

A query's relevance labels are a dict {chunk_id: grade}: 2 = the chunk holding the evidence,
1 = a near copy in another filing, missing = not relevant (grade 0). A retriever returns a ranked
list of chunk IDs, best first.

ndcg_at_k() is hand-written (the FL-M3 live-coding piece). recall_at_k() and mrr_at_k() are
added below it after review.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def _dcg(grades_in_rank_order: Iterable[int]) -> float:
    """Sum of gain / discount over grades listed best rank first (rank 1, 2, ...)."""
    total = 0.0
    for index, grade in enumerate(grades_in_rank_order):
        rank = index + 1
        total += (2**grade - 1) / math.log2(rank + 1)
    return total


def ndcg_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int = 10) -> float:
    """Normalized discounted cumulative gain of the top k of `ranked`.

    gain(g)      = 2**g - 1                 (grade 2 -> 3, grade 1 -> 1, grade 0 -> 0)
    discount(i)  = log2(i + 1)              (i = rank, starting at 1)
    DCG@k        = sum over ranks i = 1..k of gain(grade of ranked[i-1]) / discount(i)
    IDCG@k       = the same sum for the best possible ranking: every positive grade in
                   `grades`, sorted from highest to lowest, cut at k
    NDCG@k       = DCG@k / IDCG@k, or 0.0 when IDCG@k is 0 (the query has nothing relevant)

    `ranked` may be shorter than k. IDs in `ranked` that aren't in `grades` have grade 0.
    Raises ValueError if k < 1, or if an ID appears more than once in ranked[:k]: a duplicate
    means a bug upstream, and counting it twice would push NDCG above 1. Duplicates after
    rank k are ignored, since they don't affect the score.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got k={k}")

    top_k = ranked[:k]
    seen: set[str] = set()
    for chunk_id in top_k:
        if chunk_id in seen:
            raise ValueError(f"duplicate chunk ID {chunk_id!r} in ranked[:k] (k={k})")
        seen.add(chunk_id)

    dcg = _dcg(grades.get(chunk_id, 0) for chunk_id in top_k)
    idcg = _dcg(sorted((g for g in grades.values() if g > 0), reverse=True)[:k])

    if idcg == 0:
        return 0.0
    return dcg / idcg
