"""Demo database helpers: the subset join, the recall hit, the SQL shape (no database needed)."""

import numpy as np
import pandas as pd
import pytest

from filinglens.api.demo_db import (
    DDL,
    SEARCH_SQL,
    recall_hit,
    require_target,
    short_plan,
    subset_rows,
    within_one_query,
)


def test_subset_rows_keeps_the_companies_and_aligns_vectors():
    chunks = pd.DataFrame({"chunk_id": ["a#0", "b#0", "c#0"], "cik": [1, 2, 1]})
    ids = ["c#0", "b#0", "a#0"]  # the vector file's own order
    vectors = np.array([[3.0], [2.0], [1.0]])
    sub, vecs = subset_rows(chunks, ids, vectors, {1})
    assert list(sub["chunk_id"]) == ["c#0", "a#0"]  # in vector-file order
    assert vecs.ravel().tolist() == [3.0, 1.0]  # each row with its own vector


def test_subset_rows_refuses_chunks_without_a_vector():
    chunks = pd.DataFrame({"chunk_id": ["a#0", "z#0"], "cik": [1, 1]})
    with pytest.raises(SystemExit, match="no stored vector"):
        subset_rows(chunks, ["a#0"], np.ones((1, 1)), {1})


def test_recall_hit_needs_a_graded_chunk():
    assert recall_hit(["x", "y"], {"y": 1})
    assert recall_hit(["x"], {"x": 2})
    assert not recall_hit(["x"], {"y": 2, "x": 0})


def test_search_is_per_company_exact_and_parameterized():
    assert "WHERE cik = %(cik)s" in SEARCH_SQL
    assert "ORDER BY embedding <#> %(q)s LIMIT %(k)s" in SEARCH_SQL  # inner product, as FL-M3
    assert "hnsw" not in DDL.lower()  # per-company exact search needs no vector index


def test_short_plan_hides_the_query_vector():
    line = "Sort Key: ((embedding <#> '[-0.01,0.03,0.5]'::vector))"
    assert short_plan([line, "Limit"]) == ["Sort Key: ((embedding <#> <query vector>))", "Limit"]


def test_tag_must_match_where_it_runs(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    require_target("local")
    with pytest.raises(SystemExit, match="--tag neon needs"):
        require_target("neon")  # the T13c slip: the record would say neon, the rows went local
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@ep-x.us-east-1.aws.neon.tech/neondb")
    require_target("neon")
    with pytest.raises(SystemExit, match="--tag local but"):
        require_target("local")


def test_one_query_tolerance_is_counted_in_hits_not_rounded_recall():
    # T13: 122 vs 121 hits of 182 is one query, but the 4-decimal recalls differ by 0.0055 > 1/182
    assert abs(0.6703 - 0.6648) > 1 / 182
    assert within_one_query(0.6703, 0.6648, 182)
    assert within_one_query(0.6758, 0.6758, 182)
    assert not within_one_query(0.6758, 0.6648, 182)  # 123 vs 121: two queries
