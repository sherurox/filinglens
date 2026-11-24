import numpy as np
import pytest

from filinglens.retrieval import diagnose_dev as dd


def test_recall_in_and_first_relevant_rank():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    relevant = np.array([False, False, True, False])
    everything = np.ones(4, dtype=bool)
    assert not dd.recall_in(scores, everything, relevant, k=2)
    assert dd.recall_in(scores, everything, relevant, k=3)
    only_last_two = np.array([False, False, True, True])
    assert dd.recall_in(scores, only_last_two, relevant, k=1)  # the filter removes rows 0, 1
    assert not dd.recall_in(scores, np.zeros(4, dtype=bool), relevant, k=4)
    assert dd.first_relevant_rank(scores, relevant) == 3


def test_diagnose_separates_company_confusion_from_a_bug():
    # Two companies with the same boilerplate: company 2's copy scores just above company 1's.
    chunk_ids = ["a", "b", "c", "d"]
    chunk_cik = np.array([1, 2, 1, 2])
    chunk_year = np.array([2020, 2020, 2019, 2019])
    chunk_vecs = np.array([[0.8, 0.6], [0.9, 0.436], [0.1, 0.995], [1.0, 0.0]])
    query = {"cik": 1, "period_year": 2020, "grades": {"a": 2, "c": 1}}
    out = dd.diagnose(
        np.array([[1.0, 0.0]]), chunk_vecs, chunk_ids, chunk_cik, chunk_year, [query], k=1
    )
    assert out["recall_at_1"] == {"all": 0.0, "company": 1.0, "company_year": 1.0}
    assert out["first_relevant_rank_all_chunks"]["median"] == 3.0  # d, b, then a
    assert out["top_1_share_same_company"] == 0.0
    assert out["top_1_share_grade1_copies"] == 0.0


def test_ticker_in_text():
    queries = [
        {"ticker": "NDAQ", "grades": {"x": 2, "y": 1}},
        {"ticker": "GE", "grades": {"z": 2}},
    ]
    text_of = {"x": "Nasdaq (NDAQ) may face...", "y": "NDAQ", "z": "We may face GEO risks"}
    assert dd.ticker_in_text(queries, text_of) == pytest.approx(0.5)  # GE is not a word in GEO


def _files(tmp_path, chunk_order):
    import json

    import pandas as pd

    emb, qdir = tmp_path / "emb", tmp_path / "q"
    emb.mkdir()
    qdir.mkdir()
    meta = pd.DataFrame(
        {"chunk_id": ["a", "b"], "cik": [1, 2], "period_year": [2020, 2020], "text": ["x", "y"]}
    )
    meta.to_parquet(tmp_path / "chunks.parquet")
    np.save(emb / "chunks.npy", np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    (emb / "chunk_ids.txt").write_text("".join(f"{c}\n" for c in chunk_order))
    np.save(emb / "queries.dev.npy", np.array([[1.0, 0.0]], dtype=np.float32))
    (emb / "query_ids.dev.txt").write_text("q1\n")
    query = {"query_id": "q1", "cik": 1, "period_year": 2020, "ticker": "T", "grades": {"a": 2}}
    (qdir / "queries.dev.jsonl").write_text(json.dumps(query) + "\n")
    return [
        "--chunks",
        str(tmp_path / "chunks.parquet"),
        "--emb-dir",
        str(emb),
        "--query-dir",
        str(qdir),
        "--bench",
        str(tmp_path),
    ]


def test_main_writes_the_diagnostics(tmp_path):
    pytest.importorskip("pyarrow")
    import json

    dd.main(_files(tmp_path, ["a", "b"]))
    out = json.loads((tmp_path / "retrieval_dev_diagnostics.json").read_text())
    assert out["recall_at_10"] == {"all": 1.0, "company": 1.0, "company_year": 1.0}


def test_main_refuses_misaligned_vectors(tmp_path):
    pytest.importorskip("pyarrow")
    with pytest.raises(ValueError, match="different orders"):
        dd.main(_files(tmp_path, ["b", "a"]))
