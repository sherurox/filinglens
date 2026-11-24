import json

import numpy as np
import pytest

from filinglens.retrieval import bench_stores as bs


def test_sample_queries_is_seeded_and_keeps_file_order():
    queries = [{"query_id": f"q{i}"} for i in range(30)]
    a, b = bs.sample_queries(queries, 10), bs.sample_queries(queries, 10)
    assert a == b and len(a) == 10
    order = [int(q["query_id"][1:]) for q in a]
    assert order == sorted(order)
    assert bs.sample_queries(queries, 50) == queries  # fewer than asked: all of them


class FakeStore:
    def __init__(self):
        self.calls = []

    def search(self, vec, k, cik=None):
        self.calls.append(cik)
        return [f"c{int(vec[0])}"]


def test_time_searches_counts_warmup_and_timed_passes():
    store = FakeStore()
    queries = [{"cik": 7}, {"cik": 8}, {"cik": 9}]
    vecs = np.arange(3, dtype=np.float32).reshape(3, 1)
    samples = bs.time_searches(store, vecs, queries, "filtered", warmup=2, passes=5)
    assert len(samples) == 15 and all(s >= 0 for s in samples)
    assert len(store.calls) == 21 and set(store.calls) == {7, 8, 9}  # 2 warm-up + 5 timed passes
    store.calls.clear()
    bs.search_all(store, vecs, queries, "unfiltered")
    assert store.calls == [None, None, None]


def test_latency_summary():
    out = bs.latency_summary([1.0] * 95 + [10.0] * 5)
    assert out["p50_ms"] == 1.0 and out["samples"] == 100 and out["mean_ms"] == pytest.approx(1.45)
    assert out["p99_ms"] == 10.0


def test_quality_filtered_and_unfiltered():
    queries = [{"grades": {"a": 2, "b": 1}}, {"grades": {"z": 2}}]
    results = [["b", "a", "x"], ["y"]]
    reference = [["a", "b", "x"], ["z"]]
    filtered = bs.quality(results, reference, queries, "filtered", k=3)
    assert filtered["recall_at_10"] == 0.5  # query 1 hit, query 2 missed
    assert filtered["mrr_at_10"] == 0.5  # 1/1 and 0
    assert filtered["overlap_at_10_exact"] == pytest.approx(0.5)  # 3/3 and 0/1
    assert filtered["short_results"] == 1  # the second list has 1 < 3 results
    assert filtered["grade1_share_top10"] == pytest.approx((1 / 3 + 0) / 2, abs=1e-4)
    unfiltered = bs.quality(results, reference, queries, "unfiltered", k=3)
    assert "recall_at_10" not in unfiltered and "ndcg_at_10" not in unfiltered


def test_tie_aware_overlap_counts_equal_scores_as_matches():
    exact = [0.9, 0.8, 0.7]  # the exact top 3; its 3rd score is 0.7
    assert bs.tie_aware_overlap([0.9, 0.8, 0.7], exact, k=3) == 1.0
    # another chunk with the same 0.7 score (a duplicate) counts: same quality, different ID
    assert bs.tie_aware_overlap([0.9, 0.7, 0.8], exact, k=3) == 1.0
    assert bs.tie_aware_overlap([0.9, 0.8, 0.6], exact, k=3) == pytest.approx(2 / 3)
    assert bs.tie_aware_overlap([0.9], exact, k=3) == pytest.approx(1 / 3)  # short result
    assert bs.tie_aware_overlap([0.5, 0.4], [0.5, 0.4], k=10) == 1.0  # fewer than k candidates
    assert bs.tie_aware_overlap([], [], k=10) == 1.0


def test_plan_kind():
    exact = ["Limit", "  ->  Sort", "        ->  Bitmap Heap Scan on chunk_vectors"]
    hnsw = ["Limit", "  ->  Index Scan using chunk_vectors_embedding_hnsw_idx on chunk_vectors"]
    assert bs.plan_kind(exact) == "btree+sort" and bs.plan_kind(hnsw) == "hnsw"


def test_csv_rows_has_every_column():
    rows = bs.csv_rows({"s": {"filtered": {"queries": 1, "p50_ms": 2.0}, "unfiltered": {}}})
    assert [r["mode"] for r in rows] == ["filtered", "unfiltered"]
    assert set(rows[0]) == set(bs.CSV_COLUMNS) and rows[0]["p50_ms"] == 2.0


def _setup(tmp_path, monkeypatch, expected_recall=None):
    """A tiny world: 3 companies, 300 chunks, 12 dev queries, FAISS indexes on disk."""
    import pandas as pd

    from filinglens.retrieval import stores as st

    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(300, 16)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = [f"p{i}#0" for i in range(300)]
    ciks = rng.integers(1, 4, size=300)
    emb, qdir, faiss_dir = tmp_path / "emb", tmp_path / "q", tmp_path / "faiss"
    for d in (emb, qdir):
        d.mkdir()
    np.save(emb / "chunks.npy", vecs)
    (emb / "chunk_ids.txt").write_text("".join(f"{c}\n" for c in ids))
    pd.DataFrame({"chunk_id": ids, "cik": ciks}).to_parquet(tmp_path / "chunks.parquet")
    for kind in ("flat", "hnsw"):
        store = st.FaissStore(kind)
        store.build(ids, vecs, ciks)
        store.save(faiss_dir / f"{kind}.index")
    queries, qvecs, source_rows = [], [], []
    for i in range(12):
        row = i * 7
        source_rows.append(row)
        q = vecs[row] + 0.3 * rng.normal(size=16).astype(np.float32)
        qvecs.append(q / np.linalg.norm(q))
        queries.append({"query_id": f"q{i}", "cik": int(ciks[row]), "grades": {ids[row]: 2}})
    np.save(emb / "queries.dev.npy", np.asarray(qvecs, dtype=np.float32))
    (emb / "query_ids.dev.txt").write_text("".join(f"{q['query_id']}\n" for q in queries))
    (qdir / "queries.dev.jsonl").write_text("".join(json.dumps(q) + "\n" for q in queries))
    if expected_recall is None:  # the exact within-company recall, computed independently
        rows = st.company_rows(ciks)
        hits = [
            row in st.exact_top_k(vecs, qv, 10, rows=rows[q["cik"]])
            for q, qv, row in zip(queries, qvecs, source_rows, strict=True)
        ]
        expected_recall = round(float(np.mean(hits)), 4)
    diag = tmp_path / "diag.json"
    diag.write_text(json.dumps({"recall_at_10": {"company": expected_recall}}))
    monkeypatch.setattr(bs, "log_run", lambda *a, **k: "run-1")
    return [
        "--stores", "faiss-flat", "faiss-hnsw", "--emb-dir", str(emb),
        "--chunks", str(tmp_path / "chunks.parquet"), "--query-dir", str(qdir),
        "--faiss-dir", str(faiss_dir), "--diagnostics", str(diag), "--bench", str(tmp_path),
        "--results", str(tmp_path / "results.jsonl"), "--note", "test",
    ]  # fmt: skip


def test_main_with_faiss_stores(tmp_path, monkeypatch):
    pytest.importorskip("faiss")
    pytest.importorskip("pyarrow")
    monkeypatch.setattr(bs, "sample_queries", lambda qs: qs[:5])
    bs.main(_setup(tmp_path, monkeypatch))
    record = json.loads((tmp_path / "vector_stores.json").read_text())
    assert record["reproduce_t4b"]["match"] is True
    flat = record["table"]["faiss-flat"]
    assert flat["filtered"]["overlap_at_10_exact"] == 1.0 and flat["unfiltered"]["samples"] == 25
    assert flat["filtered"]["overlap_at_10_tie_aware"] == 1.0  # the reference matches itself
    assert record["pgvector_filtered_plans"] is None  # pgvector wasn't among the stores
    assert (tmp_path / "vector_stores.csv").read_text().count("\n") == 5  # header + 2 x 2 rows
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 2 * 2 * 5


def test_main_stops_when_t4b_is_not_reproduced(tmp_path, monkeypatch):
    pytest.importorskip("faiss")
    pytest.importorskip("pyarrow")
    monkeypatch.setattr(bs, "sample_queries", lambda qs: qs[:5])
    with pytest.raises(SystemExit, match="T4b"):
        bs.main(_setup(tmp_path, monkeypatch, expected_recall=0.1234))
    assert (tmp_path / "vector_stores.json").exists()  # the record is written before stopping
