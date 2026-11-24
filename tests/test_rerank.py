import csv
import json

import numpy as np
import pytest

from filinglens.retrieval import rerank as rr


def test_rerank_order_sorts_by_score_and_keeps_dense_order_on_ties():
    candidates = ["a", "b", "c", "d"]
    assert rr.rerank_order(candidates, [0.1, 0.9, 0.5, 0.9]) == ["b", "d", "c", "a"]
    assert rr.rerank_order([], []) == []


def test_per_query_metrics_before_and_after():
    grades = {"g": 2}
    dense = [f"x{i}" for i in range(12)] + ["g"]  # g at dense rank 13: outside the top 10
    reranked = ["g"] + dense[:-1]  # the reranker lifts it to rank 1
    m = rr.per_query_metrics(dense, reranked, grades)
    assert m["recall_before"] == 0.0 and m["recall_after"] == 1.0
    assert m["mrr_after"] == 1.0 and m["ndcg_after"] == 1.0
    assert m["recall_at_depth"] == 1.0  # it was among the candidates all along


def test_mean_metrics_and_latency_summary():
    rows = [{"query_id": "a", "ndcg_after": 1.0}, {"query_id": "b", "ndcg_after": 0.0}]
    assert rr.mean_metrics(rows) == {"ndcg_after": 0.5}
    lat = rr.latency_summary([10.0] * 99 + [50.0])
    assert lat["p50_ms"] == 10.0 and lat["samples"] == 100


def test_write_table(tmp_path):
    record = {
        "queries": 2,
        "metrics": {"recall_before": 0.5, "mrr_before": 0.25, "ndcg_before": 0.3,
                    "recall_after": 1.0, "mrr_after": 0.75, "ndcg_after": 0.8,
                    "recall_at_depth": 1.0},
        "rerank_latency": {"p50_ms": 20.0, "p95_ms": 30.0, "p99_ms": 40.0},
    }  # fmt: skip
    rr.write_table(tmp_path / "t.csv", record)
    rows = list(csv.DictReader((tmp_path / "t.csv").open()))
    assert [r["ndcg_at_10"] for r in rows] == ["0.3", "0.8"]
    assert rows[0]["rerank_p50_ms"] == "" and rows[1]["rerank_p95_ms"] == "30.0"
    assert rows[1]["queries"] == "2" and rows[1]["recall_at_depth"] == "1.0"


def test_candidates_step_writes_the_filtered_top_depth(tmp_path, monkeypatch):
    pytest.importorskip("faiss")
    pytest.importorskip("pyarrow")
    import pandas as pd

    from filinglens.retrieval import build_indexes, stores

    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(200, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids, ciks = [f"p{i}#0" for i in range(200)], np.array([1] * 120 + [2] * 80)
    emb, qdir, faiss_dir = tmp_path / "emb", tmp_path / "q", tmp_path / "faiss"
    for d in (emb, qdir):
        d.mkdir()
    (emb / "chunk_ids.txt").write_text("".join(f"{c}\n" for c in ids))
    pd.DataFrame({"chunk_id": ids, "cik": ciks}).to_parquet(tmp_path / "chunks.parquet")
    flat = stores.FaissStore("flat")
    flat.build(ids, vecs, ciks)
    flat.save(faiss_dir / "flat.index")
    monkeypatch.setattr(build_indexes, "FAISS_DIR", faiss_dir)
    queries = [
        {"query_id": "q0", "cik": 2, "grades": {}},
        {"query_id": "q1", "cik": 1, "grades": {}},
    ]
    np.save(emb / "queries.dev.npy", vecs[[150, 3]])
    (emb / "query_ids.dev.txt").write_text("q0\nq1\n")
    (qdir / "queries.dev.jsonl").write_text("".join(json.dumps(q) + "\n" for q in queries))
    rr.main([
        "candidates", "--split", "dev", "--query-dir", str(qdir), "--emb-dir", str(emb),
        "--chunks", str(tmp_path / "chunks.parquet"), "--ret-dir", str(tmp_path),
    ])  # fmt: skip
    rows = rr.read_jsonl(tmp_path / "candidates.dev.jsonl")
    assert [r["query_id"] for r in rows] == ["q0", "q1"]
    assert len(rows[0]["candidates"]) == rr.DEPTH and rows[0]["candidates"][0] == "p150#0"
    assert all(int(c[1:-2]) >= 120 for c in rows[0]["candidates"])  # company 2 only
    assert all(int(c[1:-2]) < 120 for c in rows[1]["candidates"])  # company 1 only


def test_candidates_step_guards_the_test_split(tmp_path):
    with pytest.raises(PermissionError):
        rr.main(["candidates", "--split", "test", "--query-dir", str(tmp_path)])


def test_score_step_with_a_fake_reranker(tmp_path, monkeypatch):
    """The whole `score` step without torch or a GPU: a stub torch module and a fake reranker
    that gives the graded chunk the top score, so 'after' must find what 'before' missed."""
    pytest.importorskip("pyarrow")
    import sys
    import types

    import pandas as pd

    from filinglens.train import tracking

    cuda = types.SimpleNamespace(
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 2**20,
        max_memory_reserved=lambda: 2**21,
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))

    class FakeReranker:
        def __init__(self, device="cuda", fp16=True):
            pass

        def count_long_pairs(self, question, texts):
            return 0

        def __call__(self, question, texts):
            return np.array([1.0 if "ANSWER" in t else 0.0 for t in texts], dtype=np.float32)

    monkeypatch.setattr(rr, "Reranker", FakeReranker)
    monkeypatch.setattr(
        rr, "check_model_files", lambda: {"architectures": ["fake"], "num_labels": 1}
    )
    monkeypatch.setattr(rr, "log_run", lambda *a, **k: "run-x")
    monkeypatch.setattr(tracking, "gpu_snapshot", lambda: {})

    qdir, ret = tmp_path / "q", tmp_path / "ret"
    qdir.mkdir()
    ret.mkdir()
    ids = [f"c{i}" for i in range(30)]
    texts = ["ANSWER" if c == "c20" else f"text {c}" for c in ids]
    pd.DataFrame({"chunk_id": ids, "embed_text": texts}).to_parquet(tmp_path / "chunks.parquet")
    queries = [
        {"query_id": "q0", "question": "?", "grades": {"c20": 2}},  # rank 21 dense, 1 reranked
        {"query_id": "q1", "question": "?", "grades": {"c0": 2}},  # rank 1 dense
    ]
    (qdir / "queries.dev.jsonl").write_text("".join(json.dumps(q) + "\n" for q in queries))
    rr.write_jsonl(
        ret / "candidates.dev.jsonl",
        [{"query_id": q["query_id"], "candidates": ids} for q in queries],
    )
    (tmp_path / "diag.json").write_text(json.dumps({"recall_at_10": {"company": 0.5}}))
    chunks, diag = str(tmp_path / "chunks.parquet"), str(tmp_path / "diag.json")
    rr.main([
        "score", "--split", "dev", "--query-dir", str(qdir), "--chunks", chunks,
        "--ret-dir", str(ret), "--bench", str(tmp_path), "--diagnostics", diag, "--note", "test",
    ])  # fmt: skip
    record = json.loads((tmp_path / "reranker_dev.json").read_text())
    assert record["metrics"]["recall_before"] == 0.5 and record["metrics"]["recall_after"] == 1.0
    assert record["reproduce_t4b"]["match"] is True
    assert record["pairs"] == 60 and record["rerank_latency"]["samples"] == 2
    per_query = rr.read_jsonl(ret / "rerank_dev_per_query.jsonl")
    # q0: c20 lifted from rank 21 to 1. q1: the fake prefers c20 for every question, so the
    # right chunk c0 drops to rank 2: a reranker can also make a query worse.
    assert per_query[0]["ndcg_after"] == 1.0
    assert per_query[1]["ndcg_after"] == pytest.approx(1 / np.log2(3))
    assert (tmp_path / "reranker.csv").read_text().count("\n") == 3  # header + 2 variants
