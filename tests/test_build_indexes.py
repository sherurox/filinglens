import json

import numpy as np
import pandas as pd
import pytest

from filinglens.retrieval import build_indexes as bi


def _inputs(tmp_path, rows=300, dim=16, order=None):
    emb = tmp_path / "emb"
    emb.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(rows, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = [f"p{i}#0" for i in range(rows)]
    np.save(emb / "chunks.npy", vectors)
    (emb / "chunk_ids.txt").write_text("".join(f"{c}\n" for c in (order or ids)))
    pd.DataFrame({"chunk_id": ids, "cik": rng.integers(1, 4, size=rows)}).to_parquet(
        tmp_path / "chunks.parquet"
    )
    return emb


def _args(tmp_path, emb, *stores):
    return [
        "--stores",
        *stores,
        "--emb-dir",
        str(emb),
        "--chunks",
        str(tmp_path / "chunks.parquet"),
        "--faiss-dir",
        str(tmp_path / "faiss"),
        "--bench",
        str(tmp_path),
        "--note",
        "test note",
    ]


def test_load_inputs_refuses_a_different_order(tmp_path):
    pytest.importorskip("pyarrow")
    emb = _inputs(tmp_path, rows=5, order=["p1#0", "p0#0", "p2#0", "p3#0", "p4#0"])
    with pytest.raises(ValueError, match="different orders"):
        bi.load_inputs(emb, tmp_path / "chunks.parquet")


def test_main_builds_faiss_and_merges_partial_rebuilds(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    pytest.importorskip("faiss")
    monkeypatch.setattr(bi, "log_run", lambda *a, **k: "run-123")  # no tracking server needed
    emb = _inputs(tmp_path)
    bi.main(_args(tmp_path, emb, "faiss-flat", "faiss-hnsw"))
    record = json.loads((tmp_path / "index_build.json").read_text())
    assert set(record["stores"]) == {"faiss-flat", "faiss-hnsw"}
    for stats in record["stores"].values():
        assert stats["self_check_top10"] is True and stats["rows"] == 300
        assert stats["index_bytes"] > 0 and stats["search_threads"] == 1
    assert record["mlflow_run_id"] == "run-123"
    assert record["command"].endswith("--note 'test note'")  # shlex keeps the quotes
    assert (tmp_path / "faiss" / "hnsw.index").exists()

    bi.main(_args(tmp_path, emb, "faiss-flat"))  # rebuild one store: the other stays recorded
    record = json.loads((tmp_path / "index_build.json").read_text())
    assert set(record["stores"]) == {"faiss-flat", "faiss-hnsw"}


def test_runlog_records_an_outage(monkeypatch, tmp_path):
    import sys
    import types

    from filinglens.retrieval import runlog
    from filinglens.train import tracking

    def down(*args, **kwargs):
        raise ConnectionError("tracking server down")

    monkeypatch.setitem(sys.modules, "mlflow", types.ModuleType("mlflow"))
    monkeypatch.setattr(tracking, "start_run", down)
    (tmp_path / "m").write_text("x")
    record: dict = {}
    assert runlog.log_run("x", record, {}, {}, tmp_path / "m", "note") is None
    assert "tracking server down" in record["mlflow_error"]
