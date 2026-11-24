import sys
import types

import numpy as np
import pytest

from filinglens.retrieval import embed as em


def test_length_order_and_restore_round_trip():
    texts = ["bb", "a", "dddd", "ccc", "ee"]
    order = em.length_order(texts)
    assert [texts[i] for i in order] == ["dddd", "ccc", "bb", "ee", "a"]  # ties keep order
    computed = np.array([[len(texts[i])] for i in order], dtype=np.float32)
    restored = em.restore_order(order, computed)
    assert restored[:, 0].tolist() == [2, 1, 4, 3, 2]


def unit(rows: int, dim: int, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=(rows, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_check_vectors():
    v = unit(5, 8)
    assert em.check_vectors(v, 5, dim=8)["norm_max"] == pytest.approx(1, abs=1e-6)
    with pytest.raises(ValueError, match="expected"):
        em.check_vectors(v, 6, dim=8)
    bad = v.copy()
    bad[2] *= 1.01
    with pytest.raises(ValueError, match="norms"):
        em.check_vectors(bad, 5, dim=8)
    bad[2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        em.check_vectors(bad, 5, dim=8)


def test_exact_recall_at_k():
    chunks = np.eye(6, dtype=np.float32)  # chunk i points along axis i
    ids = [f"c{i}" for i in range(6)]
    queries = np.zeros((3, 6), dtype=np.float32)
    queries[0, [0, 1]] = [0.9, 0.1]  # top-2: c0, c1
    queries[1, [2, 3]] = [0.6, 0.4]  # top-2: c2, c3
    queries[2, 5] = 1.0  # top-2: c5 then a tie
    grades = [{"c1": 1}, {"c4": 2}, {"c5": 2, "c0": 1}]
    # query 0 finds c1 (grade 1 counts), query 1 misses c4, query 2 finds c5
    assert em.exact_recall_at_k(queries, chunks, ids, grades, k=2, block=2) == pytest.approx(2 / 3)
    assert em.exact_recall_at_k(queries, chunks, ids, grades, k=6) == 1.0
    assert em.exact_recall_at_k(queries[:0], chunks, ids, [], k=2) == 0.0


def test_ids_round_trip(tmp_path):
    em.write_ids(tmp_path / "ids.txt", ["a#0", "b#1"])
    assert em.read_ids(tmp_path / "ids.txt") == ["a#0", "b#1"]


def test_mlflow_outage_is_recorded_not_fatal(tmp_path, monkeypatch):
    from filinglens.train import tracking

    def down(*args, **kwargs):
        raise ConnectionError("tracking server down")

    monkeypatch.setitem(sys.modules, "mlflow", types.ModuleType("mlflow"))  # no real client
    monkeypatch.setattr(tracking, "start_run", down)
    manifest = tmp_path / "m.txt"
    manifest.write_text("x")
    result = {"rows": 3, "seconds": 1.0}
    assert em.log_mlflow("embed_test", result, manifest, "note") is None
    assert "tracking server down" in result["mlflow_error"]
