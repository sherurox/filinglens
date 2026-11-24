import numpy as np
import pytest

from filinglens.retrieval import stores as st


def unit_vectors(rows: int, dim: int, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=(rows, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def corpus(rows: int = 2000, dim: int = 32, companies: int = 4, seed: int = 0):
    vectors = unit_vectors(rows, dim, seed)
    ciks = np.random.default_rng(seed + 1).integers(1, companies + 1, size=rows)
    ids = [f"c{i}#0" for i in range(rows)]
    return ids, vectors, ciks


# ---- pure helpers ----------------------------------------------------------------------------


def test_company_rows():
    rows = st.company_rows(np.array([7, 3, 7, 3, 9]))
    assert {k: v.tolist() for k, v in rows.items()} == {3: [1, 3], 7: [0, 2], 9: [4]}
    assert all(v.dtype == np.int64 for v in rows.values())


def test_exact_top_k_with_and_without_rows():
    vectors = np.array([[1.0, 0], [0.8, 0.6], [0, 1], [0.8, 0.6]], dtype=np.float32)
    query = np.array([1.0, 0], dtype=np.float32)
    assert st.exact_top_k(vectors, query, 3).tolist() == [0, 1, 3]  # tie 1 vs 3: lower row first
    assert st.exact_top_k(vectors, query, 2, rows=np.array([2, 3])).tolist() == [3, 2]
    assert st.exact_top_k(vectors, query, 10, rows=np.array([2])).tolist() == [2]


def test_overlap_at_k():
    assert st.overlap_at_k(["a", "b", "c"], ["c", "a", "x"], 3) == pytest.approx(2 / 3)
    assert st.overlap_at_k(["a"], ["a", "b"], 1) == 1.0
    assert st.overlap_at_k([], [], 10) == 1.0


def test_check_inputs():
    ids, vectors, ciks = corpus(10, 4)
    st.check_inputs(ids, vectors, ciks)
    with pytest.raises(ValueError, match="lengths"):
        st.check_inputs(ids[:-1], vectors, ciks)
    with pytest.raises(ValueError, match="float32"):
        st.check_inputs(ids, vectors.astype(np.float64), ciks)
    with pytest.raises(ValueError, match="unique"):
        st.check_inputs(["x"] * 10, vectors, ciks)


# ---- FAISS (runs where faiss-cpu is installed: the uv env) ----------------------------------


@pytest.fixture(scope="module")
def faiss_corpus():
    pytest.importorskip("faiss")
    ids, vectors, ciks = corpus()
    flat, hnsw = st.FaissStore("flat"), st.FaissStore("hnsw")
    flat.build(ids, vectors, ciks)
    stats = hnsw.build(ids, vectors, ciks)
    return ids, vectors, ciks, flat, hnsw, stats


def test_faiss_flat_equals_numpy_exact(faiss_corpus):
    ids, vectors, ciks, flat, _, _ = faiss_corpus
    for q in unit_vectors(20, vectors.shape[1], seed=5):
        expected = [ids[i] for i in st.exact_top_k(vectors, q, 10)]
        assert flat.search(q, 10) == expected
        rows = st.company_rows(ciks)[2]
        expected_2 = [ids[i] for i in st.exact_top_k(vectors, q, 10, rows=rows)]
        assert flat.search(q, 10, cik=2) == expected_2


def test_faiss_filter_returns_only_that_company(faiss_corpus):
    ids, vectors, ciks, flat, hnsw, _ = faiss_corpus
    cik_of = dict(zip(ids, ciks, strict=True))
    for store in (flat, hnsw):
        for q in unit_vectors(10, vectors.shape[1], seed=6):
            found = store.search(q, 10, cik=3)
            assert len(found) == 10 and {cik_of[c] for c in found} == {3}, store.name


def test_faiss_hnsw_close_to_exact(faiss_corpus):
    ids, vectors, _, flat, hnsw, stats = faiss_corpus
    queries = unit_vectors(50, vectors.shape[1], seed=7)
    unfiltered = np.mean(
        [st.overlap_at_k(hnsw.search(q, 10), flat.search(q, 10), 10) for q in queries]
    )
    filtered = np.mean(
        [st.overlap_at_k(hnsw.search(q, 10, cik=1), flat.search(q, 10, cik=1), 10) for q in queries]
    )
    assert unfiltered >= 0.95 and filtered >= 0.95, (unfiltered, filtered)
    assert stats["params"] == {
        "index": "IndexHNSWFlat",
        "M": 16,
        "ef_construction": 100,
        "ef_search": 100,
    }
    assert stats["rows"] == len(ids)


def test_faiss_build_uses_every_cpu_even_after_a_search(faiss_corpus):
    """A search sets OpenMP to one thread for the whole process; the next build must undo it."""
    import os

    ids, vectors, ciks, flat, _, _ = faiss_corpus
    flat.search(vectors[0], 10)
    stats = st.FaissStore("flat").build(ids, vectors, ciks)
    assert stats["build_threads"] == (os.cpu_count() or 1)


def test_faiss_small_company_and_unknown_company():
    pytest.importorskip("faiss")
    ids, vectors, _ = corpus(50, 8)
    ciks = np.array([1] * 47 + [2] * 3)
    store = st.FaissStore("flat")
    store.build(ids, vectors, ciks)
    assert sorted(store.search(vectors[0], 10, cik=2)) == ["c47#0", "c48#0", "c49#0"]
    with pytest.raises(KeyError):
        store.search(vectors[0], 10, cik=99)


def test_faiss_save_and_load_round_trip(faiss_corpus, tmp_path):
    ids, vectors, ciks, _, hnsw, _ = faiss_corpus
    size = hnsw.save(tmp_path / "hnsw.index")
    assert size > vectors.nbytes  # vectors plus the graph
    loaded = st.FaissStore.load("hnsw", tmp_path / "hnsw.index", ids, ciks)
    for q in unit_vectors(5, vectors.shape[1], seed=8):
        assert loaded.search(q, 10, cik=4) == hnsw.search(q, 10, cik=4)
    with pytest.raises(ValueError, match="rows"):
        st.FaissStore.load("hnsw", tmp_path / "hnsw.index", ids[:-1], ciks[:-1])


# ---- pgvector (runs against the compose.yaml Postgres; skipped when it isn't reachable) -----


@pytest.fixture(scope="module")
def pg_corpus():
    pytest.importorskip("psycopg")
    pytest.importorskip("pgvector")
    try:
        store = st.PgVectorStore(schema="retrieval_test")  # never the real retrieval schema
    except Exception as error:  # no password, server down, ...
        pytest.skip(f"Postgres not reachable: {error}")
    ids, vectors, ciks = corpus()
    stats = store.build(ids, vectors, ciks, maintenance_work_mem="64MB", workers=0)
    yield ids, vectors, ciks, store, stats
    store.drop_schema()
    store.close()


def test_pgvector_build_stats(pg_corpus):
    ids, _, _, _, stats = pg_corpus
    assert stats["rows"] == len(ids)
    assert stats["params"]["M"] == 16 and stats["params"]["ef_search"] == 100
    assert stats["params"]["max_scan_tuples"] > 0
    assert stats["hnsw_index_bytes"] > 0 and stats["total_bytes"] > stats["hnsw_index_bytes"]
    assert stats["pgvector_version"] == "0.8.7"
    assert isinstance(stats["notices"], list)  # NOTICEs from the build, kept for the record


def test_pgvector_close_to_exact_and_filter_honored(pg_corpus):
    ids, vectors, ciks, store, _ = pg_corpus
    cik_of = dict(zip(ids, ciks, strict=True))
    rows = st.company_rows(ciks)
    unfiltered, filtered = [], []
    for q in unit_vectors(30, vectors.shape[1], seed=9):
        exact = [ids[i] for i in st.exact_top_k(vectors, q, 10)]
        unfiltered.append(st.overlap_at_k(store.search(q, 10), exact, 10))
        found = store.search(q, 10, cik=2)
        assert len(found) == 10 and {cik_of[c] for c in found} == {2}
        exact_2 = [ids[i] for i in st.exact_top_k(vectors, q, 10, rows=rows[2])]
        filtered.append(st.overlap_at_k(found, exact_2, 10))
    assert np.mean(unfiltered) >= 0.95 and np.mean(filtered) >= 0.95, (unfiltered, filtered)


def test_pgvector_explain_names_a_plan(pg_corpus):
    _, vectors, _, store, _ = pg_corpus
    plan = store.explain(vectors[0], 10, cik=1)
    assert plan and plan[0].startswith("Limit")
    assert any("Scan" in line for line in plan)  # which scan is the finding, not asserted here
    assert not any("::vector" in line for line in plan)  # the literal is replaced
    assert any("<query vector>" in line for line in plan)


# ---- Qdrant (runs against the compose.yaml Qdrant; skipped when it isn't reachable) ---------


@pytest.fixture(scope="module")
def qdrant_corpus():
    pytest.importorskip("qdrant_client")
    try:
        store = st.QdrantStore(collection="retrieval_test")  # never the real collection
        store.client.get_collections()
    except Exception as error:
        pytest.skip(f"Qdrant not reachable: {error}")
    ids, vectors, ciks = corpus()
    # 1 KB threshold: the tiny test collection gets an HNSW graph like the real one does
    stats = store.build(ids, vectors, ciks, indexing_threshold_kb=1, timeout_seconds=120)
    yield ids, vectors, ciks, store, stats
    store.drop()
    store.close()


def test_qdrant_build_stats(qdrant_corpus):
    ids, _, _, _, stats = qdrant_corpus
    assert stats["rows"] == stats["points"] == len(ids)
    assert stats["params"]["M"] == 16 and stats["params"]["ef_construction"] == 100
    assert stats["params"]["transport"] == "gRPC"
    assert stats["indexed_vectors"] > 0  # a graph was built (Qdrant's count is approximate)
    assert stats["index_wait_exit"] in ("all_indexed", "green_10_polls")
    assert 0 <= stats["upload_seconds"] <= stats["build_seconds"] + 1  # +/- one 1 s poll


def test_qdrant_close_to_exact_and_filter_honored(qdrant_corpus):
    ids, vectors, ciks, store, _ = qdrant_corpus
    cik_of = dict(zip(ids, ciks, strict=True))
    rows = st.company_rows(ciks)
    unfiltered, filtered = [], []
    for q in unit_vectors(30, vectors.shape[1], seed=10):
        exact = [ids[i] for i in st.exact_top_k(vectors, q, 10)]
        unfiltered.append(st.overlap_at_k(store.search(q, 10), exact, 10))
        found = store.search(q, 10, cik=4)
        assert len(found) == 10 and {cik_of[c] for c in found} == {4}
        exact_4 = [ids[i] for i in st.exact_top_k(vectors, q, 10, rows=rows[4])]
        filtered.append(st.overlap_at_k(found, exact_4, 10))
    assert np.mean(unfiltered) >= 0.95 and np.mean(filtered) >= 0.95, (unfiltered, filtered)


def test_qdrant_load_checks_the_point_count(qdrant_corpus):
    ids, vectors, _, store, _ = qdrant_corpus
    loaded = st.QdrantStore.load(ids, collection="retrieval_test")
    assert loaded.search(vectors[3], 10, cik=1) == store.search(vectors[3], 10, cik=1)
    with pytest.raises(ValueError, match="points"):
        st.QdrantStore.load(ids[:-1], collection="retrieval_test")


def test_qdrant_wait_records_the_start_of_the_green_streak(monkeypatch):
    """No server needed: a fake client goes yellow, yellow, then green and never 'all indexed'."""
    import types

    clock = iter(range(100))  # perf_counter: one tick per poll
    monkeypatch.setattr(st.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(st.time, "sleep", lambda seconds: None)
    statuses = iter(["yellow", "yellow", "green", "yellow"] + ["green"] * 20)

    class Client:
        def get_collection(self, name):
            return types.SimpleNamespace(status=next(statuses), indexed_vectors_count=5)

    store = object.__new__(st.QdrantStore)
    store.client, store.collection = Client(), "x"
    store.models = types.SimpleNamespace(CollectionStatus=types.SimpleNamespace(GREEN="green"))
    # the green at tick 2 is broken by a yellow; the final streak starts at tick 4
    assert store._wait_indexed(rows=10, timeout_seconds=60) == (4, "green_10_polls")
