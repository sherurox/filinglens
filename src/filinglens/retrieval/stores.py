"""Vector stores over the same chunk embeddings, with optional company filter (FL-M3 T5).

Every store answers `search(query, k, cik=None) -> [chunk_id, ...]` (best first). With `cik`, only
that company's chunks are candidates: the analyst agent always knows which company it asks
about, and T4b showed that unfiltered search mostly returns other companies' boilerplate (exact
dev recall@10 0.2967 over all chunks vs 0.6758 within the company).

Shared settings (fixed before any result): inner product on L2-normalized vectors (= cosine),
HNSW M = 16, ef_construction = 100, ef_search = 100. Each store filters the way its own docs
recommend; a filter that keeps ~0.3% of the points is the hard case for graph search, and how
each store copes is part of what T7 measures.

- `FaissStore("flat")`: exact search (IndexFlatIP), the reference. Filter = an ID selector.
- `FaissStore("hnsw")`: IndexHNSWFlat with the same ID selector inside the graph search.
- `PgVectorStore`: table `retrieval.chunk_vectors`, a B-tree on cik and an HNSW index
  (`vector_ip_ops`); `hnsw.iterative_scan = strict_order` lets a filtered graph scan continue
  until it has k matches. The planner may instead use the B-tree and sort that company's rows
  exactly; `explain()` records which plan it picked.
- `QdrantStore`: dot-product collection over gRPC, with a payload index on cik created before
  the upload (Qdrant adds filter-aware HNSW links only for payload indexes that exist when it
  builds the graph). For a filter matching fewer points than `full_scan_threshold`, Qdrant's
  planner scans the matching points exactly instead of walking the graph; recorded, not tuned.

Searches run on one OpenMP thread (`SEARCH_THREADS`): the server stores answer a query on one
worker, so a FAISS search spread over 24 threads would not be comparable. Builds use every CPU
(`build_threads()`), set explicitly: OpenMP's thread count is per process, so a search just
before a build would otherwise leave the build on one thread.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

HNSW_M = 16
EF_CONSTRUCTION = 100
EF_SEARCH = 100
SEARCH_THREADS = 1
VECTOR_LITERAL = re.compile(r"'\[[^\]]*\]'::vector")


def build_threads() -> int:
    return os.cpu_count() or 1


class VectorStore(Protocol):
    name: str

    def search(self, query: np.ndarray, k: int, cik: int | None = None) -> list[str]: ...


# ---- pure helpers (no store library needed) ---------------------------------------------------


def company_rows(ciks: np.ndarray) -> dict[int, np.ndarray]:
    """Row numbers of each company's chunks, ascending, as int64 (FAISS's ID type)."""
    order = np.argsort(ciks, kind="stable")
    sorted_ciks = ciks[order]
    starts = np.flatnonzero(np.r_[True, sorted_ciks[1:] != sorted_ciks[:-1]])
    ends = np.r_[starts[1:], len(order)]
    return {
        int(sorted_ciks[s]): np.sort(order[s:e]).astype(np.int64)
        for s, e in zip(starts, ends, strict=True)
    }


def exact_top_k(
    vectors: np.ndarray, query: np.ndarray, k: int, rows: np.ndarray | None = None
) -> np.ndarray:
    """Row numbers of the k highest inner products (among `rows` if given), best first.

    Ties are broken by row number, so the order is reproducible.
    """
    candidates = np.arange(len(vectors)) if rows is None else rows
    scores = vectors[candidates] @ query
    order = np.lexsort((candidates, -scores))[:k]
    return candidates[order]


def overlap_at_k(found: Sequence[str], reference: Sequence[str], k: int) -> float:
    """Share of the reference top-k that `found`'s top-k also contains."""
    ref = list(reference)[:k]
    if not ref:
        return 1.0
    return len(set(found[:k]) & set(ref)) / len(ref)


def check_inputs(ids: Sequence[str], vectors: np.ndarray, ciks: np.ndarray) -> None:
    if not (len(ids) == len(vectors) == len(ciks)):
        raise ValueError(
            f"lengths differ: {len(ids)} ids, {len(vectors)} vectors, {len(ciks)} ciks"
        )
    if vectors.dtype != np.float32 or vectors.ndim != 2:
        raise ValueError(f"expected a 2-D float32 array, got {vectors.dtype} {vectors.shape}")
    if len(set(ids)) != len(ids):
        raise ValueError("chunk IDs are not unique")


# ---- FAISS ------------------------------------------------------------------------------------


class FaissStore:
    """FAISS in process: 'flat' (exact) or 'hnsw'. Row number i is chunk ids[i]."""

    def __init__(self, kind: str) -> None:
        if kind not in ("flat", "hnsw"):
            raise ValueError(f"unknown FAISS index kind {kind!r}")
        import faiss

        self.faiss = faiss
        self.kind = kind
        self.name = f"faiss-{kind}"
        self.index = None
        self.ids: list[str] = []
        self.rows: dict[int, np.ndarray] = {}
        self._params: dict[int | None, object] = {}  # keeps selectors referenced (see search)

    def _new_index(self, dim: int):
        faiss = self.faiss
        if self.kind == "flat":
            return faiss.IndexFlatIP(dim)
        index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = EF_CONSTRUCTION
        return index

    def _attach(self, ids: Sequence[str], ciks: np.ndarray) -> None:
        self.ids = list(ids)
        self.rows = company_rows(np.asarray(ciks))
        self._params = {}

    def build(self, ids: Sequence[str], vectors: np.ndarray, ciks: np.ndarray) -> dict:
        check_inputs(ids, vectors, np.asarray(ciks))
        faiss = self.faiss
        faiss.omp_set_num_threads(build_threads())
        threads = faiss.omp_get_max_threads()
        start = time.perf_counter()
        self.index = self._new_index(vectors.shape[1])
        self.index.add(vectors)
        seconds = time.perf_counter() - start
        self._attach(ids, ciks)
        return {
            "store": self.name,
            "rows": int(self.index.ntotal),
            "build_seconds": round(seconds, 1),
            "build_threads": threads,
            "params": self.params(),
        }

    def params(self) -> dict:
        if self.kind == "flat":
            return {"index": "IndexFlatIP"}
        return {
            "index": "IndexHNSWFlat",
            "M": HNSW_M,
            "ef_construction": EF_CONSTRUCTION,
            "ef_search": EF_SEARCH,
        }

    def save(self, path: Path) -> int:
        """Write the index; returns its size in bytes."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss.write_index(self.index, str(path))
        return path.stat().st_size

    @classmethod
    def load(cls, kind: str, path: Path, ids: Sequence[str], ciks: np.ndarray) -> FaissStore:
        store = cls(kind)
        store.index = store.faiss.read_index(str(path))
        if store.index.ntotal != len(ids):
            raise ValueError(f"index has {store.index.ntotal} rows, {len(ids)} ids given")
        store._attach(ids, ciks)
        return store

    def _search_params(self, cik: int | None):
        """Search parameters per company, built once. FAISS's Python objects only hold
        pointers: the selector must stay referenced as long as the parameters are used."""
        if cik in self._params:
            return self._params[cik]
        faiss = self.faiss
        params = faiss.SearchParametersHNSW() if self.kind == "hnsw" else faiss.SearchParameters()
        if self.kind == "hnsw":
            params.efSearch = EF_SEARCH
        keep = []
        if cik is not None:
            rows = self.rows.get(int(cik))
            if rows is None:
                raise KeyError(f"no chunks for cik {cik}")
            selector = faiss.IDSelectorBatch(len(rows), faiss.swig_ptr(rows))
            params.sel = selector
            keep = [rows, selector]
        self._params[cik] = (params, keep)
        return self._params[cik]

    def search(self, query: np.ndarray, k: int, cik: int | None = None) -> list[str]:
        params, _ = self._search_params(cik)
        self.faiss.omp_set_num_threads(SEARCH_THREADS)
        _, found = self.index.search(query.reshape(1, -1).astype(np.float32), k, params=params)
        return [self.ids[i] for i in found[0] if i >= 0]


# ---- pgvector ---------------------------------------------------------------------------------


def pg_conninfo() -> dict:
    """The compose.yaml Postgres (localhost:55432); the password comes from .env."""
    password = os.environ.get("POSTGRES_PASSWORD")
    if not password:
        raise KeyError("POSTGRES_PASSWORD is not set: run `set -a; source .env; set +a`")
    return {
        "host": os.environ.get("FL_PG_HOST", "127.0.0.1"),
        "port": int(os.environ.get("FL_PG_PORT", "55432")),
        "dbname": "filinglens",
        "user": "filinglens",
        "password": password,
    }


class PgVectorStore:
    """pgvector in the project's Postgres. One connection, autocommit, one query at a time."""

    name = "pgvector"

    def __init__(
        self, schema: str = "retrieval", table: str = "chunk_vectors", conninfo: dict | None = None
    ) -> None:
        import psycopg
        from pgvector.psycopg import register_vector
        from psycopg import sql

        self.psycopg, self.sql = psycopg, sql
        self.schema, self.table = schema, table
        self.conn = psycopg.connect(**(conninfo or pg_conninfo()), autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self._configure()
        tbl = sql.Identifier(schema, table)
        order = sql.SQL("ORDER BY embedding <#> %s LIMIT %s")  # <#> = negative inner product
        self._q_all = sql.SQL("SELECT chunk_id FROM {} {}").format(tbl, order)
        self._q_cik = sql.SQL("SELECT chunk_id FROM {} WHERE cik = %s {}").format(tbl, order)

    def _configure(self) -> None:
        self.conn.execute(f"SET hnsw.ef_search = {EF_SEARCH}")
        self.conn.execute("SET hnsw.iterative_scan = strict_order")
        # psycopg prepares a statement after 5 runs; a generic plan could differ from the plan
        # EXPLAIN shows for a given cik, so every run gets a plan made for its own parameters.
        self.conn.execute("SET plan_cache_mode = force_custom_plan")

    def _index_names(self) -> tuple[str, str]:
        return f"{self.table}_cik_idx", f"{self.table}_embedding_hnsw_idx"

    def build(
        self,
        ids: Sequence[str],
        vectors: np.ndarray,
        ciks: np.ndarray,
        maintenance_work_mem: str = "1GB",
        workers: int = 4,
    ) -> dict:
        """(Re)create the table, COPY the rows in, then build both indexes and ANALYZE."""
        check_inputs(ids, vectors, np.asarray(ciks))
        sql = self.sql
        schema, tbl = sql.Identifier(self.schema), sql.Identifier(self.schema, self.table)
        cik_idx, hnsw_idx = (sql.Identifier(n) for n in self._index_names())
        dim = int(vectors.shape[1])
        # pgvector reports with a NOTICE when the graph outgrows maintenance_work_mem (the build
        # then slows down a lot): keep every notice for the bench record.
        notices: list[str] = []

        def keep_notice(diag) -> None:
            notices.append(diag.message_primary)

        self.conn.add_notice_handler(keep_notice)
        with self.conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(schema))
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(tbl))
            cur.execute(
                sql.SQL(
                    "CREATE TABLE {} (chunk_id text PRIMARY KEY, cik integer NOT NULL,"
                    " embedding vector({}) NOT NULL)"
                ).format(tbl, sql.Literal(dim))
            )
            start = time.perf_counter()
            copy_sql = sql.SQL("COPY {} (chunk_id, cik, embedding) FROM STDIN WITH (FORMAT BINARY)")
            with cur.copy(copy_sql.format(tbl)) as copy:
                copy.set_types(["text", "int4", "vector"])
                for chunk_id, cik, vec in zip(ids, np.asarray(ciks), vectors, strict=True):
                    copy.write_row([chunk_id, int(cik), vec])
            load_seconds = time.perf_counter() - start
            cur.execute(
                sql.SQL("SET maintenance_work_mem = {}").format(sql.Literal(maintenance_work_mem))
            )
            cur.execute(
                sql.SQL("SET max_parallel_maintenance_workers = {}").format(sql.Literal(workers))
            )
            start = time.perf_counter()
            cur.execute(sql.SQL("CREATE INDEX {} ON {} (cik)").format(cik_idx, tbl))
            btree_seconds = time.perf_counter() - start
            start = time.perf_counter()
            cur.execute(
                sql.SQL(
                    "CREATE INDEX {} ON {} USING hnsw (embedding vector_ip_ops)"
                    " WITH (m = {}, ef_construction = {})"
                ).format(hnsw_idx, tbl, sql.Literal(HNSW_M), sql.Literal(EF_CONSTRUCTION))
            )
            hnsw_seconds = time.perf_counter() - start
            cur.execute(sql.SQL("ANALYZE {}").format(tbl))
            cur.execute("RESET maintenance_work_mem")
            cur.execute("RESET max_parallel_maintenance_workers")
        self.conn.remove_notice_handler(keep_notice)
        return {
            "store": self.name,
            "rows": len(ids),
            "load_seconds": round(load_seconds, 1),
            "btree_seconds": round(btree_seconds, 1),
            "build_seconds": round(hnsw_seconds, 1),
            "maintenance_work_mem": maintenance_work_mem,
            "max_parallel_maintenance_workers": workers,
            "notices": notices,
            "params": self.params(),
            **self.sizes(),
        }

    def params(self) -> dict:
        return {
            "index": "hnsw vector_ip_ops",
            "M": HNSW_M,
            "ef_construction": EF_CONSTRUCTION,
            "ef_search": EF_SEARCH,
            "iterative_scan": "strict_order",
            "plan_cache_mode": "force_custom_plan",
            # where an iterative scan gives up (pgvector's default 20,000), read from the server
            "max_scan_tuples": int(self.conn.execute("SHOW hnsw.max_scan_tuples").fetchone()[0]),
        }

    def sizes(self) -> dict:
        cik_idx, hnsw_idx = self._index_names()
        qualified = f"{self.schema}.{self.table}"
        row = self.conn.execute(
            "SELECT pg_total_relation_size(%s::regclass), pg_relation_size(%s::regclass),"
            " pg_relation_size(%s::regclass), (SELECT extversion FROM pg_extension"
            " WHERE extname = 'vector')",
            (qualified, f"{self.schema}.{hnsw_idx}", f"{self.schema}.{cik_idx}"),
        ).fetchone()
        return {
            "total_bytes": row[0],
            "hnsw_index_bytes": row[1],
            "cik_index_bytes": row[2],
            "pgvector_version": row[3],
        }

    def search(self, query: np.ndarray, k: int, cik: int | None = None) -> list[str]:
        vec = np.asarray(query, dtype=np.float32)
        if cik is None:
            rows = self.conn.execute(self._q_all, (vec, k)).fetchall()
        else:
            rows = self.conn.execute(self._q_cik, (int(cik), vec, k)).fetchall()
        return [r[0] for r in rows]

    def explain(self, query: np.ndarray, k: int, cik: int | None = None) -> list[str]:
        """The plan Postgres picks for this search (EXPLAIN, costs off), one line per node."""
        vec = np.asarray(query, dtype=np.float32)
        with self.psycopg.ClientCursor(self.conn) as cur:  # EXPLAIN takes literals, not $n
            if cik is None:
                stmt, args = self._q_all, (vec, k)
            else:
                stmt, args = self._q_cik, (int(cik), vec, k)
            cur.execute(self.sql.SQL("EXPLAIN (COSTS OFF) ") + stmt, args)
            # the 768 numbers of the query vector would drown the plan: name it instead
            return [VECTOR_LITERAL.sub("<query vector>", r[0]) for r in cur.fetchall()]

    def drop_schema(self) -> None:
        self.conn.execute(
            self.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                self.sql.Identifier(self.schema)
            )
        )

    def close(self) -> None:
        self.conn.close()


# ---- Qdrant -----------------------------------------------------------------------------------

QDRANT_GRPC_PORT = 56334  # compose.yaml publishes 6334 (gRPC) here


def qdrant_url() -> str:
    return os.environ.get("QDRANT_URL", "http://127.0.0.1:56333")


class QdrantStore:
    """Qdrant over gRPC. Point ID i is row i, i.e. chunk ids[i]; payload {"cik": int}."""

    name = "qdrant"

    def __init__(self, collection: str = "chunk_vectors", url: str | None = None) -> None:
        from qdrant_client import QdrantClient, models

        self.models = models
        self.collection = collection
        self.client = QdrantClient(
            url=url or qdrant_url(), grpc_port=QDRANT_GRPC_PORT, prefer_grpc=True, timeout=120
        )
        self.ids: list[str] = []
        self._filters: dict[int, object] = {}
        self._search_params = models.SearchParams(hnsw_ef=EF_SEARCH)

    def build(
        self,
        ids: Sequence[str],
        vectors: np.ndarray,
        ciks: np.ndarray,
        indexing_threshold_kb: int | None = None,
        timeout_seconds: float = 3600,
    ) -> dict:
        """(Re)create the collection, upload, and wait until the optimizer has indexed it."""
        check_inputs(ids, vectors, np.asarray(ciks))
        m, client, name = self.models, self.client, self.collection
        if client.collection_exists(name):
            client.delete_collection(name)
        optimizers = (
            m.OptimizersConfigDiff(indexing_threshold=indexing_threshold_kb)
            if indexing_threshold_kb is not None
            else None
        )
        client.create_collection(
            name,
            vectors_config=m.VectorParams(size=int(vectors.shape[1]), distance=m.Distance.DOT),
            hnsw_config=m.HnswConfigDiff(m=HNSW_M, ef_construct=EF_CONSTRUCTION),
            optimizers_config=optimizers,
        )
        client.create_payload_index(
            name, field_name="cik", field_schema=m.PayloadSchemaType.INTEGER, wait=True
        )
        start = time.perf_counter()
        client.upload_collection(
            name,
            vectors=vectors,
            payload=({"cik": int(c)} for c in np.asarray(ciks)),
            ids=range(len(ids)),
            batch_size=1024,
            wait=True,
        )
        upload_seconds = time.perf_counter() - start
        ready_at, exit_reason = self._wait_indexed(len(ids), timeout_seconds)
        self._attach(ids)
        return {
            "store": self.name,
            "rows": len(ids),
            "upload_seconds": round(upload_seconds, 1),
            # upload + background indexing, up to the first poll of the final green streak
            # (1 s polls, so +/- 1 s), not the 10 s the wait spends confirming it
            "build_seconds": round(ready_at - start, 1),
            "index_wait_exit": exit_reason,
            "params": self.params(),
            **self.sizes(),
        }

    def _wait_indexed(self, rows: int, timeout_seconds: float) -> tuple[float, str]:
        """Wait until the collection is green with every point indexed, or green for 10 polls
        in a row (segments under the indexing threshold stay unindexed by design).

        Returns (perf_counter time at which it was first seen green in the final streak, how
        the wait ended).
        """
        green_since: float | None = None
        streak, deadline = 0, time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            now = time.perf_counter()
            info = self.client.get_collection(self.collection)
            green = info.status == self.models.CollectionStatus.GREEN
            if not green:
                green_since, streak = None, 0
            else:
                green_since = now if green_since is None else green_since
                streak += 1
                if (info.indexed_vectors_count or 0) >= rows:
                    return green_since, "all_indexed"
                if streak >= 10:
                    return green_since, "green_10_polls"
            time.sleep(1)
        raise TimeoutError(f"collection {self.collection} not indexed after {timeout_seconds} s")

    def _attach(self, ids: Sequence[str]) -> None:
        self.ids = list(ids)
        self._filters = {}

    @classmethod
    def load(cls, ids: Sequence[str], collection: str = "chunk_vectors") -> QdrantStore:
        store = cls(collection)
        points = store.client.get_collection(collection).points_count
        if points != len(ids):
            raise ValueError(f"collection has {points} points, {len(ids)} ids given")
        store._attach(ids)
        return store

    def params(self) -> dict:
        config = self.client.get_collection(self.collection).config
        return {
            "index": "hnsw, dot",
            "M": config.hnsw_config.m,
            "ef_construction": config.hnsw_config.ef_construct,
            "ef_search": EF_SEARCH,
            "full_scan_threshold_kb": config.hnsw_config.full_scan_threshold,
            "indexing_threshold_kb": config.optimizer_config.indexing_threshold,
            "payload_index": "cik (integer)",
            "transport": "gRPC",
        }

    def sizes(self) -> dict:
        info = self.client.get_collection(self.collection)
        try:
            version = self.client.info().version
        except Exception:  # older clients have no info(); the version is in retrieval_env.txt
            version = None
        return {
            "points": info.points_count,
            "indexed_vectors": info.indexed_vectors_count,
            "segments": info.segments_count,
            "qdrant_version": version,
        }

    def _filter(self, cik: int | None):
        if cik is None:
            return None
        cik = int(cik)
        if cik not in self._filters:
            m = self.models
            self._filters[cik] = m.Filter(
                must=[m.FieldCondition(key="cik", match=m.MatchValue(value=cik))]
            )
        return self._filters[cik]

    def search(self, query: np.ndarray, k: int, cik: int | None = None) -> list[str]:
        result = self.client.query_points(
            self.collection,
            query=np.asarray(query, dtype=np.float32).tolist(),
            limit=k,
            query_filter=self._filter(cik),
            search_params=self._search_params,
            with_payload=False,
            with_vectors=False,
        )
        return [self.ids[point.id] for point in result.points]

    def drop(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)

    def close(self) -> None:
        self.client.close()
