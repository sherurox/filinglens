"""API routes with a fake store, encoder and model (test client; no database, no network)."""

import asyncio

import numpy as np
import pytest

pytest.importorskip("fastapi")  # the `api` dependency group: uv run --group api pytest

from fastapi.testclient import TestClient  # noqa: E402

from filinglens.api.app import Deps, create_app  # noqa: E402
from filinglens.api.logic import Hit  # noqa: E402
from filinglens.serve.backends import Reply  # noqa: E402
from filinglens.serve.cost_meter import Prices  # noqa: E402

PRICES = Prices(0.25, 1.50, 0.1817, {"vllm": 64.1, "gguf": 75.0})
DAL = {"cik": 27904, "ticker": "DAL", "sector": "transport_utilities", "chunks": 3}


class FakeStore:
    def __init__(self) -> None:
        self.searched: list[tuple] = []

    def ping(self):
        return True

    def companies(self):
        return [{"ticker": "DAL", "sector": DAL["sector"], "chunks": 3}]

    def company(self, ticker):
        return DAL if ticker.upper() == "DAL" else None

    def search(self, cik, vector, k):
        self.searched.append((cik, vector.shape, k))
        return [
            Hit(f"c{i}", "DAL", 2024, "0000027904-25-000004", "Fuel", f"t{i}", 0.9)
            for i in range(k)
        ]

    def costs(self):
        row = {"route": "answer", "backend": "gemini", "requests": 2, "prompt_tokens": 10}
        return [row | {"output_tokens": 4, "latency_p95_s": 1.0, "cost_usd": 1e-5, "errors": 0}]


class FakeEmbedder:
    def embed_queries(self, questions):
        return np.ones((len(questions), 768), dtype=np.float32) / np.sqrt(768)


class FakeLLM:
    def __init__(self, reply: Reply) -> None:
        self.reply, self.calls = reply, []

    async def chat(self, conv, schema=None, max_tokens=400, *, route, user):
        self.calls.append({"conv": conv, "max_tokens": max_tokens, "route": route, "user": user})
        return self.reply


def client(reply: Reply | None = None) -> tuple[TestClient, FakeStore, FakeLLM]:
    store = FakeStore()
    llm = FakeLLM(reply or Reply("gemini", "m", "Fuel is volatile [1].", 400, 20, 0, 1.2, "STOP"))
    app = create_app(Deps(store, FakeEmbedder(), llm, PRICES))
    return TestClient(app), store, llm


def test_health_and_companies():
    c, _, _ = client()
    with c:
        assert c.get("/health").json() == {"status": "ok", "database": True, "spend_usd": 0}
        body = c.get("/companies").json()
        assert body["count"] == 1 and body["companies"][0]["ticker"] == "DAL"


def test_search_is_per_company_and_links_filings():
    c, store, _ = client()
    with c:
        r = c.post("/search", json={"ticker": "dal", "question": "fuel price risk?", "k": 3})
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert len(hits) == 3 and hits[0]["filing_url"].startswith("https://www.sec.gov/Archives/")
    assert store.searched == [(27904, (768,), 3)]


def test_unknown_ticker_is_404_and_bad_input_is_422():
    c, _, _ = client()
    with c:
        assert c.post("/search", json={"ticker": "ZZZ", "question": "any risk?"}).status_code == 404
        assert c.post("/search", json={"ticker": "DAL", "question": "x"}).status_code == 422
        assert (
            c.post("/search", json={"ticker": "DAL", "question": "ok?", "k": 99}).status_code == 422
        )


def test_answer_cites_meters_and_prices():
    c, _, llm = client()
    with c:
        r = c.post("/answer", json={"ticker": "DAL", "question": "fuel risk?"},
                   headers={"X-User": "analyst-a"})  # fmt: skip
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Fuel is volatile [1]." and body["citations"][0]["n"] == 1
    assert llm.calls[0]["route"] == "answer" and llm.calls[0]["user"] == "analyst-a"
    assert llm.calls[0]["max_tokens"] == 300
    assert body["cost"]["actual_spend_usd"] == 0
    assert body["cost"]["list_price_usd"] == pytest.approx((400 * 0.25 + 20 * 1.50) / 1e6)


def test_answer_maps_quota_to_429_and_keeps_the_hits():
    failed = Reply("gemini", "m", "", 0, 0, 0, 0.0, None, "quota: 500 / 500 requests already sent")
    c, _, _ = client(failed)
    with c:
        r = c.post("/answer", json={"ticker": "DAL", "question": "fuel risk?"})
    assert r.status_code == 429
    assert len(r.json()["detail"]["hits"]) == 5


def test_answer_maps_other_failures_to_502():
    failed = Reply("gemini", "m", "", 0, 0, 0, 2.0, None, "HTTP 503: high demand")
    c, _, _ = client(failed)
    with c:
        assert (
            c.post("/answer", json={"ticker": "DAL", "question": "fuel risk?"}).status_code == 502
        )


def test_costs():
    c, _, _ = client()
    with c:
        body = c.get("/costs").json()
    assert body["actual_spend_usd"] == 0 and body["list_price_usd_total"] == pytest.approx(1e-5)


def test_fake_llm_is_async():  # guards the fake against drifting from MeteredBackend.chat
    llm = FakeLLM(Reply("gemini", "m", "x", 1, 1, 0, 0.1, "STOP"))
    assert asyncio.run(llm.chat([], route="r", user="u")).text == "x"
