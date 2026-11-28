"""Reranker sidecar: request checks, a real local HTTP round trip, the device rule."""

import asyncio
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from filinglens.agents.rerank import MAX_TEXTS, decide, make_handler, validate

RULE = {"fl_m3_ndcg_after": 0.488, "cuda_if": {"max_abs_ndcg_diff": 0.002, "max_p95_ms": 1000}}


def test_validate_accepts_good_bodies_and_rejects_bad_ones():
    assert validate({"question": "q", "texts": ["a", "b"]}) == ("q", ["a", "b"])
    for bad in [[], {"question": "", "texts": ["a"]}, {"question": "q", "texts": []},
                {"question": "q", "texts": ["a"] * (MAX_TEXTS + 1)},
                {"question": "q", "texts": [1]}, {"question": "x" * 2001, "texts": ["a"]}]:  # fmt: skip
        with pytest.raises(ValueError):
            validate(bad)


def result(device="cuda", errors=0, ndcg=0.4881, p95=300.0) -> dict:
    return {"device": device, "errors": errors, "metrics": {"ndcg_after": ndcg},
            "latency_ms": {"p95_ms": p95}}  # fmt: skip


def test_decide_follows_the_rule():
    assert decide(RULE, result())["device"] == "cuda"
    for bad in [result(device="cpu"), result(errors=1), result(ndcg=0.485), result(p95=1200.0)]:
        out = decide(RULE, bad)
        assert out["device"] == "cpu" and not all(out["checks"].values())


def test_committed_rule_has_what_decide_reads():
    rule = json.loads(Path("bench/m6_rerank_rule.json").read_text())
    assert decide(rule, result(ndcg=rule["fl_m3_ndcg_after"]))["device"] == "cuda"


def test_server_and_client_round_trip():
    httpx = pytest.importorskip("httpx")
    from filinglens.agents.rerank import HttpReranker, RerankError

    def score(question, texts):
        return [len(t) for t in texts]

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(score, {"device": "cpu"}))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    async def run():
        client = HttpReranker(url, client=httpx.AsyncClient())
        try:
            assert (await client.health())["device"] == "cpu"
            assert await client("q", ["a", "abc"]) == [1.0, 3.0]
            with pytest.raises(RerankError, match="HTTP 400"):
                await client("q", [])
        finally:
            await client.aclose()

    try:
        asyncio.run(run())
    finally:
        server.shutdown()
        server.server_close()
