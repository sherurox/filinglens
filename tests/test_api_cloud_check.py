"""The outside check's scoring, against a fake service (httpx MockTransport, no network)."""

import json

import pytest

httpx = pytest.importorskip("httpx")

from filinglens.api.cloud_check import ANSWER_QUESTIONS, CHECK_USER, latency_ms, run  # noqa: E402


def queries(n: int) -> list[dict]:
    tickers = ["AEP", "AEP", "DAL", "KO", "MSFT", "MSFT"]
    return [{"query_id": f"q{i}", "ticker": tickers[i % len(tickers)], "question": f"question {i}",
             "grades": {f"good{i}": 1}} for i in range(n)]  # fmt: skip


def fake_service(answer_status: int = 200):
    seen = {"answers": [], "users": []}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok", "database": True, "spend_usd": 0})
        if path == "/companies":
            return httpx.Response(200, json={"companies": [], "count": 77})
        body = json.loads(request.content) if request.content else {}
        if path == "/search":
            i = int(body["question"].split()[-1])
            if i == 5:
                return httpx.Response(503, json={"detail": "busy"})
            top = [f"good{i}" if i % 2 == 0 else "other"] + [f"x{j}" for j in range(9)]
            return httpx.Response(200, json={"hits": [{"chunk_id": c} for c in top]})
        if path == "/answer":
            seen["answers"].append(body["ticker"])
            seen["users"].append(request.headers.get("x-user"))
            if answer_status != 200:
                return httpx.Response(answer_status, json={"detail": {"error": "quota: rpd"}})
            return httpx.Response(200, json={
                "model": "gemini", "latency_s": 1.5, "tokens": {"prompt": 900, "output": 40},
                "cost": {"list_price_usd": 0.0003, "basis": "x", "actual_spend_usd": 0},
                "citations": [{"n": 1}, {"n": 3}], "answer": "Because [1] and [3]."})  # fmt: skip
        if path == "/costs":
            return httpx.Response(200, json={"by_route_backend": [], "list_price_usd_total": 0.0009,
                                             "actual_spend_usd": 0})  # fmt: skip
        return httpx.Response(404)

    client = httpx.Client(base_url="https://svc.example", transport=httpx.MockTransport(handler))
    return client, seen


def test_recall_counts_failed_searches_as_misses():
    client, _ = fake_service()
    out = run(client, queries(10), expected=0.5)
    # even i hit: 0,2,4,6,8 = 5 of 10; i=5 failed (a miss either way), still over all 10
    assert out["search"]["recall_at_10"] == 0.5
    assert out["search"]["hits"] == 5 and out["search"]["expected_hits"] == 5
    assert out["search"]["matches_expected"] is True
    assert out["search"]["errors"] == ["q5: HTTP 503"]
    assert out["search"]["client_ms"]["n"] == 9
    assert out["companies"] == 77 and out["first_request"]["body"]["database"] is True


def test_recall_outside_one_query_is_flagged():
    client, _ = fake_service()
    assert run(client, queries(10), expected=0.6)["search"]["matches_expected"] is True  # 5 vs 6
    assert run(client, queries(10), expected=0.7)["search"]["matches_expected"] is False  # 5 vs 7


def test_answers_one_per_company_and_tagged():
    client, seen = fake_service()
    out = run(client, queries(10), expected=0.5)
    assert seen["answers"] == ["AEP", "DAL", "KO"][:ANSWER_QUESTIONS]
    assert set(seen["users"]) == {CHECK_USER}
    assert out["answers_ok"] == ANSWER_QUESTIONS
    assert out["answers"][0]["citations"] == [1, 3]
    assert out["costs"]["actual_spend_usd"] == 0


def test_a_refused_answer_is_recorded_not_raised():
    client, _ = fake_service(answer_status=429)
    out = run(client, queries(10), expected=0.5)
    assert out["answers_ok"] == 0
    assert out["answers"][0]["status"] == 429 and "quota" in out["answers"][0]["error"]


def test_latency_summary():
    assert latency_ms([]) == {"n": 0}
    assert latency_ms([0.1, 0.2, 0.3]) == {"n": 3, "p50": 200.0, "p95": 290.0, "max": 300.0}
