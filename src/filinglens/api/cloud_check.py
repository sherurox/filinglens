"""FL-M5 T13: check the deployed API from outside: recall through /search, latency, /answer, /costs.

    uv run --group api python -m filinglens.api.cloud_check --url "$URL" --project "$GCP_PROJECT"

Sends the 182 ret-dev questions to POST /search on the public URL and scores recall@10 exactly
as `demo_db check` does (a hit = any graded chunk in the top 10). The same image and the same
rows answer, so the expected value is the int8 encoder's recall from bench/demo_db_check_neon.json;
a gap of at most one query is allowed, because ONNX Runtime may pick different int8 kernels on
Cloud Run's CPU than on the Ryzen. Latency is client wall time from this machine to us-east1, so
it includes the network and is an upper bound on service time.

The first request is timed on its own. Whether it was a cold start is not guessed: with
--project, the service's startup log lines ("Starting new instance", uvicorn's "Application
startup complete") from around that request are read back and recorded.

Then ANSWER_QUESTIONS calls to /answer (one per company, Gemini free tier: each is one request of
the daily quota) and /costs, which must show those calls. Writes bench/cloud_run_check.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from filinglens.api.demo_db import ONE_QUERY, dev_queries, recall_hit

ANSWER_QUESTIONS = 3
CHECK_USER = "cloud-check"
EXPECTED_FROM = [Path("bench/demo_db_check_neon.json"), Path("bench/demo_db_check_local.json")]
LOG_FILTER = (
    'resource.type="cloud_run_revision" AND resource.labels.service_name="{service}" AND '
    '(textPayload:"Starting new instance" OR textPayload:"Application startup complete")'
)


def timed(client, method: str, path: str, **kwargs):
    started = time.perf_counter()
    response = client.request(method, path, **kwargs)
    return time.perf_counter() - started, response


def latency_ms(seconds: list[float]) -> dict:
    if not seconds:
        return {"n": 0}
    a = np.asarray(seconds) * 1000
    return {
        "n": len(seconds),
        "p50": round(float(np.percentile(a, 50)), 1),
        "p95": round(float(np.percentile(a, 95)), 1),
        "max": round(float(a.max()), 1),
    }


def expected_recall() -> tuple[float, str]:
    for path in EXPECTED_FROM:
        if path.exists():
            body = json.loads(path.read_text())
            return body["results"]["int8_onnx_encoder"]["recall_at_10"], str(path)
    raise SystemExit("run `demo_db check` first: no bench/demo_db_check_*.json to compare with")


def answer_record(query: dict, seconds: float, response) -> dict:
    record = {"query_id": query["query_id"], "ticker": query["ticker"],
              "question": query["question"], "status": response.status_code,
              "client_s": round(seconds, 2)}  # fmt: skip
    body = response.json()
    if response.status_code != 200:
        return {**record, "error": str(body.get("detail"))[:200]}
    return {
        **record,
        "model": body["model"],
        "model_latency_s": body["latency_s"],
        "tokens": body["tokens"],
        "list_price_usd": body["cost"]["list_price_usd"],
        "citations": [c["n"] for c in body["citations"]],
        "answer": body["answer"],
    }


def run(client, queries: list[dict], expected: float) -> dict:
    """Everything that talks to the service; `client` is an httpx.Client on the base URL."""
    first_utc = datetime.now(UTC)
    first_s, r = timed(client, "GET", "/health")
    r.raise_for_status()
    health = r.json()
    _, r = timed(client, "GET", "/companies")
    r.raise_for_status()
    companies = r.json()["count"]

    hits, seconds, errors = 0, [], []
    for q in queries:
        body = {"ticker": q["ticker"], "question": q["question"], "k": 10}
        s, r = timed(client, "POST", "/search", json=body)
        if r.status_code != 200:
            errors.append(f"{q['query_id']}: HTTP {r.status_code}")
            continue  # counts as a miss: recall is over all queries
        seconds.append(s)
        hits += recall_hit([h["chunk_id"] for h in r.json()["hits"]], q["grades"])
    recall = round(hits / len(queries), 4)

    answers, tickers = [], set()
    for q in queries:
        if len(answers) == ANSWER_QUESTIONS:
            break
        if q["ticker"] in tickers:
            continue
        tickers.add(q["ticker"])
        body = {"ticker": q["ticker"], "question": q["question"]}
        s, r = timed(client, "POST", "/answer", json=body, headers={"X-User": CHECK_USER})
        answers.append(answer_record(q, s, r))

    _, r = timed(client, "GET", "/costs")
    r.raise_for_status()
    costs = r.json()
    return {
        "first_request": {
            "utc": first_utc.isoformat(timespec="seconds"),
            "path": "/health",
            "client_s": round(first_s, 2),
            "body": health,
        },  # fmt: skip
        "companies": companies,
        "search": {
            "queries": len(queries),
            "recall_at_10": recall,
            "expected_int8_recall_at_10": expected,
            "matches_expected": abs(recall - expected) <= ONE_QUERY + 1e-9,
            "errors": errors,
            "client_ms": latency_ms(seconds),
        },
        "answers": answers,
        "answers_ok": sum(a["status"] == 200 for a in answers),
        "costs": {
            "list_price_usd_total": costs["list_price_usd_total"],
            "actual_spend_usd": costs["actual_spend_usd"],
            "by_route_backend": costs["by_route_backend"],
        },
    }


def startup_logs(project: str, service: str, since: datetime) -> list[str] | None:
    """The service's startup lines since `since` (None if gcloud can't be run)."""
    cmd = ["gcloud", "logging", "read", LOG_FILTER.format(service=service),
           f"--project={project}", "--freshness=2h", "--limit=50",
           "--format=json(timestamp,textPayload)"]  # fmt: skip
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    cutoff = since.strftime("%Y-%m-%dT%H:%M:%S")  # gcloud timestamps are UTC ISO, so text compares
    return sorted(
        f"{e['timestamp']} {e['textPayload'].strip()[:90]}"
        for e in json.loads(out or "[]")
        if e["timestamp"] >= cutoff
    )


def main(argv: list[str] | None = None) -> int:
    import httpx

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="the service's https URL")
    parser.add_argument("--project", help="GCP project, to read startup log lines back")
    parser.add_argument("--service", default="filinglens-api")
    parser.add_argument("--out", type=Path, default=Path("bench/cloud_run_check.json"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    url = args.url.rstrip("/")
    if not url.startswith("https://"):
        raise SystemExit("--url must be the service's https URL")
    expected, expected_from = expected_recall()

    with httpx.Client(base_url=url, timeout=120) as client:
        body = run(client, dev_queries(), expected)
    first = datetime.fromisoformat(body["first_request"]["utc"])
    # read after the whole run (minutes later), so the log entries have arrived
    logs = (
        startup_logs(args.project, args.service, first - timedelta(seconds=5))
        if args.project
        else None
    )
    body["first_request"]["startup_log_lines"] = logs if args.project else "not read (no --project)"
    result = {
        "command": " ".join(["python -m filinglens.api.cloud_check", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "url": url,
        "expected_from": expected_from,
        "latency_note": "client wall time from the caller to us-east1, network included",
        **body,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    summary = {k: v for k, v in result.items() if k not in ("answers", "costs")}
    keys = ("ticker", "status", "client_s", "model_latency_s", "list_price_usd", "citations")
    summary["answers"] = [{k: a.get(k) for k in keys} for a in result["answers"]]
    summary["costs_list_price_usd_total"] = result["costs"]["list_price_usd_total"]
    print(json.dumps(summary, indent=2))
    ok = result["search"]["matches_expected"] and result["answers_ok"] == len(result["answers"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
