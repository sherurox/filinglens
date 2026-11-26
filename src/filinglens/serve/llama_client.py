r"""Run the extraction prompts through llama-server (llama.cpp, CPU), like extract.fields.

Start the server first (its own terminal; FL-M5 T4):
    ~/src/llama.cpp/build/bin/llama-server -m data/interim/serve/gguf/student-q4_k_m.gguf \
        --host 127.0.0.1 --port 58081 -c 4608 -np 1 -t 12 -tb 24 --jinja --seed 0
Then (fl-train venv: it has transformers for the HF side of `template`):
    ~/venvs/fl-train/bin/python -m filinglens.serve.llama_client template --split dev
    ~/venvs/fl-train/bin/python -m filinglens.serve.llama_client generate --split dev \
        --tag gguf-q4_k_m
    uv run python -m filinglens.extract.fields score --variant zero --split dev --tag gguf-q4_k_m

`template`: for every conversation, the server's rendering (/apply-template, thinking off) and its
token IDs (/tokenize) against the HF tokenizer of the merged model (serve.template_check). llama.cpp
renders the chat template with its own Jinja engine, so this is the check that matters most here.
`generate`: the same conversations, temperature 0, 400 tokens, the same JSON schema (llama-server
turns it into a grammar), thinking off; one request at a time. Output records use the same parser
and file names as `extract.fields`, so `fields score` and `serve.parity` work unchanged. Per-request
server timings are summed into the run file; serving speed itself is measured in T7, not here.
HTTP is standard-library only, so the client runs in any venv.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

URL = "http://127.0.0.1:58081"
TOKENIZER = Path("data/interim/serve/merged-bf16")
MAX_TOKENS = 400  # as extract.fields.generate
NO_THINKING = {"enable_thinking": False}


def chat_request(conv: list[dict], schema: dict, max_tokens: int = MAX_TOKENS) -> dict:
    """An OpenAI-style chat request with the eval's decoding settings."""
    return {
        "messages": conv,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": 0,
        "response_format": {"type": "json_schema", "json_schema": {"schema": schema}},
        "chat_template_kwargs": NO_THINKING,
    }


def content_of(response: dict) -> str:
    """The assistant text of a chat completion, or '' if the response has none."""
    choices = response.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message") or {}).get("content") or ""


def sum_timings(timings: list[dict]) -> dict:
    """Totals of llama-server's per-request `timings` (prompt and generated tokens and ms)."""
    keys = ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms")
    total = {k: sum(float(t.get(k) or 0) for t in timings) for k in keys}
    return {
        "requests_with_timings": len(timings),
        "prompt_tokens": int(total["prompt_n"]),
        "generated_tokens": int(total["predicted_n"]),
        "prompt_tok_per_s": round(1000 * total["prompt_n"] / total["prompt_ms"], 2)
        if total["prompt_ms"]
        else None,
        "generate_tok_per_s": round(1000 * total["predicted_n"] / total["predicted_ms"], 2)
        if total["predicted_ms"]
        else None,
    }


def post(url: str, path: str, body: dict, timeout: float = 600) -> dict:
    request = urllib.request.Request(
        url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def get(url: str, path: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url + path, timeout=timeout) as response:
        return json.loads(response.read())


def wait_healthy(url: str, seconds: float = 120) -> None:
    deadline = time.monotonic() + seconds
    while True:
        try:
            if get(url, "/health").get("status") == "ok":
                return
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            pass
        if time.monotonic() > deadline:
            raise TimeoutError(f"llama-server at {url} not healthy after {seconds:.0f} s")
        time.sleep(2)


def server_props(url: str) -> dict:
    """What the server says it is running (model file, context, build), for the run record."""
    try:
        props = get(url, "/props")
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
        return {"error": str(error)}
    settings = props.get("default_generation_settings") or {}
    return {
        "model_path": props.get("model_path"),
        "build_info": props.get("build_info"),
        "n_ctx": settings.get("n_ctx", props.get("n_ctx")),
        "total_slots": props.get("total_slots"),
    }


def run_template(args: argparse.Namespace) -> dict:
    from filinglens.serve.template_check import compare_prompts, conversations_for, render

    ids, convs = conversations_for(args.split, args.allow_test)
    server = []
    for conv in convs:
        body = {"messages": conv, "chat_template_kwargs": NO_THINKING}
        prompt = post(args.url, "/apply-template", body)["prompt"]
        tokens = post(
            args.url, "/tokenize", {"content": prompt, "add_special": False, "parse_special": True}
        )["tokens"]
        server.append({"text": prompt, "ids": tokens})
    return {
        "a": f"HF tokenizer {args.tokenizer.as_posix()}",
        "b": f"llama-server {args.url} (/apply-template + /tokenize)",
        "server": server_props(args.url),
        **compare_prompts(ids, render(args.tokenizer, convs), server),
    }


def run_generate(args: argparse.Namespace) -> dict:
    from filinglens.extract.fields import (
        OUT_DIR,
        PROMPT,
        TASKS,
        conversations,
        load_rows,
        task_view,
    )
    from filinglens.extract.silver_label import parse, schema

    rows = load_rows(args.split, args.allow_test)
    tasks = {t["para_id"]: t for t in json.loads(TASKS.read_text())}
    convs = conversations(rows, tasks, PROMPT.read_text(), [])
    raws, timings, errors = [], [], []
    started = time.perf_counter()
    for row, conv in zip(rows, convs, strict=True):
        try:
            response = post(args.url, "/v1/chat/completions", chat_request(conv, schema()))
        except (urllib.error.URLError, OSError) as error:  # recorded; the row scores as invalid
            errors.append({"para_id": row["para_id"], "error": str(error)[:200]})
            raws.append("")
            continue
        raws.append(content_of(response))
        if response.get("timings"):
            timings.append(response["timings"])
    elapsed = time.perf_counter() - started
    records = [parse(raw, task_view(row, tasks)) for raw, row in zip(raws, rows, strict=True)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"fields_zero_{args.split}_{args.tag}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    return {
        "server": server_props(args.url),
        "variant": "zero",
        "rows": len(rows),
        "valid": sum(r["valid"] for r in records),
        "request_errors": errors,
        "elapsed_s": round(elapsed, 1),
        "timings": sum_timings(timings),
        "predictions": str(out),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["template", "generate"])
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--url", default=URL)
    parser.add_argument("--tag", default="gguf-q4_k_m", help="output suffix, as fields --tag")
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--bench", type=Path, default=Path("bench"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    wait_healthy(args.url)
    body = run_template(args) if args.command == "template" else run_generate(args)
    result = {
        "command": " ".join(["python -m filinglens.serve.llama_client", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": args.split,
        **body,
    }
    if args.command == "template":
        out = args.bench / f"template_{args.tag}-vs-hf.json"
    else:
        out = args.bench / f"fields_zero_{args.split}_{args.tag}_run.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "ids_differ"}, indent=2))
    return 0 if result.get("ok", True) and not result.get("request_errors") else 1


if __name__ == "__main__":
    sys.exit(main())
