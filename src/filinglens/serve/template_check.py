"""Do two tokenizers turn the same conversations into the same prompt? (FL-M5 T2; reused in T4.)

    ~/venvs/fl-train/bin/python -m filinglens.serve.template_check --name adapter-vs-base \
        --a data/interim/serve/student-v1/adapter --b data/interim/serve/merged-bf16

Builds the exact `extract.fields` conversations (zero-shot variant, dev split by default), renders
each with both tokenizers' chat templates (generation prompt on, thinking off, as `fields.generate`
asks vLLM to), tokenizes the text without adding special tokens again, and compares text and IDs.
Byte-comparing tokenizer files can't answer this: the same tokenizer saved by two transformers
versions gives different bytes. Writes bench/template_<name>.json; exit 1 if any prompt differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def first_diff(a: list | str, b: list | str) -> int | None:
    """First index where two sequences differ (or the shorter length); None if equal."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare_prompts(ids: list[str], a: list[dict], b: list[dict]) -> dict:
    """a, b: per conversation {'text': str, 'ids': list[int]}, in the same order as ids."""
    if not (len(ids) == len(a) == len(b)):
        raise ValueError("ids, a and b must have the same length")
    text_diff = [i for i, x, y in zip(ids, a, b, strict=True) if x["text"] != y["text"]]
    ids_diff = [i for i, x, y in zip(ids, a, b, strict=True) if x["ids"] != y["ids"]]
    example = None
    if ids_diff:
        k = ids.index(ids_diff[0])
        at = first_diff(a[k]["ids"], b[k]["ids"])
        example = {
            "para_id": ids_diff[0],
            "first_diff_token": at,
            "a_ids": a[k]["ids"][max(0, at - 3) : at + 5],
            "b_ids": b[k]["ids"][max(0, at - 3) : at + 5],
        }
    return {
        "conversations": len(ids),
        "text_identical": len(ids) - len(text_diff),
        "ids_identical": len(ids) - len(ids_diff),
        "ids_differ": ids_diff,
        "first_difference": example,
        "prompt_tokens_min_max": [
            min(len(x["ids"]) for x in a),
            max(len(x["ids"]) for x in a),
        ],
        "ok": not ids_diff,
    }


def render(tokenizer_dir: Path, convs: list[list[dict]]) -> list[dict]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    out = []
    for conv in convs:
        text = tok.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        out.append({"text": text, "ids": tok(text, add_special_tokens=False)["input_ids"]})
    return out


def conversations_for(split: str, allow_test: bool) -> tuple[list[str], list[list[dict]]]:
    from filinglens.extract.fields import PROMPT, TASKS, conversations, load_rows

    rows = load_rows(split, allow_test)
    tasks = {t["para_id"]: t for t in json.loads(TASKS.read_text())}
    return [r["para_id"] for r in rows], conversations(rows, tasks, PROMPT.read_text(), [])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="output: bench/template_<name>.json")
    parser.add_argument("--a", type=Path, required=True, help="tokenizer folder A")
    parser.add_argument("--b", type=Path, required=True, help="tokenizer folder B")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--bench", type=Path, default=Path("bench"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    ids, convs = conversations_for(args.split, args.allow_test)
    result = {
        "command": " ".join(["python -m filinglens.serve.template_check", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "a": args.a.as_posix(),
        "b": args.b.as_posix(),
        "split": args.split,
        "render": "apply_chat_template(add_generation_prompt=True, enable_thinking=False)",
        **compare_prompts(ids, render(args.a, convs), render(args.b, convs)),
    }
    (args.bench / f"template_{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
