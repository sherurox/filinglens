"""SFT examples for the FL-M4 student: silver only, teacher only, and both (FL-M4 T7).

Every example is a prompt-completion pair in chat form (TRL's conversational prompt-completion
format). The prompt is built by `fields.conversations(..., shots=[])`, the exact zero-shot prompt
the FL-M2 baseline was scored with, so the student is trained on the same input view it is
evaluated on. The completion is the full schema answer (rationale, category, severity,
first_seen_year, unsure) as compact JSON in schema key order: the eval schema forces the same
fields at inference.

- silver: the 191 train rows of gold/risk_paragraphs.train.jsonl; labels from the silver row,
  rationale and `unsure` from the primary labeler's raw answer (gold/silver_raw/).
- teacher: rows kept by the teacher filter (T6): {"para_id", "task", "answer", ...} per line.
  not_a_risk answers are dropped by default, so both arms share the label space of the eval set
  (gold_set dropped not_a_risk rows too).
- both: silver + teacher; a para_id in both keeps the silver row.

Guard: every example must come from a train-split company (gold/splits.csv), or nothing is written.
Outputs: data/interim/sft/{silver,teacher,both}.jsonl + MANIFEST.json (its sha256 is the data
version logged with every training run) and bench/sft_data.json.

    uv run python -m filinglens.train.sft_data [--teacher data/interim/teacher/filtered.jsonl]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from filinglens.extract.fields import conversations
from filinglens.extract.gold_set import load_split
from filinglens.extract.silver_label import check, messages

GOLD = Path("gold")
TASKS = Path("data/interim/label_tasks/risk_tasks.json")
PROMPT = Path("docs/labeling/silver_prompt.md")
SILVER_RAW = Path("gold/silver_raw/ibm-granite-granite-4-1-3b.jsonl")
OUT = Path("data/interim/sft")
ANSWER_KEYS = ("rationale", "category", "severity", "first_seen_year", "unsure")
ARMS = ("silver", "teacher", "both")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completion(answer: dict) -> list[dict]:
    """The assistant turn: the answer's schema fields, in schema order, as compact JSON."""
    body = {k: answer[k] for k in ANSWER_KEYS}
    content = json.dumps(body, ensure_ascii=False, separators=(",", ":"))  # as vLLM's JSON output
    return [{"role": "assistant", "content": content}]


def silver_examples(
    rows: list[dict], tasks: dict[str, dict], raw: dict[str, dict], system_prompt: str
) -> list[dict]:
    """One example per silver train row. Labels come from the row; the raw answer must agree."""
    prompts = conversations(rows, tasks, system_prompt, shots=[])
    examples = []
    for row, prompt in zip(rows, prompts, strict=True):
        answer_raw = raw[row["para_id"]]
        for key in ("category", "severity", "first_seen_year"):
            if answer_raw[key] != row[key]:
                raise ValueError(f"{row['para_id']}: silver row and raw answer differ on {key}")
        answer = {k: row[k] for k in ("category", "severity", "first_seen_year")}
        answer |= {"rationale": answer_raw["rationale"], "unsure": bool(answer_raw["unsure"])}
        examples.append(
            {
                "id": row["para_id"],
                "source": "silver",
                "cik": int(row["cik"]),
                "category": answer["category"],
                "prompt": prompt,
                "completion": completion(answer),
            }
        )
    return examples


def teacher_examples(
    rows: list[dict], system_prompt: str, drop_not_a_risk: bool = True
) -> tuple[list[dict], Counter]:
    """One example per filtered teacher row; rows that break the labeling rules raise."""
    examples, dropped = [], Counter()
    for row in rows:
        task, answer = row["task"], row["answer"]
        problems = check(answer, int(task["period_year"]))
        if problems:
            raise ValueError(f"{row['para_id']}: teacher answer breaks the rules: {problems}")
        if drop_not_a_risk and answer["category"] == "not_a_risk":
            dropped["not_a_risk"] += 1
            continue
        examples.append(
            {
                "id": row["para_id"],
                "source": "teacher",
                "cik": int(task["cik"]),
                "category": answer["category"],
                "prompt": messages(task, system_prompt),
                "completion": completion(answer),
            }
        )
    return examples, dropped


def combine(silver: list[dict], teacher: list[dict]) -> tuple[list[dict], int]:
    """Silver + teacher; a para_id present in both keeps the silver example."""
    ids = {e["id"] for e in silver}
    extra = [e for e in teacher if e["id"] not in ids]
    return silver + extra, len(teacher) - len(extra)


def outside_train(examples: list[dict], train_ciks: set[int]) -> int:
    return sum(e["cik"] not in train_ciks for e in examples)


def describe(examples: list[dict]) -> dict:
    lengths = sorted(len(e["completion"][0]["content"]) for e in examples)
    return {
        "examples": len(examples),
        "by_source": dict(sorted(Counter(e["source"] for e in examples).items())),
        "categories": dict(Counter(e["category"] for e in examples).most_common()),
        "completion_chars_median": lengths[len(lengths) // 2] if lengths else None,
        "completion_chars_max": lengths[-1] if lengths else None,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--tasks", type=Path, default=TASKS)
    parser.add_argument("--prompt", type=Path, default=PROMPT)
    parser.add_argument("--silver-raw", type=Path, default=SILVER_RAW)
    parser.add_argument("--teacher", type=Path, default=None, help="filtered teacher rows (T6)")
    parser.add_argument("--keep-not-a-risk", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--stats", type=Path, default=Path("bench/sft_data.json"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    system_prompt = args.prompt.read_text()
    tasks = {t["para_id"]: t for t in json.loads(args.tasks.read_text())}
    raw = {r["para_id"]: r for r in read_jsonl(args.silver_raw)}
    splits = pd.read_csv(args.gold / "splits.csv")
    train_ciks = set(splits.loc[splits["split"] == "train", "cik"].astype(int))

    arms = {
        "silver": silver_examples(load_split("train", gold=args.gold), tasks, raw, system_prompt)
    }
    teacher_stats: dict = {"file": None}
    if args.teacher is not None:
        teacher, dropped = teacher_examples(
            read_jsonl(args.teacher), system_prompt, drop_not_a_risk=not args.keep_not_a_risk
        )
        arms["teacher"] = teacher
        arms["both"], overlap = combine(arms["silver"], teacher)
        teacher_stats = {
            "file": args.teacher.as_posix(),
            "sha256": sha256_file(args.teacher),
            "dropped": dict(dropped),
            "overlap_with_silver_dropped": overlap,
        }

    leaks = {arm: outside_train(examples, train_ciks) for arm, examples in arms.items()}
    if any(leaks.values()):
        raise SystemExit(f"examples from non-train companies, nothing written: {leaks}")

    args.out.mkdir(parents=True, exist_ok=True)
    for name in [*(f"{a}.jsonl" for a in ARMS), "MANIFEST.json"]:
        (args.out / name).unlink(missing_ok=True)  # no stale arm from an earlier run
    files = []
    for arm, examples in arms.items():
        path = args.out / f"{arm}.jsonl"
        path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples))
        files.append(path)

    inputs = [args.gold / "MANIFEST.json", args.prompt, args.silver_raw]
    summary = {
        "command": " ".join(["uv run python -m filinglens.train.sft_data", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "prompt": "fields.conversations zero-shot (silver_prompt.md + user_message)",
        "completion": f"compact JSON, keys {list(ANSWER_KEYS)}",
        "drop_not_a_risk": not args.keep_not_a_risk,
        "teacher": teacher_stats,
        "examples_from_non_train_companies": leaks,
        "arms": {arm: describe(examples) for arm, examples in arms.items()},
        "inputs": {p.as_posix(): sha256_file(p) for p in inputs if p.exists()},
    }
    manifest = summary | {"files": {p.name: sha256_file(p) for p in files}}
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stats = manifest | {"data_version": sha256_file(args.out / "MANIFEST.json")}
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k not in ("files", "inputs")}, indent=2))


if __name__ == "__main__":
    main()
