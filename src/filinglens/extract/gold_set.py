"""Write the silver-labeled risk-paragraph set with its splits, and guard the test split (FL-M2 T6).

The labels are silver: they come from a model (see extract/silver_label.py), not from a person.
Each row keeps the primary labeler's answer as the label and the second labeler's answer next to
it, with an agreement flag, so results can be reported on all rows and on the rows where the two
models agree. Rows are dropped, and counted, when the primary answer broke the labeling rules or
said the paragraph isn't a risk.

Outputs (committed):
- gold/risk_paragraphs.{train,dev,test}.jsonl
- gold/silver_raw/<model>.jsonl: both labelers' raw answers, rationale included
- gold/MANIFEST.json: provenance, counts and sha256 of every file

Reading the test split needs `load_split("test", allow_test=True)`: prompt work and training must
never touch it by accident (M2 baseline prompts are tuned on dev; M4 trains on train).
"""

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from filinglens.extract.silver_label import slug

GOLD = Path("gold")
SPLITS = ("train", "dev", "test")
PRIMARY = "ibm-granite/granite-4.1-3b"
SECONDARY = "Qwen/Qwen3-4B-AWQ"
SILVER_DIR = Path("data/interim/silver")
TASKS = Path("data/interim/label_tasks/risk_tasks.json")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows(
    sample: pd.DataFrame, tasks: list[dict], primary: list[dict], secondary: list[dict]
) -> tuple[list[dict], Counter]:
    """Silver rows (one per kept paragraph) and the reasons rows were dropped."""
    by_id = {t["para_id"]: t for t in tasks}
    second = {r["para_id"]: r for r in secondary}
    dropped: Counter = Counter()
    rows = []
    for answer in primary:
        meta = sample.loc[answer["para_id"]]
        task = by_id[answer["para_id"]]
        if hashlib.sha256(task["text"].encode()).hexdigest() != meta["text_sha256"]:
            raise ValueError(f"text changed since the sample was drawn: {answer['para_id']}")
        if not answer["valid"]:
            dropped["primary answer broke the rules"] += 1
            continue
        if answer["category"] == "not_a_risk":
            dropped["primary said not_a_risk"] += 1
            continue
        check = second.get(answer["para_id"], {})
        check_ok = bool(check.get("valid"))
        rows.append(
            {
                "para_id": answer["para_id"],
                "task_no": int(meta["task_no"]),
                "cik": int(meta["cik"]),
                "ticker": meta["ticker"],
                "sector": meta["sector"],
                "period_year": int(meta["period_year"]),
                "split": meta["split"],
                "sampled_as": meta["source"],
                "heading": task["heading"],
                "text": task["text"],
                "text_sha256": meta["text_sha256"],
                "label_source": f"silver:{PRIMARY}",
                "category": answer["category"],
                "severity": answer["severity"],
                "first_seen_year": answer["first_seen_year"],
                "check_model": SECONDARY,
                "check_category": check.get("category") if check_ok else None,
                "check_severity": check.get("severity") if check_ok else None,
                "check_first_seen_year": check.get("first_seen_year") if check_ok else None,
                "models_agree_category": check_ok and check["category"] == answer["category"],
            }
        )
    return rows, dropped


def write_split_files(rows: list[dict], gold: Path) -> dict[str, Path]:
    paths = {}
    for split in SPLITS:
        path = gold / f"risk_paragraphs.{split}.jsonl"
        part = sorted((r for r in rows if r["split"] == split), key=lambda r: r["task_no"])
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part))
        paths[split] = path
    return paths


def load_split(split: str, gold: Path = GOLD, allow_test: bool = False) -> list[dict]:
    """Rows of one split. The test split is only for final, one-off evaluation runs."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if split == "test" and not allow_test:
        raise PermissionError("the test split is reserved for final evaluation: allow_test=True")
    return read_jsonl(gold / f"risk_paragraphs.{split}.jsonl")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--tasks", type=Path, default=TASKS)
    parser.add_argument("--silver-dir", type=Path, default=SILVER_DIR)
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    sample = pd.read_csv(args.gold / "risk_sample.csv").set_index("para_id")
    tasks = json.loads(args.tasks.read_text())
    raw = {m: args.silver_dir / f"{slug(m)}.jsonl" for m in (PRIMARY, SECONDARY)}
    rows, dropped = build_rows(sample, tasks, read_jsonl(raw[PRIMARY]), read_jsonl(raw[SECONDARY]))

    raw_dir = args.gold / "silver_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_copies = [Path(shutil.copy(p, raw_dir / p.name)) for p in raw.values()]
    split_files = write_split_files(rows, args.gold)
    inputs = [args.gold / "splits.csv", args.gold / "risk_sample.csv"]
    files = [*inputs, *split_files.values(), *raw_copies]
    manifest = {
        "command": " ".join(["uv run python -m filinglens.extract.gold_set", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "labels": "silver (model-labeled, not hand-labeled)",
        "primary_labeler": PRIMARY,
        "check_labeler": SECONDARY,
        "rules": "docs/labeling/silver_prompt.md (condensed docs/LABELING_GUIDE.md v1.0)",
        "sampled": len(sample),
        "kept": len(rows),
        "dropped": dict(dropped),
        "by_split": dict(sorted(Counter(r["split"] for r in rows).items())),
        "agree_category_by_split": {
            s: sum(r["models_agree_category"] for r in rows if r["split"] == s) for s in SPLITS
        },
        "categories": dict(Counter(r["category"] for r in rows).most_common()),
        "files": {str(p): sha256_file(p) for p in files},
    }
    (args.gold / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
