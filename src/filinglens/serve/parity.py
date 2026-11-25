"""Do two serving builds give the same answers on the same prompts? (FL-M5 T2, reused in T6.)

    uv run python -m filinglens.serve.parity --name merged-vs-lora \
        --a data/interim/fields/fields_zero_dev_lora-v034.jsonl \
        --b data/interim/fields/fields_zero_dev_merged-bf16.jsonl \
        --scores-a bench/fields_zero_dev_lora-v034.json \
        --scores-b bench/fields_zero_dev_merged-bf16.json

Both files are `extract.fields generate` outputs (one record per paragraph, with the raw text).
Writes bench/parity_<name>.json: per-paragraph agreement on the raw output and on each label, the
paragraphs whose raw output differs, and each side's category macro-F1 if its score file is given.
Agreement is reported, never thresholded here: what counts as "the same model" is decided in the
pre-registered rule (T7), not by this tool.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

FIELDS = ("valid", "category", "severity", "first_seen_year")


def load(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_id = {r["para_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"duplicate para_id in {path}")
    return by_id


def compare(a: dict[str, dict], b: dict[str, dict]) -> dict:
    both = sorted(a.keys() & b.keys())
    if not both:
        raise ValueError("the two files share no paragraph")
    raw_diff = [p for p in both if a[p].get("raw") != b[p].get("raw")]
    return {
        "rows_a": len(a),
        "rows_b": len(b),
        "rows_compared": len(both),
        "only_a": sorted(a.keys() - b.keys()),
        "only_b": sorted(b.keys() - a.keys()),
        "raw_identical": len(both) - len(raw_diff),
        "agree": {f: sum(a[p].get(f) == b[p].get(f) for p in both) for f in FIELDS},
        "raw_differs": raw_diff,
        "category_changes": [
            {"para_id": p, "a": a[p].get("category"), "b": b[p].get("category")}
            for p in both
            if a[p].get("category") != b[p].get("category")
        ],
    }


def macro_f1_from(score_file: Path | None) -> float | None:
    if score_file is None:
        return None
    return json.loads(score_file.read_text())["all_rows"]["category_macro_f1"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="output: bench/parity_<name>.json")
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--scores-a", type=Path, default=None)
    parser.add_argument("--scores-b", type=Path, default=None)
    parser.add_argument("--bench", type=Path, default=Path("bench"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    result = {
        "command": " ".join(["python -m filinglens.serve.parity", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "a": args.a.as_posix(),
        "b": args.b.as_posix(),
        "category_macro_f1_a": macro_f1_from(args.scores_a),
        "category_macro_f1_b": macro_f1_from(args.scores_b),
        **compare(load(args.a), load(args.b)),
    }
    (args.bench / f"parity_{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
