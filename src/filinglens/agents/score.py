r"""Score crew answers with docs/AGENT_RUBRIC.md: R1 and R2 automatically, R3-R5 from a hand sheet.

    uv run python -m filinglens.agents.score auto --split dev --tag dev1
    # fill R3, R4, R5 (0/1) and notes in bench/agent_score_<split>_<tag>.csv, then:
    uv run python -m filinglens.agents.score totals --split dev --tag dev1

auto: for each question and its answer record (data/interim/agents/answers_<split>_<tag>.jsonl)
- R1 figures: every figure the question asks for appears in the answer and matches its reference
  (agents.numbers: the figure as written, figure_matches, sign-lenient as the checker). Asked for,
  by template: metric_margin_risk revenue, operating income and the margin; peer_compare the two
  net margins; growth the growth; risk_change_metric the change, stated as the change itself or as
  both years' values (revenue in both years, or the margin in both years);
  for a question with a missing figure, the answer states no dollar amount at all; a blocked or
  failed answer fails R1. (Any other figure was already checked by the numbers checker: an answered
  status means every figure in it matched a value the tools gave.)
- R2 citations: the answer exists, every sentence cites (the crew's check), and every (company,
  fiscal year) the question's reference needs is cited by some [n] / [Fn] that resolves to that
  company's filing for that year (the CIK is in the EDGAR URL).
Writes the sheet (one row per question: question, status, answer, the cited excerpts in full, the
figure lines, R1, R2, and empty R3 / R4 / R5 / note columns) and bench/agent_score_<...>_auto.json.

totals: reads the filled sheet back (R3-R5 must be 0 or 1 on every row), computes task success
(all five pass; answerable and unanswerable apart, blocked counted apart) with Wilson 95%
intervals, per item and per template, and writes bench/agent_score_<split>_<tag>.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from filinglens.agents.figures import figure_matches
from filinglens.agents.numbers import find_figures
from filinglens.agents.questions import load_questions

ANSWERS = Path("data/interim/agents")
HAND = ("R3", "R4", "R5")
SHEET_FIELDS = [
    "qid",
    "template",
    "answerable",
    "question",
    "status",
    "answer",
    "cited_excerpts",
    "figure_lines",
    "risk_evidence",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "note",
]
KIND_FOR = {
    "ratio": ("percent", "bare"),
    "growth": ("percent", "bare"),
    "difference": ("points", "percent", "bare"),
}


def wilson(successes: int, n: int, z: float = 1.959964) -> list[float] | None:
    if n == 0:
        return None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def figure_present(answer: str, value: float, kinds: tuple[str, ...]) -> bool:
    for f in find_figures(answer):
        if f.kind in kinds:
            actual = value if f.signed else abs(value)
            if figure_matches(f.stated, f.decimals, f.scale, actual):
                return True
    return False


def _item_ok(answer: str, item: dict) -> bool:
    if "first_reported_value" in item:  # a reference figure: last or first reported value
        return any(
            figure_present(answer, item[k], ("money", "bare"))
            for k in ("value", "first_reported_value")
        )
    return figure_present(answer, item["value"], KIND_FOR[item["kind"]])


def asked(q: dict) -> list[tuple[str, list[list[dict]]]]:
    """What the question asks for: (label, alternatives), each alternative a list of reference
    items that must all appear. A change ("how did X change") may be stated as the change itself
    or as both years' values; everything else must appear itself."""
    ref = q["reference"]
    items = {x["label"]: x for x in ref["figures"] + ref["derived"]}
    if q["template"] == "metric_margin_risk":
        return [(x["label"], [[x]]) for x in ref["figures"] + ref["derived"]]
    if q["template"] in ("peer_compare", "growth"):
        return [(d["label"], [[d]]) for d in ref["derived"]]
    if q["template"] == "risk_change_metric":
        change = ref["derived"][-1]  # growth (revenue) or the margin difference, built last
        return [(change["label"], [[change], [items[label] for label in change["of"]]])]
    return []  # missing_data: checked separately


def r1(q: dict, rec: dict) -> tuple[bool, list[str]]:
    if rec["status"] != "answered":
        return False, [f"status {rec['status']}"]
    answer, missing = rec["answer"], []
    for label, alternatives in asked(q):
        if not any(all(_item_ok(answer, it) for it in alt) for alt in alternatives):
            missing.append(label)
    if q["reference"]["missing"] and any(f.kind == "money" for f in find_figures(answer)):
        missing.append("states a dollar amount for a figure that is missing")
    return not missing, missing


def required_years(q: dict) -> set[tuple[int, int]]:
    """(cik, fiscal year) pairs the answer must cite: risk evidence and figures."""
    ref = q["reference"]
    need = {(e["cik"], e["fiscal_year"]) for e in ref["risk_evidence"]}
    need |= {(f["cik"], f["fiscal_year"]) for f in ref["figures"]}
    return need


def r2(q: dict, rec: dict) -> tuple[bool, list[str]]:
    if rec["status"] != "answered":
        return False, [f"status {rec['status']}"]
    cited = [(int(y), url) for c in rec["citations"].values() for y, url in c.items() if url]
    gaps = []
    for cik, fy in sorted(required_years(q)):
        if not any(y == fy and f"/data/{cik}/" in url for y, url in cited):
            gaps.append(f"no citation of CIK {cik}'s fiscal-year-{fy} filing")
    return not gaps, gaps


def cited_excerpts(rec: dict) -> str:
    """The full text of every excerpt the answer cites, from the context it was given."""
    blocks = re.split(r"\n\n(?=\[\d+\] )", rec.get("excerpts") or "")
    by_n = {}
    for b in blocks:
        m = re.match(r"\[(\d+)\]", b)
        if m:
            by_n[m.group(1)] = b
    used = [t for t in rec["citations"] if not t.startswith("F")]
    return "\n\n".join(by_n[t] for t in sorted(used, key=int) if t in by_n)


def evidence_summary(q: dict) -> str:
    return (
        "; ".join(
            f"{e['label']}: {e['theme_chunks_total']} matching chunk(s)"
            for e in q["reference"]["risk_evidence"]
        )
        or "(none)"
    )


def read_answers(split: str, tag: str) -> dict[str, dict]:
    path = ANSWERS / f"answers_{split}_{tag}.jsonl"
    return {r["qid"]: r for r in map(json.loads, path.read_text().splitlines()) if r}


def run_auto(args: argparse.Namespace) -> dict:
    questions = load_questions(args.split, allow_test=args.allow_test)
    answers = read_answers(args.split, args.tag)
    rows, details = [], {}
    for q in questions:
        rec = answers.get(q["qid"])
        if rec is None:
            raise SystemExit(f"{q['qid']} has no answer in the run {args.tag!r}")
        ok1, why1 = r1(q, rec)
        ok2, why2 = r2(q, rec)
        details[q["qid"]] = {
            "status": rec["status"],
            "R1": ok1,
            "R1_why": why1,
            "R2": ok2,
            "R2_why": why2,
        }
        rows.append(
            {
                "qid": q["qid"],
                "template": q["template"],
                "answerable": q["answerable"],
                "question": q["question"],
                "status": rec["status"],
                "answer": rec.get("answer") or rec.get("last_draft") or "",
                "cited_excerpts": cited_excerpts(rec),
                "figure_lines": "\n".join(rec["figure_lines"]),
                "risk_evidence": evidence_summary(q),
                "R1": int(ok1),
                "R2": int(ok2),
                "R3": "",
                "R4": "",
                "R5": "",
                "note": "",
            }
        )
    sheet = Path(f"bench/agent_score_{args.split}_{args.tag}.csv")
    if sheet.exists() and not args.overwrite:
        raise SystemExit(f"{sheet} exists (it may hold hand scores): pass --overwrite on purpose")
    with sheet.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    n = len(rows)
    return {
        "sheet": str(sheet),
        "questions": n,
        "status": {
            s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})
        },
        "R1_pass": sum(r["R1"] for r in rows),
        "R2_pass": sum(r["R2"] for r in rows),
        "R1_and_R2_pass": sum(r["R1"] and r["R2"] for r in rows),
        "per_question": details,
    }


def run_totals(args: argparse.Namespace) -> dict:
    sheet = Path(f"bench/agent_score_{args.split}_{args.tag}.csv")
    rows = list(csv.DictReader(sheet.open(newline="")))
    bad = [f"{r['qid']} {k}={r[k]!r}" for r in rows for k in HAND if r[k] not in ("0", "1")]
    if bad:
        raise SystemExit("every R3/R4/R5 must be 0 or 1: " + ", ".join(bad[:10]))
    zero_without_note = [
        r["qid"] for r in rows if "0" in (r["R3"], r["R4"], r["R5"]) and not r["note"].strip()
    ]
    if zero_without_note:
        raise SystemExit(f"a 0 needs a one-line note (rubric): {zero_without_note}")
    for r in rows:
        r["success"] = all(r[k] == "1" for k in ("R1", "R2", *HAND))
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups["all"].append(r)
        groups["answerable" if r["answerable"] == "True" else "unanswerable"].append(r)
        groups[f"template:{r['template']}"].append(r)

    def summary(rs: list[dict]) -> dict:
        n, s = len(rs), sum(r["success"] for r in rs)
        return {
            "n": n,
            "success": s,
            "rate": round(s / n, 4) if n else None,
            "wilson95": wilson(s, n),
            "items": {k: sum(r[k] == "1" for r in rs) for k in ("R1", "R2", *HAND)},
            "blocked": sum(r["status"] == "blocked" for r in rs),
            "error": sum(r["status"] == "error" for r in rs),
        }

    return {"sheet": str(sheet), **{g: summary(rs) for g, rs in sorted(groups.items())}}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["auto", "totals"])
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--overwrite", action="store_true")
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    result = run_auto(args) if args.step == "auto" else run_totals(args)
    record = {
        "command": " ".join(["uv run python -m filinglens.agents.score", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **result,
    }
    suffix = "_auto" if args.step == "auto" else ""
    out = Path(f"bench/agent_score_{args.split}_{args.tag}{suffix}.json")
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "per_question"}, indent=2))


if __name__ == "__main__":
    main()
