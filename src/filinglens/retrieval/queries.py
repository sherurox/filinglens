"""Labeled retrieval queries from the synthetic Q&A, split into ret-dev and ret-test (FL-M3 T3).

The queries are FL-M2's synthetic questions (gold/synthetic/risk_qa.train.jsonl, written by
Qwen3-4B-AWQ from paragraphs of train-split companies). Relevance is graded automatically; no
person judged it:

- one query per source paragraph (its first question by qa_id), so no two queries share a
  paragraph and each query is an independent unit for the bootstrap;
- grade 2: the chunk(s) of the source paragraph that hold the evidence quote, matched the way
  FL-M2 checked it (`synthetic_qa.squash`: case, whitespace and curly quotes ignored; the quote
  may come from the heading, which every part of a split paragraph carries). If a split moved
  the quote across two parts, the part sharing the most of its content words is used;
- grade 1: chunks of the same company in *other* filings whose text is a near copy of a grade-2
  chunk: overlap coefficient of content words >= 0.8 (FL-M2's `content_words` and `overlap`),
  and at least half as many content words as the grade-2 chunk, so a short fragment whose few
  words all appear in the source doesn't count. Risk text repeats almost word for word from year
  to year; the questions name a fiscal year, so those copies are near misses, not equal answers.

Companies, not queries, are split (sha256 of seed:cik, alternating within each sector), so the
boilerplate of one company never sits on both sides. Reading ret-test needs
`load_queries("test", allow_test=True)`.

Outputs: gold/retrieval/queries.{dev,test}.jsonl (committed: they fix the evaluation before any
retrieval result) and bench/ret_queries.json (counts, grade-1 distribution, SHA-256 of inputs and
outputs).

    uv run python -m filinglens.retrieval.queries
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from filinglens.extract.gold_sample import content_words, overlap
from filinglens.extract.synthetic_qa import squash
from filinglens.retrieval.chunks import sha256_file
from filinglens.retrieval.config import CHUNKS

QA = Path("gold/synthetic/risk_qa.train.jsonl")
SPLITS = Path("gold/splits.csv")
OUT_DIR = Path("gold/retrieval")
RET_SPLITS = ("dev", "test")
SEED = 3
DUP_OVERLAP = 0.8
DUP_MIN_SIZE_RATIO = 0.5
CHUNK_COLUMNS = ["chunk_id", "para_id", "cik", "accession", "period_year", "heading", "text"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pick_queries(qa: list[dict]) -> list[dict]:
    """The first question (lowest qa_id) of each source paragraph."""
    first: dict[str, dict] = {}
    for row in sorted(qa, key=lambda r: r["qa_id"]):
        first.setdefault(row["para_id"], row)
    return sorted(first.values(), key=lambda r: r["qa_id"])


def _hash_key(seed: int, cik: int) -> str:
    return hashlib.sha256(f"{seed}:{cik}".encode()).hexdigest()  # as in extract.gold_sample


def assign_splits(companies: pd.DataFrame, seed: int = SEED) -> dict[int, str]:
    """cik -> 'dev' / 'test': ordered by sha256(seed:cik) within each sector, alternating."""
    out: dict[int, str] = {}
    for _, group in companies.groupby("sector", sort=True):
        ciks = sorted(group["cik"], key=lambda c: _hash_key(seed, c))
        for i, cik in enumerate(ciks):
            out[int(cik)] = RET_SPLITS[i % 2]
    return out


def _heading(value: object) -> str:
    return value if isinstance(value, str) else ""  # a missing heading is NaN in Parquet


def evidence_chunks(evidence: str, parts: pd.DataFrame) -> tuple[list[str], str]:
    """Chunk IDs of the source paragraph that hold the quote, and how they were found."""
    quote = squash(evidence)
    found = [
        row.chunk_id
        for row in parts.itertuples(index=False)
        if quote in squash(row.text) or quote in squash(_heading(row.heading))
    ]
    if found:
        return found, "verbatim"
    quote_words = content_words(evidence)
    shared = [len(quote_words & content_words(t)) for t in parts["text"]]
    return [parts["chunk_id"].iloc[int(np.argmax(shared))]], "most_shared_words"


def near_copies(
    source_texts: list[str], candidates: pd.DataFrame, words_of: dict[str, frozenset[str]]
) -> list[str]:
    """Chunk IDs among `candidates` that are near copies of any of the source texts.

    `words_of` caches each chunk's content words: a company's chunks are compared with every
    query of that company.
    """
    sources = [content_words(t) for t in source_texts]
    out = []
    for row in candidates.itertuples(index=False):
        words = words_of.get(row.chunk_id)
        if words is None:
            words = words_of[row.chunk_id] = content_words(row.text)
        for src in sources:
            big_enough = len(words) >= DUP_MIN_SIZE_RATIO * len(src)
            if big_enough and overlap(src, words) >= DUP_OVERLAP:
                out.append(row.chunk_id)
                break
    return out


def build_queries(qa: list[dict], chunks: pd.DataFrame, splits: dict[int, str]) -> list[dict]:
    """One labeled query per source paragraph, with its split and graded chunk IDs."""
    by_para = {pid: g for pid, g in chunks.groupby("para_id", sort=False)}
    by_cik = {int(cik): g for cik, g in chunks.groupby("cik", sort=False)}
    words_of: dict[str, frozenset[str]] = {}
    rows = []
    for q in pick_queries(qa):
        parts = by_para.get(q["para_id"])
        if parts is None:
            raise ValueError(f"source paragraph not in the chunks: {q['para_id']}")
        grade2, method = evidence_chunks(q["evidence"], parts)
        source_texts = parts.set_index("chunk_id").loc[grade2, "text"].tolist()
        company = by_cik[int(q["cik"])]
        others = company[company["accession"] != parts["accession"].iloc[0]]
        grade1 = near_copies(source_texts, others, words_of)
        rows.append(
            {
                "query_id": q["qa_id"],
                "split": splits[int(q["cik"])],
                "question": q["question"],
                "type": q["type"],
                "cik": int(q["cik"]),
                "ticker": q["ticker"],
                "sector": q["sector"],
                "period_year": int(q["period_year"]),
                "source_para_id": q["para_id"],
                "evidence_match": method,
                "grades": {**{c: 1 for c in grade1}, **{c: 2 for c in grade2}},
                "synthetic": True,
                "generator": q["generator"],
            }
        )
    return rows


def leakage(rows: list[dict]) -> int:
    """Companies with queries in both splits (must be 0)."""
    seen: dict[int, set[str]] = {}
    for r in rows:
        seen.setdefault(r["cik"], set()).add(r["split"])
    return sum(len(s) > 1 for s in seen.values())


def summarize(rows: list[dict]) -> dict:
    out: dict = {"queries": len(rows), "leakage_companies_in_both_splits": leakage(rows)}
    for split in RET_SPLITS:
        part = [r for r in rows if r["split"] == split]
        n_grade1 = [sum(g == 1 for g in r["grades"].values()) for r in part]
        out[split] = {
            "queries": len(part),
            "companies": len({r["cik"] for r in part}),
            "by_sector": dict(sorted(Counter(r["sector"] for r in part).items())),
            "by_type": dict(sorted(Counter(r["type"] for r in part).items())),
            "evidence_match": dict(sorted(Counter(r["evidence_match"] for r in part).items())),
            "grade2_chunks": sum(sum(g == 2 for g in r["grades"].values()) for r in part),
            "grade1_per_query": {
                "mean": round(float(np.mean(n_grade1)), 3) if part else 0.0,
                "max": max(n_grade1, default=0),
                "zero": sum(n == 0 for n in n_grade1),
            },
        }
    return out


def write_split_files(rows: list[dict], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in RET_SPLITS:
        path = out_dir / f"queries.{split}.jsonl"
        part = [r for r in rows if r["split"] == split]
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part))
        paths[split] = path
    return paths


def load_queries(split: str, out_dir: Path = OUT_DIR, allow_test: bool = False) -> list[dict]:
    """Queries of one split. ret-test is only for the one final A/B and peer runs."""
    if split not in RET_SPLITS:
        raise ValueError(f"unknown retrieval split {split!r}")
    if split == "test" and not allow_test:
        raise PermissionError("ret-test is reserved for the final A/B run: allow_test=True")
    return read_jsonl(out_dir / f"queries.{split}.jsonl")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qa", type=Path, default=QA)
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--splits", type=Path, default=SPLITS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--bench", type=Path, default=Path("bench"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    qa = read_jsonl(args.qa)
    m2_splits = pd.read_csv(args.splits)
    train = set(m2_splits.loc[m2_splits["split"] == "train", "cik"].astype(int))
    outside = {int(r["cik"]) for r in qa} - train
    if outside:
        raise ValueError(f"synthetic Q&A from non-train companies: {sorted(outside)[:5]}")
    companies = m2_splits[m2_splits["cik"].isin({int(r["cik"]) for r in qa})]
    chunks = pd.read_parquet(args.chunks, columns=CHUNK_COLUMNS)
    rows = build_queries(qa, chunks, assign_splits(companies))
    if leakage(rows):
        raise ValueError("a company has queries in both ret-dev and ret-test")
    paths = write_split_files(rows, args.out_dir)
    result = {
        "command": "python -m filinglens.retrieval.queries " + " ".join(argv),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "labels": "synthetic queries, relevance graded automatically (no human judgment)",
        "rule": {
            "seed": SEED,
            "grade2": "source-paragraph chunks holding the evidence quote (synthetic_qa.squash)",
            "grade1": f"same company, other filing, overlap >= {DUP_OVERLAP}, "
            f"size >= {DUP_MIN_SIZE_RATIO} x grade-2 chunk (content words)",
        },
        "qa_pairs": len(qa),
        "inputs_sha256": {str(p): sha256_file(p) for p in (args.qa, args.chunks, args.splits)},
        **summarize(rows),
        "outputs_sha256": {str(p): sha256_file(p) for p in paths.values()},
    }
    (args.bench / "ret_queries.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
