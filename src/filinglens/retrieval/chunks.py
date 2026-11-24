"""Turn the risk-factor paragraphs into retrieval chunks (FL-M3 T2).

One chunk per paragraph of Item 1A. The text that gets embedded is the paragraph's risk heading
(its first 40 words) on one line and the paragraph below it, because most paragraphs continue the
risk named in the heading and say little on their own ("In the last three years, approximately 80%
of our revenues came from ..."). The heading is left out when the paragraph already starts with it.

bge-base reads at most 512 tokens, 510 of them text. A paragraph whose embedding text is longer is
split at sentence ends into parts that fit, about equal in length, and every part keeps the
heading. A sentence that alone is too long is cut between words. Splitting relies on
WordPiece token counts adding up across whitespace (the tokenizer splits on whitespace first), and
the result is recounted with the real tokenizer: any part over the budget stops the run.

Chunk IDs are `<para_id>#<part>`, stable across runs. Output: data/interim/retrieval/chunks.parquet
and bench/chunks.json (counts, token lengths, input and output SHA-256).

    ~/venvs/fl-train/bin/python -m filinglens.retrieval.chunks
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from filinglens.retrieval.config import (
    CHUNKS,
    CORPUS,
    EMBED_MODEL,
    EMBED_REVISION,
    ITEM,
    MAX_SEQ_TOKENS,
    TEXT_TOKEN_BUDGET,
)

Counter = Callable[[Sequence[str]], list[int]]  # token counts, without special tokens

HEADING_WORDS = 40
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"“(\[A-Z0-9•●▪·\-–])")
META_COLUMNS = ["para_id", "cik", "ticker", "sector", "period_year", "accession", "para_index"]


def heading_prefix(heading: str | None, text: str) -> str:
    """The heading to put above the paragraph: '' if there is none or the text starts with it."""
    words = (heading if isinstance(heading, str) else "").split()[:HEADING_WORDS]
    if not words:
        return ""
    prefix = " ".join(words)
    if " ".join(text.split()).lower().startswith(prefix.lower()):
        return ""
    return prefix


def embed_text(prefix: str, text: str) -> str:
    return f"{prefix}\n{text}" if prefix else text


def split_sentences(text: str) -> list[str]:
    """Split at ., ! or ? followed by whitespace and a capital, digit, quote or bullet."""
    return [s for s in SENTENCE_END.split(" ".join(text.split())) if s]


def _word_windows(sentence: str, budget: int, count: Counter) -> list[tuple[str, int]]:
    """Cut one over-long sentence between words into pieces of at most `budget` tokens."""
    words = sentence.split()
    pieces: list[tuple[str, int]] = []
    current: list[str] = []
    used = 0
    for word, n in zip(words, count(words), strict=True):
        if n > budget:
            raise ValueError(f"a single word is longer than the budget: {word[:40]!r} ({n})")
        if current and used + n > budget:
            pieces.append((" ".join(current), used))
            current, used = [], 0
        current.append(word)
        used += n
    if current:
        pieces.append((" ".join(current), used))
    return pieces


def pack(sentences: Sequence[str], counts: Sequence[int], budget: int, count: Counter) -> list[str]:
    """Join sentences, in order, into parts of at most `budget` tokens and about equal length.

    It aims for ceil(total / budget) parts: a part closes once it has reached the target length
    total / ceil(total / budget), or before a sentence that would take it past the budget, so the
    last part isn't a stub. Sentences longer than the budget are first cut between words.
    """
    units: list[tuple[str, int]] = []
    for sentence, n in zip(sentences, counts, strict=True):
        units.extend(_word_windows(sentence, budget, count) if n > budget else [(sentence, n)])
    total = sum(n for _, n in units)
    target = math.ceil(total / math.ceil(total / budget)) if total else budget
    parts: list[str] = []
    current: list[str] = []
    used = 0
    for unit, n in units:
        if current and (used + n > budget or used >= target):
            parts.append(" ".join(current))
            current, used = [], 0
        current.append(unit)
        used += n
    if current:
        parts.append(" ".join(current))
    return parts


def _verify(chunks: pd.DataFrame, rows: np.ndarray, count: Counter, budget: int) -> None:
    """Recount the given chunk rows with the real counter; stop if any is over the budget."""
    if len(rows) == 0:
        return
    texts = chunks.loc[rows, "embed_text"].tolist()
    counts = np.asarray(count(texts))
    chunks.loc[rows, "n_tokens"] = counts
    over = chunks.loc[rows[counts > budget], "chunk_id"].tolist()
    if over:
        raise ValueError(f"{len(over)} chunks over {budget} tokens after splitting: {over[:5]}")


def build_chunks(
    corpus: pd.DataFrame, count: Counter, budget: int = TEXT_TOKEN_BUDGET
) -> pd.DataFrame:
    """One row per chunk, in corpus order, with its metadata, texts and token count."""
    pairs = zip(corpus["heading"], corpus["text"], strict=True)
    prefixes = [heading_prefix(h, t) for h, t in pairs]
    texts = [embed_text(p, t) for p, t in zip(prefixes, corpus["text"], strict=True)]
    counts = count(texts)
    records: list[dict] = []
    split_rows: list[int] = []
    for i, row in enumerate(corpus[META_COLUMNS + ["heading", "text"]].itertuples(index=False)):
        meta = dict(zip(META_COLUMNS, row[: len(META_COLUMNS)], strict=True))
        prefix = prefixes[i]
        if counts[i] <= budget:
            parts, n_tokens = [row.text], [counts[i]]
        else:
            sentences = split_sentences(row.text)
            room = budget - (count([prefix])[0] if prefix else 0)
            parts = pack(sentences, count(sentences), room, count)
            n_tokens = [-1] * len(parts)  # recounted below
        for part_no, (part, n) in enumerate(zip(parts, n_tokens, strict=True)):
            if n < 0:
                split_rows.append(len(records))
            records.append(
                {
                    "chunk_id": f"{meta['para_id']}#{part_no}",
                    **meta,
                    "item": ITEM,
                    "part": part_no,
                    "n_parts": len(parts),
                    "heading": row.heading,
                    "heading_prefixed": bool(prefix),
                    "text": part,
                    "embed_text": embed_text(prefix, part),
                    "n_tokens": n,
                }
            )
    chunks = pd.DataFrame.from_records(records)
    _verify(chunks, np.asarray(split_rows, dtype=int), count, budget)
    chunks["n_tokens"] = chunks["n_tokens"].astype("int64")
    return chunks


def summarize(corpus: pd.DataFrame, chunks: pd.DataFrame) -> dict:
    tokens = chunks["n_tokens"]
    split = chunks[chunks["n_parts"] > 1]
    parts = split.groupby("para_id")["n_parts"].first().value_counts().sort_index()
    return {
        "paragraphs": len(corpus),
        "chunks": len(chunks),
        "paragraphs_split": int(split["para_id"].nunique()),
        "parts_per_split_paragraph": {str(k): int(v) for k, v in parts.items()},
        "heading_prefixed": int(chunks["heading_prefixed"].sum()),
        "companies": int(chunks["cik"].nunique()),
        "filings": int(chunks["accession"].nunique()),
        "tokens": {
            "p50": float(np.percentile(tokens, 50)),
            "p95": float(np.percentile(tokens, 95)),
            "p99": float(np.percentile(tokens, 99)),
            "max": int(tokens.max()),
            "budget": TEXT_TOKEN_BUDGET,
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bge_counter(batch: int = 4096) -> Counter:
    """Token counts from the pinned bge-base tokenizer (no special tokens)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, revision=EMBED_REVISION)
    if tokenizer.model_max_length != MAX_SEQ_TOKENS:
        raise ValueError(f"expected a {MAX_SEQ_TOKENS}-token model: {tokenizer.model_max_length}")

    def count(texts: Sequence[str]) -> list[int]:
        out: list[int] = []
        for start in range(0, len(texts), batch):
            encoded = tokenizer(list(texts[start : start + batch]), add_special_tokens=False)
            out.extend(len(x) for x in encoded["input_ids"])
        return out

    return count


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--out", type=Path, default=CHUNKS)
    parser.add_argument("--bench", type=Path, default=Path("bench"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    start = time.perf_counter()
    corpus = pd.read_parquet(args.corpus)
    chunks = build_chunks(corpus, bge_counter())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    chunks.to_parquet(args.out, index=False)
    result = {
        "command": "python -m filinglens.retrieval.chunks " + " ".join(argv),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "tokenizer": f"{EMBED_MODEL}@{EMBED_REVISION}",
        "item": ITEM,
        "heading_words": HEADING_WORDS,
        "corpus": str(args.corpus),
        "corpus_sha256": sha256_file(args.corpus),
        "out": str(args.out),
        "out_sha256": sha256_file(args.out),
        **summarize(corpus, chunks),
        "seconds": round(time.perf_counter() - start, 1),
    }
    (args.bench / "chunks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
