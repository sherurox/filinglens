"""Draw the gold-set paragraphs, fix the company splits, and suggest first-seen years (FL-M2 T3).

1. Splits are by company: every company in the corpus goes to train, dev or test (about
   60/20/20 within each sector), ordered by a seeded hash of its CIK. A company never appears in
   two splits, so its boilerplate can't leak from training into the test set.
2. The sample is 270 random paragraphs (9 per sector x year cell, years 2019, 2020, 2022, 2024,
   2025) plus 60 "boosted" ones (20 per rare category, picked by keyword) so the rare categories
   have enough examples for per-class F1. Each row records its source, so results can also be
   reported on the random part alone. At most one paragraph per filing and three per company.
3. For each sampled paragraph, the most similar paragraph of the same company in each earlier
   year is found by word overlap; the earliest year above a threshold is the suggested
   first-seen year. It is only a suggestion: the labeler confirms or corrects it.

Outputs: gold/splits.csv and gold/risk_sample.csv (committed: they fix the gold set before any
label exists), data/interim/label_tasks/risk_tasks.json (tasks with text and candidates, for
Label Studio) and bench/gold_sample.json (counts).
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

YEARS = (2019, 2020, 2022, 2024, 2025)
PER_CELL = 9  # 6 sectors x 5 years x 9 = 270
BOOST_PER_CATEGORY = 20
COMPANY_CAP = 3
MIN_WORDS, MAX_WORDS = 40, 400
FIRST_YEAR = 2018
SUGGEST_THRESHOLD = 0.5
CONTEXT_WORDS = 60
SAMPLE_COLUMNS = [
    "task_no",
    "para_id",
    "cik",
    "ticker",
    "sector",
    "period_year",
    "split",
    "source",
    "n_words",
    "text_sha256",
]

BULLET = re.compile(r"^[•●▪·\-–]")
BOOST = {
    "people": r"key (personnel|employees|executives|management)|talent|\blabor\b|union|workforce"
    r"|retain and attract|attract and retain|qualified (employees|personnel)",
    "health_pandemic": r"covid|coronavirus|pandemic|epidemic|public health",
    "geopolitical_international": r"international operation|geopolitic|\bwar\b|ukraine|russia"
    r"|china|sanction|tariff|trade polic|political (instability|conditions)|export control",
}
BOOST_PATTERNS = {name: re.compile(rx) for name, rx in BOOST.items()}
STOPWORDS = frozenset(
    "a an and are as at be by can could do for from has have if in into is it its may might"
    " not of on or our such that the their these this to us we which will with would".split()
)


def _hash_key(seed: int, value: object) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def assign_splits(companies: pd.DataFrame, seed: int) -> pd.DataFrame:
    """One row per company (cik, ticker, sector) with split = train | dev | test."""
    parts = []
    for _, group in companies.groupby("sector", sort=True):
        group = group.assign(_k=group["cik"].map(lambda c: _hash_key(seed, c))).sort_values("_k")
        n = len(group)
        n_test, n_dev = round(0.2 * n), round(0.2 * n)
        labels = ["test"] * n_test + ["dev"] * n_dev + ["train"] * (n - n_test - n_dev)
        parts.append(group.assign(split=labels).drop(columns="_k"))
    return pd.concat(parts).sort_values("cik").reset_index(drop=True)


def eligible(corpus: pd.DataFrame) -> pd.DataFrame:
    """Paragraphs that can be sampled: not summary bullets, 40-400 words, a sampled year."""
    keep = (
        ~corpus["text"].str.match(BULLET)
        & corpus["n_words"].between(MIN_WORDS, MAX_WORDS)
        & corpus["period_year"].isin(YEARS)
    )
    return corpus[keep]


def _boost_text(row: pd.Series) -> str:
    return (row["heading"] or " ".join(row["text"].split()[:40])).lower()


def pick(pool: pd.DataFrame, n: int, rng: random.Random, used: set, per_company: Counter) -> list:
    """Up to n para_ids from pool, in a seeded random order, respecting one paragraph per filing
    and COMPANY_CAP per company. `used` and `per_company` are updated in place."""
    order = sorted(pool["para_id"])
    rng.shuffle(order)
    rows = pool.set_index("para_id")
    chosen = []
    for para_id in order:
        if len(chosen) == n:
            break
        row = rows.loc[para_id]
        if row["accession"] in used or per_company[row["cik"]] >= COMPANY_CAP:
            continue
        used.add(row["accession"])
        per_company[row["cik"]] += 1
        chosen.append(para_id)
    return chosen


def draw_sample(pool: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The 270 random + 60 boosted paragraphs, shuffled, with source and task_no."""
    rng = random.Random(seed)
    used: set = set()
    per_company: Counter = Counter()
    picks: list[tuple[str, str]] = []
    for _, cell in pool.groupby(["sector", "period_year"], sort=True):
        picks += [(p, "random") for p in pick(cell, PER_CELL, rng, used, per_company)]
    texts = pool.apply(_boost_text, axis=1)
    for name, rx in BOOST_PATTERNS.items():
        matched = pool[texts.map(lambda t, rx=rx: bool(rx.search(t)))]
        chosen = pick(matched, BOOST_PER_CATEGORY, rng, used, per_company)
        picks += [(p, f"boost:{name}") for p in chosen]
    rng.shuffle(picks)  # boosted items must not cluster in the labeling order
    order = pd.DataFrame(picks, columns=["para_id", "source"])
    order["task_no"] = range(1, len(order) + 1)
    return order.merge(pool, on="para_id", how="left")


def content_words(text: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"[a-z][a-z\-]+", text.lower()) if w not in STOPWORDS)


def overlap(a: frozenset[str], b: frozenset[str]) -> float:
    """Overlap coefficient: shared words / words in the smaller set (0 if either is empty)."""
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def _context(row) -> str:
    return f"{row['heading']} {' '.join(row['text'].split()[:CONTEXT_WORDS])}"


def first_seen(row: pd.Series, company: pd.DataFrame) -> dict:
    """Best match in each earlier year of the same company, and the suggested first-seen year."""
    query = content_words(_context(row))
    candidates = []
    for year in range(FIRST_YEAR, int(row["period_year"])):
        same_year = company[company["period_year"] == year]
        if same_year.empty:
            continue
        scores = same_year.apply(lambda r: overlap(query, content_words(_context(r))), axis=1)
        best = same_year.loc[scores.idxmax()]
        candidates.append(
            {
                "year": year,
                "score": round(float(scores.max()), 3),
                "para_id": best["para_id"],
                "heading": best["heading"],
                "text": " ".join(best["text"].split()[:CONTEXT_WORDS]),
            }
        )
    above = [c["year"] for c in candidates if c["score"] >= SUGGEST_THRESHOLD]
    return {"suggested": min(above, default=int(row["period_year"])), "candidates": candidates}


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def summarize(sample: pd.DataFrame, splits: pd.DataFrame, tasks: list[dict]) -> dict:
    suggested_earlier = sum(t["first_seen"]["suggested"] < t["period_year"] for t in tasks)
    cells = Counter(zip(sample["sector"], sample["period_year"], strict=True))
    return {
        "companies": len(splits),
        "companies_by_split": dict(sorted(Counter(splits["split"]).items())),
        "paragraphs": len(sample),
        "by_source": dict(sorted(Counter(sample["source"]).items())),
        "by_split": dict(sorted(Counter(sample["split"]).items())),
        "by_sector_year": {f"{s} {y}": n for (s, y), n in sorted(cells.items())},
        "distinct_companies": int(sample["cik"].nunique()),
        "distinct_filings": int(sample["accession"].nunique()),
        "first_seen_suggested_before_own_year": suggested_earlier,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=Path("data/interim/risk_paragraphs.parquet"))
    parser.add_argument("--gold", type=Path, default=Path("gold"))
    parser.add_argument(
        "--tasks", type=Path, default=Path("data/interim/label_tasks/risk_tasks.json")
    )
    parser.add_argument("--stats", type=Path, default=Path("bench/gold_sample.json"))
    parser.add_argument("--seed", type=int, default=20261119)
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    corpus = pd.read_parquet(args.corpus)
    companies = corpus[["cik", "ticker", "sector"]].drop_duplicates("cik")
    splits = assign_splits(companies, args.seed)
    sample = draw_sample(eligible(corpus), args.seed).merge(splits[["cik", "split"]], on="cik")
    sample = sample.sort_values("task_no").reset_index(drop=True)

    tasks = []
    for _, row in sample.iterrows():
        company = corpus[corpus["cik"] == row["cik"]]
        tasks.append(
            {
                "task_no": int(row["task_no"]),
                "para_id": row["para_id"],
                "ticker": row["ticker"],
                "sector": row["sector"],
                "period_year": int(row["period_year"]),
                "heading": row["heading"],
                "text": row["text"],
                "first_seen": first_seen(row, company),
            }
        )  # source and split are left out on purpose: the labeler must not see them

    args.gold.mkdir(parents=True, exist_ok=True)
    splits.to_csv(args.gold / "splits.csv", index=False)
    sample_out = sample.assign(text_sha256=sample["text"].map(text_sha256))
    sample_out[SAMPLE_COLUMNS].to_csv(args.gold / "risk_sample.csv", index=False)
    args.tasks.parent.mkdir(parents=True, exist_ok=True)
    args.tasks.write_text(json.dumps(tasks, indent=1, ensure_ascii=False) + "\n")
    stats = {
        "command": " ".join(["uv run python -m filinglens.extract.gold_sample", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": args.seed,
        **summarize(sample, splits, tasks),
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
