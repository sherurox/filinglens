"""Check the draft risk categories against the paragraph corpus before writing the guide (FL-M2 T2).

Analysis only: the labels come from a person following docs/LABELING_GUIDE.md, never from these
keyword rules. The rules only answer two questions before the guide is written:
1. Is any draft category so rare that a 300-paragraph sample would hold almost none of it?
2. Do the categories move by year the way known events say they should (COVID in 2020)?
It also prints real paragraphs to use as examples in the guide.

Each paragraph is matched on its heading (a one-line summary of the risk) or, if it has none,
its first 40 words. A paragraph can match several categories or none.

Run from the repo root (needs data/interim/risk_paragraphs.parquet from T1):
    uv run python notebooks/explore_risk_taxonomy.py > data/interim/taxonomy_explore.txt
"""

import json
import re
import textwrap
from pathlib import Path

import pandas as pd

CORPUS = Path("data/interim/risk_paragraphs.parquet")
STATS = Path("bench/risk_taxonomy_explore.json")
SEED = 20261119

# Draft categories with indicative keywords (lowercase text). Indicative only: see the docstring.
CATEGORIES: dict[str, str] = {
    "macro_market": r"economic (conditions|downturn|slowdown)|recession|inflation|interest rate"
    r"|foreign (currency|exchange)|exchange rate|commodity price|market volatility",
    "competition_demand": r"compet|market share|pricing pressure|demand for our|customer demand"
    r"|consumer (spending|preferences)|largest customer|customer concentration",
    "regulatory_legal": r"regulat|legislation|\blaws?\b|litigation|lawsuit|legal proceeding"
    r"|\btax|antitrust|intellectual property|government investigation",
    "technology_cyber": r"cyber|security breach|data breach|ransomware|information technology"
    r"|information systems|artificial intelligence|\bai\b|technolog",
    "operations_supply": r"supply chain|supplier|manufactur|raw material|logistic|product recall"
    r"|product liability|disruption",
    "financial_capital": r"indebtedness|\bdebt\b|liquidity|credit rating|impairment|goodwill"
    r"|pension|capital requirement|borrower|credit risk|covenant",
    "people": r"key (personnel|employees|executives|management)|talent|\blabor\b|union|workforce",
    "health_pandemic": r"covid|coronavirus|pandemic|epidemic|public health",
    "climate_environment": r"climate|greenhouse|emission|environmental|carbon|\besg\b",
    "geopolitical_international": r"international operation|geopolitic|\bwar\b|ukraine|russia"
    r"|china|sanction|tariff|trade polic|political (instability|conditions)",
    "strategy_corporate": r"acquisition|merger|divestiture|restructuring|stock price"
    r"|market price of our|dividend|anti-takeover|stockholder|shareholder",
}
PATTERNS = {name: re.compile(rx) for name, rx in CATEGORIES.items()}
BULLET = re.compile(r"^[•●▪·\-–]")


def match_text(row: pd.Series) -> str:
    return (row["heading"] or " ".join(row["text"].split()[:40])).lower()


def matches(text: str) -> list[str]:
    return [name for name, rx in PATTERNS.items() if rx.search(text)]


def show(row: pd.Series, words: int = 70) -> str:
    body = " ".join(row["text"].split()[:words])
    head = f"heading: {row['heading'][:150]}\n" if row["heading"] else "heading: (none)\n"
    meta = f"[{row['para_id']} {row['ticker']} {row['sector']} {row['period_year']}]\n"
    return meta + head + textwrap.fill(body + " ...", width=100)


def main() -> None:
    df = pd.read_parquet(CORPUS)
    total = len(df)
    df = df[~df["text"].str.match(BULLET)]  # risk-summary bullets are not sampled (T3)
    df = df.assign(cats=[matches(match_text(r)) for _, r in df.iterrows()])
    df = df.assign(n_cats=df["cats"].str.len())

    share = {c: round(100 * df["cats"].map(lambda cs, c=c: c in cs).mean(), 1) for c in CATEGORIES}
    by_year = {
        c: {
            int(y): round(100 * g["cats"].map(lambda cs, c=c: c in cs).mean(), 1)
            for y, g in df.groupby("period_year")
        }
        for c in ("health_pandemic", "geopolitical_international", "macro_market")
    }
    stats = {
        "command": "uv run python notebooks/explore_risk_taxonomy.py",
        "paragraphs": total,
        "after_dropping_bullets": len(df),
        "with_heading_pct": round(100 * (df["heading"] != "").mean(), 1),
        "pct_matching_category": share,
        "pct_matching_none": round(100 * (df["n_cats"] == 0).mean(), 1),
        "pct_matching_two_or_more": round(100 * (df["n_cats"] >= 2).mean(), 1),
        "pct_by_year": by_year,
        "ai_mention_pct_by_year": {
            int(y): round(100 * g["text"].str.contains(r"artificial intelligence|\bAI\b").mean(), 1)
            for y, g in df.groupby("period_year")
        },
    }
    STATS.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))

    print("\n=== 3 examples per category (paragraphs matching only that category) ===")
    single = df[df["n_cats"] == 1]
    for c in CATEGORIES:
        pool = single[single["cats"].map(lambda cs, c=c: cs == [c])]
        print(f"\n--- {c} ({len(pool)} single-category paragraphs) ---")
        for _, row in pool.sample(min(3, len(pool)), random_state=SEED).iterrows():
            print(show(row), "\n")

    print("\n=== 12 random paragraphs (no keyword filter): try labeling these with the guide ===")
    for _, row in df.sample(12, random_state=SEED + 1).iterrows():
        print(show(row, words=90), f"\nkeyword guess: {row['cats']}\n")


if __name__ == "__main__":
    main()
