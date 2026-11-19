"""Gold sample: company splits, the 270 + 60 draw and its constraints, first-seen suggestions."""

import json
from collections import Counter

import pandas as pd
import pytest

from filinglens.extract.gold_sample import (
    COMPANY_CAP,
    PER_CELL,
    YEARS,
    assign_splits,
    content_words,
    draw_sample,
    eligible,
    first_seen,
    main,
    overlap,
)

SECTORS = ["finance", "manufacturing", "mining_energy", "retail", "services", "transport"]
HEADINGS = [
    "Competition may reduce our margins",
    "The COVID-19 pandemic could disrupt our business",
    "We depend on key personnel",
    "Tariffs and trade policy could raise our costs",
    "Our debt could limit our flexibility",
]


def corpus(companies_per_sector: int = 40) -> pd.DataFrame:
    rows = []
    for s, sector in enumerate(SECTORS):
        for c in range(companies_per_sector):
            cik = 1000 * (s + 1) + c
            for year in (2018, *YEARS):
                accession = f"{cik}-{year}"
                for i in range(2):
                    heading = HEADINGS[(c + i + year) % len(HEADINGS)]
                    rows.append(
                        {
                            "para_id": f"{accession}:{i}",
                            "cik": cik,
                            "ticker": f"T{cik}",
                            "sector": sector,
                            "report_date": f"{year}-12-31",
                            "period_year": year,
                            "accession": accession,
                            "para_index": i,
                            "heading": heading,
                            "text": f"{heading}. " + " ".join(["word"] * 50),
                            "n_words": 56,
                        }
                    )
    return pd.DataFrame(rows)


def test_splits_are_per_company_about_60_20_20_and_seeded():
    companies = corpus(10)[["cik", "ticker", "sector"]].drop_duplicates("cik")
    splits = assign_splits(companies, seed=1)
    assert len(splits) == 60 and splits["cik"].is_unique
    per_sector = splits.groupby("sector")["split"].value_counts().unstack()
    assert (per_sector["test"] == 2).all() and (per_sector["dev"] == 2).all()
    assert (per_sector["train"] == 6).all()
    assert splits.equals(assign_splits(companies, seed=1))
    assert not splits["split"].equals(assign_splits(companies, seed=2)["split"])


def test_eligible_drops_bullets_short_paragraphs_and_other_years():
    df = corpus(1)
    df.loc[0, "text"] = "• " + df.loc[0, "text"]
    df.loc[1, "n_words"] = 39
    kept = eligible(df)
    assert df.loc[0, "para_id"] not in set(kept["para_id"])
    assert df.loc[1, "para_id"] not in set(kept["para_id"])
    assert set(kept["period_year"]) == set(YEARS)  # 2018 is only used for first-seen


def test_draw_sample_sizes_constraints_and_determinism():
    pool = eligible(corpus())
    sample = draw_sample(pool, seed=7)
    by_source = Counter(sample["source"])
    assert by_source["random"] == len(SECTORS) * len(YEARS) * PER_CELL == 270
    assert {k for k in by_source if k.startswith("boost:")} == {
        "boost:people",
        "boost:health_pandemic",
        "boost:geopolitical_international",
    }
    assert sample["accession"].is_unique  # one paragraph per filing
    assert sample["cik"].value_counts().max() <= COMPANY_CAP
    assert list(sample["task_no"]) == list(range(1, len(sample) + 1))
    assert sample["para_id"].tolist() == draw_sample(pool, seed=7)["para_id"].tolist()
    random_part = sample[sample["source"] == "random"]
    cells = Counter(zip(random_part["sector"], random_part["period_year"], strict=True))
    assert set(cells.values()) == {PER_CELL}


def test_boosted_items_are_spread_through_the_task_order():
    sample = draw_sample(eligible(corpus()), seed=7)
    boosted = sample[sample["source"] != "random"]["task_no"]
    assert boosted.min() < 50 and boosted.max() > len(sample) - 50


def test_overlap_coefficient():
    a, b = (
        content_words("Supply chain disruptions hurt suppliers"),
        content_words("the supply chain"),
    )
    assert overlap(a, b) == 1.0  # every word of the smaller set is shared
    assert overlap(a, frozenset()) == 0.0


def test_first_seen_suggests_earliest_matching_year_and_skips_missing_years():
    def para(year, heading):
        text = f"{heading}. " + " ".join(["filler"] * 5)
        return {"para_id": f"x-{year}:0", "period_year": year, "heading": heading, "text": text}

    same = "Disruptions at our suppliers could interrupt production of our turbines"
    company = pd.DataFrame(
        [
            para(2018, "Competition from new entrants may reduce prices"),
            para(2020, same),
            para(2021, same + " and engines"),
            para(2022, same),
        ]
    )
    result = first_seen(company.iloc[3], company)
    assert result["suggested"] == 2020
    assert [c["year"] for c in result["candidates"]] == [2018, 2020, 2021]  # 2019 has no 10-K
    assert result["candidates"][0]["score"] < 0.5


def test_first_seen_defaults_to_own_year():
    row = pd.Series({"period_year": 2019, "heading": "New risk", "text": "brand new words"})
    other = pd.DataFrame([{"para_id": "a", "period_year": 2018, "heading": "x", "text": "zzz"}])
    assert first_seen(row, other)["suggested"] == 2019


def test_main_writes_gold_files_without_split_or_source_in_tasks(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "corpus.parquet"
    corpus().to_parquet(path, index=False)
    main(
        [
            *("--corpus", str(path), "--gold", str(tmp_path / "gold")),
            *("--tasks", str(tmp_path / "tasks.json"), "--stats", str(tmp_path / "s.json")),
        ]
    )
    sample = pd.read_csv(tmp_path / "gold" / "risk_sample.csv")
    tasks = json.loads((tmp_path / "tasks.json").read_text())
    assert len(tasks) == len(sample) and "split" not in tasks[0] and "source" not in tasks[0]
    splits = pd.read_csv(tmp_path / "gold" / "splits.csv")
    assert (
        sample.merge(splits, on="cik")["split_x"]
        .eq(sample.merge(splits, on="cik")["split_y"])
        .all()
    )
