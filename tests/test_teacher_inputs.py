"""Teacher inputs: pool exclusions, leakage check, round-robin draw, candidates, test guard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from filinglens.extract import gold_sample
from filinglens.extract.synthetic_qa import text_key
from filinglens.train import teacher_inputs as ti

# cik -> (ticker, sector, split)
COMPANIES = {
    1: ("AAA", "manufacturing", "train"),
    2: ("BBB", "finance", "train"),
    3: ("CCC", "manufacturing", "test"),
    4: ("DDD", "finance", "dev"),
}
YEARS = (2018, 2019, 2020)  # 2018 is not a sampled year: candidates only
SHARED = "shared"  # text of AAA 2020 #0 also appears at CCC 2019 #1
REPEATED = "repeated"  # AAA reuses its 2019 #2 text in 2020 #2
SAMPLED = "0000000001-2019:1"


def words(base: str, n: int = 45) -> str:
    """n distinct letter-only words (content_words ignores digits)."""
    return " ".join(base + chr(97 + i % 26) + chr(97 + i // 26) for i in range(n))


def text_for(cik: int, year: int, i: int) -> str:
    if (cik, year, i) in {(1, 2020, 0), (3, 2019, 1)}:
        return words(SHARED)
    if (cik, year, i) in {(1, 2019, 2), (1, 2020, 2)}:
        return words(REPEATED)
    # paragraph i of every year shares its first words, so earlier-year candidates score high
    return words(f"topic{'abc'[i]}", 30) + " " + words(f"c{'abcd'[cik - 1]}y{year % 100}x", 15)


def make_corpus() -> pd.DataFrame:
    rows = []
    for cik, (ticker, sector, _) in COMPANIES.items():
        for year in YEARS:
            accession = f"{cik:010d}-{year}"
            for i in range(3):
                text = text_for(cik, year, i)
                rows.append(
                    {
                        "para_id": f"{accession}:{i}",
                        "cik": cik,
                        "ticker": ticker,
                        "sector": sector,
                        "period_year": year,
                        "accession": accession,
                        "heading": f"Heading {i} of {ticker}",
                        "text": text,
                        "n_words": len(text.split()),
                    }
                )
    return pd.DataFrame(rows)


def make_splits() -> pd.DataFrame:
    return pd.DataFrame(
        [{"cik": c, "ticker": t, "sector": s, "split": sp} for c, (t, s, sp) in COMPANIES.items()]
    )


def test_first_seen_matches_gold_sample() -> None:
    corpus = make_corpus()
    company = corpus[corpus["cik"] == 1]
    index = ti.candidate_index(company)
    for _, row in company[company["period_year"] > 2018].iterrows():
        expected = gold_sample.first_seen(row, company)
        assert ti.first_seen(row.to_dict(), index) == expected
        assert expected["candidates"]  # the fixture exercises real candidates


def test_build_pool_exclusions() -> None:
    corpus = make_corpus()
    pool, stats, train, held_keys = ti.build_pool(corpus, make_splits(), {SAMPLED})
    assert set(pool["cik"]) == {1, 2}
    assert train == {1, 2}
    assert SAMPLED not in set(pool["para_id"])
    assert words(SHARED) not in set(pool["text"])
    assert text_key(words(SHARED)) in held_keys
    assert stats["dropped_in_gold_sample"] == 1
    assert stats["dropped_text_also_at_dev_or_test_company"] == 1
    assert 2018 not in set(pool["period_year"])  # not eligible: candidates only
    assert stats["pool_paragraphs"] == len(pool) == 2 * 2 * 3 - 2


def test_draw_round_robin_and_no_repeated_text() -> None:
    corpus = make_corpus()
    pool, *_ = ti.build_pool(corpus, make_splits(), {SAMPLED})
    order, stats = ti.draw(pool, per_cell=3, per_filing=10, seed=1)
    cells = [(r["sector"], r["period_year"]) for r in order]
    assert cells[:4] == sorted(set(cells))  # first round: one row per cell, cells sorted
    keys = [r["text_key"] for r in order]
    assert len(keys) == len(set(keys))
    assert stats["skipped_while_drawing"].get("repeated_text", 0) <= 1
    again, _ = ti.draw(pool, per_cell=3, per_filing=10, seed=1)
    assert [r["para_id"] for r in again] == [r["para_id"] for r in order]  # seeded


def test_draw_per_filing_cap() -> None:
    pool, *_ = ti.build_pool(make_corpus(), make_splits(), set())
    order, stats = ti.draw(pool, per_cell=3, per_filing=1, seed=1)
    accessions = [r["accession"] for r in order]
    assert len(accessions) == len(set(accessions))
    assert stats["skipped_while_drawing"]["filing_cap"] > 0


def test_leakage_counts() -> None:
    rows = [
        {"cik": 3, "para_id": "x:0", "text": "held company"},
        {"cik": 1, "para_id": "y:0", "text": words(SHARED)},
        {"cik": 1, "para_id": SAMPLED, "text": "sampled"},
        {"cik": 1, "para_id": "z:0", "text": "clean"},
    ]
    assert ti.leakage(rows, {1, 2}, {text_key(words(SHARED))}, {SAMPLED}) == {
        "rows_not_from_train_company": 1,
        "rows_with_text_at_dev_or_test_company": 1,
        "rows_in_gold_sample": 1,
    }


def write_fixture(tmp_path: Path) -> list[str]:
    pytest.importorskip("pyarrow")
    corpus = make_corpus()
    corpus.to_parquet(tmp_path / "corpus.parquet")
    gold = tmp_path / "gold"
    gold.mkdir()
    make_splits().to_csv(gold / "splits.csv", index=False)
    pd.DataFrame({"para_id": [SAMPLED]}).to_csv(gold / "risk_sample.csv", index=False)
    held = {"dev": "0000000004-2019:0", "test": "0000000003-2020:0"}
    tasks = []
    for split, para_id in held.items():
        (gold / f"risk_paragraphs.{split}.jsonl").write_text(
            json.dumps({"para_id": para_id}) + "\n"
        )
        row = corpus.set_index("para_id").loc[para_id]
        tasks.append({"para_id": para_id, "ticker": row["ticker"], "first_seen": {}})
    (tmp_path / "tasks.json").write_text(json.dumps(tasks))
    (tmp_path / "prompt.md").write_text("rules\n")
    return [
        "--corpus",
        str(tmp_path / "corpus.parquet"),
        "--gold",
        str(gold),
        "--tasks",
        str(tmp_path / "tasks.json"),
        "--prompt",
        str(tmp_path / "prompt.md"),
        "--out",
        str(tmp_path / "out"),
        "--stats",
        str(tmp_path / "bench.json"),
        "--per-cell",
        "3",
    ]


def test_main_without_test(tmp_path: Path) -> None:
    argv = write_fixture(tmp_path)
    ti.main(argv)
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == [
        "MANIFEST.json",
        "dev.jsonl",
        "pool.jsonl",
        "silver_prompt.md",
    ]
    pool = [json.loads(line) for line in (out / "pool.jsonl").read_text().splitlines()]
    assert [t["task_no"] for t in pool] == list(range(1, len(pool) + 1))
    assert {t["split"] for t in pool} == {"train"}
    assert all("first_seen" in t for t in pool)
    stats = json.loads((tmp_path / "bench.json").read_text())
    assert stats["leakage"] == {
        "rows_not_from_train_company": 0,
        "rows_with_text_at_dev_or_test_company": 0,
        "rows_in_gold_sample": 0,
    }
    assert stats["held_out_rows"] == {"dev": 1}
    manifest = json.loads((out / "MANIFEST.json").read_text())
    assert set(manifest["files"]) == {"dev.jsonl", "pool.jsonl", "silver_prompt.md"}


def test_main_with_test_then_without_removes_it(tmp_path: Path) -> None:
    argv = write_fixture(tmp_path)
    ti.main([*argv, "--allow-test"])
    out = tmp_path / "out"
    test_rows = [json.loads(line) for line in (out / "test.jsonl").read_text().splitlines()]
    assert [r["split"] for r in test_rows] == ["test"]
    ti.main(argv)
    assert not (out / "test.jsonl").exists()  # no stale test file left behind
