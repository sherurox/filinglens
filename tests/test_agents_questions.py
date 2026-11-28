"""Analyst questions: fiscal-year matching, templates, selection, checks (no database, no files)."""

import json
import math

import pytest

from filinglens.agents import questions as qs

SECTORS = ("finance", "retail")
YEARS = range(2018, 2026)


def chunk(cik: int, year: int, i: int, text: str, heading: str | None = None) -> dict:
    return {
        "chunk_id": f"{cik}-{year}:{i}#0",
        "cik": cik,
        "accession": f"{cik}-{year}",
        "heading": heading,
        "text": text,
    }


def universe(per_sector_split: int = 4, bank: int | None = None) -> tuple[qs.Data, dict]:
    """Companies 1000+ (dev) and 2000+ (test), 2 sectors; FY ends 28 Sep, facts dated 30 Sep."""
    companies, filing_rows, facts, chunks, split_of = {}, [], [], [], {}
    for base, split in ((1000, "dev"), (2000, "test")):
        for s, sector in enumerate(SECTORS):
            for j in range(per_sector_split):
                cik = base + 100 * s + j
                split_of[cik] = split
                companies[cik] = {
                    "cik": cik,
                    "ticker": f"T{cik}",
                    "name": f"Co {cik}",
                    "sector": sector,
                }
                for y in YEARS:
                    filing_rows.append(
                        {
                            "cik": cik,
                            "period_year": y,
                            "accession": f"{cik}-{y}",
                            "report_date": f"{y}-09-28",
                        }
                    )
                    chunks.append(
                        chunk(
                            cik,
                            y,
                            0,
                            "Breaches of our systems could hurt us.",
                            heading="Cybersecurity incidents",
                        )
                    )
                    if y >= 2022:
                        chunks.append(chunk(cik, y, 1, "Rising inflation raises our costs."))
                    base_row = {
                        "cik": cik,
                        "value_date": f"{y}-09-30",
                        "was_restated": False,
                        "last_accession_number": f"{cik}-{y}",
                    }
                    revenue = 1000.0 + 10 * (y - 2018) + j
                    tags = {"NetIncomeLoss": 50.0 + j, "OperatingIncomeLoss": 80.0 + y - 2018}
                    if cik != bank:
                        tags["Revenues"] = revenue
                    for tag, value in tags.items():
                        facts.append(
                            base_row | {"tag": tag, "value": value, "first_reported_value": value}
                        )
    filings = qs.filings_table(filing_rows)
    figures, no_dates = qs.annual_figures(facts, filings)
    return qs.Data(companies, filings, figures, no_dates, qs.theme_hits(chunks)), split_of


SMALL = {
    "dev": {
        "risk_change_metric": 1,
        "metric_margin_risk": 1,
        "peer_compare": 1,
        "growth": 1,
        "missing_data": 1,
    },
    "test": {
        "risk_change_metric": 1,
        "metric_margin_risk": 1,
        "peer_compare": 1,
        "growth": 1,
        "missing_data": 1,
    },
}


def test_revenue_tags_match_fl_m1():
    bench_ratios = pytest.importorskip("filinglens.features.bench_ratios")
    assert qs.REVENUE_TAGS == bench_ratios.REVENUE_TAGS


def test_month_end_fact_dates_match_and_ambiguous_years_are_skipped():
    filings = qs.filings_table(
        [
            {"cik": 1, "period_year": 2024, "accession": "a", "report_date": "2024-09-28"},
            {"cik": 1, "period_year": 2023, "accession": "b", "report_date": "2023-09-30"},
            {"cik": 1, "period_year": 2022, "accession": "c", "report_date": "2022-06-30"},
        ]
    )
    row = {
        "cik": 1,
        "tag": "Revenues",
        "value": 5.0,
        "first_reported_value": 5.0,
        "was_restated": False,
        "last_accession_number": "a",
    }
    facts = [
        row | {"value_date": "2024-09-30"},  # 2 days after the report date: matched
        row | {"value_date": "2023-09-30"},
        row | {"value_date": "2023-10-15"},  # a second date in FY2023's window: ambiguous
    ]
    figures, no_dates = qs.annual_figures(facts, filings)
    assert figures[(1, 2024)]["revenue"]["value_date"] == "2024-09-30"
    assert (1, 2023) not in figures and (1, 2023) not in no_dates
    assert (1, 2022) in no_dates


def test_revenue_takes_the_first_tag_in_priority_order():
    filings = qs.filings_table(
        [{"cik": 1, "period_year": 2024, "accession": "a", "report_date": "2024-12-31"}]
    )
    row = {
        "cik": 1,
        "value_date": "2024-12-31",
        "first_reported_value": 1.0,
        "was_restated": False,
        "last_accession_number": "a",
    }
    facts = [
        row | {"tag": "SalesRevenueNet", "value": 1.0},
        row | {"tag": "RevenueFromContractWithCustomerExcludingAssessedTax", "value": 2.0},
    ]
    figures, _ = qs.annual_figures(facts, filings)
    rev = figures[(1, 2024)]["revenue"]
    assert rev["tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert rev["value"] == 2.0


def test_a_year_with_two_10ks_is_left_out():
    filings = qs.filings_table(
        [
            {"cik": 1, "period_year": 2024, "accession": "a", "report_date": "2024-12-31"},
            {"cik": 1, "period_year": 2024, "accession": "b", "report_date": "2024-06-30"},
            {"cik": 1, "period_year": 2023, "accession": "c", "report_date": "2023-12-31"},
            {"cik": 1, "period_year": 2023, "accession": "c", "report_date": "2023-12-31"},
        ]
    )
    assert set(filings) == {(1, 2023)}


def test_theme_hits_read_heading_and_text_and_ai_is_a_whole_word():
    hits = qs.theme_hits(
        [
            chunk(1, 2024, 0, "Our systems could fail.", heading="Cybersecurity risks"),
            chunk(1, 2024, 1, "We use AI models in our products."),
            chunk(1, 2024, 2, "Our main said goal is waiting."),  # 'ai' inside words only
        ]
    )
    assert hits[("1-2024", "cybersecurity")] == ["1-2024:0#0"]
    assert hits[("1-2024", "artificial_intelligence")] == ["1-2024:1#0"]


def test_generate_fills_the_counts_deterministically_and_keeps_splits_apart():
    data, split_of = universe()
    first = qs.generate(data, split_of, SMALL)
    second = qs.generate(data, split_of, SMALL)
    assert first == second
    for split in qs.SPLITS:
        rows = first[split]
        assert [q["qid"] for q in rows] == [f"{split}-{i:03d}" for i in range(1, 6)]
        assert [q["template"] for q in rows] == list(qs.TEMPLATES)
        assert all(q["split"] == split for q in rows)
        ciks = [c["cik"] for q in rows for c in q["companies"]]
        assert len(ciks) == len(set(ciks))  # a company appears once per split
        assert {split_of[c] for c in ciks} == {split}


def test_derived_values_recompute_exactly():
    data, split_of = universe()
    for rows in qs.generate(data, split_of, SMALL).values():
        for q in rows:
            ref = q["reference"]
            values = {x["label"]: x for x in ref["figures"] + ref["derived"]}
            for d in ref["derived"]:
                assert qs.recompute(d, values) == d["value"]


def test_metric_margin_question_has_the_margin_of_its_own_figures():
    data, _ = universe()
    q = qs.build_metric_margin_risk(data, 1000)
    rev, op = q["reference"]["figures"]
    (margin,) = q["reference"]["derived"]
    assert (rev["metric"], op["metric"]) == ("revenue", "operating_income")
    assert margin["value"] == op["value"] / rev["value"]
    assert q["answerable"] and q["reference"]["risk_evidence"][0]["theme_chunks_total"] >= 1
    assert q["reference"]["risk_evidence"][0]["filing_url"].startswith(
        "https://www.sec.gov/Archives/edgar/data/1000/"
    )


def test_risk_change_years_are_far_enough_apart_and_share_the_theme():
    data, _ = universe()
    q = qs.build_risk_change_metric(data, 1001)
    y1, y2 = q["fiscal_years"]
    assert y2 - y1 >= qs.MIN_YEAR_GAP
    assert all(ev["theme_chunks_total"] >= 1 for ev in q["reference"]["risk_evidence"])
    assert q["theme"] in qs.THEMES and qs.THEMES[q["theme"]][0] in q["question"]


def test_peer_compare_pairs_two_companies_of_one_sector():
    data, split_of = universe()
    dev = qs.generate(data, split_of, SMALL)["dev"]
    q = next(q for q in dev if q["template"] == "peer_compare")
    assert len({c["sector"] for c in q["companies"]}) == 1
    assert len(q["companies"]) == 2 and len(q["reference"]["derived"]) == 2


def test_missing_data_prefers_a_real_gap_then_the_early_year():
    data, split_of = universe(bank=1000)
    real = qs.build_missing_data(data, 1000)
    assert real is not None and not real["answerable"]
    (miss,) = real["reference"]["missing"]
    assert miss["metric"] == "revenue" and miss["tags"] == list(qs.REVENUE_TAGS)
    assert real["reference"]["figures"] == [] and real["reference"]["risk_evidence"]
    assert qs.build_missing_data(data, 1001) is None  # it has revenue every year
    early = qs.build_missing_early(data, 1001)
    assert early["fiscal_years"] == [qs.EARLY_YEAR] and not early["answerable"]
    dev = qs.generate(data, split_of, SMALL)["dev"]
    assert next(q for q in dev if q["template"] == "missing_data")["companies"][0]["cik"] == 1000


def test_too_few_candidates_stops_the_run():
    data, split_of = universe(per_sector_split=1)
    counts = {s: {"risk_change_metric": 5} for s in qs.SPLITS}
    with pytest.raises(SystemExit, match="only 2 of 5 risk_change_metric"):
        qs.generate(data, split_of, counts)


def fct_rows(questions: list[dict]) -> dict:
    return {
        (f["cik"], f["tag"], f["value_date"]): (
            f["value"],
            f["first_reported_value"],
            f["was_restated"],
            f["last_accession"],
        )
        for q in questions
        for f in q["reference"]["figures"]
    }


def accessions(questions: list[dict]) -> dict:
    return {
        cid: ev["accession"]
        for q in questions
        for ev in q["reference"]["risk_evidence"]
        for cid in ev["theme_chunk_ids"]
    }


def test_check_passes_on_unchanged_data_and_flags_every_kind_of_change():
    data, split_of = universe(bank=1000)
    rows = qs.generate(data, split_of, SMALL)["dev"]
    table = fct_rows(rows)

    def lookup(f):
        key = (f["cik"], f["tag"], f["value_date"])
        return [table[key]] if key in table else []

    ok = qs.check_questions(rows, lookup, lambda m: 0, accessions(rows))
    assert ok["ok"] and ok["figures_checked"] == len(table) and ok["missing_checked"] == 1

    key = next(iter(table))
    table[key] = (table[key][0] + 1, *table[key][1:])  # a value changed in fct
    bad_chunks = {cid: "other" for cid in accessions(rows)}
    bad = qs.check_questions(rows, lookup, lambda m: 3, bad_chunks)
    text = " ".join(bad["problems"])
    assert not bad["ok"] and "fct has" in text and "expected none" in text
    assert "not a chunk of" in text


def test_load_questions_guards_the_test_split(tmp_path):
    qs.write_jsonl(tmp_path / "questions.test.jsonl", [{"qid": "test-001"}])
    with pytest.raises(PermissionError):
        qs.load_questions("test", tmp_path)
    assert qs.load_questions("test", tmp_path, allow_test=True) == [{"qid": "test-001"}]
    with pytest.raises(ValueError):
        qs.load_questions("train", tmp_path)


def test_summary_holds_counts_only():
    data, split_of = universe()
    generated = qs.generate(data, split_of, SMALL)
    summary = qs.summarize(generated)
    text = json.dumps(summary)
    assert summary["test"]["questions"] == 5 and summary["test"]["answerable"] == 4
    for q in generated["test"]:
        assert q["question"] not in text and q["companies"][0]["ticker"] not in text
    assert not math.isnan(summary["dev"]["figures"])
