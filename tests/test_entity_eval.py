import pytest

from filinglens.extract.entities import Resolver
from filinglens.extract.entity_eval import (
    blocking,
    by_method,
    choose,
    decide,
    diagnostics,
    errors,
    half,
    metrics,
    precision_of,
    raw_matches,
    sample_pairs,
    wilson,
)

REFERENCE = [
    (1, "Delta Air Lines, Inc."),
    (2, "Delta Apparel, Inc."),
    (3, "Duke Energy CORP"),
    (4, "Acme Widgets Inc"),
]
QUERIES = [
    {"cik": "1", "name": "DELTA AIR LINES INC /DE/"},  # exact, right
    {"cik": "4", "name": "ACME WIDGETS HOLDINGS INC"},  # fuzzy, right
    {"cik": "99", "name": "DUKE ENERGY CAROLINAS, LLC"},  # not listed: a match is a false merge
    {"cik": "98", "name": "ZZZ UNKNOWN CORP"},  # no candidates
]


@pytest.fixture
def rows():
    resolver = Resolver(REFERENCE, max_df=3)
    return raw_matches(resolver, QUERIES, {c for c, _ in REFERENCE})


def test_half_is_deterministic_and_balanced():
    assert half(320193) == half(320193)
    halves = [half(c) for c in range(2000)]
    assert 900 < halves.count("tune") < 1100


def test_wilson_known_values():
    assert wilson(0, 0) == (None, None)
    lo, hi = wilson(95, 100)
    assert lo == pytest.approx(0.8882, abs=1e-4) and hi == pytest.approx(0.9785, abs=1e-4)
    assert wilson(100, 100)[1] == 1.0 and wilson(100, 100)[0] < 1.0


def test_raw_matches_fields(rows):
    by_name = {r["name"]: r for r in rows}
    assert by_name["DELTA AIR LINES INC /DE/"]["method"] == "exact"
    acme = by_name["ACME WIDGETS HOLDINGS INC"]
    assert (acme["method"], acme["best_cik"], acme["reachable"]) == ("fuzzy", 4, True)
    assert by_name["ZZZ UNKNOWN CORP"]["method"] == "none"
    assert by_name["DUKE ENERGY CAROLINAS, LLC"]["gold_in_reference"] is False


def test_decide_applies_threshold_only_to_fuzzy(rows):
    exact, acme, duke, _ = rows
    assert decide(exact, 0.99, 0.99) == 1
    assert decide(acme, 0.0, 0.0) == 4
    assert decide(acme, 0.99, 0.0) is None
    assert decide(duke, 0.99, 0.0) is None


def test_metrics_counts_false_merges(rows):
    loose = metrics(rows, 0.0, 0.0)  # Duke Carolinas gets merged into Duke Energy: wrong
    assert (loose["matched"], loose["correct"], loose["gold_in_reference"]) == (3, 2, 2)
    assert loose["precision"] == pytest.approx(2 / 3, abs=1e-4) and loose["recall"] == 1.0
    strict = metrics(rows, 0.99, 0.0)
    assert (strict["matched"], strict["correct"], strict["precision"]) == (1, 1, 1.0)


def test_choose_prefers_f05_then_stricter():
    sweep = [
        {"f0_5": 0.9, "threshold": 0.6, "margin": 0.0},
        {"f0_5": 0.9, "threshold": 0.8, "margin": 0.0},
        {"f0_5": 0.8, "threshold": 0.95, "margin": 0.2},
    ]
    assert choose(sweep)["threshold"] == 0.8


def test_by_method_and_errors(rows):
    report = by_method(rows, 0.0, 0.0)
    assert report["exact"]["matched"] == 1 and report["exact"]["precision"] == 1.0
    assert report["fuzzy"]["matched"] == 2 and report["fuzzy"]["correct"] == 1
    assert report["compact"]["precision"] is None
    err = errors(rows, 0.0, 0.0)
    assert err["wrong"] == 1
    assert err["kinds"] == {"true company not listed (false merge)": 1}


def test_blocking_stats(rows):
    stats = blocking(rows, entries=4)
    assert stats["fuzzy_queries"] == 3  # acme, duke, zzz
    assert stats["listed_fuzzy_queries"] == 1 and stats["pair_completeness"] == 1.0
    assert 0 <= stats["reduction_ratio"] < 1


def test_sample_pairs_and_precision(rows):
    pairs = sample_pairs(rows, 0.0, 0.0, 100, "all", seed=1)
    assert len(pairs) == 3  # fewer matches than asked: take them all
    assert pairs == sample_pairs(rows, 0.0, 0.0, 100, "all", seed=1)
    summary = precision_of(pairs)
    assert (summary["pairs"], summary["correct"]) == (3, 2)
    assert summary["wilson_ci95"][0] < summary["precision"] < summary["wilson_ci95"][1]


def test_diagnostics_bands_and_reach(rows):
    resolver = Resolver(REFERENCE, max_df=3)
    d = diagnostics(rows, resolver)
    assert (d["listed_not_exact"], d["share_a_token_with_true_name"]) == (1, 1)
    assert d["reachable_via_rare_token"] == 1
    assert [b["score"] for b in d["fuzzy_score_bands"]][-1] == [0.9, 1.0]
    assert sum(b["queries"] for b in d["fuzzy_score_bands"]) == sum(
        r["method"] == "fuzzy" and r["score"] >= 0.5 for r in rows
    )
