import csv
import io
import zipfile

import pytest

from filinglens.extract.entities import (
    Resolver,
    compact,
    dedup_report,
    master_row,
    normalize,
    read_fsds_names,
    ticker_rows,
)


@pytest.mark.parametrize(
    ("raw", "tokens"),
    [
        ("Apple Inc.", ("APPLE",)),
        ("APPLE INC /CA/", ("APPLE",)),
        ("COCA COLA CO", ("COCA", "COLA")),
        ("The Coca-Cola Company", ("COCA", "COLA")),
        ("PROCTER & GAMBLE Co", ("PROCTER", "AND", "GAMBLE")),
        ("JPMORGAN CHASE & CO", ("JPMORGAN", "CHASE", "AND")),
        ("J.P. Morgan Chase & Co.", ("JP", "MORGAN", "CHASE", "AND")),
        ("MOODY'S CORP /DE/", ("MOODYS",)),
        ("XYZ HLDGS INC\\DE\\", ("XYZ", "HOLDINGS")),
        ("Nestlé S.A.", ("NESTLE",)),
        ("THE COMPANY", ("THE", "COMPANY")),  # nothing but filler: keep the words
    ],
)
def test_normalize(raw, tokens):
    assert normalize(raw) == tokens


def test_state_tag_alone_is_not_stripped_to_nothing():
    assert normalize("/DE/") == ("DE",)


def test_compact_joins_split_initials():
    assert compact(normalize("J P MORGAN CHASE & CO")) == compact(normalize("JPMORGAN CHASE & CO"))


REFERENCE = [
    (1, "Delta Air Lines, Inc."),
    (2, "Delta Apparel, Inc."),
    (3, "Walmart Inc."),
    (4, "WAL-MART STORES INC"),  # another CIK with a near-identical name, for the margin rule
    (5, "Duke Energy CORP"),
    (6, "First Financial Corp"),
    (7, "First Financial Bankshares Inc"),
    (8, "Acme Widgets Holdings Inc"),
    (8, "Acme Widgets Inc"),  # a second alias of the same CIK
    (9, "Twin Name Inc"),
    (10, "TWIN NAME CORP"),  # normalizes to the same key as CIK 9
]


@pytest.fixture
def resolver():
    return Resolver(REFERENCE, max_df=3)


def test_exact_after_normalization(resolver):
    m = resolver.resolve("DELTA AIR LINES INC /DE/")
    assert (m.cik, m.method, m.name, m.score) == (1, "exact", "Delta Air Lines, Inc.", 1.0)


def test_compact_key_catches_spacing(resolver):
    m = resolver.resolve("WALMART STORES INC")  # "WAL MART STORES" vs "WALMART STORES"
    assert (m.cik, m.method) == (4, "compact")


def test_ambiguous_exact_key_returns_no_cik(resolver):
    m = resolver.resolve("Twin Name Company")
    assert (m.cik, m.method) == (None, "ambiguous")


def test_fuzzy_match_above_threshold(resolver):
    m = resolver.resolve("Delta Air Lines Holdings", threshold=0.6)
    assert (m.cik, m.method) == (1, "fuzzy")
    assert 0.6 <= m.score < 1.0 and m.margin > 0 and m.candidates >= 1


def test_fuzzy_below_threshold_is_none_but_reports_best(resolver):
    m = resolver.resolve("Delta Air Lines Holdings", threshold=0.99)
    assert (m.cik, m.method, m.name) == (None, "none", "Delta Air Lines, Inc.")


def test_margin_rule_rejects_close_runner_up(resolver):
    # "WALMART" alone scores the same against CIK 3 and is far from CIK 4: accepted.
    assert resolver.resolve("Walmart Stores Holdings", threshold=0.3, min_margin=0.0).cik in {3, 4}
    # Demanding a large margin over the other Walmart CIK rejects it.
    m = resolver.resolve("Walmart Group", threshold=0.1, min_margin=0.9)
    assert m.method == "none"


def test_best_alias_per_cik_counts_once(resolver):
    m = resolver.resolve("Acme Widgets Group", threshold=0.3)
    assert m.cik == 8
    assert m.margin == m.score  # both aliases are CIK 8, so there is no runner-up CIK


def test_blocking_common_tokens_create_no_candidates():
    # With max_df=1, "FIRST" and "FINANCIAL" (in two names each) can't create candidates.
    r = Resolver(REFERENCE, max_df=1)
    m = r.resolve("First Financial Group")  # GROUP is in no reference name
    assert (m.method, m.candidates) == ("none", 0)


def test_unseen_tokens_pull_the_score_down(resolver):
    # CAROLINAS is in no reference name, so it gets the highest weight: 2 of 3 equal weights.
    near = resolver.resolve("Duke Energy Carolinas", threshold=0.0, min_margin=0.0)
    assert (near.cik, near.score) == (5, pytest.approx(2 / 3, abs=1e-4))
    # At the default threshold the subsidiary is not merged into its parent.
    assert resolver.resolve("Duke Energy Carolinas").method == "none"


def test_ticker_rows_one_per_ticker():
    data = {
        "1": {"cik_str": 1652044, "ticker": "GOOG", "title": "Alphabet Inc."},
        "0": {"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet Inc."},
    }
    assert [r["ticker"] for r in ticker_rows(data)] == ["GOOGL", "GOOG"]


def fsds_zip(path, rows):
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    writer.writerow(["adsh", "cik", "name", "form", "filed"])
    writer.writerows(rows)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("sub.txt", buf.getvalue())


def test_read_fsds_names_aggregates_across_quarters(tmp_path):
    fsds_zip(
        tmp_path / "2019q1.zip",
        [["a1", "1326801", "FACEBOOK INC", "10-K", "20190131"], ["a2", "5", "X CORP", "10-Q", ""]],
    )
    fsds_zip(
        tmp_path / "2022q1.zip",
        [
            ["a3", "1326801", "META PLATFORMS, INC.", "10-K", "20220203"],
            ["a4", "1326801", "META PLATFORMS, INC.", "10-Q", "20220427"],
            ["a5", "", "NO CIK", "10-K", "20220101"],
        ],
    )
    rows = read_fsds_names(sorted(tmp_path.glob("*.zip")))
    meta = [r for r in rows if r["cik"] == 1326801]
    assert [r["name"] for r in meta] == ["FACEBOOK INC", "META PLATFORMS, INC."]
    assert meta[1] | {} == {
        "cik": 1326801,
        "name": "META PLATFORMS, INC.",
        "first_filed": "20220203",
        "last_filed": "20220427",
        "n_filings": 2,
    }
    assert {r["cik"] for r in rows} == {1326801, 5}


def test_master_row_collects_identifiers_and_variants():
    company = {
        "cik": "1326801",
        "name": "Meta Platforms, Inc.",
        "sector": "services",
        "sic": "7370",
    }
    subs = {
        "name": "Meta Platforms, Inc.",
        "tickers": ["META"],
        "exchanges": ["Nasdaq", ""],
        "sic": "7370",
        "ein": "201665019",
        "stateOfIncorporation": "DE",
        "fiscalYearEnd": "1231",
        "formerNames": [{"name": "FACEBOOK INC", "from": "2004", "to": "2021"}],
    }
    row = master_row(company, subs, ["META"], ["FACEBOOK INC", "META PLATFORMS, INC."])
    assert row["tickers"] == "META" and row["exchanges"] == "Nasdaq"
    assert row["former_names"] == "FACEBOOK INC"
    assert row["n_name_variants"] == 2  # META PLATFORMS and FACEBOOK
    assert row["in_ticker_file"] is True


def test_dedup_report_counts():
    tickers = [
        {"cik": 1, "ticker": "GOOG", "name": "Alphabet Inc."},
        {"cik": 1, "ticker": "GOOGL", "name": "Alphabet Inc."},
        {"cik": 2, "ticker": "TW1", "name": "Twin Name Inc"},
        {"cik": 3, "ticker": "TW2", "name": "TWIN NAME CORP"},
    ]
    fsds = [
        {"cik": 4, "name": "FACEBOOK INC"},
        {"cik": 4, "name": "META PLATFORMS, INC."},
        {"cik": 4, "name": "Meta Platforms Inc"},  # same normalized name: not a rename
    ]
    master = [{"in_ticker_file": True, "former_names": "X", "tickers": "A|B", "n_name_variants": 2}]
    report = dedup_report(tickers, fsds, master)
    assert report["ticker_file"]["ciks_with_multiple_tickers"] == 1
    assert report["ticker_file_name_collisions"]["normalized_names_with_multiple_ciks"] == 1
    assert report["fsds"]["ciks_with_multiple_normalized_names"] == 1
    assert report["master"]["with_multiple_tickers"] == 1
