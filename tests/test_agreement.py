"""cohen_kappa (hand-written) against known values and sklearn; the agreement report."""

import json
import math
import random

import pytest

from filinglens.extract.agreement import agreement_report, cohen_kappa, main, paired


def test_textbook_example():
    # 50 items: yes/yes 20, yes/no 5, no/yes 10, no/no 15 -> p_o = 0.70, p_e = 0.50, kappa = 0.40
    a = ["yes"] * 25 + ["no"] * 25
    b = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_perfect_agreement_and_worse_than_chance():
    assert cohen_kappa([1, 2, 3, 1], [1, 2, 3, 1]) == pytest.approx(1.0)
    assert cohen_kappa(["x", "y", "x", "y"], ["y", "x", "y", "x"]) == pytest.approx(-1.0)


def test_label_used_by_only_one_rater():
    a, b = ["a", "a", "b", "b"], ["a", "c", "b", "b"]
    assert -1.0 <= cohen_kappa(a, b) < 1.0


def test_weights_give_partial_credit_to_near_misses():
    a = [1, 2, 3, 1, 2, 3, 1, 2]
    near = [2, 2, 3, 1, 3, 3, 1, 1]  # misses by one step
    far = [3, 2, 3, 1, 1, 3, 1, 3]  # some misses by two steps
    assert cohen_kappa(a, near, "quadratic") > cohen_kappa(a, far, "quadratic")
    assert cohen_kappa(a, near, "quadratic") > cohen_kappa(a, near)


def test_constant_identical_raters_give_nan():
    assert math.isnan(cohen_kappa([2, 2, 2], [2, 2, 2]))


@pytest.mark.parametrize("bad", [([1, 2], [1]), ([], [])])
def test_bad_lengths_raise(bad):
    with pytest.raises(ValueError):
        cohen_kappa(*bad)


def test_unknown_weights_raise():
    with pytest.raises(ValueError):
        cohen_kappa([1, 2], [1, 2], weights="cubic")


@pytest.mark.parametrize("weights", [None, "linear", "quadratic"])
def test_matches_sklearn(weights):
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = random.Random(7)
    for _ in range(20):
        n = rng.randint(5, 80)
        a = [rng.randint(1, 4) for _ in range(n)]
        b = [x if rng.random() < 0.6 else rng.randint(1, 4) for x in a]
        expected = sklearn_metrics.cohen_kappa_score(a, b, weights=weights)
        assert cohen_kappa(a, b, weights) == pytest.approx(expected, abs=1e-12)


def answer(pid, category, severity=2, year=2020, valid=True):
    base = {"para_id": pid, "valid": valid, "category": category}
    return base | {"severity": severity, "first_seen_year": year}


def test_paired_keeps_only_both_valid():
    primary = [answer("a", "people"), answer("b", "people"), answer("c", "people", valid=False)]
    check = [answer("a", "people"), answer("b", "people", valid=False), answer("c", "people")]
    assert [p["para_id"] for p, _ in paired(primary, check)] == ["a"]


def test_report_and_main(tmp_path):
    cats = ["people", "macro_market", "people", "regulatory_legal", "macro_market", "people"]
    other = ["people", "competition_demand", "people", "regulatory_legal", "macro_market", "people"]
    primary = [answer(str(i), c, severity=1 + i % 3) for i, c in enumerate(cats)]
    check = [answer(str(i), c, severity=1 + (i + 1) % 3) for i, c in enumerate(other)]
    report = agreement_report(paired(primary, check))
    assert report["category"]["n"] == 6 and report["category"]["observed_agreement"] == 0.833
    assert report["top_category_disagreements"] == [
        {"pair": ["competition_demand", "macro_market"], "count": 1}
    ]
    raw = tmp_path / "raw"
    raw.mkdir()
    files = {"ibm-granite-granite-4-1-3b.jsonl": primary, "qwen-qwen3-4b-awq.jsonl": check}
    for name, rows in files.items():
        (raw / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    main(["--raw", str(raw), "--out", str(tmp_path / "kappa.json")])
    saved = json.loads((tmp_path / "kappa.json").read_text())
    assert saved["what"].startswith("agreement between two model labelers")
    assert len(saved["category"]["ci95"]) == 2
    # every first-seen year is 2020 in this fixture: kappa is undefined, saved as null, not NaN
    assert saved["first_seen_year_quadratic"]["kappa"] is None
    assert saved["first_seen_year_quadratic"]["ci95"] == [None, None]
