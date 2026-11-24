import json
import math
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from filinglens.retrieval import peers as pr

SIC = {1: "3571", 2: "3572", 3: "3674", 4: "3674", 5: "6021", 6: "3661"}


def test_peer_sets_by_level():
    four = pr.peer_sets(SIC, 4)
    assert four[3] == {4} and four[1] == set()
    three = pr.peer_sets(SIC, 3)
    assert three[1] == {2} and three[5] == set()
    assert pr.peer_sets(SIC, 2)[3] == {4, 6} and pr.peer_sets(SIC, 2)[5] == set()


def test_choose_sic_level_takes_the_finest_level_with_enough_coverage():
    level, table = pr.choose_sic_level(SIC, min_coverage=0.6)
    assert level == 3  # 4 digits: 2/6 have a peer; 3 digits: 4/6; 2 digits: 5/6
    assert table["4"]["companies_with_peer"] == 2 and table["3"]["coverage"] == 0.6667
    assert table["2"]["groups"] == 3 and table["2"]["max_peers"] == 2
    assert pr.choose_sic_level(SIC, min_coverage=0.3)[0] == 4
    with pytest.raises(ValueError, match="coverage"):
        pr.choose_sic_level(SIC, min_coverage=0.9)


def test_random_hit_rate_matches_brute_force():
    assert pr.random_hit_rate(10, 1, 5) == pytest.approx(0.5)
    assert pr.random_hit_rate(10, 0, 5) == 0.0
    assert pr.random_hit_rate(7, 3, 5) == 1.0  # only 4 non-peers: 5 draws must hit
    for n, p in [(9, 2), (8, 3), (12, 1)]:
        draws = list(combinations(range(n), 5))
        brute = sum(any(i < p for i in d) for d in draws) / len(draws)
        assert pr.random_hit_rate(n, p, 5) == pytest.approx(brute)


def test_latest_filing_mask_takes_the_latest_year_then_the_last_accession():
    chunks = pd.DataFrame(
        {
            "cik": [1, 1, 1, 2, 2, 2],
            "period_year": [2023, 2024, 2024, 2025, 2025, 2022],
            "accession": ["a23", "a24", "a24", "b25", "b25z", "b22"],
        }
    )
    assert pr.latest_filing_mask(chunks).tolist() == [False, True, True, False, True, False]


def test_company_vectors_are_renormalized_means_in_order():
    ciks = np.array([7, 7, 9])
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [3.0, 4.0]], dtype=np.float32)
    out = pr.company_vectors(ciks, vectors, [9, 7])
    assert out[0] == pytest.approx([0.6, 0.8])
    assert out[1] == pytest.approx([math.sqrt(0.5)] * 2)
    with pytest.raises(ValueError, match="no chunk"):
        pr.company_vectors(ciks, vectors, [8])


def test_ratio_ranks_use_the_latest_year_and_fill_missing_with_the_median():
    rows = []
    for cik, year, margin in [(1, 2024, 9.0), (1, 2025, 0.1), (2, 2025, 0.3), (3, 2025, np.nan)]:
        row = dict.fromkeys(pr.RATIO_FEATURES, np.nan)
        row.update(cik=cik, value_date=pd.Timestamp(f"{year}-12-31"), net_margin=margin)
        rows.append(row)
    ranks, info = pr.ratio_ranks(pd.DataFrame(rows), [1, 2, 3, 4])
    col = list(pr.RATIO_FEATURES).index("net_margin")
    assert ranks[:, col].tolist() == [0.5, 1.0, 0.5, 0.5]  # 1 uses 0.1 (2025), not 9.0
    assert (ranks[:, col + 1] == 0.5).all()  # a ratio missing everywhere carries no signal
    assert info["companies_without_ratios"] == 2  # cik 3 (all NaN) and cik 4 (no row)
    assert info["latest_ratio_year"] == {"2025": 3}


def test_similarities_and_offdiagonal_zscore():
    vecs = np.eye(3)
    assert pr.text_similarity(vecs).tolist() == np.eye(3).tolist()
    ranks = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
    sim = pr.ratio_similarity(ranks)
    assert sim[0, 1] == -5.0 and sim[0, 2] == 0.0
    z = pr.zscore_offdiag(sim)
    off = z[~np.eye(3, dtype=bool)]
    assert off.mean() == pytest.approx(0.0) and off.std() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="all equal"):
        pr.zscore_offdiag(np.ones((3, 3)))


def test_top_k_skips_self_and_breaks_ties_by_index():
    score = np.array([[9.0, 1.0, 1.0, 0.0], [5.0, 9.0, 5.0, 6.0], [0.0] * 4, [0.0] * 4])
    top = pr.top_k(score, k=2)
    assert top[0].tolist() == [1, 2]  # never itself, even with the highest score
    assert top[1].tolist() == [3, 0]  # 0 and 2 tie at 5.0: the lower index wins
    assert top[2].tolist() == [0, 1]
    only = pr.top_k(score, k=1, candidates=[2, 3])  # e.g. the selection pool
    assert only[:, 0].tolist() == [2, 3, 3, 2]  # never a company outside the pool, nor itself


def test_restrict_keeps_only_allowed_peers():
    peers = {1: {2, 3}, 2: {1}, 3: {1}}
    assert pr.restrict(peers, {1, 2}) == {1: {2}, 2: {1}, 3: {1}}


def test_hits_and_scorable_companies():
    order = [10, 20, 30, 40]
    peers = {10: {20}, 20: {10}, 30: set(), 40: {10}}
    split_of = {10: "test", 20: "train", 30: "test", 40: "test"}
    queries = pr.scorable(order, peers, split_of, {"test"})
    assert queries == [0, 3]  # 30 has no peer: it can't be scored
    top = np.array([[1, 2], [0, 2], [0, 1], [1, 2]])
    assert pr.hits_at_k(top, order, peers, queries).tolist() == [1.0, 0.0]


def test_choose_alpha_picks_the_better_signal_and_breaks_ties_toward_half():
    order = list(range(6))
    peers = {i: {i ^ 1} for i in order}  # pairs (0,1), (2,3), (4,5)
    good = np.full((6, 6), -1.0)
    for i in order:
        good[i, i ^ 1] = 5.0
    noise = np.random.default_rng(0).normal(size=(6, 6))
    alpha, grid = pr.choose_alpha(good, noise, order, peers, order, alphas=(0.0, 0.5, 1.0))
    assert grid["1.0"] == 1.0 and alpha in (0.5, 1.0)
    if grid["0.5"] == 1.0:
        assert alpha == 0.5  # equal best: the alpha nearest 0.5 wins
    alpha, grid = pr.choose_alpha(good, good, order, peers, order, alphas=(0.0, 0.3, 0.8, 1.0))
    assert set(grid.values()) == {1.0} and alpha == 0.3  # |0.3 - 0.5| < |0.8 - 0.5|


# ---- the two steps on a tiny fake universe ----------------------------------------------------


@pytest.fixture
def fake_inputs(tmp_path, monkeypatch):
    """12 companies in 4 SIC groups; each company's text vector points at its group's axis."""
    rng = np.random.default_rng(1)
    ciks = list(range(101, 113))
    sic = {c: ["2834", "3674", "6021", "7372"][i % 4] for i, c in enumerate(ciks)}
    split = {c: ["train", "dev", "test"][i % 3] for i, c in enumerate(ciks)}
    chunk_rows, vectors = [], []
    for i, cik in enumerate(ciks):
        for year in (2023, 2024):
            for part in range(3):
                chunk_rows.append(
                    {
                        "chunk_id": f"{cik}-{year}:{part}#0",
                        "cik": cik,
                        "period_year": year,
                        "accession": f"{cik}-{year}",
                    }
                )
                v = rng.normal(scale=0.05, size=8)
                v[i % 4] += 1.0 if year == 2024 else 0.0  # only the latest 10-K carries signal
                vectors.append(v / np.linalg.norm(v))
    chunks = pd.DataFrame(chunk_rows)
    emb = tmp_path / "emb"
    emb.mkdir()
    np.save(emb / "chunks.npy", np.array(vectors, dtype=np.float32))
    (emb / "chunk_ids.txt").write_text("".join(f"{c}\n" for c in chunks["chunk_id"]))
    universe = pd.DataFrame(
        {"cik": ciks + [999], "ticker": "X", "sic": [sic[c] for c in ciks] + ["1311"]}
    )
    universe.to_csv(tmp_path / "universe.csv", index=False)
    pd.DataFrame({"cik": ciks, "split": [split[c] for c in ciks]}).to_csv(
        tmp_path / "splits.csv", index=False
    )
    ratio_rows = []
    for c in ciks:
        row = {f: float(rng.normal()) for f in pr.RATIO_FEATURES}
        ratio_rows.append({"cik": c, "value_date": pd.Timestamp("2024-12-31"), **row})
    ratios = pd.DataFrame(ratio_rows)
    tables = {"chunks.parquet": chunks, "ratios.parquet": ratios}

    def read_parquet(path, columns=None):
        df = tables[path.name]
        return df[columns] if columns else df

    monkeypatch.setattr(pr.pd, "read_parquet", read_parquet)
    monkeypatch.setattr(pr, "log_run", lambda *a, **k: "run-id")
    bench = tmp_path / "bench"
    bench.mkdir()
    common = [
        "--note",
        "test",
        "--chunks",
        str(tmp_path / "chunks.parquet"),
        "--emb-dir",
        str(emb),
        "--universe",
        str(tmp_path / "universe.csv"),
        "--splits",
        str(tmp_path / "splits.csv"),
        "--ratios",
        str(tmp_path / "ratios.parquet"),
        "--select",
        str(bench / "peers_select.json"),
        "--bench",
        str(bench),
    ]
    return common, bench


def test_select_then_test(fake_inputs, monkeypatch, capsys):
    common, bench = fake_inputs
    pr.main(["select", *common])
    chosen = json.loads((bench / "peers_select.json").read_text())
    assert chosen["sic_level"] == 4 and chosen["selection_companies_scored"] == 8
    assert chosen["selection_pool"] == 8 and chosen["selection_companies_without_peer"] == 0
    assert chosen["data"]["latest_10k_chunks"] == 36  # 12 companies x 3 chunks of 2024
    assert chosen["data"]["latest_10k_period_year"] == {"2024": 12}
    assert chosen["hit_rate_by_alpha"]["1.0"] == 1.0  # text alone finds every group

    with pytest.raises(PermissionError):
        pr.main(["test", *common])
    monkeypatch.setattr(pr, "rule_is_committed", lambda path: False)
    with pytest.raises(SystemExit, match="committed"):
        pr.main(["test", "--allow-test", *common])

    monkeypatch.setattr(pr, "rule_is_committed", lambda path: True)
    pr.main(["test", "--allow-test", *common])
    out = json.loads((bench / "peers.json").read_text())
    assert out["test_companies_scored"] == 4 and out["alpha"] == chosen["alpha"]
    assert out["results"]["text_only"]["hit_rate_at_5"] == 1.0
    # 11 candidates, 2 peers each, 5 draws: 1 - C(9,5)/C(11,5)
    assert out["results"]["combined"]["random_baseline"] == round(1 - 126 / 462, 4)
    assert out["mlflow_run_id"] == "run-id"
    assert '"step": "test"' in capsys.readouterr().out


def test_misaligned_embeddings_are_refused(fake_inputs):
    common, _ = fake_inputs
    emb = __import__("pathlib").Path(common[common.index("--emb-dir") + 1])
    ids = (emb / "chunk_ids.txt").read_text().splitlines()
    (emb / "chunk_ids.txt").write_text("".join(f"{i}\n" for i in reversed(ids)))
    with pytest.raises(ValueError, match="row for row"):
        pr.main(["select", *common])


def test_select_never_sees_a_test_company(fake_inputs, monkeypatch):
    common, _ = fake_inputs
    seen = {}

    def spy(z_text, z_ratio, order, peers, queries, candidates=None):
        seen.update(order=order, peers=peers, queries=queries, candidates=candidates)
        return 0.5, {"0.5": 1.0}

    monkeypatch.setattr(pr, "choose_alpha", spy)
    pr.main(["select", *common])
    split = dict(zip(range(101, 113), ["train", "dev", "test"] * 4, strict=True))
    order = seen["order"]
    assert all(split[order[i]] != "test" for i in seen["candidates"])
    assert all(split[order[i]] != "test" for i in seen["queries"])
    assert all(split[p] != "test" for i in seen["queries"] for p in seen["peers"][order[i]])
