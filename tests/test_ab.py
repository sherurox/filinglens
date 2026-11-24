import json
import subprocess

import numpy as np
import pytest

from filinglens.retrieval import ab


def test_vectorized_bootstrap_equals_a_plain_loop():
    d = np.random.default_rng(3).normal(0.05, 0.2, size=37)
    rng = np.random.default_rng(0)
    loop = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(500)])
    assert np.array_equal(ab.bootstrap_means(d, n_boot=500, seed=0), loop)


def test_cluster_bootstrap_with_singleton_clusters_is_the_query_bootstrap():
    d = np.random.default_rng(4).normal(0.0, 1.0, size=25)
    plain = ab.bootstrap_means(d, n_boot=300, seed=7)
    singletons = ab.bootstrap_means(d, n_boot=300, seed=7, clusters=list(range(25)))
    assert np.allclose(plain, singletons)


def test_cluster_bootstrap_widens_the_ci_when_companies_move_together():
    rng = np.random.default_rng(8)
    clusters = np.repeat(np.arange(20), 10)
    d = 0.05 + rng.normal(0, 0.2, 20)[clusters] + rng.normal(0, 0.05, 200)  # strong company effect
    by_query = ab.paired_bootstrap(d, n_boot=2000)
    by_company = ab.paired_bootstrap(d, n_boot=2000, clusters=clusters)

    def width(r):
        return r["ci95"][1] - r["ci95"][0]

    assert width(by_company) > 2 * width(by_query)
    assert by_company["clusters"] == 20 and by_query["clusters"] is None


def test_icc_estimator_recovers_a_known_value():
    rng = np.random.default_rng(9)
    clusters = np.repeat(np.arange(400), 4)
    d = rng.normal(0, np.sqrt(0.3), 400)[clusters] + rng.normal(0, np.sqrt(0.7), 1600)
    assert ab.icc_oneway(d, clusters) == pytest.approx(0.3, abs=0.05)
    independent = rng.normal(0, 1, 1600)
    assert ab.icc_oneway(independent, clusters) < 0.05  # clipped at 0, never negative
    with pytest.raises(ValueError):
        ab.icc_oneway([1.0, 2.0], [0, 0])


def test_n_clustered():
    assert ab.n_clustered(100, icc=0.0, n_clusters=50) == 100
    # n = 100 * (1 + (n/50 - 1) * 0.1)  ->  n = 100 * 0.9 / (1 - 100 * 0.1 / 50) = 112.5
    assert ab.n_clustered(100, icc=0.1, n_clusters=50) == 113
    assert ab.n_clustered(100, icc=0.5, n_clusters=50) is None  # ICC >= C / n_t: impossible


def test_icc_from_bootstraps():
    query = {"n": 200, "clusters": None, "ci95": [0.0, 0.10]}
    cluster = {"n": 200, "clusters": 100, "ci95": [0.0, 0.10 * 1.2**0.5]}  # design effect 1.2
    assert ab.icc_from_bootstraps(cluster, query) == pytest.approx(0.2)  # (1.2 - 1) / (2 - 1)
    narrower = {**cluster, "ci95": [0.0, 0.09]}
    assert ab.icc_from_bootstraps(narrower, query) == 0.0  # clipped, never negative


def test_power_clustered_matches_the_planning_numbers():
    # the planning table given in chat (sd 0.2752, 75 companies, delta 0.03), now from code
    assert ab.power_clustered(0.2752, 0.1434, 0.03, 75, 15) == pytest.approx(0.56, abs=0.01)
    assert ab.power_clustered(0.2752, 0.03, 0.03, 75, 15) == pytest.approx(0.87, abs=0.01)
    # no clustering: the plain normal-approximation power with n = C * m
    from scipy.stats import norm

    se = 0.2752 / (75 * 9) ** 0.5
    plain = norm.cdf(0.03 / se - norm.ppf(0.975)) + norm.cdf(-0.03 / se - norm.ppf(0.975))
    assert ab.power_clustered(0.2752, 0.0, 0.03, 75, 9) == pytest.approx(plain)


def test_simulated_power_clustered():
    assert ab.simulated_power_clustered(0.2, 0.2, 0.3, 30, 3, n_sim=50, n_boot=300) == 1.0
    assert ab.simulated_power_clustered(0.2, 0.2, 0.0, 30, 3, n_sim=200, n_boot=300) <= 0.1


def test_paired_bootstrap_known_answers():
    same = ab.paired_bootstrap([0.1] * 20, n_boot=200)
    assert same["mean_diff"] == 0.1 and same["ci95"] == [0.1, 0.1]
    assert same["share_not_better"] == 0.0 and same["effect_size_dz"] is None  # sd = 0
    mixed = ab.paired_bootstrap([0.2, -0.1, 0.0, 0.3], n_boot=200)
    assert (mixed["queries_better"], mixed["queries_worse"], mixed["queries_tied"]) == (2, 1, 1)
    assert mixed["effect_size_dz"] == pytest.approx(
        0.1 / np.std([0.2, -0.1, 0.0, 0.3], ddof=1), abs=1e-4
    )


def test_bootstrap_ci_covers_the_true_mean_about_95_percent_of_the_time():
    rng = np.random.default_rng(11)
    covered = 0
    for i in range(400):
        d = rng.normal(0.05, 0.2, size=60)
        low, high = ab.paired_bootstrap(d, n_boot=1000, seed=i)["ci95"]
        covered += low <= 0.05 <= high
    assert 0.91 <= covered / 400 <= 0.98, covered  # percentile CIs run a little narrow at n=60


def test_sample_size_matches_published_values():
    # one-sample / paired t, alpha 0.05 two-sided, power 0.8 (G*Power): d=0.5 -> 34, d=0.2 -> 199
    assert ab.n_t(sd=1.0, mde=0.5) == 34
    assert ab.n_t(sd=1.0, mde=0.2) == 199
    assert ab.n_normal(sd=1.0, mde=0.5) == 32  # the normal approximation is a little optimistic
    assert ab.power_t(34, 1.0, 0.5) >= 0.8 > ab.power_t(33, 1.0, 0.5)
    assert ab.n_t(sd=0.2, mde=0.1) == ab.n_t(sd=1.0, mde=0.5)  # only sd / mde matters


def test_mde_at_inverts_the_power():
    mde = ab.mde_at(34, 1.0)
    assert 0.48 < mde <= 0.5 and ab.power_t(34, 1.0, mde) == pytest.approx(0.8, abs=1e-6)
    for n in (10, 175, 1000, 5000):  # large n: scipy's noncentral t can give NaN; must not
        assert ab.power_t(n, 1.0, ab.mde_at(n, 1.0)) == pytest.approx(0.8, abs=1e-4)


def test_simulated_power_of_the_bootstrap_rule():
    d = np.random.default_rng(5).normal(0.0, 0.2, size=150)
    assert ab.simulated_power(d, effect=0.2, n=100, n_sim=100, n_boot=300) == 1.0
    assert ab.simulated_power(d, effect=0.0, n=100, n_sim=200, n_boot=300) <= 0.08  # ~2.5%


def test_decide():
    ok = {"ci95": [0.01, 0.08]}
    assert ab.decide(ok, p95_ms=280, budget_ms=500)["ship_rerank"] is True
    assert ab.decide(ok, p95_ms=600, budget_ms=500)["ship_rerank"] is False
    assert ab.decide({"ci95": [-0.01, 0.08]}, p95_ms=100, budget_ms=500)["ship_rerank"] is False


def test_differences():
    rows = [{"ndcg_after": 0.5, "ndcg_before": 0.2}, {"ndcg_after": 0.0, "ndcg_before": 0.1}]
    assert ab.differences(rows, "ndcg") == pytest.approx([0.3, -0.1])


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _world(tmp_path, monkeypatch):
    """A git repo with dev + test queries, per-query rerank results and a reranker_test.json."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ab, "log_run", lambda *a, **k: "run-ab")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    qdir, ret, bench = tmp_path / "q", tmp_path / "ret", tmp_path / "bench"
    for d in (qdir, ret, bench):
        d.mkdir()
    rng = np.random.default_rng(0)
    for split, n in (("dev", 40), ("test", 30)):
        ids = [f"{split}{i}" for i in range(n)]
        queries = [
            {"query_id": q, "cik": 1000 + j // 2} for j, q in enumerate(ids)
        ]  # 2 per company
        (qdir / f"queries.{split}.jsonl").write_text("".join(json.dumps(q) + "\n" for q in queries))
        rows = []
        for i in ids:
            before = float(rng.uniform(0, 0.6))
            row = {"query_id": i}
            for m in ab.METRICS:
                row[f"{m}_before"], row[f"{m}_after"] = before, before + float(rng.normal(0.1, 0.1))
            rows.append(row)
        (ret / f"rerank_{split}_per_query.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    (bench / "reranker_test.json").write_text(
        json.dumps({"metrics": {"ndcg_after": 0.5}, "rerank_latency": {"p95_ms": 280.0}})
    )
    common = ["--ret-dir", str(ret), "--query-dir", str(qdir), "--bench", str(bench),
              "--rule", "bench/ab_rule.json"]  # fmt: skip
    return common, bench


def test_pilot_then_test_with_a_committed_rule(tmp_path, monkeypatch):
    common, bench = _world(tmp_path, monkeypatch)
    ab.main(["pilot", "--mde", "0.08", "--latency-budget-ms", "500", *common])
    power = json.loads((bench / "ab_power.json").read_text())
    assert power["pilot_queries"] == 40 and power["pilot_companies"] == 20
    assert power["n_test_available"] == 30 and power["test_companies"] == 15
    assert power["n_required_t_independent"] >= power["n_required_normal_independent"]
    assert power["feasible_with_test_companies"] and power["n_required_clustered"] <= 30
    rule = json.loads((bench / "ab_rule.json").read_text())
    assert rule["n_test_basis"] == "power analysis"
    assert rule["n_test_required"] == power["n_required_clustered"]

    with pytest.raises(PermissionError):
        ab.main(["test", "--note", "x", *common])  # no --allow-test
    with pytest.raises(SystemExit, match="committed"):
        ab.main(["test", "--allow-test", "--note", "x", *common])  # rule not committed yet
    _git(tmp_path, "add", "bench/ab_rule.json")
    _git(tmp_path, "commit", "-q", "-m", "rule")
    ab.main(["test", "--allow-test", "--note", "x", *common])
    record = json.loads((bench / "ab_test.json").read_text())
    assert record["primary"]["ndcg_at_10"]["n"] == 30
    assert record["primary"]["ndcg_at_10"]["clusters"] == 15
    assert record["decision"]["ship_rerank"] is True  # mean +0.1, p95 280 <= 500
    assert record["mlflow_run_id"] == "run-ab"

    rule = bench / "ab_rule.json"
    rule.write_text(rule.read_text().replace('"mde": 0.08', '"mde": 0.5'))  # edited after commit
    with pytest.raises(SystemExit, match="committed"):
        ab.main(["test", "--allow-test", "--note", "x", *common])


def test_test_step_refuses_too_few_queries(tmp_path, monkeypatch):
    common, bench = _world(tmp_path, monkeypatch)
    ab.main(["pilot", "--mde", "0.08", "--latency-budget-ms", "500", *common])
    rule = json.loads((bench / "ab_rule.json").read_text())
    rule["n_test_required"] = 31  # one more than ret-test holds
    (bench / "ab_rule.json").write_text(json.dumps(rule))
    _git(tmp_path, "add", "bench/ab_rule.json")
    _git(tmp_path, "commit", "-q", "-m", "rule")
    with pytest.raises(SystemExit, match="at least 31"):
        ab.main(["test", "--allow-test", "--note", "x", *common])
