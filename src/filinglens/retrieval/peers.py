"""Peer recommender: filing text + financial ratios, scored against SIC peers (FL-M3 T11).

For each company, rank the other universe companies by
    score = alpha * z(text cosine) + (1 - alpha) * z(ratio similarity)
and count a hit if one of the top 5 shares the company's SIC code (hit-rate@5).

- Text: the mean of the company's bge-base chunk embeddings from its latest 10-K (Item 1A),
  normalized again; similarity = cosine.
- Ratios: the six FL-M1 ratios of the company's latest fiscal year, each turned into a percentile
  rank across the companies (heavy tails can't set the scale), a missing ratio = the median 0.5;
  similarity = minus the Euclidean distance between the rank vectors.
- Each similarity matrix is z-scored over its off-diagonal entries, so alpha weighs two signals on
  one scale. alpha = 1 is text only, alpha = 0 ratios only.
- SIC level: the finest of 4, 3 or 2 digits at which at least 80% of companies have one or more
  peers, fixed from the label counts alone before any score is computed. A company with no peer
  at that level can't be scored and is left out (counted in the record).
- Splits: the FL-M2 company splits (gold/splits.csv). alpha is chosen on train + dev (nothing is
  trained, so the larger set gives a less noisy choice); test is scored once. Candidates are
  always all other companies: a candidate's SIC code is only the label being scored.
- Baseline: the exact hit-rate of 5 random candidates, 1 - C(N - p, 5) / C(N, 5) for N
  candidates and p peers, averaged over the scored companies.

Two steps, like the A/B (choose, commit, then test once):

    uv run python -m filinglens.retrieval.peers select --note "..."
    git add bench/peers_select.json && git commit -m "..."
    uv run python -m filinglens.retrieval.peers test --allow-test --note "..."
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from filinglens.retrieval.ab import bootstrap_means, rule_is_committed
from filinglens.retrieval.config import CHUNKS, EMB_DIR
from filinglens.retrieval.runlog import log_run

K = 5
SIC_LEVELS = (4, 3, 2)
MIN_COVERAGE = 0.80
ALPHAS = tuple(round(a / 10, 1) for a in range(11))
RATIO_FEATURES = (
    "net_margin",
    "operating_margin",
    "liabilities_to_equity",
    "equity_multiplier",
    "revenue_growth",
    "assets_growth",
)
N_BOOT, SEED = 10_000, 0
UNIVERSE = Path("dbt/seeds/universe.csv")
SPLITS = Path("gold/splits.csv")
RATIOS = Path("data/features/annual_ratios.parquet")
SELECT = Path("bench/peers_select.json")


# ---- labels -----------------------------------------------------------------------------------


def peer_sets(sic_by_cik: dict[int, str], level: int) -> dict[int, set[int]]:
    """For each company, the other companies sharing the first `level` digits of its SIC."""
    groups: dict[str, set[int]] = {}
    for cik, sic in sic_by_cik.items():
        groups.setdefault(sic[:level], set()).add(cik)
    return {cik: groups[sic[:level]] - {cik} for cik, sic in sic_by_cik.items()}


def choose_sic_level(
    sic_by_cik: dict[int, str],
    levels: Sequence[int] = SIC_LEVELS,
    min_coverage: float = MIN_COVERAGE,
) -> tuple[int, dict]:
    """The finest level where >= min_coverage of companies have a peer, and the counts per level."""
    table = {}
    for level in levels:
        sizes = sorted(len(p) for p in peer_sets(sic_by_cik, level).values())
        with_peer = sum(s > 0 for s in sizes)
        table[str(level)] = {
            "groups": len({sic[:level] for sic in sic_by_cik.values()}),
            "companies_with_peer": with_peer,
            "coverage": round(with_peer / len(sizes), 4),
            "median_peers": float(np.median(sizes)),
            "max_peers": sizes[-1],
        }
    for level in sorted(levels, reverse=True):
        if table[str(level)]["coverage"] >= min_coverage:
            return level, table
    raise ValueError(f"no SIC level in {list(levels)} gives {min_coverage:.0%} coverage")


def random_hit_rate(n_candidates: int, n_peers: int, k: int = K) -> float:
    """P(at least one peer among k candidates drawn without replacement from n_candidates)."""
    if n_peers <= 0:
        return 0.0
    if n_candidates - n_peers < k:
        return 1.0
    return 1.0 - math.comb(n_candidates - n_peers, k) / math.comb(n_candidates, k)


# ---- features ---------------------------------------------------------------------------------


def latest_filing_mask(chunks: pd.DataFrame) -> np.ndarray:
    """Rows of each company's latest 10-K: its highest period_year, then the last accession."""
    key = chunks[["cik", "period_year", "accession"]]
    latest = key.sort_values(["cik", "period_year", "accession"]).groupby("cik").tail(1)
    chosen = set(latest["accession"])
    return chunks["accession"].isin(chosen).to_numpy()


def company_vectors(ciks: np.ndarray, vectors: np.ndarray, order: Sequence[int]) -> np.ndarray:
    """Mean of each company's row vectors (rows labeled by `ciks`), renormalized, in `order`."""
    out = np.empty((len(order), vectors.shape[1]), dtype=np.float64)
    for i, cik in enumerate(order):
        rows = np.flatnonzero(ciks == cik)
        if len(rows) == 0:
            raise ValueError(f"company {cik} has no chunk in its latest 10-K")
        mean = np.asarray(vectors[rows], dtype=np.float64).mean(axis=0)
        out[i] = mean / np.linalg.norm(mean)
    return out


def ratio_ranks(ratios: pd.DataFrame, order: Sequence[int]) -> tuple[np.ndarray, dict]:
    """Latest-year ratios as percentile ranks across `order` (missing = 0.5), plus coverage."""
    latest = ratios.sort_values(["cik", "value_date"]).groupby("cik").tail(1).set_index("cik")
    table = latest.reindex(list(order))[list(RATIO_FEATURES)].astype("float64")
    ranks = table.rank(pct=True).fillna(0.5).to_numpy()
    info = {
        "companies_without_ratios": int(table.isna().all(axis=1).sum()),
        "missing_share_per_feature": {
            f: round(float(table[f].isna().mean()), 4) for f in RATIO_FEATURES
        },
        "latest_ratio_year": dict(
            sorted(
                Counter(
                    latest.reindex(list(order))["value_date"]
                    .dt.year.dropna()
                    .astype(int)
                    .astype(str)
                ).items()
            )
        ),
    }
    return ranks, info


def text_similarity(company_vecs: np.ndarray) -> np.ndarray:
    return company_vecs @ company_vecs.T


def ratio_similarity(ranks: np.ndarray) -> np.ndarray:
    diff = ranks[:, None, :] - ranks[None, :, :]
    return -np.sqrt((diff**2).sum(axis=2))


def zscore_offdiag(sim: np.ndarray) -> np.ndarray:
    """Standardize a similarity matrix by the mean and sd of its off-diagonal entries."""
    off = sim[~np.eye(len(sim), dtype=bool)]
    sd = off.std()
    if sd == 0:
        raise ValueError("off-diagonal similarities are all equal")
    return (sim - off.mean()) / sd


# ---- ranking and scoring ----------------------------------------------------------------------


def top_k(score: np.ndarray, k: int = K) -> np.ndarray:
    """Indices of each row's k best other companies; ties keep index order (stable sort)."""
    s = score.astype(np.float64, copy=True)
    np.fill_diagonal(s, -np.inf)
    return np.argsort(-s, axis=1, kind="stable")[:, :k]


def hits_at_k(
    top: np.ndarray, order: Sequence[int], peers: dict[int, set[int]], queries: Sequence[int]
) -> np.ndarray:
    """1.0 / 0.0 per query company (by row index) whose peers are non-empty."""
    out = []
    for q in queries:
        wanted = peers[order[q]]
        out.append(float(any(order[j] in wanted for j in top[q])))
    return np.array(out)


def scorable(
    order: Sequence[int], peers: dict[int, set[int]], split_of: dict[int, str], splits: set[str]
) -> list[int]:
    """Row indices of companies in `splits` that have at least one peer."""
    return [i for i, cik in enumerate(order) if split_of[cik] in splits and peers[cik]]


def summarize(
    hits: np.ndarray, order: Sequence[int], peers: dict[int, set[int]], queries: Sequence[int]
) -> dict:
    means = bootstrap_means(hits, N_BOOT, SEED)
    low, high = np.percentile(means, [2.5, 97.5])
    n_candidates = len(order) - 1
    baseline = np.mean([random_hit_rate(n_candidates, len(peers[order[q]])) for q in queries])
    return {
        "companies": int(len(hits)),
        "hit_rate_at_5": round(float(hits.mean()), 4),
        "ci95": [round(float(low), 4), round(float(high), 4)],
        "random_baseline": round(float(baseline), 4),
    }


def choose_alpha(
    z_text: np.ndarray,
    z_ratio: np.ndarray,
    order: Sequence[int],
    peers: dict[int, set[int]],
    queries: Sequence[int],
    alphas: Sequence[float] = ALPHAS,
) -> tuple[float, dict]:
    """The alpha with the best hit-rate@5 on `queries`; ties go to the alpha nearest 0.5, then
    the smaller one (a fixed rule, so the choice doesn't depend on the grid's order)."""
    rates = {}
    for a in alphas:
        top = top_k(a * z_text + (1 - a) * z_ratio)
        rates[a] = float(hits_at_k(top, order, peers, queries).mean())
    best = max(rates.values())
    alpha = min((a for a in alphas if rates[a] == best), key=lambda a: (abs(a - 0.5), a))
    return alpha, {str(a): round(r, 4) for a, r in rates.items()}


# ---- data and steps ---------------------------------------------------------------------------


def load_inputs(args: argparse.Namespace) -> dict:
    chunks = pd.read_parquet(args.chunks, columns=["chunk_id", "cik", "period_year", "accession"])
    ids = (args.emb_dir / "chunk_ids.txt").read_text().splitlines()
    if ids != chunks["chunk_id"].tolist():
        raise ValueError("chunk_ids.txt doesn't match chunks.parquet row for row")
    vectors = np.load(args.emb_dir / "chunks.npy", mmap_mode="r")
    if len(vectors) != len(chunks):
        raise ValueError(f"{len(vectors)} vectors for {len(chunks)} chunks")
    mask = latest_filing_mask(chunks)
    universe = pd.read_csv(args.universe, dtype={"sic": str})
    splits = pd.read_csv(args.splits)
    order = sorted(set(chunks["cik"]))
    sic_by_cik = dict(zip(universe["cik"], universe["sic"].str.zfill(4), strict=True))
    split_of = dict(zip(splits["cik"], splits["split"], strict=True))
    missing = [c for c in order if c not in sic_by_cik or c not in split_of]
    if missing:
        raise ValueError(f"{len(missing)} corpus companies lack a SIC code or split: {missing[:5]}")
    sic_by_cik = {c: sic_by_cik[c] for c in order}
    rows = np.flatnonzero(mask)
    vecs = company_vectors(chunks["cik"].to_numpy()[rows], vectors[rows], order)
    ranks, ratio_info = ratio_ranks(pd.read_parquet(args.ratios), order)
    level, sic_table = choose_sic_level(sic_by_cik)
    latest = chunks.loc[mask]
    return {
        "order": order,
        "split_of": split_of,
        "peers": peer_sets(sic_by_cik, level),
        "level": level,
        "sic_table": sic_table,
        "z_text": zscore_offdiag(text_similarity(vecs)),
        "z_ratio": zscore_offdiag(ratio_similarity(ranks)),
        "info": {
            "companies": len(order),
            "latest_10k_chunks": int(mask.sum()),
            "latest_10k_period_year": dict(
                sorted(Counter(latest.groupby("cik")["period_year"].first().astype(str)).items())
            ),
            "ratios": ratio_info,
        },
    }


def base_record(step: str, argv: list[str], data: dict) -> dict:
    return {
        "command": "python -m filinglens.retrieval.peers " + shlex.join(argv),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "step": step,
        "k": K,
        "sic_rule": f"finest of {list(SIC_LEVELS)} digits with coverage >= {MIN_COVERAGE}",
        "sic_level": data["level"],
        "sic_levels": data["sic_table"],
        "text": "mean bge-base chunk embedding of the latest 10-K (Item 1A), renormalized",
        "ratios": f"latest-year percentile ranks of {list(RATIO_FEATURES)}, missing = 0.5",
        "score": "alpha * z(text cosine) + (1 - alpha) * z(-euclidean on ratio ranks)",
        "data": data["info"],
    }


def run_select(args: argparse.Namespace, argv: list[str]) -> dict:
    data = load_inputs(args)
    queries = scorable(data["order"], data["peers"], data["split_of"], {"train", "dev"})
    alpha, grid = choose_alpha(
        data["z_text"], data["z_ratio"], data["order"], data["peers"], queries
    )
    record = base_record("select", argv, data)
    record.update(
        {
            "selection_splits": ["train", "dev"],
            "selection_companies_scored": len(queries),
            "selection_companies_without_peer": sum(
                1
                for c in data["order"]
                if data["split_of"][c] in {"train", "dev"} and not data["peers"][c]
            ),
            "hit_rate_by_alpha": grid,
            "alpha": alpha,
        }
    )
    metrics = {"alpha": alpha, "hit_rate_selection": grid[str(alpha)]}
    params = {"sic_level": data["level"], "k": K, "companies": len(queries)}
    record["mlflow_run_id"] = log_run(
        "peers_select", record, params, metrics, args.emb_dir / "chunk_ids.txt", args.note
    )
    args.select.write_text(json.dumps(record, indent=2) + "\n")
    return record


def run_test(args: argparse.Namespace, argv: list[str]) -> dict:
    if not args.allow_test:
        raise PermissionError("the peers test step scores FL-M2 test companies: pass --allow-test")
    if not rule_is_committed(args.select):
        raise SystemExit(f"{args.select} must be committed (and unchanged) before the test run")
    chosen = json.loads(args.select.read_text())
    data = load_inputs(args)
    if data["level"] != chosen["sic_level"]:
        raise ValueError(f"SIC level {data['level']} != the committed {chosen['sic_level']}")
    order, peers = data["order"], data["peers"]
    queries = scorable(order, peers, data["split_of"], {"test"})
    variants = {"combined": chosen["alpha"], "text_only": 1.0, "ratios_only": 0.0}
    results = {}
    for name, a in variants.items():
        top = top_k(a * data["z_text"] + (1 - a) * data["z_ratio"])
        results[name] = {
            "alpha": a,
            **summarize(hits_at_k(top, order, peers, queries), order, peers, queries),
        }
    record = base_record("test", argv, data)
    record.update(
        {
            "select_file": str(args.select),
            "alpha": chosen["alpha"],
            "test_companies_scored": len(queries),
            "test_companies_without_peer": sum(
                1 for c in order if data["split_of"][c] == "test" and not peers[c]
            ),
            "results": results,
        }
    )
    metrics = {f"hit_rate_{n}": r["hit_rate_at_5"] for n, r in results.items()}
    metrics["random_baseline"] = results["combined"]["random_baseline"]
    params = {"alpha": chosen["alpha"], "sic_level": data["level"], "companies": len(queries)}
    record["mlflow_run_id"] = log_run(
        "peers_test", record, params, metrics, args.emb_dir / "chunk_ids.txt", args.note
    )
    (args.bench / "peers.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    select, test = sub.add_parser("select"), sub.add_parser("test")
    test.add_argument("--allow-test", action="store_true")
    for p in (select, test):
        p.add_argument("--note", required=True)
        p.add_argument("--chunks", type=Path, default=CHUNKS)
        p.add_argument("--emb-dir", type=Path, default=EMB_DIR)
        p.add_argument("--universe", type=Path, default=UNIVERSE)
        p.add_argument("--splits", type=Path, default=SPLITS)
        p.add_argument("--ratios", type=Path, default=RATIOS)
        p.add_argument("--select", type=Path, default=SELECT)
        p.add_argument("--bench", type=Path, default=Path("bench"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    record = (run_select if args.step == "select" else run_test)(args, argv)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
