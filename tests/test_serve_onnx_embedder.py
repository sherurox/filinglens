"""Pure helpers of the ONNX embedder check: cosines, top-k agreement, the rule decision."""

import json
from pathlib import Path

import numpy as np
import pytest

from filinglens.serve.onnx_embedder import latency_summary, min_cosine, rule_decision, same_top_k

RULE = json.loads(Path("bench/m5_embed_rule.json").read_text())


def unit(rows) -> np.ndarray:
    a = np.asarray(rows, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_min_cosine():
    a = unit([[1, 0], [0, 1]])
    b = unit([[1, 0], [1, 1]])
    assert min_cosine(a, b) == pytest.approx(np.sqrt(0.5))


def test_same_top_k_counts_identical_sets():
    rng = np.random.default_rng(0)
    chunks = unit(rng.normal(size=(50, 8)))
    q = unit(rng.normal(size=(3, 8)))
    assert same_top_k(q, q, chunks, k=5) == 3
    assert same_top_k(q, -q, chunks, k=5) == 0  # opposite queries rank the reverse


def results(fp32_cos: float, int8_recall: float) -> dict:
    return {
        "opt-fp32": {"min_cosine_vs_fl_m3": fp32_cos},
        "int8": {"exact_recall_at_10": int8_recall},
    }


def test_rule_ships_int8_within_tolerance():
    d = rule_decision(RULE, results(0.9999, 0.2967 - 0.0055))  # one query lost
    assert d["int8_ok"] and d["ship"] == "int8" and d["int8_recall_drop"] == pytest.approx(0.0055)


def test_rule_falls_back_to_fp32_when_int8_loses_too_much():
    d = rule_decision(RULE, results(0.9999, 0.2967 - 0.011))  # two queries lost
    assert not d["int8_ok"] and d["ship"] == "opt-fp32"


def test_rule_ships_nothing_if_the_pipeline_is_wrong():
    d = rule_decision(RULE, results(0.98, 0.30))
    assert not d["pipeline_ok"] and d["ship"] is None


def test_latency_summary():
    s = latency_summary([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4 and s["p50_ms"] == 2.5 and s["mean_ms"] == 2.5
