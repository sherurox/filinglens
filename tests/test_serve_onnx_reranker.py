import csv

import numpy as np
import pytest

from filinglens.serve import onnx_reranker as orr


def test_variant_name_and_keys():
    assert orr.variant_name("ort", "cuda", "fp32") == "ort-cuda-fp32"
    assert orr.variant_name("ort", "cuda", "fp16", "opt-fp16") == "ort-cuda-fp16-opt"
    assert orr.variant_name("ort", "cpu", "fp32", "opt-fp32") == "ort-cpu-fp32-opt"
    assert orr.graph_dtype("opt-fp16") == "fp16" and orr.graph_dtype("plain") == "fp32"
    assert orr.variant_name("ort", "cuda", "fp16", "eager-opt-fp16") == "ort-cuda-fp16-eager-opt"
    assert orr.variant_name("ort", "cpu", "fp32", "eager") == "ort-cpu-fp32-eager"
    assert orr.graph_dtype("eager-opt-fp16") == "fp16"
    assert orr.query_keys(3) == ["q00000", "q00001", "q00002"]


def test_test_split_needs_allow_test():
    with pytest.raises(PermissionError):
        orr.guard_split("test", allow_test=False)
    orr.guard_split("test", allow_test=True)
    orr.guard_split("dev", allow_test=False)


def test_arrays_round_trip(tmp_path):
    arrays = {
        "q00000_ids": np.array([[0, 5, 2]], dtype=np.int64),
        "q00000": np.ones(2, np.float32),
    }
    orr.save_arrays(tmp_path / "x.npz", arrays)
    back = orr.load_arrays(tmp_path / "x.npz")
    assert back.keys() == arrays.keys()
    assert all(np.array_equal(back[k], arrays[k]) for k in arrays)


def test_max_abs_diff():
    a = {"q0": np.array([1.0, 2.0], np.float32), "q1": np.array([0.5], np.float32)}
    b = {"q0": np.array([1.0, 2.25], np.float32), "q1": np.array([0.0], np.float32)}
    assert orr.max_abs_diff(a, b) == 0.5
    with pytest.raises(ValueError):
        orr.max_abs_diff(a, {"q0": a["q0"]})


def test_same_top_k_order_and_stable_ties():
    a = np.array([0.1, 0.9, 0.5, 0.5])
    assert orr.same_top_k(a, a.copy(), 3)
    swapped = np.array([0.1, 0.9, 0.49, 0.51])  # candidates 2 and 3 change places
    assert not orr.same_top_k(a, swapped, 3)
    assert orr.same_top_k(a, swapped, 1)  # the top 1 is still candidate 1


def test_latency_summary():
    s = orr.latency_summary([1.0, 2.0, 3.0, 4.0])
    assert s["p50_ms"] == 2.5 and s["samples"] == 4 and s["mean_ms"] == 2.5


def test_write_csv(tmp_path):
    result = {
        "k": 10,
        "queries": 2,
        "tokenize_latency": {"p50_ms": 3.0},
        "variants": {
            "ort-cpu-fp32": {
                "metrics": {"ndcg_after": 0.488, "recall_after": 0.7308, "mrr_after": 0.5708},
                "forward_latency": {"p50_ms": 100.0, "p95_ms": 120.0, "p99_ms": 130.0},
                "max_abs_logit_diff_vs_torch-cpu-fp32": 1e-5,
                "same_top10_vs_torch-cpu-fp32": 2,
            }
        },
    }
    orr.write_csv(result, tmp_path / "r.csv")
    rows = list(csv.DictReader((tmp_path / "r.csv").open()))
    assert rows[0]["variant"] == "ort-cpu-fp32" and rows[0]["ndcg_at_10"] == "0.488"
    assert rows[0]["same_top10_vs_torch-cpu-fp32"] == "2"
    assert rows[0]["max_abs_logit_diff_vs_torch-cuda-fp16"] == ""


def test_rule_checks_pass_and_fail():
    rule = {
        "reference_variant": "torch-cuda-fp16",
        "candidate_variant": "ort-cuda-fp16-eager-opt",
        "reproduce_tolerance": 0.001,
        "match_tolerance": 0.005,
        "match_metrics": ["ndcg_after", "recall_after", "mrr_after"],
    }
    ref = {"ndcg_after": 0.5125, "recall_after": 0.7, "mrr_after": 0.6}
    ok = {"ndcg_after": 0.5130, "recall_after": 0.7, "mrr_after": 0.598}
    variants = {
        "torch-cuda-fp16": {"metrics": ref},
        "ort-cuda-fp16-eager-opt": {"metrics": ok},
    }
    result = orr.rule_checks(rule, variants, 0.5125)
    assert result["pipeline_reproduces_fl_m3"] and result["export_matches"]
    assert result["candidate_vs_reference_gaps"]["mrr_after"] == 0.002

    variants["ort-cuda-fp16-eager-opt"] = {"metrics": {**ok, "recall_after": 0.69}}
    assert not orr.rule_checks(rule, variants, 0.5125)["export_matches"]
    assert not orr.rule_checks(rule, variants, 0.514)["pipeline_reproduces_fl_m3"]
