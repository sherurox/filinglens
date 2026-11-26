import json

from filinglens.serve import serving_table as st


def test_kernel_from_vllm_log_facts_and_cpu():
    facts = [
        "GPU KV cache size: 15,872 tokens",
        "Using MarlinLinearKernel for CompressedTensorsWNA16",
        "Using MarlinLinearKernel for mixed-precision linear",
    ]
    assert st.kernel_from(facts, "GPU") == "MarlinLinearKernel"
    assert st.kernel_from(["Selected CutlassFP8ScaledMMLinearKernel for X"], "GPU") == (
        "CutlassFP8ScaledMMLinearKernel"
    )
    assert st.kernel_from([], "GPU").startswith("bf16")
    assert st.kernel_from(facts, "CPU").startswith("llama.cpp")


def test_kv_tokens_from():
    assert st.kv_tokens_from(["GPU KV cache size: 15,872 tokens"]) == 15872
    assert st.kv_tokens_from(["nothing"]) is None


def test_test_label_follows_the_rule():
    assert st.test_label([-0.2, -0.01]) == "lower than BF16"
    assert st.test_label([-0.2, 0.0]) == "no difference detected"
    assert st.test_label([-0.1, 0.05]) == "no difference detected"
    assert st.test_label(None) == ""


def test_build_rows_copies_values_and_lists_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "BENCH", tmp_path)
    load = {
        "power_note": "AC power, Best performance",
        "server_log_facts": ["GPU KV cache size: 15,872 tokens"],
        "levels": [
            {
                "concurrency": 1,
                "requests": 64,
                "errors": 0,
                "ttft_s": {"p50": 0.12, "p95": 0.17},
                "latency_s": {"p50": 0.9, "p95": 1.0},
                "output_tok_per_s": 58.0,
                "decode_tok_per_s_per_request_mean": 60.0,
                "requests_per_s": 1.1,
                "completion_tokens_mean": 50.0,
                "gpu": {"server_vram_mib_peak": 4541.0, "temperature_c_max": 55.0},
            }
        ],
    }
    (tmp_path / "load_merged-bf16.json").write_text(json.dumps(load))
    (tmp_path / "merge.json").write_text(json.dumps({"out_weights_bytes": 3441185608}))
    (tmp_path / "fields_zero_dev_merged-bf16.json").write_text(
        json.dumps({"all_rows": {"category_macro_f1": 0.7414}})
    )
    missing: list[str] = []
    rows = st.build_rows(missing)
    bf16 = [r for r in rows if r["build"] == "merged-bf16"]
    assert len(bf16) == 1
    row = bf16[0]
    assert row["kv_cache_tokens"] == 15872 and row["weights_bytes"] == 3441185608
    assert row["server_vram_mib_peak"] == 4541.0 and row["dev_macro_f1"] == 0.7414
    assert row["test_macro_f1"] is None and row["test_label"] == ""
    assert "fields_zero_test_merged-bf16.json" in missing
    assert "load_gptq-w4a16.json" in missing
    # every build still gets a row (empty) so the table shape doesn't depend on progress
    assert {r["build"] for r in rows} == set(st.BUILDS)
    st.write_csv(rows, tmp_path / "serving.csv")
    header = (tmp_path / "serving.csv").read_text().splitlines()[0].split(",")
    assert header == st.COLUMNS


def test_dodge_keeps_order_and_min_gap():
    ys = [100.0, 102.0, 300.0, 101.0]
    out = st.dodge(ys, 10.0)
    order = sorted(range(4), key=lambda i: ys[i])
    placed = [out[i] for i in order]
    assert all(b - a >= 10.0 - 1e-9 for a, b in zip(placed, placed[1:], strict=False))
    assert out[2] >= 300.0 - 10.0  # a far-away label barely moves
    assert st.dodge([5.0], 10.0) == [5.0]
    assert st.dodge([], 10.0) == []
