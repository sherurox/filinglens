"""bench/serving.csv, serving.png and serving_commands.md from committed bench files (T7, D1).

    uv run python -m filinglens.serve.serving_table

One row per build and concurrency level. Every value is copied from a bench file written by a run
(load_<build>.json, fields_zero_{dev,test}_<build>.json, compare_test_merged-bf16_vs_<build>.json,
merge.json, quant_<build>.json, gguf_build.json); nothing is computed here except the label the rule
defines (bench/m5_serving_rule.json). A file that doesn't exist yet leaves its cells empty and is
listed, so the table can be built after the load phase and again after the test phase.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

BENCH = Path("bench")
BUILDS = ["merged-bf16", "fp8-dynamic", "gptq-w4a16", "awq-w4a16", "gguf-q4_k_m"]
ENGINE = {b: ("vLLM 0.34", "GPU") for b in BUILDS[:4]} | {"gguf-q4_k_m": ("llama.cpp", "CPU")}
COLUMNS = [
    "build", "engine", "device", "kernel", "weights_bytes", "concurrency", "requests", "errors",
    "ttft_p50_s", "ttft_p95_s", "latency_p50_s", "latency_p95_s", "output_tok_per_s",
    "decode_tok_per_s_per_request", "requests_per_s", "completion_tokens_mean",
    "server_vram_mib_peak", "kv_cache_tokens", "gpu_temp_max_c", "gpu_power_mean_w",
    "power_mode", "dev_macro_f1", "test_macro_f1", "test_macro_f1_ci95", "test_rule_valid",
    "test_diff_vs_bf16", "test_diff_ci95", "test_label",
]  # fmt: skip
# Fixed categorical order (the dataviz reference palette, light mode): color follows the build.
COLORS = dict(zip(BUILDS, ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac"], strict=True))


def read(name: str, missing: list[str]) -> dict | None:
    path = BENCH / name
    if not path.is_file():
        missing.append(name)
        return None
    return json.loads(path.read_text())


def kernel_from(facts: list[str], device: str) -> str:
    """The linear-layer kernel vLLM logged ('Using X for' / 'Selected X for'), or the CPU note."""
    if device == "CPU":
        return "llama.cpp CPU (ggml)"
    names = []
    for fact in facts:
        match = re.search(r"(?:Using|Selected) (\w+) for", fact)
        if match and match.group(1) not in names:
            names.append(match.group(1))
    return "+".join(names) if names else "bf16 (no quantized linear kernel)"


def kv_tokens_from(facts: list[str]) -> int | None:
    for fact in facts:
        match = re.search(r"GPU KV cache size: ([\d,]+) tokens", fact)
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def weights_bytes(build: str, missing: list[str]) -> int | None:
    if build == "merged-bf16":
        record = read("merge.json", missing)
        return record and record.get("out_weights_bytes")
    if build == "gguf-q4_k_m":
        record = read("gguf_build.json", missing)
        return record and record.get("q4_k_m_gguf_bytes")
    record = read(f"quant_{build}.json", missing)
    return record and record.get("out_weights_bytes")


def test_label(diff_ci: list[float] | None) -> str:
    """The rule's label: 'lower than BF16' iff the CI's upper end is below 0."""
    if diff_ci is None:
        return ""
    return "lower than BF16" if diff_ci[1] < 0 else "no difference detected"


def build_rows(missing: list[str]) -> list[dict]:
    rows = []
    for build in BUILDS:
        engine, device = ENGINE[build]
        load = read(f"load_{build}.json", missing)
        dev = read(f"fields_zero_dev_{build}.json", missing)
        test = read(f"fields_zero_test_{build}.json", missing)
        cmp = None
        if build != "merged-bf16":
            cmp = read(f"compare_test_merged-bf16_vs_{build}.json", missing)
        facts = (load or {}).get("server_log_facts", [])
        dev_all = (dev or {}).get("all_rows", {})
        test_all = (test or {}).get("all_rows", {})
        diff = (cmp or {}).get("category_macro_f1", {})
        common = {
            "build": build,
            "engine": engine,
            "device": device,
            "kernel": kernel_from(facts, device),
            "weights_bytes": weights_bytes(build, missing),
            "kv_cache_tokens": kv_tokens_from(facts),
            "power_mode": (load or {}).get("power_note"),
            "dev_macro_f1": dev_all.get("category_macro_f1"),
            "test_macro_f1": test_all.get("category_macro_f1"),
            "test_macro_f1_ci95": test_all.get("category_macro_f1_ci95"),
            "test_rule_valid": f"{test_all['rule_valid']}/{test_all['rows']}" if test_all else None,
            "test_diff_vs_bf16": diff.get("diff_b_minus_a"),
            "test_diff_ci95": diff.get("diff_ci95"),
            "test_label": test_label(diff.get("diff_ci95")) if build != "merged-bf16" else "",
        }
        for level in (load or {}).get("levels") or [{}]:
            gpu = level.get("gpu", {})
            rows.append(
                {
                    **common,
                    "concurrency": level.get("concurrency"),
                    "requests": level.get("requests"),
                    "errors": level.get("errors"),
                    "ttft_p50_s": level.get("ttft_s", {}).get("p50"),
                    "ttft_p95_s": level.get("ttft_s", {}).get("p95"),
                    "latency_p50_s": level.get("latency_s", {}).get("p50"),
                    "latency_p95_s": level.get("latency_s", {}).get("p95"),
                    "output_tok_per_s": level.get("output_tok_per_s"),
                    "decode_tok_per_s_per_request": level.get("decode_tok_per_s_per_request_mean"),
                    "requests_per_s": level.get("requests_per_s"),
                    "completion_tokens_mean": level.get("completion_tokens_mean"),
                    "server_vram_mib_peak": gpu.get("server_vram_mib_peak")
                    if device == "GPU"
                    else None,
                    "gpu_temp_max_c": gpu.get("temperature_c_max"),
                    "gpu_power_mean_w": gpu.get("power_w_mean"),
                }
            )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row[k] for k in COLUMNS})


def plot(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)
    panels = [
        (axes[0], "output_tok_per_s", "Output tokens/s (all requests)"),
        (axes[1], "latency_p95_s", "p95 end-to-end latency (s)"),
    ]
    for ax, key, title in panels:
        for build in BUILDS:
            pts = [(r["concurrency"], r[key]) for r in rows if r["build"] == build and r[key]]
            if not pts:
                continue
            xs, ys = zip(*sorted(pts), strict=True)
            ax.plot(xs, ys, color=COLORS[build], linewidth=2, marker="o", markersize=8,
                    markeredgecolor="white", markeredgewidth=2, label=build)  # fmt: skip
            ax.annotate(build, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        fontsize=8, color="#444444", va="center")  # fmt: skip
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16], labels=["1", "4", "16"])
        ax.set_xlabel("Concurrent requests (closed loop)")
        ax.set_title(title, loc="left", fontsize=11)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[2]
    first = {r["build"]: r for r in rows}
    builds = [b for b in BUILDS if first.get(b, {}).get("test_macro_f1") is not None]
    for i, build in enumerate(builds):
        value = first[build]["test_macro_f1"]
        low, high = first[build]["test_macro_f1_ci95"] or (value, value)
        ax.bar(i, value, width=0.6, color=COLORS[build], edgecolor="white", linewidth=2)
        ax.errorbar(i, value, yerr=[[value - low], [high - value]], color="#444444", capsize=4)
        ax.annotate(f"{value:.3f}", (i, high), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=8, color="#444444")  # fmt: skip
    ax.set_xticks(range(len(builds)), labels=builds, rotation=20, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Test category macro-F1 (65 rows, 95% CI)", loc="left", fontsize=11)
    if not builds:
        ax.text(0.5, 0.5, "test phase not run yet", ha="center", transform=ax.transAxes)

    power = sorted({r["power_mode"] for r in rows if r["power_mode"]})
    fig.suptitle(
        "FilingLens student (Qwen3-1.7B + LoRA, merged) by serving format; RTX 5070 Laptop 8 GB / "
        f"Ryzen 9 8940HX; power: {', '.join(power) or 'n/a'}",
        fontsize=10,
        x=0.01,
        ha="left",
    )
    fig.savefig(path, dpi=150)


def commands_md(missing: list[str]) -> str:
    lines = ["# Commands that produced bench/serving.csv and bench/serving.png", ""]
    rule = BENCH / "m5_serving_rule.json"
    if rule.is_file():
        r = json.loads(rule.read_text())
        lines += ["Rule (committed before the test run): `bench/m5_serving_rule.json`", ""]
        lines += [f"- load: `{r['load']['command']}`", f"- test: `{r['accuracy']['command']}`"]
        lines += ["- table: `bash scripts/fl_m5_serving.sh table`", ""]
    lines += ["## Per build (copied from the bench files)", ""]
    for build in BUILDS:
        path = BENCH / f"load_{build}.json"
        if path.is_file():
            load = json.loads(path.read_text())
            lines += [f"- {build}: server `{load['server_cmd']}`; load `{load['command']}`"]
    if missing:
        lines += ["", "Not yet present: " + ", ".join(sorted(set(missing)))]
    return "\n".join(lines) + "\n"


def main() -> int:
    missing: list[str] = []
    rows = build_rows(missing)
    write_csv(rows, BENCH / "serving.csv")
    plot(rows, BENCH / "serving.png")
    (BENCH / "serving_commands.md").write_text(commands_md(missing))
    print(f"{len(rows)} rows -> bench/serving.csv, bench/serving.png, bench/serving_commands.md")
    if missing:
        print("missing:", ", ".join(sorted(set(missing))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
