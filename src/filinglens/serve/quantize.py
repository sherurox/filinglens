r"""Quantize the merged BF16 student with llm-compressor: AWQ / GPTQ W4A16 and FP8 dynamic (T3).

    PYTHONPATH=src ~/venvs/fl-quant/bin/python -m filinglens.serve.quantize --scheme awq \
        --note "AC power, Best performance"

Input: data/interim/serve/merged-bf16 (T2). Output: data/interim/serve/<build>/ (compressed-tensors
safetensors + the same tokenizer files) and bench/quant_<build>.json.

Recipes (llmcompressor 0.15; `lm_head` is never quantized: it is tied to the embeddings):
- awq:  AWQ scale search (a transform in 0.15), then W4A16 round-to-nearest, group 128;
- gptq: GPTQ W4A16, group 128 (columns rounded with second-order error correction);
- fp8:  FP8_DYNAMIC: fp8 weights with per-channel scales, activation scales per token at run time,
        so no calibration data.

Calibration (awq, gptq): the first N gold-train rows by task number (train only; dev and test are
never seen), each rendered exactly as `extract.fields` prompts the model (same system prompt, same
chat template, thinking off) plus the silver answer as the assistant turn, so the activations the
scales are fitted on are the ones the served model sees. Runs in its own venv (fl-quant) so the
quantizer's torch/transformers pins can't touch the vLLM or training venvs.

Memory (25 Nov): AWQ with 128 x 2,560 calibration tokens filled the 8 GB GPU and WSL spilled into
system RAM: still on layer 0 after 84 min, grid steps slowing 36 s -> 1,246 s, the laptop barely
usable. Now (1) PyTorch is capped at the VRAM free at start minus a margin, so overflowing raises
an out-of-memory error (recorded in the bench file) instead of spilling, as in FL-M4 training; and
(2) AWQ and GPTQ use the same 32 x 2,048-token calibration set (about 65k tokens, the AWQ paper's
budget of 128 x 512), so the two 4-bit builds differ only in the algorithm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from filinglens.serve.merge import is_tokenizer_file
from filinglens.serve.student import tree_sha256

MERGED = Path("data/interim/serve/merged-bf16")
OUT_ROOT = Path("data/interim/serve")
BUILDS = {"awq": "awq-w4a16", "gptq": "gptq-w4a16", "fp8": "fp8-dynamic"}
NEEDS_DATA = {"awq", "gptq"}
VRAM_MARGIN_MIB = 256


def vram_cap_fraction(free_mib: float, total_mib: float, margin_mib: float = 256) -> float:
    """Fraction of total VRAM PyTorch may use: what is free now, minus a margin for the driver.

    In WSL, memory.used includes the Windows desktop, so the free amount, not the total, is what
    this process can have without spilling into system RAM."""
    if total_mib <= 0 or free_mib <= margin_mib:
        raise ValueError(f"not enough free VRAM: {free_mib:.0f} MiB free, margin {margin_mib}")
    return round((free_mib - margin_mib) / total_mib, 4)


def calibration_rows(train: list[dict], n: int) -> list[dict]:
    """The first n train rows by task number: deterministic, no sampling seed to record."""
    if n < 1:
        raise ValueError("n must be >= 1")
    rows = sorted(train, key=lambda r: r["task_no"])
    if len(rows) < n:
        raise ValueError(f"asked for {n} calibration rows, train has {len(rows)}")
    return rows[:n]


def answer(row: dict, rationales: dict[str, str]) -> str:
    """The assistant turn: the silver labels in the eval's JSON shape (as fields.example_turns)."""
    return json.dumps(
        {
            "rationale": rationales.get(row["para_id"], ""),
            "category": row["category"],
            "severity": row["severity"],
            "first_seen_year": row["first_seen_year"],
            "unsure": False,
        }
    )


def ids_digest(para_ids: list[str]) -> str:
    """SHA-256 of the calibration paragraph IDs in order: proves which rows were used."""
    return hashlib.sha256("\n".join(para_ids).encode()).hexdigest()


def quantization_summary(config: dict) -> dict:
    """The parts of config.json's quantization_config that say what was built."""
    q = config.get("quantization_config") or config.get("compression_config") or {}
    groups = {}
    for name, group in (q.get("config_groups") or {}).items():
        w = group.get("weights") or {}
        a = group.get("input_activations") or {}
        groups[name] = {
            "targets": group.get("targets"),
            "weights": {k: w.get(k) for k in ("num_bits", "type", "strategy", "group_size")},
            "input_activations": {k: a.get(k) for k in ("num_bits", "type", "dynamic")}
            if a
            else None,
        }
    return {
        "quant_method": q.get("quant_method"),
        "format": q.get("format"),
        "ignore": q.get("ignore"),
        "config_groups": groups,
    }


def recipe(scheme: str) -> list:
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier

    if scheme == "awq":
        from llmcompressor.modifiers.transform.awq import AWQModifier

        return [
            AWQModifier(),
            QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
        ]
    if scheme == "gptq":
        return [GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])]
    if scheme == "fp8":
        return [QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])]
    raise ValueError(f"unknown scheme {scheme!r}")


def calibration_dataset(tokenizer, n: int, max_len: int):
    from datasets import Dataset

    from filinglens.extract.fields import PROMPT, RATIONALES, TASKS, conversations, load_rows

    rows = calibration_rows(load_rows("train", allow_test=False), n)
    tasks = {t["para_id"]: t for t in json.loads(TASKS.read_text())}
    rationales = {
        r["para_id"]: r.get("rationale", "")
        for r in (json.loads(line) for line in RATIONALES.read_text().splitlines() if line)
    }
    convs = conversations(rows, tasks, PROMPT.read_text(), [])
    ids, lengths = [], []
    for row, conv in zip(rows, convs, strict=True):
        full = [*conv, {"role": "assistant", "content": answer(row, rationales)}]
        text = tokenizer.apply_chat_template(full, tokenize=False, enable_thinking=False)
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        lengths.append(len(tokens))
        ids.append(tokens[:max_len])
    data = Dataset.from_dict({"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]})
    stats = {
        "rows": len(rows),
        "para_ids_sha256": ids_digest([r["para_id"] for r in rows]),
        "tokens_min_max": [min(lengths), max(lengths)],
        "truncated_to_max_len": sum(n_tok > max_len for n_tok in lengths),
    }
    return data, stats


def apply_vram_cap(margin_mib: float) -> dict:
    import torch

    free, total = (x / 2**20 for x in torch.cuda.mem_get_info())
    fraction = vram_cap_fraction(free, total, margin_mib)
    torch.cuda.set_per_process_memory_fraction(fraction)
    return {
        "free_mib_at_start": round(free),
        "total_mib": round(total),
        "margin_mib": margin_mib,
        "fraction": fraction,
        "cap_mib": round(fraction * total),
    }


def quantize(scheme: str, merged: Path, out_root: Path, n: int, max_len: int, note: str) -> dict:
    import torch
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from filinglens.train.tracking import gpu_snapshot

    out = out_root / BUILDS[scheme]
    tokenizer = AutoTokenizer.from_pretrained(merged)
    gpu_before = gpu_snapshot()
    vram_cap = apply_vram_cap(VRAM_MARGIN_MIB)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(merged, dtype=torch.bfloat16, device_map="cuda")
    kwargs, calib = {}, None
    if scheme in NEEDS_DATA:
        data, calib = calibration_dataset(tokenizer, n, max_len)
        kwargs = {
            "dataset": data,
            "num_calibration_samples": len(data),
            "max_seq_length": max_len,
            "shuffle_calibration_samples": False,
        }
    steps = recipe(scheme)
    oneshot(model=model, tokenizer=tokenizer, recipe=steps, **kwargs)
    quantize_s = time.perf_counter() - started
    peak_alloc = torch.cuda.max_memory_allocated() // 2**20
    peak_reserved = torch.cuda.max_memory_reserved() // 2**20
    gpu_after = gpu_snapshot()

    if out.exists():
        shutil.rmtree(out)  # never mix files from an earlier build
    model.save_pretrained(out, save_compressed=True)
    for path in sorted(merged.iterdir()):
        if path.is_file() and is_tokenizer_file(path.name):
            shutil.copy2(path, out / path.name)  # same tokenizer files as the BF16 build
    config = json.loads((out / "config.json").read_text())
    from importlib.metadata import version

    return {
        "scheme": scheme,
        "build": BUILDS[scheme],
        "recipe": [repr(step) for step in steps],
        "source": merged.as_posix(),
        "source_weights_sha256": tree_sha256(merged).get("model.safetensors"),
        "calibration": calib,
        "max_seq_length": max_len if scheme in NEEDS_DATA else None,
        "quantize_s": round(quantize_s, 1),
        "vram_cap": vram_cap,
        "peak_vram_allocated_mib": peak_alloc,
        "peak_vram_reserved_mib": peak_reserved,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "power_note": note,
        "out": out.as_posix(),
        "out_weights_bytes": sum(p.stat().st_size for p in out.glob("*.safetensors")),
        "out_files": tree_sha256(out),
        "quantization_config": quantization_summary(config),
        "versions": {d: version(d) for d in ("llmcompressor", "compressed-tensors", "torch")},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scheme", choices=sorted(BUILDS), required=True)
    parser.add_argument("--merged", type=Path, default=MERGED)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--note", required=True, help="power mode, e.g. 'AC power, Best perf'")
    parser.add_argument("--bench", type=Path, default=Path("bench"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    import torch

    code = 0
    try:
        body = quantize(
            args.scheme, args.merged, args.out_root, args.samples, args.max_seq_len, args.note
        )
    except torch.OutOfMemoryError as error:  # a finding for the table, not a crash to retry
        code = 1
        body = {
            "scheme": args.scheme,
            "build": BUILDS[args.scheme],
            "samples": args.samples,
            "max_seq_length": args.max_seq_len,
            "power_note": args.note,
            "error": f"OutOfMemoryError under the VRAM cap: {str(error)[:400]}",
        }
    result = {
        "command": " ".join(["python -m filinglens.serve.quantize", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **body,
    }
    out = args.bench / f"quant_{BUILDS[args.scheme]}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "out_files"}, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
