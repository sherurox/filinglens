"""Merge the registry LoRA adapter into the Qwen3-1.7B base: the BF16 serving build (FL-M5 T2).

    ~/venvs/fl-train/bin/python -m filinglens.serve.merge

Reads the verified registry copy (T1: data/interim/serve/student-v1/adapter) and writes
data/interim/serve/merged-bf16/ (safetensors + the base's tokenizer files) and bench/merge.json.

Why merge: AWQ, GPTQ, FP8 and GGUF all quantize a plain checkpoint, so every serving format starts
from the same merged weights. Why fp32 on the CPU: W + (alpha / r) * B @ A is computed once at full
precision and rounded to bf16 once; merging in bf16 rounds the product and the sum separately.
1.7B parameters x 4 bytes is about 7 GB of RAM, and no VRAM.

Recorded checks: no LoRA tensor is left; the base tensors that changed are exactly the LoRA target
modules in every layer (nothing else, nothing missing); the saved config says bfloat16 (vLLM's
dtype "auto" would load an fp32 config as fp16); the tokenizer files are the base's.

Tied head (found on the first run, 25 Nov): Qwen3-1.7B ties lm_head to embed_tokens. The base
checkpoint stores both; transformers 5 saves the tied copy once. A missing `lm_head.weight` is
accepted only if the saved config says tie_word_embeddings and the base's lm_head equals its
embed_tokens bit for bit (embed_tokens itself is checked to be unchanged by the merge).
"""

from __future__ import annotations

import argparse
import json
import re
import resource
import shutil
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from filinglens.serve.student import tree_sha256
from filinglens.train.tracking import file_sha256

BASE = "Qwen/Qwen3-1.7B"
BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"  # bench/serving_env.txt (T1)
ADAPTER = Path("data/interim/serve/student-v1/adapter")
OUT = Path("data/interim/serve/merged-bf16")
TOKENIZER_FILES = re.compile(
    r"(tokenizer.*\.json|vocab\.json|merges\.txt|special_tokens_map\.json|.*\.jinja"
    r"|generation_config\.json)"
)
LAYER_PARAM = re.compile(r"layers\.\d+\.(.+)\.weight$")


def is_tokenizer_file(name: str) -> bool:
    return TOKENIZER_FILES.fullmatch(name) is not None


def module_kind(name: str) -> str:
    """'model.layers.3.self_attn.q_proj.weight' -> 'self_attn.q_proj'; other names unchanged."""
    match = LAYER_PARAM.search(name)
    return match.group(1) if match else name


def expected_changed(target_modules: list[str] | str, n_layers: int) -> int | None:
    """Tensors a merge should change: one weight per target module per layer.

    None when PEFT was given a pattern ('all-linear' or a regex string): then the count is
    reported but there is no number to check it against."""
    if isinstance(target_modules, str):
        return None
    return len(set(target_modules)) * n_layers


def config_dtype(config: dict) -> str | None:
    """transformers 5 writes `dtype`; older versions wrote `torch_dtype`."""
    return config.get("dtype") or config.get("torch_dtype")


def changed_tensors(base_files: list[Path], merged_files: list[Path]) -> dict:
    """Compare every tensor of the two checkpoints, one tensor at a time (low memory)."""
    import torch
    from safetensors import safe_open

    def index(files: list[Path]) -> dict[str, Path]:
        out = {}
        for path in files:
            with safe_open(str(path), framework="pt") as f:
                out |= dict.fromkeys(f.keys(), path)
        return out

    base, merged = index(base_files), index(merged_files)
    changed, dtypes = [], Counter()
    for name in sorted(base.keys() & merged.keys()):
        with safe_open(str(base[name]), framework="pt") as fb:
            b = fb.get_tensor(name)
        with safe_open(str(merged[name]), framework="pt") as fm:
            m = fm.get_tensor(name)
        dtypes[str(m.dtype)] += 1
        if b.shape != m.shape or not torch.equal(b, m):
            changed.append(name)
    return {
        "changed_names": changed,
        "tensors_base": len(base),
        "tensors_merged": len(merged),
        "only_base": sorted(base.keys() - merged.keys()),
        "only_merged": sorted(merged.keys() - base.keys()),
        "lora_tensors_left": sorted(n for n in merged if "lora_" in n),
        "merged_dtypes": dict(dtypes),
        "changed": len(changed),
        "changed_by_module": dict(sorted(Counter(map(module_kind, changed)).items())),
    }


def tied_head_equal(base_files: list[Path]) -> bool:
    """True when the base checkpoint's lm_head.weight equals its embed_tokens.weight exactly."""
    import torch
    from safetensors import safe_open

    found = {}
    for path in base_files:
        with safe_open(str(path), framework="pt") as f:
            for name in ("lm_head.weight", "model.embed_tokens.weight"):
                if name in f.keys():
                    found[name] = f.get_tensor(name)
    if len(found) != 2:
        return False
    return torch.equal(found["lm_head.weight"], found["model.embed_tokens.weight"])


def only_base_ok(only_base: list[str], tie_word_embeddings: bool, head_equal: bool) -> bool:
    """Tensors missing from the merged checkpoint are fine only as the tied head."""
    if not only_base:
        return True
    return only_base == ["lm_head.weight"] and tie_word_embeddings and head_equal


def merge(adapter: Path, out: Path, base: str, revision: str) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    adapter_config = json.loads((adapter / "adapter_config.json").read_text())
    if adapter_config.get("base_model_name_or_path") not in {base, None}:
        raise ValueError(f"adapter was trained on {adapter_config['base_model_name_or_path']!r}")
    snapshot = Path(snapshot_download(base, revision=revision, local_files_only=True))

    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.float32, device_map="cpu")
    n_layers = model.config.num_hidden_layers
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload(safe_merge=True)  # safe_merge: raises on NaN in the result
    model = model.to(torch.bfloat16)
    model.config.dtype = torch.bfloat16
    if out.exists():
        shutil.rmtree(out)  # a rerun never mixes files from an earlier merge
    model.save_pretrained(out)
    merge_s = time.perf_counter() - started

    copied = []
    for path in sorted(snapshot.iterdir()):
        if path.is_file() and is_tokenizer_file(path.name):
            shutil.copy2(path, out / path.name)  # the files FL-M4's vLLM evals used
            copied.append(path.name)

    saved_config = json.loads((out / "config.json").read_text())
    if config_dtype(saved_config) != "bfloat16":
        raise ValueError(f"saved config dtype is {config_dtype(saved_config)!r}, not bfloat16")

    base_files = sorted(snapshot.glob("*.safetensors"))
    diff = changed_tensors(base_files, sorted(out.glob("*.safetensors")))
    expected = expected_changed(adapter_config["target_modules"], n_layers)
    base_tok = {p.name: file_sha256(p) for p in snapshot.iterdir() if is_tokenizer_file(p.name)}
    adapter_tok = {n: h for n, h in tree_sha256(adapter).items() if is_tokenizer_file(n)}
    out_hashes = tree_sha256(out)
    tied = bool(saved_config.get("tie_word_embeddings"))
    head_equal = tied_head_equal(base_files) if diff["only_base"] else None
    missing_ok = only_base_ok(diff["only_base"], tied, bool(head_equal))
    ok = (
        not diff["lora_tensors_left"]
        and missing_ok
        and not diff["only_merged"]
        and "model.embed_tokens.weight" not in diff["changed_names"]
        and (expected is None or diff["changed"] == expected)
        and set(diff["merged_dtypes"]) == {"torch.bfloat16"}
    )
    return {
        "base": base,
        "base_revision": revision,
        "adapter": adapter.as_posix(),
        "adapter_weights_sha256": tree_sha256(adapter).get("adapter_model.safetensors"),
        "lora_r": adapter_config.get("r"),
        "lora_alpha": adapter_config.get("lora_alpha"),
        "target_modules": adapter_config["target_modules"],
        "merge": "fp32 on CPU, safe_merge, cast to bfloat16 once",
        "merge_and_save_s": round(merge_s, 1),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
        "out": out.as_posix(),
        "out_files": out_hashes,
        "out_weights_bytes": sum(p.stat().st_size for p in out.glob("*.safetensors")),
        "config_dtype": config_dtype(saved_config),
        "tokenizer_files_copied_from_base": copied,
        "adapter_tokenizer_files_bytes_equal_base": {
            n: adapter_tok[n] == base_tok.get(n) for n in sorted(adapter_tok)
        },
        "tokenizer_note": "byte equality only; serve/template_check.py compares rendered token IDs",
        "tie_word_embeddings": tied,
        "base_lm_head_equals_embed_tokens": head_equal,
        "only_base_ok": missing_ok,
        "layers": n_layers,
        "expected_changed": expected,
        **{k: v for k, v in diff.items() if k != "changed_names"},
        "ok": ok,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", type=Path, default=ADAPTER)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--revision", default=BASE_REVISION)
    parser.add_argument("--bench", type=Path, default=Path("bench"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    body = merge(args.adapter, args.out, args.base, args.revision)
    result = {
        "command": " ".join(["python -m filinglens.serve.merge", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **body,
    }
    (args.bench / "merge.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "out_files"}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
