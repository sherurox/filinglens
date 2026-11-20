"""QLoRA smoke test for the FL-M4 student on the laptop GPU.

Loads Qwen3-1.7B in 4-bit NF4 (double quantization), attaches LoRA adapters to
all attention and MLP projections, runs a few optimizer steps on full-length
random-token batches (the worst case for activation memory), and records peak
VRAM and step time. Writes a JSON record to bench/ (also when the run hits OOM).

VRAM cap: on WSL2 the driver can silently spill CUDA memory into system RAM even with the
"Prefer No Sysmem Fallback" policy set (seen 20 Nov: 8,186 MiB reserved with 2,412 MiB already
used by others, at half the tokens/s). So PyTorch is capped at the memory that is free when the
script starts, minus a headroom; going over raises OutOfMemoryError instead of spilling.

--liger uses Liger-Kernel's fused linear cross-entropy (BSD-2-Clause): the loss is computed in
chunks, so the full [tokens x 151k vocabulary] logits tensor is never built.

Run in the training environment (~/venvs/fl-train), not the vLLM one.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import bitsandbytes
import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MIB = 2**20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--liger", action="store_true", help="fused linear cross-entropy (Liger)")
    p.add_argument("--headroom-mib", type=int, default=300, help="kept free under the VRAM cap")
    p.add_argument("--note", default="", help="power mode etc., stored as-is")
    p.add_argument("--out", type=Path, default=Path("bench/train_smoke.json"))
    return p.parse_args()


def cap_to_free_vram(headroom_mib: int) -> dict:
    """Limit this process to the VRAM free right now (minus headroom); returns the numbers."""
    free_b, total_b = torch.cuda.mem_get_info()
    cap_b = max(free_b - headroom_mib * MIB, 0)
    torch.cuda.set_per_process_memory_fraction(cap_b / total_b, 0)
    return {
        "gpu_total_mib": round(total_b / MIB),
        "gpu_free_at_start_mib": round(free_b / MIB),
        "used_by_others_at_start_mib": round((total_b - free_b) / MIB),
        "vram_cap_mib": round(cap_b / MIB),
    }


def main() -> None:
    a = parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA not available in this environment")
    torch.manual_seed(0)
    vram = cap_to_free_vram(a.headroom_mib)

    liger_version = None
    if a.liger:
        try:
            import liger_kernel
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3
        except ImportError:
            sys.exit("--liger needs liger-kernel in this environment")
        # Only the loss is replaced: module names stay the same for bitsandbytes and LoRA.
        apply_liger_kernel_to_qwen3(
            rope=False,
            rms_norm=False,
            swiglu=False,
            cross_entropy=False,
            fused_linear_cross_entropy=True,
        )
        liger_version = getattr(liger_kernel, "__version__", "unknown")

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(a.model)

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        a.model, quantization_config=quant, device_map={"": 0}, dtype=torch.bfloat16
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=a.rank,
            lora_alpha=2 * a.rank,
            lora_dropout=0.05,
            target_modules=LORA_TARGETS,
            task_type="CAUSAL_LM",
        ),
    )
    load_s = time.perf_counter() - t0
    weights_mib = torch.cuda.memory_allocated() / MIB
    trainable, total = model.get_nb_trainable_parameters()

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
    ids = torch.randint(100, len(tok) - 1, (a.batch_size, a.seq_len), device="cuda")
    model.train()

    torch.cuda.reset_peak_memory_stats()  # peak from here on still includes the resident weights
    losses: list[float] = []
    step_times: list[float] = []
    oom = None
    try:
        for _ in range(a.steps):
            t = time.perf_counter()
            out = model(input_ids=ids, labels=ids)
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t)
            losses.append(round(out.loss.item(), 4))
    except torch.OutOfMemoryError as exc:
        oom = str(exc).splitlines()[0][:300]

    steady = step_times[1:] or step_times  # first step includes warm-up
    sec_per_step = sum(steady) / len(steady) if steady else None
    record = {
        "command": " ".join(
            [Path(sys.executable).name, "-m", "filinglens.train.smoke_qlora", *sys.argv[1:]]
        ),
        "utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": a.note,
        "model": a.model,
        "quant": "nf4 + double quant, bf16 compute",
        "lora": {"rank": a.rank, "alpha": 2 * a.rank, "targets": LORA_TARGETS},
        "liger_fused_linear_cross_entropy": a.liger,
        "liger_version": liger_version,
        "trainable_params": trainable,
        "total_params": total,
        "batch_size": a.batch_size,
        "seq_len": a.seq_len,
        "gradient_checkpointing": True,
        "steps_requested": a.steps,
        "steps_done": len(step_times),
        "oom": oom,
        **vram,
        "load_s": round(load_s, 1),
        "weights_after_load_mib": round(weights_mib),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / MIB),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / MIB),
        "sec_per_step": round(sec_per_step, 3) if sec_per_step else None,
        "tokens_per_s": round(a.batch_size * a.seq_len / sec_per_step) if sec_per_step else None,
        "losses": losses,
        "env": {
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__,
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(record, indent=2) + "\n")
    keys = ("oom", "vram_cap_mib", "peak_allocated_mib", "peak_reserved_mib", "sec_per_step")
    print(json.dumps({k: record[k] for k in (*keys, "tokens_per_s", "losses")}, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
