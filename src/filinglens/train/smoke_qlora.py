"""QLoRA smoke test for the FL-M4 student on the laptop GPU.

Loads Qwen3-1.7B in 4-bit NF4 (double quantization), attaches LoRA adapters to
all attention and MLP projections, runs a few optimizer steps on full-length
random-token batches (the worst case for activation memory), and records peak
VRAM and step time. Writes a JSON record to bench/.

Run in the training environment (~/venvs/fl-train), not the vLLM one.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
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
    p.add_argument("--note", default="", help="power mode etc., stored as-is")
    p.add_argument("--out", type=Path, default=Path("bench/train_smoke.json"))
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA not available in this environment")
    torch.manual_seed(0)

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
    for _ in range(a.steps):
        t = time.perf_counter()
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t)
        losses.append(round(out.loss.item(), 4))

    steady = step_times[1:] or step_times  # first step includes warm-up
    sec_per_step = sum(steady) / len(steady)
    free_b, total_b = torch.cuda.mem_get_info()

    record = {
        "command": " ".join([Path(sys.executable).name, "-m", "filinglens.train.smoke_qlora", *sys.argv[1:]]),
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": a.note,
        "model": a.model,
        "quant": "nf4 + double quant, bf16 compute",
        "lora": {"rank": a.rank, "alpha": 2 * a.rank, "targets": LORA_TARGETS},
        "trainable_params": trainable,
        "total_params": total,
        "batch_size": a.batch_size,
        "seq_len": a.seq_len,
        "gradient_checkpointing": True,
        "steps": a.steps,
        "load_s": round(load_s, 1),
        "weights_after_load_mib": round(weights_mib),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / MIB),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / MIB),
        "gpu_total_mib": round(total_b / MIB),
        "gpu_free_at_end_mib": round(free_b / MIB),
        "sec_per_step": round(sec_per_step, 3),
        "tokens_per_s": round(a.batch_size * a.seq_len / sec_per_step),
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
    print(json.dumps({k: record[k] for k in (
        "peak_allocated_mib", "peak_reserved_mib", "sec_per_step", "tokens_per_s",
        "trainable_params", "losses")}, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
