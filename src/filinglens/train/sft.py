"""QLoRA SFT of the FL-M4 student (Qwen3-1.7B) on one SFT arm (FL-M4 T8).

Pipeline: data/interim/sft/<arm>.jsonl (T7) -> render each prompt with the tokenizer's chat
template exactly as vLLM renders it at evaluation (add_generation_prompt=True,
enable_thinking=False) -> tokenize prompt and completion separately (the model sees the prompt's
tokens and then generates, as at inference) -> labels = -100 on prompt tokens, so the loss is on
the answer only -> transformers Trainer + PEFT LoRA on a 4-bit NF4 base.

Why not TRL's SFTTrainer: it truncates over-long examples from the end, which here would cut the
answer the loss is about, and its token-accuracy metric needs the full logits, which Liger's
fused cross-entropy never builds. Here an example longer than --max-length is dropped and
counted, never cut. Memory: VRAM capped to what is free at start (no silent WSL sysmem spill)
and Liger fused linear cross-entropy (bench/train_smoke_4096_liger.json: 4,096 tokens fit).

Every run logs to MLflow (experiment fl-m4-sft): config, data version (sha256 of the SFT
manifest), git state, per-step loss / learning rate / grad norm / GPU temperature, peak VRAM,
and the adapter as an artifact. A JSON summary goes to bench/. MLflow must answer when the run
starts; after that an outage doesn't stop training: per-step metrics also go to
<adapters>/<run>/steps.jsonl, the missed steps are replayed at the end, and the bench summary
says whether MLflow got everything (`mlflow_synced`).

    ~/venvs/fl-train/bin/python -m filinglens.train.sft --arm silver --dry-run
    ~/venvs/fl-train/bin/python -m filinglens.train.sft --arm silver --max-steps 30 --note "..."

Top-level imports are standard library only, so CI can test the pure helpers without torch.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SFT_DIR = Path("data/interim/sft")
ADAPTERS = Path("data/interim/adapters")
IGNORE = -100  # label value the loss skips


@dataclass(frozen=True)
class TrainConfig:
    arm: str = "silver"
    model: str = "Qwen/Qwen3-1.7B"
    rank: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: float = 2.0
    max_steps: int = -1  # > 0 overrides epochs (smoke runs)
    grad_accum: int = 8  # batch 1 x 8 = effective batch 8
    max_length: int = 4096
    warmup_frac: float = 0.05
    seed: int = 0
    liger: bool = True
    teacher_limit: int = -1  # > 0: only the first N teacher examples (a balanced prefix)

    @property
    def alpha(self) -> int:
        return 2 * self.rank

    def run_name(self) -> str:
        length = f"s{self.max_steps}" if self.max_steps > 0 else f"e{self.epochs:g}"
        subset = (
            f"-n{self.teacher_limit}" if self.teacher_limit > 0 and self.arm != "silver" else ""
        )
        return f"{self.arm}{subset}-r{self.rank}-lr{self.learning_rate:g}-{length}"

    def params(self) -> dict[str, Any]:
        return asdict(self) | {"lora_alpha": self.alpha, "batch_size": 1}


# ---------- pure helpers (tested in CI with a fake tokenizer) ----------


def render(example: dict, tokenizer: Any) -> tuple[str, str]:
    """Prompt text as vLLM's llm.chat renders it, and the completion text ending in EOS."""
    prompt = tokenizer.apply_chat_template(
        example["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return prompt, example["completion"][0]["content"] + tokenizer.eos_token


def encode(prompt: str, completion: str, tokenizer: Any, max_length: int) -> dict | None:
    """input_ids and labels (prompt masked with -100); None if longer than max_length."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(completion_ids) > max_length:
        return None
    return {
        "input_ids": prompt_ids + completion_ids,
        "labels": [IGNORE] * len(prompt_ids) + completion_ids,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(completion_ids),
    }


def prepare(examples: list[dict], tokenizer: Any, max_length: int) -> tuple[list[dict], dict]:
    """Encoded examples that fit, and length statistics (dropped ones are counted, not cut)."""
    encoded, dropped = [], []
    for example in examples:
        item = encode(*render(example, tokenizer), tokenizer, max_length)
        if item is None:
            dropped.append(example["id"])
        else:
            encoded.append(item | {"id": example["id"]})
    totals = sorted(len(e["input_ids"]) for e in encoded)
    stats = {
        "examples_in": len(examples),
        "examples_kept": len(encoded),
        "dropped_too_long": len(dropped),
        "dropped_ids": dropped[:20],
        "tokens_total": sum(totals),
        "tokens_median": statistics.median(totals) if totals else None,
        "tokens_max": totals[-1] if totals else None,
        "completion_tokens_median": (
            statistics.median(e["completion_tokens"] for e in encoded) if encoded else None
        ),
    }
    return encoded, stats


def pad_batch(features: list[dict], pad_id: int) -> dict[str, list[list[int]]]:
    """Right-pad input_ids (pad_id), labels (-100) and attention_mask (0) to the longest."""
    width = max(len(f["input_ids"]) for f in features)
    batch: dict[str, list[list[int]]] = {"input_ids": [], "labels": [], "attention_mask": []}
    for f in features:
        gap = width - len(f["input_ids"])
        batch["input_ids"].append(f["input_ids"] + [pad_id] * gap)
        batch["labels"].append(f["labels"] + [IGNORE] * gap)
        batch["attention_mask"].append([1] * len(f["input_ids"]) + [0] * gap)
    return batch


def optimizer_steps(n_examples: int, grad_accum: int, epochs: float, max_steps: int) -> int:
    """Optimizer updates the Trainer will make (batch size 1)."""
    if max_steps > 0:
        return max_steps
    return max(1, math.ceil(epochs * math.ceil(n_examples / grad_accum)))


def warmup_steps(total_steps: int, frac: float) -> int:
    return max(1, round(frac * total_steps))


def limit_teacher(examples: list[dict], limit: int) -> list[dict]:
    """All silver examples plus the first `limit` teacher examples (file order is the teacher
    pool's round-robin order, so the prefix stays balanced across sectors and years)."""
    if limit <= 0:
        return examples
    silver = [e for e in examples if e["source"] == "silver"]
    teacher = [e for e in examples if e["source"] == "teacher"][:limit]
    return silver + teacher


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------- training (fl-train environment only) ----------


def train(cfg: TrainConfig, examples: list[dict], args: argparse.Namespace) -> dict:
    import mlflow
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    from filinglens.train.smoke_qlora import LORA_TARGETS, MIB, cap_to_free_vram
    from filinglens.train.tracking import StepLog, gpu_snapshot, run_tags, start_run

    vram = cap_to_free_vram(args.headroom_mib)
    if cfg.liger:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3

        apply_liger_kernel_to_qwen3(
            rope=False,
            rms_norm=False,
            swiglu=False,
            cross_entropy=False,
            fused_linear_cross_entropy=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    encoded, length_stats = prepare(examples, tokenizer, cfg.max_length)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    class Examples(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(encoded)

        def __getitem__(self, i: int) -> dict:
            return {k: encoded[i][k] for k in ("input_ids", "labels")}

    def collate(features: list[dict]) -> dict:
        return {k: torch.tensor(v) for k, v in pad_batch(features, pad_id).items()}

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=LORA_TARGETS,
            task_type="CAUSAL_LM",
        ),
    )

    total = optimizer_steps(len(encoded), cfg.grad_accum, cfg.epochs, cfg.max_steps)
    run_name = args.run_name or cfg.run_name()
    out_dir = args.adapters / run_name
    targs = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps(total, cfg.warmup_frac),
        optim="adamw_torch",
        bf16=True,
        logging_steps=1,
        logging_first_step=True,
        save_strategy="no",
        report_to="none",
        seed=cfg.seed,
        data_seed=cfg.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    temps: list[float] = []
    step_log = StepLog(out_dir / "steps.jsonl", lambda m, s: mlflow.log_metrics(m, step=s))

    class MlflowSteps(TrainerCallback):
        """Per optimizer step: loss, learning rate, grad norm and GPU temperature (to the local
        steps.jsonl always, to MLflow while it answers)."""

        def on_log(self, _args, state, _control, logs=None, **_kwargs):
            metrics = {k: float(v) for k, v in (logs or {}).items() if isinstance(v, int | float)}
            gpu = gpu_snapshot()
            if gpu.get("gpu.temperature.gpu"):
                temps.append(float(gpu["gpu.temperature.gpu"]))
                metrics["gpu_temp_c"] = temps[-1]
            metrics["vram_reserved_mib"] = torch.cuda.memory_reserved() / MIB
            step_log.log(metrics, state.global_step)

    def finish_mlflow(summary: dict, adapter_dir: Path) -> bool:
        """Missed steps, summary metrics, the step log and the adapter; False if MLflow is still
        unreachable (everything is in bench/ and steps.jsonl, so nothing is lost)."""
        if not step_log.sync():
            return False
        try:
            mlflow.log_metrics(
                {k: v for k, v in summary.items() if isinstance(v, int | float) and k != "run_id"}
            )
            mlflow.log_artifact(str(step_log.path))
            mlflow.log_artifacts(str(adapter_dir), artifact_path="adapter")
        except Exception as e:
            step_log.record_error("finish", e)
            return False
        return True

    manifest = args.data / "MANIFEST.json"
    tags = run_tags(data_manifest=manifest, note=args.note) | {"arm": cfg.arm}
    summary: dict[str, Any] | None = None
    try:
        with start_run("sft", run_name, tags=tags) as run:
            mlflow.log_params(cfg.params() | {"optimizer_steps": total})
            mlflow.log_params(
                {f"len.{k}": v for k, v in length_stats.items() if k != "dropped_ids"}
            )
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            trainer = Trainer(
                model=model,
                args=targs,
                train_dataset=Examples(),
                data_collator=collate,
                callbacks=[MlflowSteps()],
            )
            result = trainer.train()
            elapsed = time.perf_counter() - started
            adapter_dir = out_dir / "adapter"
            model.save_pretrained(adapter_dir)
            tokenizer.save_pretrained(adapter_dir)
            summary = {
                "run_id": run.info.run_id,
                "run_name": run_name,
                "train_runtime_s": round(elapsed, 1),
                "optimizer_steps": result.global_step,
                "final_train_loss": round(float(result.training_loss), 4),
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / MIB),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / MIB),
                "gpu_temp_max_c": max(temps) if temps else None,
                "sec_per_optimizer_step": round(elapsed / max(result.global_step, 1), 2),
                "train_samples_per_s": result.metrics.get("train_samples_per_second"),
                "adapter_dir": adapter_dir.as_posix(),
                "steps_log": step_log.path.as_posix(),
            }
            summary["mlflow_synced"] = finish_mlflow(summary, adapter_dir)
    except Exception as e:
        if summary is None:
            raise  # training itself failed, or MLflow was down before training began
        summary["mlflow_synced"] = False  # only closing the run failed (end_run)
        step_log.record_error("end_run", e)
    summary["mlflow_offline_at_step"] = step_log.offline_at_step
    summary["mlflow_errors"] = step_log.errors
    return {"config": cfg.params(), "vram": vram, "lengths": length_stats, **summary}


# ---------- CLI ----------


def dry_run(cfg: TrainConfig, examples: list[dict]) -> dict:
    """Tokenizer only (no GPU): length statistics and what the loss sees for one example."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    encoded, stats = prepare(examples, tokenizer, cfg.max_length)
    first = encoded[0]
    loss_tokens = [
        t for t, lab in zip(first["input_ids"], first["labels"], strict=True) if lab != IGNORE
    ]
    prompt_text, _ = render(examples[0], tokenizer)
    return stats | {
        "optimizer_steps": optimizer_steps(len(encoded), cfg.grad_accum, cfg.epochs, cfg.max_steps),
        "first_example_prompt_ends_with": prompt_text[-60:],
        "first_example_loss_text": tokenizer.decode(loss_tokens),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=["silver", "teacher", "both"], default="silver")
    parser.add_argument("--data", type=Path, default=SFT_DIR)
    parser.add_argument("--adapters", type=Path, default=ADAPTERS)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--teacher-limit", type=int, default=-1, help="first N teacher examples")
    parser.add_argument("--no-liger", action="store_true")
    parser.add_argument("--headroom-mib", type=int, default=300)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--note", default="", help="power mode etc.")
    parser.add_argument("--dry-run", action="store_true", help="tokenizer only, no training")
    parser.add_argument("--bench", type=Path, default=Path("bench"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    cfg = TrainConfig(
        arm=args.arm,
        rank=args.rank,
        learning_rate=args.lr,
        epochs=args.epochs,
        max_steps=args.max_steps,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        seed=args.seed,
        liger=not args.no_liger,
        teacher_limit=args.teacher_limit,
    )
    examples = limit_teacher(read_jsonl(args.data / f"{args.arm}.jsonl"), cfg.teacher_limit)
    if args.dry_run:
        print(json.dumps(dry_run(cfg, examples), indent=2))
        return
    body = train(cfg, examples, args)
    record = {
        "command": "python -m filinglens.train.sft " + shlex.join(argv),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": args.note,
        **body,
    }
    out = args.bench / f"sft_{body['run_name']}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
