"""SFT trainer helpers: rendering like vLLM, completion-only labels, drop-not-cut, padding, steps.

A fake tokenizer stands in for Qwen's, so these run in CI without torch or transformers.
"""

from __future__ import annotations

import pytest

from filinglens.train import sft

EXAMPLE = {
    "id": "p1",
    "prompt": [{"role": "system", "content": "rules"}, {"role": "user", "content": "para"}],
    "completion": [{"role": "assistant", "content": '{"category":"people"}'}],
}


class FakeTokenizer:
    """One token per character; records how the chat template was called."""

    eos_token = "|"

    def __init__(self) -> None:
        self.template_calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs) -> str:
        self.template_calls.append(kwargs)
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        assert add_special_tokens is False  # no BOS/EOS added behind our back
        return {"input_ids": [ord(c) for c in text]}


def test_render_matches_vllm_chat_call() -> None:
    tok = FakeTokenizer()
    prompt, completion = sft.render(EXAMPLE, tok)
    assert tok.template_calls == [
        {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    ]
    assert prompt == "<system>rules<user>para<assistant>"
    assert completion == '{"category":"people"}|'  # ends with EOS so the model learns to stop


def test_encode_masks_prompt_and_keeps_completion() -> None:
    tok = FakeTokenizer()
    item = sft.encode("abc", "de|", tok, max_length=10)
    assert item["input_ids"] == [ord(c) for c in "abcde|"]
    assert item["labels"] == [-100, -100, -100, ord("d"), ord("e"), ord("|")]
    assert (item["prompt_tokens"], item["completion_tokens"]) == (3, 3)


def test_encode_drops_instead_of_truncating() -> None:
    tok = FakeTokenizer()
    assert sft.encode("abc", "de|", tok, max_length=6) is not None  # exactly fits
    assert sft.encode("abc", "de|", tok, max_length=5) is None  # one over: dropped, not cut


def test_prepare_counts_dropped_examples() -> None:
    tok = FakeTokenizer()
    long = EXAMPLE | {"id": "p2", "prompt": [{"role": "user", "content": "x" * 500}]}
    encoded, stats = sft.prepare([EXAMPLE, long], tok, max_length=100)
    assert [e["id"] for e in encoded] == ["p1"]
    assert stats["examples_kept"] == 1
    assert stats["dropped_too_long"] == 1
    assert stats["dropped_ids"] == ["p2"]
    assert stats["tokens_total"] == len(encoded[0]["input_ids"])


def test_pad_batch() -> None:
    batch = sft.pad_batch(
        [{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]}, {"input_ids": [8], "labels": [8]}],
        pad_id=0,
    )
    assert batch["input_ids"] == [[5, 6, 7], [8, 0, 0]]
    assert batch["labels"] == [[-100, 6, 7], [8, -100, -100]]  # padding never counts in the loss
    assert batch["attention_mask"] == [[1, 1, 1], [1, 0, 0]]


@pytest.mark.parametrize(
    ("n", "accum", "epochs", "max_steps", "expected"),
    [(191, 8, 2.0, -1, 48), (191, 8, 1.0, -1, 24), (191, 8, 2.0, 30, 30), (3, 8, 1.0, -1, 1)],
)
def test_optimizer_steps(n: int, accum: int, epochs: float, max_steps: int, expected: int) -> None:
    assert sft.optimizer_steps(n, accum, epochs, max_steps) == expected


def test_warmup_steps_at_least_one() -> None:
    assert sft.warmup_steps(48, 0.05) == 2
    assert sft.warmup_steps(5, 0.05) == 1


def test_config_names_and_params() -> None:
    cfg = sft.TrainConfig(rank=8, learning_rate=1e-4, max_steps=30)
    assert cfg.alpha == 16
    assert cfg.run_name() == "silver-r8-lr0.0001-s30"
    assert sft.TrainConfig(epochs=2).run_name() == "silver-r16-lr0.0002-e2"
    params = cfg.params()
    assert params["lora_alpha"] == 16 and params["batch_size"] == 1 and params["liger"] is True


def test_limit_teacher_keeps_silver_and_a_teacher_prefix() -> None:
    examples = [{"id": f"s{i}", "source": "silver"} for i in range(2)]
    examples += [{"id": f"t{i}", "source": "teacher"} for i in range(5)]
    assert [e["id"] for e in sft.limit_teacher(examples, 3)] == ["s0", "s1", "t0", "t1", "t2"]
    assert sft.limit_teacher(examples, -1) == examples


def test_heldout_teacher_is_a_teacher_slice_after_the_training_prefix() -> None:
    examples = [{"id": f"s{i}", "source": "silver"} for i in range(2)]
    examples += [{"id": f"t{i}", "source": "teacher"} for i in range(6)]
    heldout = sft.heldout_teacher(examples, 3, 2)
    assert [e["id"] for e in heldout] == ["t3", "t4"]  # silver never counts as held out
    trained = {e["id"] for e in sft.limit_teacher(examples, 3)}
    assert trained.isdisjoint(e["id"] for e in heldout)
    assert sft.heldout_teacher(examples, 5, 10) == [examples[-1]]  # short file: fewer rows


def test_completion_positions_predict_exactly_the_completion_tokens() -> None:
    tok = FakeTokenizer()
    item = sft.encode("abc", "de|", tok, max_length=10)  # prompt 3 tokens, completion 3
    start, end = sft.completion_positions(item["prompt_tokens"], len(item["input_ids"]))
    assert (start, end) == (2, 5)
    targets = item["input_ids"][start + 1 : end + 1]  # what positions start..end-1 predict
    assert targets == [t for t in item["labels"] if t != -100]  # = the training loss tokens
    assert end - start == item["completion_tokens"]
    with pytest.raises(ValueError):
        sft.completion_positions(0, 5)  # no position predicts the first token
    with pytest.raises(ValueError):
        sft.completion_positions(5, 5)  # no completion


def test_write_record_creates_the_bench_folder(tmp_path) -> None:
    out = sft.write_record(tmp_path / "optuna" / "bench", "run-a", {"heldout_loss": 0.5})
    assert out == tmp_path / "optuna" / "bench" / "sft_run-a.json"
    assert out.read_text() == '{\n  "heldout_loss": 0.5\n}\n'


def test_token_weighted_mean() -> None:
    # 2 tokens at loss 1.0 and 6 tokens at loss 3.0: per token 2.5, per example would be 2.0
    assert sft.token_weighted_mean([1.0, 3.0], [2, 6]) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="no completion tokens"):
        sft.token_weighted_mean([], [])


@pytest.mark.parametrize("limit", ["-1", "1600"])
def test_cli_refuses_heldout_overlapping_training(limit: str) -> None:
    with pytest.raises(SystemExit):  # parser.error, before any file is read
        sft.main(["--arm", "teacher", "--teacher-limit", limit, "--heldout-n", "300"])


def test_run_name_with_teacher_limit() -> None:
    assert sft.TrainConfig(arm="teacher", teacher_limit=1500).run_name() == (
        "teacher-n1500-r16-lr0.0002-e2"
    )
    assert sft.TrainConfig(arm="silver", teacher_limit=1500).run_name() == "silver-r16-lr0.0002-e2"
