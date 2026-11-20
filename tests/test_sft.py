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
