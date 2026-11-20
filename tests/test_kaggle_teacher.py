"""Kaggle teacher helpers: shard selection, the category step, confidence, records, test guard.

A tiny regex tokenizer stands in for Qwen's, so these run in CI without vLLM.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from filinglens.train import kaggle_teacher as kt

TASK = {"task_no": 1, "para_id": "p1", "period_year": 2022, "split": "train", "cik": 1}
ANSWER = {
    "rationale": "Inflation risk.",
    "category": "macro_market",
    "severity": 2,
    "first_seen_year": 2019,
    "unsure": False,
}


class FakeTokenizer:
    """Words, underscores and single characters are tokens; ids are assigned on first sight.
    With merge_quote=True a double quote merges with the letters after it (like some BPEs)."""

    def __init__(self, merge_quote: bool = False) -> None:
        word = r'"[a-z]+|[A-Za-z]+' if merge_quote else r"[A-Za-z]+"
        self.pattern = re.compile(rf"{word}|\d+|\s+|.")
        self.vocab: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        return [self.vocab.setdefault(t, len(self.vocab)) for t in self.pattern.findall(text)]

    def decode(self, ids: list[int]) -> str:
        inverse = {v: k for k, v in self.vocab.items()}
        return "".join(inverse[i] for i in ids)


def generated(tok: FakeTokenizer, answer: dict = ANSWER) -> tuple[str, list[int]]:
    raw = json.dumps(answer, separators=(",", ":"))
    return raw, tok.encode(raw)


def test_select_prefix_then_shard() -> None:
    rows = [{"i": i} for i in range(10)]
    assert [r["i"] for r in kt.select(rows, 6, 0, 2)] == [0, 2, 4]
    assert [r["i"] for r in kt.select(rows, 6, 1, 2)] == [1, 3, 5]
    assert len(kt.select(rows, None, 0, 1)) == 10
    with pytest.raises(ValueError):
        kt.select(rows, None, 2, 2)


@pytest.mark.parametrize("merge_quote", [False, True])
def test_category_step_finds_the_value_token(merge_quote: bool) -> None:
    tok = FakeTokenizer(merge_quote=merge_quote)
    raw, ids = generated(tok)
    step = kt.category_step(ids, tok.decode, tok.encode)
    assert step is not None
    assert step["lead"] == ('"' if merge_quote else "")
    assert tok.decode([ids[step["index"]]]) == ('"macro' if merge_quote else "macro")
    assert step["first_tokens"]["macro_market"] == step["generated_token"]
    assert len(set(step["first_tokens"].values())) == len(kt.CATEGORIES)  # all distinct


def test_category_step_none_without_category_field() -> None:
    tok = FakeTokenizer()
    ids = tok.encode('{"rationale":"x"}')
    assert kt.category_step(ids, tok.decode, tok.encode) is None


def test_build_record_confidence_and_consistency() -> None:
    tok = FakeTokenizer()
    raw, ids = generated(tok)
    step = kt.category_step(ids, tok.decode, tok.encode)
    first = step["first_tokens"]
    logprobs = [{} for _ in ids]
    logprobs[step["index"]] = {
        first["macro_market"]: -0.1,
        first["competition_demand"]: -2.5,
        tok.encode("{")[0]: -6.0,  # not a category token: ignored
    }
    record = kt.build_record(TASK, raw, ids, logprobs, tok.decode, tok.encode, "teacher-x")
    assert record["valid"] is True and record["category"] == "macro_market"
    assert record["first_token_consistent"] is True
    assert record["confidence"]["macro_market"] == pytest.approx(0.9168, abs=1e-4)
    assert sum(record["confidence"].values()) == pytest.approx(1.0)
    assert record["confidence_error"] is None
    assert record["task"] == TASK and record["teacher"] == "teacher-x"
    assert set(record["category_step"]["top_logprobs"]) == {str(k) for k in logprobs[step["index"]]}


def test_build_record_shared_first_token_is_reported() -> None:
    tok = FakeTokenizer()
    raw, ids = generated(tok)

    def same_first(text: str) -> list[int]:  # every category starts with the same token
        return [7, *tok.encode(text)]

    record = kt.build_record(TASK, raw, ids, [{} for _ in ids], tok.decode, same_first, "t")
    assert record["confidence"] is None
    assert "share first token" in record["confidence_error"]


def test_build_record_invalid_json() -> None:
    tok = FakeTokenizer()
    ids = tok.encode("not json")
    record = kt.build_record(TASK, "not json", ids, [{} for _ in ids], tok.decode, tok.encode, "t")
    assert record["valid"] is False
    assert record["confidence_error"] == "no category field in the output"


def test_summarize_counts() -> None:
    tok = FakeTokenizer()
    raw, ids = generated(tok)
    step = kt.category_step(ids, tok.decode, tok.encode)
    logprobs = [{} for _ in ids]
    logprobs[step["index"]] = {step["first_tokens"]["macro_market"]: -0.2}
    good = kt.build_record(TASK, raw, ids, logprobs, tok.decode, tok.encode, "t")
    bad = kt.build_record(TASK, "x", tok.encode("x"), [{}], tok.decode, tok.encode, "t")
    stats = kt.summarize([good, bad])
    assert stats["rows"] == 2 and stats["rule_valid"] == 1
    assert stats["first_token_consistent"] == 1
    assert stats["chosen_category_confidence_median"] == pytest.approx(1.0)


def test_find_inputs_nested(tmp_path: Path) -> None:
    nested = tmp_path / "datasets" / "user" / "filinglens-teacher-inputs"
    nested.mkdir(parents=True)
    (nested / "pool.jsonl").write_text("")
    assert kt.find_inputs(tmp_path) == nested
    with pytest.raises(FileNotFoundError):
        kt.find_inputs(tmp_path / "datasets" / "missing")


def test_test_split_needs_allow_test() -> None:
    with pytest.raises(SystemExit, match="allow-test"):
        kt.main(["--split", "test"])
