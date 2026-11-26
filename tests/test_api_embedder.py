"""The API's query encoder, with a fake ONNX session and tokenizer (no onnxruntime needed)."""

from types import SimpleNamespace

import numpy as np
import pytest

from filinglens.api import embedder
from filinglens.api.embedder import OnnxEmbedder, l2_normalize
from filinglens.retrieval import config


def test_constants_match_fl_m3():
    assert embedder.QUERY_INSTRUCTION == config.QUERY_INSTRUCTION
    assert embedder.MAX_SEQ_TOKENS == config.MAX_SEQ_TOKENS
    assert embedder.EMBED_DIM == config.EMBED_DIM


def test_l2_normalize():
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert out.dtype == np.float32
    assert np.allclose(out, [[0.6, 0.8], [0.0, 1.0]])


class FakeTokenizer:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode_batch(self, texts):
        self.seen.extend(texts)
        return [SimpleNamespace(ids=[101, 7, 102], attention_mask=[1, 1, 1]) for _ in texts]


class FakeSession:
    def __init__(self) -> None:
        self.feeds: list[dict] = []

    def run(self, outputs, feed):
        assert outputs == ["cls"]
        self.feeds.append(feed)
        return [np.tile(np.array([[3.0, 4.0]]), (len(feed["input_ids"]), 1))]


def fake_embedder(inputs=("input_ids", "attention_mask")) -> OnnxEmbedder:
    e = object.__new__(OnnxEmbedder)  # skip __init__: no onnxruntime or tokenizers needed
    e.session, e.tokenizer, e.inputs = FakeSession(), FakeTokenizer(), set(inputs)
    return e


def test_embed_queries_prefixes_the_instruction_and_normalizes():
    e = fake_embedder()
    out = e.embed_queries(["what risks?", "why?"])
    assert e.tokenizer.seen == [
        config.QUERY_INSTRUCTION + "what risks?",
        config.QUERY_INSTRUCTION + "why?",
    ]
    assert out.shape == (2, 2) and np.allclose(out, [[0.6, 0.8]] * 2)
    feed = e.session.feeds[0]
    assert set(feed) == {"input_ids", "attention_mask"}
    assert feed["input_ids"].dtype == np.int64


@pytest.mark.parametrize("with_types", [True, False])
def test_token_type_ids_only_if_the_graph_takes_them(with_types):
    names = ("input_ids", "attention_mask") + (("token_type_ids",) if with_types else ())
    e = fake_embedder(names)
    e.embed_texts(["x"])
    assert ("token_type_ids" in e.session.feeds[0]) is with_types
