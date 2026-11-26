"""Query encoder for the API: bge-base-en-v1.5 as an ONNX graph on the CPU (FL-M5 T12).

Same recipe as FL-M3's retrieval.embed (the corpus vectors were made with it): the model card's
query instruction is prefixed, the text is tokenized to at most 512 tokens, the [CLS] hidden state
is taken and L2-normalized in fp32. Only onnxruntime, tokenizers and numpy are needed (no torch, no
transformers), so the API image stays small. The graph and tokenizer.json come from
serve/onnx_embedder.py; which graph ships (fp32 or int8) is decided by bench/m5_embed_rule.json.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

# Kept in step with retrieval/config.py (tested), without importing that module's pandas.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
MAX_SEQ_TOKENS = 512
EMBED_DIM = 768


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class OnnxEmbedder:
    """`embed_queries(["..."])` -> (n, 768) float32 unit vectors."""

    def __init__(self, model: Path, tokenizer: Path, threads: int = 1) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads  # Cloud Run: one vCPU per instance
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.tokenizer.enable_truncation(max_length=MAX_SEQ_TOKENS)
        pad = self.tokenizer.token_to_id("[PAD]")
        self.tokenizer.enable_padding(pad_id=0 if pad is None else pad, pad_token="[PAD]")

    def encode_ids(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        batch = self.tokenizer.encode_batch(list(texts))
        ids = np.array([e.ids for e in batch], dtype=np.int64)
        mask = np.array([e.attention_mask for e in batch], dtype=np.int64)
        return ids, mask

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        ids, mask = self.encode_ids(texts)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        (cls,) = self.session.run(["cls"], {k: v for k, v in feed.items() if k in self.inputs})
        return l2_normalize(cls)

    def embed_queries(self, questions: Sequence[str]) -> np.ndarray:
        return self.embed_texts([QUERY_INSTRUCTION + q for q in questions])
