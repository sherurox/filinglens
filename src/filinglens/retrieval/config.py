"""Shared FL-M3 settings: the pinned models and the data paths.

The revisions are the Hugging Face commit SHAs recorded in bench/retrieval_env.txt (T1), so every
step loads exactly the weights and tokenizer whose license was checked.
"""

from pathlib import Path

EMBED_MODEL = "BAAI/bge-base-en-v1.5"  # MIT
EMBED_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
RERANK_MODEL = "BAAI/bge-reranker-base"  # MIT
RERANK_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"

MAX_SEQ_TOKENS = 512  # bge-base input length, including [CLS] and [SEP]
TEXT_TOKEN_BUDGET = MAX_SEQ_TOKENS - 2  # what a chunk's own text may use

ITEM = "1A"  # Risk Factors: the only 10-K item in the FL-M2 corpus (Item 7 is a LATER row)

CORPUS = Path("data/interim/risk_paragraphs.parquet")
RET_DIR = Path("data/interim/retrieval")
CHUNKS = RET_DIR / "chunks.parquet"
