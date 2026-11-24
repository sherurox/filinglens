import random

import pandas as pd
import pytest

from filinglens.retrieval import chunks as ch


def words(texts):
    """Stand-in tokenizer: one token per whitespace-separated word (additive, like WordPiece)."""
    return [len(t.split()) for t in texts]


def sentence(i: int, n: int) -> str:
    return "S" + " ".join([str(i)] + ["w"] * (n - 2)) + " end."


def corpus(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "para_id": f"acc-{i}:{i}",
                "cik": 100 + i,
                "ticker": f"T{i}",
                "sector": "finance",
                "period_year": 2020,
                "accession": f"acc-{i}",
                "para_index": i,
                "heading": heading,
                "text": text,
                "n_words": len(text.split()),
            }
            for i, (heading, text) in enumerate(rows)
        ]
    )


def test_heading_prefix():
    assert ch.heading_prefix(None, "Some text.") == ""
    assert ch.heading_prefix(float("nan"), "Some text.") == ""  # missing heading from Parquet
    assert ch.heading_prefix("  ", "Some text.") == ""
    assert ch.heading_prefix("Copper prices", "Our revenue depends on it.") == "Copper prices"
    # the paragraph already starts with its heading (case and spacing ignored): no repeat
    assert ch.heading_prefix("Copper  Prices", "copper prices may fall. We sell copper.") == ""
    long = " ".join(f"h{i}" for i in range(60))
    assert ch.heading_prefix(long, "Text.").split() == long.split()[: ch.HEADING_WORDS]


def test_embed_text():
    assert ch.embed_text("", "Body.") == "Body."
    assert ch.embed_text("Head", "Body.") == "Head\nBody."


def test_split_sentences():
    text = "First risk.  Second one! U.S. law applies? e.g. not here. 2025 was hard. (a) item"
    assert ch.split_sentences(text) == [
        "First risk.",
        "Second one!",
        "U.S. law applies? e.g. not here.",
        "2025 was hard.",
        "(a) item",
    ]


def test_pack_balances_parts():
    sentences = [sentence(i, 10) for i in range(10)]  # 100 words
    parts = ch.pack(sentences, words(sentences), budget=60, count=words)
    assert words(parts) == [50, 50]  # greedy filling would give 60 + 40
    assert " ".join(parts).split() == " ".join(sentences).split()


def test_pack_cuts_an_overlong_sentence_between_words():
    long = sentence(0, 25)
    parts = ch.pack([long], [25], budget=10, count=words)
    assert all(n <= 10 for n in words(parts))
    assert " ".join(parts).split() == long.split()


def test_pack_never_exceeds_the_budget_and_keeps_every_word():
    rng = random.Random(0)
    for _ in range(300):
        budget = rng.randint(5, 40)
        sentences = [sentence(i, rng.randint(2, 60)) for i in range(rng.randint(1, 12))]
        parts = ch.pack(sentences, words(sentences), budget, words)
        assert all(n <= budget for n in words(parts))
        assert " ".join(parts).split() == " ".join(sentences).split()


def test_a_word_longer_than_the_budget_stops():
    with pytest.raises(ValueError, match="single word"):
        ch.pack(["one enormous"], [12], budget=1, count=lambda ws: [2 for _ in ws])


def test_build_chunks_short_and_long_paragraphs():
    long_text = " ".join(sentence(i, 6) for i in range(5))  # 30 words
    frame = corpus([("Copper prices", "Short paragraph here."), ("Rates", long_text)])
    out = ch.build_chunks(frame, words, budget=12)

    short = out[out["para_id"] == "acc-0:0"]
    assert short["chunk_id"].tolist() == ["acc-0:0#0"]
    assert short["n_parts"].tolist() == [1]
    assert short["embed_text"].iloc[0] == "Copper prices\nShort paragraph here."

    long = out[out["para_id"] == "acc-1:1"]
    assert long["chunk_id"].tolist() == [f"acc-1:1#{k}" for k in range(len(long))]
    assert len(long) > 1 and (long["n_parts"] == len(long)).all()
    assert long["embed_text"].str.startswith("Rates\n").all()  # every part keeps the heading
    assert " ".join(long["text"]).split() == long_text.split()

    assert (out["n_tokens"] == words(out["embed_text"].tolist())).all()
    assert (out["n_tokens"] <= 12).all()
    assert (out["item"] == "1A").all()
    assert out["cik"].tolist() == [100] + [101] * len(long)


def test_build_chunks_is_deterministic():
    frame = corpus([("H", " ".join(sentence(i, 7) for i in range(9))), ("", "Tiny.")])
    pd.testing.assert_frame_equal(
        ch.build_chunks(frame, words, budget=20), ch.build_chunks(frame, words, budget=20)
    )


def test_recount_over_the_budget_stops_the_run():
    """If token counts don't add up across whitespace, the real recount catches the overflow."""

    def newline_costs_a_token(texts):
        return [len(t.split()) + ("\n" in t) for t in texts]

    # room after the heading = 9; three 3-word sentences fill it, so heading + newline = 11
    frame = corpus([("Head", " ".join(sentence(i, 3) for i in range(6)))])
    with pytest.raises(ValueError, match="over 10 tokens"):
        ch.build_chunks(frame, newline_costs_a_token, budget=10)


def test_summarize():
    frame = corpus([("H", " ".join(sentence(i, 6) for i in range(5))), ("", "Tiny.")])
    out = ch.build_chunks(frame, words, budget=12)
    summary = ch.summarize(frame, out)
    assert summary["paragraphs"] == 2
    assert summary["chunks"] == len(out)
    assert summary["paragraphs_split"] == 1
    assert summary["parts_per_split_paragraph"] == {str(len(out) - 1): 1}
    assert summary["heading_prefixed"] == len(out) - 1
    assert summary["tokens"]["max"] <= 12


def test_main_writes_chunks_and_bench(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    import json

    src = tmp_path / "risk_paragraphs.parquet"
    corpus([("H", " ".join(sentence(i, 6) for i in range(200))), ("", "Tiny.")]).to_parquet(src)
    monkeypatch.setattr(ch, "bge_counter", lambda: words)
    out, bench = tmp_path / "chunks.parquet", tmp_path / "bench"
    bench.mkdir()
    ch.main(["--corpus", str(src), "--out", str(out), "--bench", str(bench)])

    chunks = pd.read_parquet(out)
    record = json.loads((bench / "chunks.json").read_text())
    assert record["chunks"] == len(chunks) > 2
    assert record["paragraphs"] == 2 and record["paragraphs_split"] == 1
    assert record["corpus_sha256"] == ch.sha256_file(src)
    assert record["out_sha256"] == ch.sha256_file(out)
    assert record["tokens"]["max"] <= record["tokens"]["budget"] == 510
