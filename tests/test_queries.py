import json

import pandas as pd
import pytest

from filinglens.retrieval import queries as rq

RISK = (
    "Fluctuations in copper prices could reduce our revenue, margins and cash flow because most "
    "of our sales come from copper concentrate shipped to smelters in Asia and Europe."
)
OTHER = "We depend on key personnel and may be unable to retain senior engineers or executives."


def chunk(chunk_id, para_id, cik, accession, year, text, heading="Commodity prices"):
    return {
        "chunk_id": chunk_id,
        "para_id": para_id,
        "cik": cik,
        "accession": accession,
        "period_year": year,
        "heading": heading,
        "text": text,
    }


def qa(qa_id, para_id, cik=1, evidence="copper prices could reduce our revenue", **extra):
    return {
        "qa_id": qa_id,
        "para_id": para_id,
        "cik": cik,
        "ticker": f"T{cik}",
        "sector": "mining_energy",
        "period_year": 2020,
        "type": "factual",
        "question": f"What does T{cik}'s FY2020 10-K say about copper?",
        "evidence": evidence,
        "generator": "Qwen/Qwen3-4B-AWQ",
        **extra,
    }


def test_pick_queries_keeps_the_first_question_per_paragraph():
    rows = [qa("p2:q1", "p2"), qa("p1:q2", "p1"), qa("p1:q1", "p1")]
    assert [r["qa_id"] for r in rq.pick_queries(rows)] == ["p1:q1", "p2:q1"]


def test_assign_splits_alternates_within_each_sector():
    companies = pd.DataFrame({"cik": range(1, 12), "sector": ["finance"] * 6 + ["retail"] * 5})
    splits = rq.assign_splits(companies)
    assert set(splits) == set(range(1, 12))
    for sector, ciks in (("finance", range(1, 7)), ("retail", range(7, 12))):
        counts = pd.Series([splits[c] for c in ciks]).value_counts()
        assert abs(counts.get("dev", 0) - counts.get("test", 0)) <= 1, sector
    assert rq.assign_splits(companies) == splits  # deterministic
    assert rq.assign_splits(companies, seed=4) != splits  # the seed matters


def test_evidence_found_verbatim_ignoring_case_spacing_and_curly_quotes():
    parts = pd.DataFrame([chunk("p#0", "p", 1, "a", 2020, "Our customers’  RISK appetite fell.")])
    assert rq.evidence_chunks("our customers' risk appetite", parts) == (["p#0"], "verbatim")


def test_evidence_in_the_heading_marks_every_part():
    parts = pd.DataFrame(
        [
            chunk("p#0", "p", 1, "a", 2020, "First half.", heading="Copper prices may fall"),
            chunk("p#1", "p", 1, "a", 2020, "Second half.", heading="Copper prices may fall"),
        ]
    )
    assert rq.evidence_chunks("copper prices may fall", parts) == (["p#0", "p#1"], "verbatim")


def test_evidence_across_a_split_falls_back_to_the_part_sharing_most_words():
    parts = pd.DataFrame(
        [
            chunk("p#0", "p", 1, "a", 2020, "Unrelated opening sentence about weather. Copper"),
            chunk("p#1", "p", 1, "a", 2020, "prices could reduce our revenue and margins."),
        ]
    )
    found = rq.evidence_chunks("Copper prices could reduce our revenue", parts)
    assert found == (["p#1"], "most_shared_words")


def test_near_copies_rule():
    candidates = pd.DataFrame(
        [
            chunk("same", "x", 1, "b", 2019, RISK.replace("Asia", "China")),  # near copy
            chunk("diff", "y", 1, "b", 2019, OTHER),
            chunk("frag", "z", 1, "b", 2019, "copper prices revenue margins"),  # too small
            chunk("half", "w", 1, "b", 2019, " ".join(RISK.split()[:12]) + " " + OTHER),
        ]
    )
    cache: dict = {}
    assert rq.near_copies([RISK], candidates, cache) == ["same"]
    assert set(cache) == {"same", "diff", "frag", "half"}  # each chunk's words computed once


def test_build_queries_grades():
    chunks = pd.DataFrame(
        [
            chunk("acc20:0#0", "acc20:0", 1, "acc20", 2020, RISK),  # the source paragraph
            chunk("acc20:1#0", "acc20:1", 1, "acc20", 2020, RISK),  # same filing: not graded
            chunk("acc19:0#0", "acc19:0", 1, "acc19", 2019, RISK),  # other year: grade 1
            chunk("acc19:1#0", "acc19:1", 1, "acc19", 2019, OTHER),
            chunk("bcc20:0#0", "bcc20:0", 2, "bcc20", 2020, RISK),  # other company: not graded
        ]
    )
    rows = rq.build_queries([qa("acc20:0:q1", "acc20:0")], chunks, {1: "dev", 2: "test"})
    assert len(rows) == 1
    row = rows[0]
    assert row["grades"] == {"acc20:0#0": 2, "acc19:0#0": 1}
    assert row["split"] == "dev" and row["evidence_match"] == "verbatim"
    assert row["source_para_id"] == "acc20:0" and row["synthetic"] is True

    with pytest.raises(ValueError, match="not in the chunks"):
        rq.build_queries([qa("missing:q1", "missing")], chunks, {1: "dev"})


def test_leakage_and_summary():
    rows = [
        {
            "cik": 1,
            "split": "dev",
            "sector": "s",
            "type": "factual",
            "evidence_match": "verbatim",
            "grades": {"a": 2, "b": 1, "c": 1},
        },
        {
            "cik": 2,
            "split": "test",
            "sector": "s",
            "type": "cause",
            "evidence_match": "verbatim",
            "grades": {"d": 2},
        },
    ]
    assert rq.leakage(rows) == 0
    assert rq.leakage(rows + [{**rows[0], "split": "test"}]) == 1
    summary = rq.summarize(rows)
    assert summary["dev"]["grade1_per_query"] == {"mean": 2.0, "max": 2, "zero": 0}
    assert summary["test"]["grade1_per_query"]["zero"] == 1
    assert summary["dev"]["grade2_chunks"] == 1


def test_load_queries_guards_the_test_split(tmp_path):
    rq.write_split_files(
        [{"query_id": "a", "split": "dev"}, {"query_id": "b", "split": "test"}], tmp_path
    )
    assert [r["query_id"] for r in rq.load_queries("dev", tmp_path)] == ["a"]
    with pytest.raises(PermissionError):
        rq.load_queries("test", tmp_path)
    assert [r["query_id"] for r in rq.load_queries("test", tmp_path, allow_test=True)] == ["b"]
    with pytest.raises(ValueError):
        rq.load_queries("train", tmp_path)


def _inputs(tmp_path, qa_rows, splits_rows):
    paths = {name: tmp_path / name for name in ("qa.jsonl", "chunks.parquet", "splits.csv")}
    paths["qa.jsonl"].write_text("".join(json.dumps(r) + "\n" for r in qa_rows))
    pd.DataFrame(
        [
            chunk("acc20:0#0", "acc20:0", 1, "acc20", 2020, RISK),
            chunk("acc19:0#0", "acc19:0", 1, "acc19", 2019, RISK),
            chunk("bcc20:0#0", "bcc20:0", 2, "bcc20", 2020, OTHER),
        ]
    ).to_parquet(paths["chunks.parquet"])
    pd.DataFrame(splits_rows).to_csv(paths["splits.csv"], index=False)
    return paths


def _args(paths, tmp_path):
    return [
        "--qa",
        str(paths["qa.jsonl"]),
        "--chunks",
        str(paths["chunks.parquet"]),
        "--splits",
        str(paths["splits.csv"]),
        "--out-dir",
        str(tmp_path / "out"),
        "--bench",
        str(tmp_path),
    ]


def test_main_writes_both_splits_and_bench(tmp_path):
    pytest.importorskip("pyarrow")
    qa_rows = [
        qa("acc20:0:q1", "acc20:0"),
        qa("bcc20:0:q1", "bcc20:0", cik=2, evidence="retain senior engineers"),
    ]
    splits = [
        {"cik": 1, "ticker": "T1", "sector": "mining_energy", "split": "train"},
        {"cik": 2, "ticker": "T2", "sector": "mining_energy", "split": "train"},
    ]
    paths = _inputs(tmp_path, qa_rows, splits)
    rq.main(_args(paths, tmp_path))

    record = json.loads((tmp_path / "ret_queries.json").read_text())
    assert record["queries"] == 2 and record["leakage_companies_in_both_splits"] == 0
    assert record["dev"]["queries"] + record["test"]["queries"] == 2
    written = [
        r for s in ("dev", "test") for r in rq.load_queries(s, tmp_path / "out", allow_test=True)
    ]
    assert {r["query_id"] for r in written} == {"acc20:0:q1", "bcc20:0:q1"}
    assert len(record["outputs_sha256"]) == 2


def test_main_refuses_qa_from_non_train_companies(tmp_path):
    pytest.importorskip("pyarrow")
    splits = [{"cik": 1, "ticker": "T1", "sector": "mining_energy", "split": "dev"}]
    paths = _inputs(tmp_path, [qa("acc20:0:q1", "acc20:0")], splits)
    with pytest.raises(ValueError, match="non-train"):
        rq.main(_args(paths, tmp_path))


def test_check_only_additions():
    dev = [{"query_id": "d1", "split": "dev", "grades": {"a": 2}}]
    test = [{"query_id": "t1", "split": "test", "grades": {"b": 2}}]
    added = {"query_id": "t2", "split": "test", "grades": {"c": 2}}
    assert rq.check_only_additions({}, dev + test) == {"previous_files": False}
    ok = rq.check_only_additions({"dev": dev, "test": test}, dev + test + [added])
    assert ok["added_test_queries"] == 1 and ok["existing_test_queries_identical"] == 1
    with pytest.raises(ValueError, match="ret-dev"):
        rq.check_only_additions({"dev": dev, "test": test}, test + [{**added, "split": "dev"}])
    regraded = {**test[0], "grades": {"b": 1}}
    with pytest.raises(ValueError, match="existing ret-test"):
        rq.check_only_additions({"dev": dev, "test": test}, dev + [regraded])


def test_main_rebuild_with_an_extra_file_only_adds_test_queries(tmp_path):
    """In this tiny world company 1 is ret-dev and company 2 ret-test (seed 3, fixed)."""
    pytest.importorskip("pyarrow")
    qa_rows = [
        qa("acc20:0:q1", "acc20:0"),
        qa("bcc20:0:q1", "bcc20:0", cik=2, evidence="retain senior engineers"),
    ]
    splits = [
        {"cik": 1, "ticker": "T1", "sector": "mining_energy", "split": "train"},
        {"cik": 2, "ticker": "T2", "sector": "mining_energy", "split": "train"},
    ]
    paths = _inputs(tmp_path, qa_rows, splits)
    chunks = pd.read_parquet(paths["chunks.parquet"])
    more = chunk("bcc19:0#0", "bcc19:0", 2, "bcc19", 2019, "Tariffs on imported parts raise costs.")
    pd.concat([chunks, pd.DataFrame([more])]).to_parquet(paths["chunks.parquet"])
    rq.main(_args(paths, tmp_path))
    out = tmp_path / "out"
    dev_before = (out / "queries.dev.jsonl").read_text()
    assert {r["cik"] for r in rq.read_jsonl(out / "queries.test.jsonl")} == {2}

    def rebuild_with(extra_row):
        extra = tmp_path / "extra.jsonl"
        extra.write_text(json.dumps({**extra_row, "batch": "ret_test_extra_v1"}) + "\n")
        args = _args(paths, tmp_path)
        at = args.index("--qa") + 1
        args[at : at + 1] = [str(paths["qa.jsonl"]), str(extra)]
        rq.main(args)

    rebuild_with(qa("bcc19:0:q1", "bcc19:0", cik=2, evidence="tariffs on imported parts"))
    record = json.loads((tmp_path / "ret_queries.json").read_text())
    assert record["rebuild"]["dev_identical"] is True
    assert record["rebuild"]["added_test_queries"] == 1
    assert record["qa_pairs_by_batch"] == {"fl_m2_t14": 2, "ret_test_extra_v1": 1}
    assert (out / "queries.dev.jsonl").read_text() == dev_before

    with pytest.raises(ValueError, match="ret-dev"):  # a question from a dev company is refused
        rebuild_with(qa("acc19:0:q1", "acc19:0", cik=1, evidence="copper prices could reduce"))
