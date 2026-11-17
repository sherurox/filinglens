"""Universe export on tiny CSV fixtures (DuckDB in memory) and the bq load command."""

from pathlib import Path

import duckdb

from filinglens.ingest.load_bigquery import bq_load_command, export_universe

FIXTURE = {
    "universe": "rank,cik,ticker\n1,1,AAA\n",
    "sub": "adsh,cik,form\nA,1,10-K\nB,2,10-K\n",
    "num": "adsh,tag,version,value\nA,Revenues,us-gaap/2024,10\nB,Assets,us-gaap/2024,20\n",
    "pre": "adsh,report,line\nA,1,1\nB,1,1\n",
    "tag": "tag,version,tlabel\nRevenues,us-gaap/2024,Rev\nAssets,us-gaap/2024,Ast\nX,v,X\n",
}


def test_export_keeps_only_universe_filings_and_their_tags(tmp_path: Path):
    for name, text in FIXTURE.items():
        (tmp_path / f"{name}.csv").write_text(text)
    sources = {t: f"read_csv('{(tmp_path / f'{t}.csv').as_posix()}')" for t in FIXTURE}
    stats = export_universe(duckdb.connect(), sources, tmp_path / "universe.csv", tmp_path / "out")
    assert {name: s["rows"] for name, s in stats.items()} == {
        "sub": 1,
        "num": 1,
        "pre": 1,
        "tag": 1,
    }
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "num.parquet",
        "pre.parquet",
        "sub.parquet",
        "tag.parquet",
    ]  # no .part files left behind


def test_bq_load_command_replaces_table_from_parquet():
    cmd = bq_load_command("proj", "filinglens_raw", "num", Path("data/x/num.parquet"))
    assert cmd[:3] == ["bq", "--project_id=proj", "--location=US"]
    assert "--source_format=PARQUET" in cmd and "--replace" in cmd
    decimal = [a for a in cmd if a.startswith("--decimal_target_types=")]
    assert decimal == ["--decimal_target_types=NUMERIC", "--decimal_target_types=BIGNUMERIC"]
    assert cmd[-2:] == ["filinglens_raw.num", "data/x/num.parquet"]
