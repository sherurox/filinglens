"""Table pages: number parsing, EDGAR layout cells, logical rows, captions, filing choice."""

import csv
import random

import pytest
from lxml import html

from filinglens.extract.table_pages import (
    canonical,
    caption,
    choose_filings,
    is_candidate,
    kind_of,
    logical_rows,
    page_html,
    row_cells,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("391,035", "391035"),
        ("$391,035", "391035"),
        ("(1,234)", "-1234"),
        ("$(1,234)", "-1234"),
        ("12.50%", "12.5"),
        ("0.0", "0"),
        ("-7", "-7"),
        ("2025", "2025"),
        ("—", "—"),
        ("—%", "—"),
        ("$—", "—"),
        ("Net sales", None),
        ("1,23", None),
        ("N/A", None),
    ],
)
def test_canonical(text, expected):
    assert canonical(text) == expected


def tr(cells: str):
    return html.fragment_fromstring(f"<table><tr>{cells}</tr></table>").find(".//tr")


def test_row_cells_join_dollar_parenthesis_and_percent_cells():
    row = tr(
        "<td>Net income</td><td>$</td><td>(1,234</td><td>)</td><td></td><td>12.5</td><td>%</td>"
    )
    assert row_cells(row) == ["Net income", "$(1,234)", "12.5%"]


TABLE = html.fragment_fromstring(
    """<div><p>CONSOLIDATED STATEMENTS OF OPERATIONS</p><table>
    <tr><td></td><td colspan="2">2025</td><td></td><td colspan="2">2024</td></tr>
    <tr><td>Net sales</td><td>$</td><td>391,035</td><td></td><td>$</td><td>383,285</td></tr>
    <tr><td>Cost of sales</td><td></td><td>(210,352</td><td>)</td>
        <td></td><td>(214,137</td><td>)</td></tr>
    <tr><td>Other income</td><td></td><td>&#8212;</td><td></td><td></td><td>565</td></tr>
    <tr><td>Operating expenses:</td></tr>
    </table></div>"""
)


def test_logical_rows_keep_labels_and_numbers_in_reading_order():
    rows = logical_rows(TABLE.find("table"))
    assert rows == [
        {"label": "", "values": ["2025", "2024"]},
        {"label": "Net sales", "values": ["391035", "383285"]},
        {"label": "Cost of sales", "values": ["-210352", "-214137"]},
        {"label": "Other income", "values": ["—", "565"]},
        {"label": "Operating expenses:", "values": []},
    ]


def test_caption_kind_and_candidate_rule():
    table = TABLE.find("table")
    assert caption(table) == "CONSOLIDATED STATEMENTS OF OPERATIONS"
    assert kind_of(caption(table)) == "income_statement"
    assert kind_of("Consolidated Statements of Cash Flows") == "cash_flow"
    assert kind_of("Selected quarterly data") == "other"
    rows = logical_rows(table)
    assert not is_candidate(rows)  # only 8 numbers
    assert is_candidate(rows * 2)


def test_page_html_escapes_the_caption():
    out = page_html(TABLE.find("table"), "Q&A <b>")
    assert "Q&amp;A &lt;b&gt;" in out and "391,035" in out


def test_choose_filings_one_per_company_round_robin_by_sector(tmp_path):
    rows = []
    for sector in ("finance", "retail"):
        for cik in range(3):
            for year in (2018, 2020, 2024):
                path = tmp_path / f"{sector}{cik}_{year}.htm"
                path.write_text("<html></html>")
                meta = {"cik": f"{sector}{cik}", "ticker": "T", "sector": sector}
                rows.append(
                    meta
                    | {"report_date": f"{year}-12-31", "accession": f"a{cik}{year}"}
                    | {"path": str(path)}
                )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    chosen = choose_filings(manifest, random.Random(1))
    assert len(chosen) == 6 and len({c["cik"] for c in chosen}) == 6
    assert [c["sector"] for c in chosen[:2]] == ["finance", "retail"]
    assert all(c["report_date"][:4] != "2018" for c in chosen)
