"""Item 1A splitter: TOC entries lose to the real section, split and fallback headings, hidden
inline-XBRL text, emphasis detection, and paragraphs with their headings."""

from filinglens.extract.sections import (
    find_risk_section,
    html_to_blocks,
    risk_paragraphs,
    split_risk_factors,
)

BODY = " ".join(["risk"] * 40)  # one 40-word paragraph


def doc(body: str) -> bytes:
    return f"<html><head><title>10-K</title></head><body>{body}</body></html>".encode()


TOC = """
<table>
  <tr><td>Item 1.</td><td>Business</td><td>3</td></tr>
  <tr><td>Item 1A.</td><td>Risk Factors</td><td>12</td></tr>
  <tr><td>Item 1B.</td><td>Unresolved Staff Comments</td><td>30</td></tr>
</table>
"""

REAL = f"""
<div><span style="font-weight:bold">Item 1A. Risk Factors</span></div>
<p><b>Risks Related to Our Business</b></p>
<p><b><i>Competition could reduce our margins.</i></b></p>
<p>{BODY}</p>
<p>Too short to keep.</p>
<p><span style="font-weight: 700">Supply chain problems could hurt sales.</span></p>
<p>{BODY} supply</p>
<div><b>Item 1B. Unresolved Staff Comments</b></div>
<p>None.</p>
"""


def test_toc_entry_loses_to_the_real_section():
    blocks = html_to_blocks(doc(TOC + REAL))
    start, end, method = find_risk_section(blocks)
    assert method == "item"
    assert blocks[start].text == "Risks Related to Our Business"
    assert blocks[end].text == "Item 1B. Unresolved Staff Comments"


def test_paragraphs_keep_the_latest_emphasized_heading():
    section = split_risk_factors(doc(TOC + REAL), min_chars=100)
    assert section.status == "ok"
    assert [(p.index, p.heading) for p in section.paragraphs] == [
        (0, "Competition could reduce our margins."),
        (1, "Supply chain problems could hurt sales."),
    ]
    assert section.paragraphs[1].text.endswith("risk supply")


def test_heading_split_over_two_blocks():
    html = f"<p>ITEM 1A.</p><p>RISK FACTORS</p><p>{BODY}</p><p>ITEM 2. PROPERTIES</p>"
    start, end, method = find_risk_section(html_to_blocks(doc(html)))
    assert (start, end, method) == (2, 3, "item")


def test_table_cells_are_separated_by_spaces():
    blocks = html_to_blocks(doc("<table><tr><td>Item&#160;1A.</td><td>Risk Factors</td></tr>"))
    assert blocks == [blocks[0]] and blocks[0].text == "Item 1A. Risk Factors"


def test_hidden_inline_xbrl_header_and_display_none_are_skipped():
    html = (
        '<div style="display: none"><ix:header>dei:DocumentType 10-K</ix:header></div>'
        "<ix:header>more hidden</ix:header><p>visible</p><!-- a comment -->"
    )
    assert [b.text for b in html_to_blocks(doc(html))] == ["visible"]


def test_partly_bold_block_is_not_emphasized():
    blocks = html_to_blocks(doc("<p><b>Lead-in.</b> Then normal text.</p><p><u>All</u></p>"))
    assert [(b.text, b.emphasized) for b in blocks] == [
        ("Lead-in. Then normal text.", False),
        ("All", True),
    ]


def test_fallback_bare_risk_factors_heading():
    html = f"<p><b>Risk Factors</b></p><p>{BODY}</p><p><b>Unresolved SEC Staff Comments</b></p>"
    section = split_risk_factors(doc(html), min_chars=100)
    assert (section.status, section.method, len(section.paragraphs)) == ("ok", "heading", 1)


def test_short_item_entry_falls_back_to_the_bare_heading():
    # GE-style 10-K: a cross-reference index lists Item 1A with nothing under it, and the
    # real section has the bare heading "RISK FACTORS".
    html = (
        "<p>Item 1A. Risk Factors</p><p>Item 1B. Unresolved Staff Comments</p>"
        f"<p><b>RISK FACTORS</b></p><p>{BODY}</p><p><b>LEGAL PROCEEDINGS</b></p>"
    )
    blocks = html_to_blocks(doc(html))
    start, end, method = find_risk_section(blocks, min_chars=100)
    assert (method, blocks[start].text, blocks[end].text) == ("heading", BODY, "LEGAL PROCEEDINGS")


def test_short_section_is_reported_not_split():
    html = "<p>Item 1A. Risk Factors</p><p>Not applicable.</p><p>Item 1B. None</p>"
    section = split_risk_factors(doc(html))
    assert (section.status, section.chars, section.paragraphs) == ("too_short", 15, [])


def test_no_heading():
    assert split_risk_factors(doc(f"<p>{BODY}</p>")).status == "no_heading"


def test_candidate_without_an_end_is_dropped():
    html = f"<p>Item 1A. Risk Factors</p><p>{BODY}</p>"  # no Item 1B/2 after it
    assert find_risk_section(html_to_blocks(doc(html))) is None


def test_table_of_contents_label_is_not_a_heading():
    blocks = html_to_blocks(
        doc(f"<p><b>Real heading here</b></p><p><b>Table of Contents</b></p><p>{BODY}</p>")
    )
    assert risk_paragraphs(blocks)[0].heading == "Real heading here"
