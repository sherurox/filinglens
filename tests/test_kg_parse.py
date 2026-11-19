import pytest

from filinglens.kg.parse import (
    Subsidiary,
    clean_name,
    parse_ex21,
    parse_signatures,
    person_key,
    roles,
    split_line,
    split_signer,
)


def table(*rows: list[str]) -> bytes:
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<html><body><table>{body}</table></body></html>".encode()


def test_ex21_table_with_header_and_footnotes():  # DoorDash / ANSYS layouts
    data = table(
        [""],
        ["Name of Subsidiary", "State or Jurisdiction of Organization"],
        ["Wolt Oy (46)", "Finland"],
        ["DoorDash Essentials HoldCo, LLC (3)2", "Delaware (United States)"],
        ["ANSYS Luxembourg Holding Company S.à.r.l.", "Luxembourg"],
        ["Wolt Oy", "Finland"],  # duplicate after cleaning
    )
    assert parse_ex21(data) == [
        Subsidiary("Wolt Oy", "Finland"),
        Subsidiary("DoorDash Essentials HoldCo, LLC", "Delaware (United States)"),
        Subsidiary("ANSYS Luxembourg Holding Company S.à.r.l.", "Luxembourg"),
    ]


def test_ex21_three_columns_title_row_and_group_heading():  # Aon layout
    data = table(
        ["Aon plc - Worldwide Subsidiaries as of December 31, 2025", "Exhibit 21"],
        ["Legal Entity Name", "Country Incorporated", "State/Province"],
        ["Argentina"],
        ["Admiseg S.A.", "Argentina"],
        ["Aon Risk Services, Inc. of Illinois", "United States", "Illinois"],
        ["Aon UK Holdings Ltd", "United Kingdom", "100%"],
    )
    assert parse_ex21(data) == [
        Subsidiary("Admiseg S.A.", "Argentina"),
        Subsidiary("Aon Risk Services, Inc. of Illinois", "United States, Illinois"),
        Subsidiary("Aon UK Holdings Ltd", "United Kingdom"),
    ]


def test_ex21_text_format():
    data = (
        b"<html><body><p>SUBSIDIARIES OF THE REGISTRANT</p>"
        b"<p>Acme Widgets, Inc. (Delaware)</p><p>Acme Europe B.V. - Netherlands</p>"
        b"<p>Acme Holdings LLC, a Delaware limited liability company</p>"
        b"Acme (Pty) Ltd<br/>Acme Trading Co.<br/>"
        b"<p>All other subsidiaries have been omitted pursuant to Item 601(b)(21).</p>"
        b"</body></html>"
    )
    assert parse_ex21(data) == [
        Subsidiary("Acme Widgets, Inc.", "Delaware"),
        Subsidiary("Acme Europe B.V.", "Netherlands"),
        Subsidiary("Acme Holdings LLC", "Delaware"),
        Subsidiary("Acme (Pty) Ltd", ""),
        Subsidiary("Acme Trading Co.", ""),
    ]


def test_ex21_pre_text_with_tabs():
    lines = b"Name\tJurisdiction\nAlpha Corp\tDelaware\nBeta GmbH   Germany\n"
    data = b"<html><body><pre>" + lines + b"</pre></body></html>"
    assert parse_ex21(data) == [
        Subsidiary("Alpha Corp", "Delaware"),
        Subsidiary("Beta GmbH", "Germany"),
    ]


def test_ex21_pdf_and_empty():
    assert parse_ex21(b"%PDF-1.7 binary") == []
    assert parse_ex21(b"   ") == []


@pytest.mark.parametrize(
    ("raw", "name"),
    [("Wolt Oy (46)", "Wolt Oy"), ("1. Beta Inc.", "Beta Inc."), ("Gamma LLC*", "Gamma LLC")],
)
def test_clean_name(raw, name):
    assert clean_name(raw) == name


def test_split_line_keeps_inner_parentheses():
    assert split_line("Synopsys (India) Private Limited") == Subsidiary(
        "Synopsys (India) Private Limited", ""
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("PHILIP B. DANIELE, III", ("PHILIP B. DANIELE, III", "")),
        (
            "MICHAEL K. WIRTH Michael K. Wirth, Chairman of the Board and Chief Executive Officer",
            ("MICHAEL K. WIRTH", "Chairman of the Board and Chief Executive Officer"),
        ),
        (
            "EIMEAR P. BONNEREimear P. Bonner, Chief Financial Officer",
            ("EIMEAR P. BONNER", "Chief Financial Officer"),
        ),
        ("JD Smith", ("JD Smith", "")),
        (
            "Jane Doe, Senior Vice President, Finance",
            ("Jane Doe", "Senior Vice President, Finance"),
        ),
    ],
)
def test_split_signer(text, expected):
    assert split_signer(text) == expected


def test_person_key_ignores_case_initials_and_suffixes():
    assert person_key("PHILIP B. DANIELE, III") == person_key("Philip Daniele") == "philip daniele"
    assert person_key("Mary O'Neil Jr.") == "mary oneil"


def test_roles():
    assert roles("President and Chief Executive Officer (Principal Executive Officer)") == ("ceo",)
    assert roles("Chairman of the Board and Chief Executive Officer") == ("ceo", "director")
    assert roles("Vice President and Controller") == ("cao",)
    assert roles("Senior Vice President") == ()


def page(*parts: str) -> bytes:
    return ("<html><body>" + "".join(parts) + "</body></html>").encode()


def rows(*rs: list[str]) -> str:
    return (
        "<table>"
        + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rs)
        + "</table>"
    )


def test_signatures_azo_layout_and_auditor_ignored():
    data = page(
        rows(["/s/ PricewaterhouseCoopers LLP"]),  # auditor's report, before SIGNATURES
        "<p><b>SIGNATURES</b></p>",
        rows(["​", "By:", "/s/ Philip B. Daniele, III"]),
        rows(
            ["SIGNATURE", "TITLE", "DATE"],
            [
                "/s/ PHILIP B. DANIELE, III",
                "​",
                "President and Chief Executive Officer",
                "October 27, 2025",
            ],
            ["/s/ JAMERE JACKSON", "Chief Financial Officer", "October 27, 2025"],
            ["/s/ WILLIAM C. RHODES, III", "Executive Chairman", "October 27, 2025"],
        ),
    )
    signers = parse_signatures(data)
    assert [(s.name, s.title, s.roles) for s in signers] == [
        ("PHILIP B. DANIELE, III", "President and Chief Executive Officer", ("ceo",)),
        ("JAMERE JACKSON", "Chief Financial Officer", ("cfo",)),
        ("WILLIAM C. RHODES, III", "Executive Chairman", ("director",)),
    ]


CEO = "Chief Executive Officer"


def test_load_html_respects_declared_charset():
    from filinglens.kg.parse import load_html

    latin = "<html><head><meta charset='iso-8859-1'></head><body>S.\xe0 r.l.</body></html>"
    assert "S.à r.l." in load_html(latin.encode("latin-1")).text_content()
    assert "S.à r.l." in load_html("<p>S.à r.l.</p>".encode()).text_content()


def test_signatures_cvx_single_cell_and_grid():
    data = page(
        "<p>Signatures</p>",
        rows(["/s/ MICHAEL K. WIRTH Michael K. Wirth, Chairman of the Board and " + CEO]),
        rows(
            ["/s/ ALANA K. KNOWLES", "/s/ JOHN B. FRANK"],
            ["Alana K. Knowles, Controller", "Director"],
        ),
        rows(["*By: /s/ Jane Roe, Attorney-in-Fact"]),
    )
    assert [(s.name, s.title) for s in parse_signatures(data)] == [
        ("MICHAEL K. WIRTH", "Chairman of the Board and " + CEO),
        ("ALANA K. KNOWLES", "Controller"),
        ("JOHN B. FRANK", "Director"),
    ]


def test_signatures_without_heading_reads_all_rows():
    data = page(rows(["/s/ Chris Turner", "Chief Executive Officer"]))
    assert [s.name for s in parse_signatures(data)] == ["Chris Turner"]


def test_signatures_abt_name_repeated_and_md_suffix():
    data = page(
        "<p>SIGNATURES</p>",
        rows(
            ["/s/ ROBERT B. FORD", "Robert B. Ford", "Chairman of the Board and " + CEO],
            ["/s/ NITA AHUJA, M.D.", "Nita Ahuja, M.D.", "Director"],
        ),
    )
    assert [(s.name, s.title) for s in parse_signatures(data)] == [
        ("ROBERT B. FORD", "Chairman of the Board and " + CEO),
        ("NITA AHUJA, M.D.", "Director"),
    ]


def test_signatures_amzn_stacked_layout():
    data = page(
        "<p>SIGNATURES</p>",
        rows(
            ["/s/ Andrew R. Jassy"],
            ["Andrew R. Jassy"],
            ["President and " + CEO + " (Principal Executive Officer)", "February 6, 2026"],
            ["/s/ Jeffrey P. Bezos"],
            ["Jeffrey P. Bezos"],
            ["Executive Chair"],
            ["/s/ Keith B. Alexander"],
            ["Keith B. Alexander"],
        ),
    )
    assert [(s.name, s.title, s.roles) for s in parse_signatures(data)] == [
        ("Andrew R. Jassy", "President and " + CEO + " (Principal Executive Officer)", ("ceo",)),
        ("Jeffrey P. Bezos", "Executive Chair", ("director",)),
        ("Keith B. Alexander", "", ()),
    ]
