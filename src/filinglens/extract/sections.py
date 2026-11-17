"""Find Item 1A (Risk Factors) in a 10-K HTML document and split it into paragraphs (FL-M2 T1).

Steps:
1. html_to_blocks: HTML -> text blocks, one per block-level element (p, div, tr, li, ...),
   skipping hidden content (the inline-XBRL header, display:none). Each block records whether
   all of its text is emphasized (bold, italic or underlined): risk-factor headings usually are.
2. find_risk_section: the table of contents also lists "Item 1A. Risk Factors", so every
   matching heading is a candidate; each candidate runs to the next item heading, and the
   longest candidate wins. A TOC entry is only a line long, so the real section always wins.
   Fallback for 10-Ks laid out like an annual report: a bare "Risk Factors" heading.
3. risk_paragraphs: body blocks with at least `min_words` words, each with the latest
   emphasized block above it as its heading.
"""

import argparse
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from lxml import html

BLOCK_TAGS = frozenset(
    {"p", "div", "li", "ul", "ol", "table", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}
    | {"center", "blockquote", "section", "article", "hr", "dt", "dd", "pre"}
)
SPACED_TAGS = frozenset({"td", "th", "br"})  # inline, but their text must not run together
SKIP_TAGS = frozenset({"script", "style", "head", "title", "noscript", "ix:header"})
EMPHASIS_TAGS = frozenset({"b", "strong", "i", "em", "u"})
EMPHASIS_STYLE = re.compile(
    r"font-weight:(bold|bolder|[6-9]00)|font-style:italic|text-decoration:[^;]*underline"
)
HEADING_MAX_WORDS = 12  # item headings are short; longer blocks are body text that mentions them

ITEM_1A = re.compile(r"item\s*1a\b[\s.:\-–—]*")
END_HEADING = re.compile(
    r"(item\s*(1b|1c|[2-9]|1[0-6])\b"
    r"|unresolved (sec )?staff comments\b"
    r"|cybersecurity\.?$|properties\.?$|legal proceedings\.?$)"
)
NOT_A_HEADING = re.compile(r"(table of contents|index|page|\d+)\.?")


@dataclass(frozen=True)
class Block:
    text: str
    emphasized: bool


@dataclass(frozen=True)
class Paragraph:
    index: int  # 0-based position among the kept paragraphs of the section
    heading: str  # latest emphasized block above the paragraph ("" if none)
    text: str


@dataclass(frozen=True)
class RiskSection:
    status: str  # ok | too_short | no_heading
    method: str = ""  # item (Item 1A heading) | heading (bare "Risk Factors" heading)
    chars: int = 0  # characters in the section body
    paragraphs: list[Paragraph] = field(default_factory=list)


def normalize(text: str) -> str:
    """NFKC (turns non-breaking spaces into spaces), drop zero-width spaces, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).replace("​", "")
    return re.sub(r"\s+", " ", text).strip()


def _style(el) -> str:
    return (el.get("style") or "").lower().replace(" ", "")


def _walk(el, emphasized: bool, pieces: list[tuple[str, bool]], flush) -> None:
    """Depth-first: text goes into `pieces`; block-level elements start and end a block."""
    tag = el.tag.lower() if isinstance(el.tag, str) else None  # None: comment or PI
    if tag is not None and tag not in SKIP_TAGS and "display:none" not in _style(el):
        mine = emphasized or tag in EMPHASIS_TAGS or bool(EMPHASIS_STYLE.search(_style(el)))
        if tag in BLOCK_TAGS:
            flush()
        if tag in SPACED_TAGS:
            pieces.append((" ", mine))
        if el.text:
            pieces.append((el.text, mine))
        for child in el:
            _walk(child, mine, pieces, flush)
        if tag in BLOCK_TAGS:
            flush()
        elif tag in SPACED_TAGS:
            pieces.append((" ", mine))
    if el.tail:  # text after the element belongs to the parent's style
        pieces.append((el.tail, emphasized))


def html_to_blocks(data: bytes) -> list[Block]:
    """Text blocks in document order. A block is emphasized if all its visible text is."""
    blocks: list[Block] = []
    pieces: list[tuple[str, bool]] = []

    def flush() -> None:
        text = normalize("".join(t for t, _ in pieces))
        if text:
            blocks.append(Block(text, all(e for t, e in pieces if t.strip())))
        pieces.clear()

    _walk(html.document_fromstring(data), False, pieces, flush)
    flush()
    return blocks


def _is_short(block: Block) -> bool:
    return len(block.text.split()) <= HEADING_MAX_WORDS


def _item_start(blocks: list[Block], i: int) -> int | None:
    """Index of the first body block if blocks[i] starts an Item 1A heading, else None."""
    text = blocks[i].text.lower()
    m = ITEM_1A.match(text)
    if not m or not _is_short(blocks[i]):
        return None
    rest = text[m.end() :]
    if rest.startswith("risk factors"):
        return i + 1
    nxt = i + 1 < len(blocks) and blocks[i + 1].text.lower().startswith("risk factors")
    if rest == "" and nxt:  # heading split over two blocks: "ITEM 1A." / "RISK FACTORS"
        return i + 2
    return None


def _heading_start(blocks: list[Block], i: int) -> int | None:
    """Fallback: a block that is exactly "Risk Factors" (annual-report style 10-Ks)."""
    return i + 1 if blocks[i].text.lower().rstrip(".:") == "risk factors" else None


def _end(blocks: list[Block], start: int) -> int | None:
    for j in range(start, len(blocks)):
        if _is_short(blocks[j]) and END_HEADING.match(blocks[j].text.lower()):
            return j
    return None


def _longest(blocks: list[Block], starts) -> tuple[int, int, int] | None:
    """(chars, start, end) of the longest candidate found by `starts`, or None.

    A candidate with no following item heading is dropped: it would run to the end of the
    document.
    """
    best: tuple[int, int, int] | None = None
    for i in range(len(blocks)):
        start = starts(blocks, i)
        if start is None:
            continue
        end = _end(blocks, start)
        if end is None:
            continue
        chars = sum(len(b.text) for b in blocks[start:end])
        if best is None or chars > best[0]:
            best = (chars, start, end)
    return best


def find_risk_section(blocks: list[Block], min_chars: int = 2000) -> tuple[int, int, str] | None:
    """(first body block, end, method) of the risk-factor section, or None.

    Item 1A headings come first. Only if the best Item 1A section is shorter than `min_chars`
    (a cross-reference index entry, or "incorporated by reference") is the bare "Risk Factors"
    heading tried too, and the longer of the two wins (Item 1A on a tie).
    """
    item = _longest(blocks, _item_start)
    if item is not None and item[0] >= min_chars:
        return item[1], item[2], "item"
    heading = _longest(blocks, _heading_start)
    options = [(found, method) for found, method in ((item, "item"), (heading, "heading")) if found]
    if not options:
        return None
    (_, start, end), method = max(options, key=lambda option: option[0][0])
    return start, end, method


def risk_paragraphs(blocks: list[Block], min_words: int = 30) -> list[Paragraph]:
    """Body paragraphs of a section, each with the latest emphasized block above it."""
    heading, out = "", []
    for block in blocks:
        words = len(block.text.split())
        if block.emphasized and words >= 3 and not NOT_A_HEADING.fullmatch(block.text.lower()):
            heading = block.text
        elif not block.emphasized and words >= min_words:
            out.append(Paragraph(len(out), heading, block.text))
    return out


def split_risk_factors(data: bytes, min_chars: int = 2000, min_words: int = 30) -> RiskSection:
    """The whole pipeline for one 10-K. Sections under `min_chars` are reported, not split."""
    blocks = html_to_blocks(data)
    found = find_risk_section(blocks, min_chars)
    if found is None:
        return RiskSection("no_heading")
    start, end, method = found
    body = blocks[start:end]
    chars = sum(len(b.text) for b in body)
    if chars < min_chars:  # e.g. "Not applicable" or incorporated by reference
        return RiskSection("too_short", method, chars)
    return RiskSection("ok", method, chars, risk_paragraphs(body, min_words))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check the Item 1A split on a few 10-K files.")
    parser.add_argument("paths", nargs="+", type=Path)
    for path in parser.parse_args(argv).paths:
        section = split_risk_factors(path.read_bytes())
        headings = sorted({p.heading for p in section.paragraphs if p.heading})
        print(
            f"{path}: status={section.status} method={section.method} chars={section.chars} "
            f"paragraphs={len(section.paragraphs)} headings={len(headings)}"
        )
        for p in section.paragraphs[:2]:
            print(f"  [{p.index}] heading: {p.heading[:90]!r}")
            print(f"       text: {p.text[:140]!r}")


if __name__ == "__main__":
    main()
