"""Parse Exhibit 21 subsidiary lists and 10-K signature pages. Pure functions, no I/O.

Exhibit 21: most filers use a table (name | jurisdiction [| more]); some use lines of text
("Name (Delaware)", "Name - Delaware", "Name, a Delaware corporation"). PDFs are not parsed.

Signatures: only rows after the last "SIGNATURES" heading count, so the auditor's "/s/" signature
earlier in the 10-K is ignored. The name follows "/s/"; the title is the next cell that isn't a
date, or the text after the first comma when name and title share one cell.
"""

import re
from dataclasses import dataclass

from lxml import html

from filinglens.extract.sections import BLOCK_TAGS, normalize

ENTITY_HINT = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|llc|l\.l\.c|ltd|limited|lp|l\.p|llp|plc"
    r"|gmbh|ag|sa|s\.a|sas|s\.a\.s|sarl|s\.a\.r\.l|s\.à\.?\s?r\.l|srl|s\.r\.l|bv|b\.v|nv|n\.v"
    r"|pty|pte|kk|k\.k|ab|as|oy|spa|s\.p\.a|kft|sro|s\.r\.o|ulc|lllp|trust|holdings?|partners"
    r"|group|bank|fund|de c\.v|ltda|limitada|sociedad)\b\.?",
    re.IGNORECASE,
)
HEADER_WORDS = re.compile(
    r"\b(name|jurisdiction|subsidiar(y|ies)|incorporat\w*|organi[sz]ation|organized|country"
    r"|state|province|domicile|exhibit|entity|ownership|percent)\b",
    re.IGNORECASE,
)
BOILERPLATE = re.compile(
    r"subsidiaries of|exhibit 21|omitted|pursuant to|item 601|the following|as of |listed below"
    r"|significant subsidiar|in the aggregate|wholly[- ]owned|registrant",
    re.IGNORECASE,
)
FOOTNOTES = re.compile(r"(\s*\(\d+\))+\d*$|[*†‡§¹²³⁴⁵⁶⁷⁸⁹]+$|\s+\d{1,2}$")
LEADING = re.compile(r"^([•·\-–—*]|\d{1,3}[.)])\s+")
NUMERIC = re.compile(r"^[\d.,%\s()]+$")
TRAILING_PARENS = re.compile(r"^(.*\S)\s*\(([^()]{2,40})\)$")
DASH_SPLIT = re.compile(r"\s+[-–—]\s+")
A_STATE_CORP = re.compile(
    r"^(.*?),\s+an?\s+(.+?)\s+(corporation|limited liability company|limited partnership"
    r"|company|partnership|trust)$",
    re.IGNORECASE,
)
MAX_NAME_CHARS = 150

SIGNATURES_HEADING = re.compile(r"^signatures?[.:]?$", re.IGNORECASE)
SLASH_S = re.compile(r"/s/\s*", re.IGNORECASE)
DATE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|^date",
    re.IGNORECASE,
)
SUFFIX = r"(jr|sr|ii|iii|iv|v|phd|ph\.d|md|m\.d|esq|cpa)\.?"
NAME_SUFFIX = re.compile(r",?\s+" + SUFFIX + r"$", re.IGNORECASE)
CAPS_THEN_MIXED = re.compile(r"^([A-Z][A-Z .,'\-]+?)(?=[A-Z][a-z])")
TITLE_HINT = re.compile(
    r"officer|director|president|chair|controller|treasurer|secretary|counsel|trustee|principal"
    r"|executive|chief|vice|manager|partner",
    re.IGNORECASE,
)
FIRM = re.compile(r"\b(llp|llc|inc|corporation|company|& co)\b\.?", re.IGNORECASE)
ROLES = {
    "ceo": re.compile(r"chief executive|principal executive", re.IGNORECASE),
    "cfo": re.compile(r"chief financial|principal financial", re.IGNORECASE),
    "cao": re.compile(r"chief accounting|principal accounting|controller", re.IGNORECASE),
    "director": re.compile(r"director|chair(man|woman|person)?\b|trustee", re.IGNORECASE),
}


@dataclass(frozen=True)
class Subsidiary:
    name: str
    jurisdiction: str


@dataclass(frozen=True)
class Signer:
    name: str
    title: str
    roles: tuple[str, ...]


def load_html(data: bytes):
    """Parse HTML bytes. A declared charset is left to lxml; otherwise UTF-8 (else cp1252),
    because lxml reads undeclared bytes as Latin-1 and mangles "à" or zero-width spaces."""
    head = data[:2048].lower()
    if b"charset" in head or b"encoding=" in head:
        return html.document_fromstring(data)
    try:
        return html.document_fromstring(data.decode("utf-8"))
    except UnicodeDecodeError:
        return html.document_fromstring(data.decode("cp1252", errors="replace"))


# ---------- Exhibit 21 ----------


def clean_name(text: str) -> str:
    text = LEADING.sub("", normalize(text))
    for _ in range(2):  # "(3)2" and "Name 1" style marks, possibly stacked
        text = FOOTNOTES.sub("", text).strip()
    return text.strip(" ,;")


def is_header(cells: list[str]) -> bool:
    joined = " ".join(cells)
    return bool(HEADER_WORDS.search(joined)) and not ENTITY_HINT.search(cells[0])


def keep(name: str, jurisdiction: str) -> bool:
    """A plausible subsidiary: not boilerplate, not too long, an entity hint or a jurisdiction."""
    if not name or len(name) > MAX_NAME_CHARS or not re.search(r"[A-Za-z]", name):
        return False
    if BOILERPLATE.search(name):
        return False
    return bool(ENTITY_HINT.search(name)) or bool(jurisdiction)


def from_rows(rows: list[list[str]]) -> list[Subsidiary]:
    out = []
    for cells in rows:
        cells = [c for c in (normalize(c) for c in cells) if c]
        if not cells or is_header(cells):
            continue
        name = clean_name(cells[0])
        jurisdiction = ", ".join(c for c in cells[1:] if not NUMERIC.match(c))
        if len(cells) == 1 and not ENTITY_HINT.search(name):
            continue  # a lone word such as a country group heading
        if keep(name, jurisdiction):
            out.append(Subsidiary(name, jurisdiction))
    return out


def split_line(line: str) -> Subsidiary:
    """Name and jurisdiction from one line of a text-format Exhibit 21."""
    parts = [p for p in re.split(r"\t|\s{3,}", line.strip()) if p.strip()]
    if len(parts) >= 2 and not NUMERIC.match(parts[1]):
        return Subsidiary(clean_name(parts[0]), normalize(parts[1]))
    line = clean_name(line)
    if m := A_STATE_CORP.match(line):
        return Subsidiary(m.group(1).strip(), m.group(2).strip())
    if (m := TRAILING_PARENS.match(line)) and not NUMERIC.match(m.group(2)):
        if not ENTITY_HINT.fullmatch(m.group(2).strip()):  # "(Pty) Ltd" style is part of a name
            return Subsidiary(m.group(1).strip(" ,"), m.group(2).strip())
    pieces = DASH_SPLIT.split(line)
    if len(pieces) >= 2 and len(pieces[-1].split()) <= 4:
        return Subsidiary(" - ".join(pieces[:-1]).strip(), pieces[-1].strip())
    return Subsidiary(line, "")


def text_lines(doc) -> list[str]:
    """Lines of an HTML or text document: block tags and <br> end a line, as do newlines."""
    for el in doc.iter():
        tag = el.tag.lower() if isinstance(el.tag, str) else ""
        if tag in BLOCK_TAGS or tag == "br":
            el.tail = "\n" + (el.tail or "")
    return [line for line in doc.text_content().splitlines() if line.strip()]


def parse_ex21(data: bytes) -> list[Subsidiary]:
    """Subsidiaries listed in one Exhibit 21 document, in order, without duplicates."""
    if data[:5] == b"%PDF-" or not data.strip():
        return []
    doc = load_html(data)
    rows = [[td.text_content() for td in tr.xpath("./td|./th")] for tr in doc.xpath("//tr")]
    multi = [r for r in rows if len([c for c in r if normalize(c)]) >= 2]
    if len(multi) >= 3:
        subs = from_rows(rows)
    else:
        subs = []
        for line in text_lines(doc):
            sub = split_line(line)
            if keep(sub.name, sub.jurisdiction) and not is_header([sub.name]):
                subs.append(sub)
    seen, unique = set(), []
    for s in subs:
        if s.name.lower() not in seen:
            seen.add(s.name.lower())
            unique.append(s)
    return unique


# ---------- signatures ----------


def roles(title: str) -> tuple[str, ...]:
    return tuple(role for role, rx in ROLES.items() if rx.search(title))


def person_key(name: str) -> str:
    """Matching key: lower case, no punctuation, suffixes or middle initials."""
    name = NAME_SUFFIX.sub("", normalize(name))
    tokens = re.sub(r"[^a-z\s'-]", " ", name.lower()).replace("'", "").split()
    tokens = [t for t in tokens if len(t) > 1 and not re.fullmatch(SUFFIX, t)]
    return " ".join(tokens)


def split_signer(text: str) -> tuple[str, str]:
    """(name, title) from the text after "/s/" when one cell may hold both.

    "PHILIP B. DANIELE, III" -> the suffix stays in the name; "MICHAEL K. WIRTH Michael K.
    Wirth, Chairman" -> the ALL-CAPS copy is the name, and the title follows the comma.
    """
    parts = [p.strip() for p in normalize(text).split(",")]
    head, rest = parts[0], parts[1:]
    if rest and re.fullmatch(SUFFIX, rest[0], re.IGNORECASE):
        head, rest = f"{head}, {rest[0]}", rest[1:]
    m = CAPS_THEN_MIXED.match(head)
    if m and head[m.end() :].upper().startswith(m.group(1).split()[0]):
        head = m.group(1)  # only when the mixed-case repeat of the same name follows
    return head.strip(" ,"), ", ".join(rest).strip()


def title_in(cells: list[str], key: str) -> str:
    """The first cell that reads like a title: not a date, a signature or the name again."""
    for cell in cells:
        if DATE.search(cell) or SLASH_S.search(cell) or person_key(cell) == key:
            continue
        if TITLE_HINT.search(cell):
            return cell
    return ""


def signature_rows(doc) -> list[list[str]]:
    """Table rows after the last SIGNATURES heading (empty cells dropped)."""
    order = {el: i for i, el in enumerate(doc.iter())}
    marks = [
        order[el]
        for el in doc.iter()
        if isinstance(el.tag, str)
        and len(el.text_content()) < 40
        and SIGNATURES_HEADING.match(normalize(el.text_content()))
    ]
    start = max(marks, default=-1)
    rows = []
    for tr in doc.xpath("//tr"):
        if order[tr] > start:
            cells = [normalize(td.text_content()) for td in tr.xpath("./td|./th")]
            rows.append([c for c in cells if c])
    return rows


def parse_signatures(data: bytes) -> list[Signer]:
    """People who signed the 10-K, with title and roles; one entry per person."""
    doc = load_html(data)
    rows = signature_rows(doc)
    found: dict[str, Signer] = {}
    for i, cells in enumerate(rows):
        slash = [j for j, c in enumerate(cells) if SLASH_S.search(c)]
        for j in slash:
            if "attorney" in cells[j].lower():
                continue  # "*By: /s/ X, Attorney-in-Fact" signs for others, not as an officer
            name, title = split_signer(SLASH_S.split(cells[j], maxsplit=1)[1])
            key = person_key(name)
            if title and not TITLE_HINT.search(title):
                title = ""  # "M.D." or a repeated name is not a title
            if not title and len(slash) == 1:
                title = title_in(cells[j + 1 :], key)
                for below in rows[i + 1 : i + 4]:  # stacked layout: title in the rows below
                    if title or any(SLASH_S.search(c) for c in below):
                        break
                    title = title_in(below, key)
            if not title and len(slash) > 1:  # grid layout: the title sits under the signature
                for below in rows[i + 1 : i + 3]:
                    if j < len(below) and not SLASH_S.search(below[j]):
                        cell = split_signer(below[j])[1] or below[j]
                        if title := title_in([cell], key):
                            break
            if not key or len(key.split()) > 5 or FIRM.search(name):
                continue
            if cells[0].lower().rstrip(":") == "by" and key in found:
                continue
            old = found.get(key)
            if old is None or (not old.title and title):
                found[key] = Signer(name, title, roles(title))
    return list(found.values())
