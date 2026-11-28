"""Numbers checker: every figure in an answer must match a reference value, or it is blocked (T6).

The spec's rule (M6, numbers checker): every figure must match fct_financials or the answer is
blocked. The reference values are exactly what the writer was given by the tools (get_financials
and compare_peers): each fact as last reported and as first reported (a filing quotes the original
figure; FL-M1 kept both because 51,866 facts were restated), the margins and revenue growth
computed from those facts, and the change in a margin between two of the years given.

find_figures(text) finds each figure and how it is written:
- money: "$1.23 billion", "-$0.4 billion", "US$812 million", "$5,512"; a bare number with a
  thousands separator ("1,234,567,000") counts as money too;
- percent: "12.5%", "12.5 percent"; points: "3.2 percentage points", "3.2 pp";
- any other bare number ("3", "0.25") is a figure as well: the writer is told to state no number
  except the figures it was given, so a stray number is a finding, not noise.
Not figures: years (a bare integer 1900-2100), FY2023, citation markers ([1], [2][3]), form names
(10-K, 10-Q), Item 1A, ordinals (1st), the day in a date (December 31, 2023), digits inside URLs or
words.

check_answer(text, allowed) compares each figure with the allowed values of a compatible kind
through figure_matches() (agents/figures.py, hand-written by the user): money -> usd values;
percent -> ratio or points values; points -> points values; bare -> usd, ratio or points. A figure
written with an explicit sign must match the value's sign; an unsigned figure is compared with the
absolute value, so "a net loss of $0.4 billion" matches -412,000,000 ("+" counts as a sign). The
cost of that leniency: "revenue rose 2%" would pass against an actual -2%; direction words are
judged by the rubric (R4), and this limit is documented.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from filinglens.agents.figures import figure_matches

SCALE_WORDS = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
MONTHS = (
    "january february march april may june july august september october november "
    "december jan feb mar apr jun jul aug sep sept oct nov dec"
).split()
YEAR_RANGE = (1900, 2100)

FIGURE = re.compile(
    r"(?<![\w./\-])"  # not inside a word, a number, a URL path or "10-K"-style tokens
    r"(?P<sign>[-+\u2212])?\s?"
    r"(?P<cur>US\$|\$)?\s?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?P<pct>%|percent\b)"
    r"|\s?(?P<pts>percentage points?\b|pp\b)"
    r"|\s(?P<unit>thousand|million|billion|trillion)\b)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Figure:
    text: str
    start: int
    end: int
    stated: float  # with the sign when written; unsigned otherwise
    decimals: int
    scale: float
    kind: str  # money | percent | points | bare
    signed: bool


@dataclass(frozen=True)
class Allowed:
    label: str
    value: float
    kind: str  # usd | ratio | points


@dataclass
class Check:
    ok: bool
    figures: list[dict] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _skip(text: str, m: re.Match) -> bool:
    """True for numbers that are not figures (see the module docstring)."""
    if m.group("cur") or m.group("unit") or m.group("pct") or m.group("pts"):
        return False
    num, start, end = m.group("num"), m.start("num"), m.end("num")
    after = text[end : end + 2]
    if text[start - 1 : start] == "[" and after[:1] == "]":
        return True  # citation marker
    if after[:1].isalpha() or (after[:1] == "-" and after[1:2].isalpha()):
        return True  # 1A, 1st, 10-K
    if "," not in num and "." not in num and YEAR_RANGE[0] <= int(num) <= YEAR_RANGE[1]:
        return True  # a year
    before = text[max(0, start - 12) : start].lower().rstrip()
    next_word = text[end : end + 12].lower().split()[:1]
    if "," not in num and "." not in num and int(num) <= 31:
        if (
            any(before.endswith(month) for month in MONTHS)
            or next_word[:1]
            and (next_word[0].strip(",.") in MONTHS)
        ):
            return True  # the day in "December 31, 2023" or "31 December 2023"
    return False


def find_figures(text: str) -> list[Figure]:
    out = []
    for m in FIGURE.finditer(text):
        if _skip(text, m):
            continue
        num = m.group("num")
        decimals = len(num.split(".")[1]) if "." in num else 0
        value = float(num.replace(",", ""))
        signed = m.group("sign") is not None
        if signed and m.group("sign") != "+":
            value = -value
        if m.group("pct"):
            kind, scale = "percent", 0.01
        elif m.group("pts"):
            kind, scale = "points", 0.01
        elif m.group("cur") or m.group("unit") or "," in num:
            kind = "money"
            scale = SCALE_WORDS[m.group("unit").lower()] if m.group("unit") else 1.0
        else:
            kind, scale = "bare", 1.0
        out.append(
            Figure(m.group(0).strip(), m.start(), m.end(), value, decimals, scale, kind, signed)
        )
    return out


COMPATIBLE = {
    "money": ("usd",),
    "percent": ("ratio", "points"),
    "points": ("points",),
    "bare": ("usd", "ratio", "points"),
}


def matches(fig: Figure, a: Allowed) -> bool:
    if a.kind not in COMPATIBLE[fig.kind]:
        return False
    actual = a.value if fig.signed else abs(a.value)
    return figure_matches(fig.stated, fig.decimals, fig.scale, actual)


def check_answer(text: str, allowed: Iterable[Allowed]) -> Check:
    allowed = list(allowed)
    result = Check(ok=True)
    for fig in find_figures(text):
        hit = next((a for a in allowed if matches(fig, a)), None)
        result.figures.append(
            {"text": fig.text, "kind": fig.kind, "matched": hit.label if hit else None}
        )
        if hit is None:
            result.ok = False
            result.problems.append(
                f"figure {fig.text!r} matches no reference value (it must equal a figure from"
                " get_financials, as written to its last digit)"
            )
    return result


# ---------------------------------------------------------------- reference values from the tools


def year_values(ticker: str, entry: dict) -> list[Allowed]:
    """The values of one get_financials year entry (or a compare_peers 'financials' entry)."""
    fy = entry["fiscal_year"]
    out = []
    for metric, f in entry.get("figures", {}).items():
        for which in ("value", "first_reported_value"):
            out.append(Allowed(f"{ticker} FY{fy} {metric} ({which})", float(f[which]), "usd"))
    for name, d in entry.get("derived", {}).items():
        out.append(Allowed(f"{ticker} FY{fy} {name}", float(d["value"]), "ratio"))
    return out


def margin_changes(ticker: str, entries: list[dict]) -> list[Allowed]:
    """Each margin's change between every two of the years given (later minus earlier)."""
    out = []
    by_year = {e["fiscal_year"]: e.get("derived", {}) for e in entries}
    years = sorted(by_year)
    for i, y1 in enumerate(years):
        for y2 in years[i + 1 :]:
            for name in set(by_year[y1]) & set(by_year[y2]):
                change = by_year[y2][name]["value"] - by_year[y1][name]["value"]
                out.append(Allowed(f"{ticker} {name} change FY{y1}-FY{y2}", change, "points"))
    return out


def allowed_values(financials: Iterable[dict] = (), peers: Iterable[dict] = ()) -> list[Allowed]:
    """Every reference value the writer was given: get_financials and compare_peers results."""
    out: list[Allowed] = []
    for res in financials:
        ticker = res["company"]["ticker"]
        for entry in res["years"]:
            out += year_values(ticker, entry)
        out += margin_changes(ticker, res["years"])
        for g in res.get("growth", []):
            out.append(
                Allowed(
                    f"{ticker} revenue growth FY{g['from']}-FY{g['to']}",
                    float(g["revenue_growth"]),
                    "ratio",
                )
            )
    for res in peers:
        if "company_financials" in res:
            out += year_values(res["company"]["ticker"], res["company_financials"])
        for p in res.get("peers", []):
            if "financials" in p:
                out += year_values(p["ticker"], p["financials"])
    return out
