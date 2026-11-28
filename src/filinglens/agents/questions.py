"""Analyst questions for the FL-M6 crew, generated from templates and data (FL-M6 T2).

Each question asks for figures from marts.fct_financials and for what a company's 10-K Item 1A
says about one risk theme. Everything a scorer needs is stored with it: the exact reference
figures (tag, period, unit, value as last and as first reported, accession), the derived values
computed from them, the figures that are missing on purpose, and the chunks of each 10-K whose
text matches the theme (the scorer's aid for rubric item R4). No model is involved, so the
references are exact by construction. The rubric is docs/AGENT_RUBRIC.md.

Splits are FL-M2's company splits (gold/splits.csv): dev questions from FL-M2 dev companies,
test questions from FL-M2 test companies. Neither set trained the student or tuned retrieval
(FL-M3's queries come from FL-M2 train companies), and no company is in both splits.

Fiscal year = the year of a 10-K's report date (period end), as `period_year` in the corpus, so
"FY2020" in a label means the fiscal year that ended in calendar 2020; for a company whose year
ends in late January that is its own "fiscal 2019". The question text therefore never says
"FY2020": it says "the fiscal year that ended in 2020", which has one meaning for every company,
and each filing's exact report date is stored with its evidence. Its
annual figures are the facts with duration_quarters = 4 and unit USD dated within
DATE_TOLERANCE_DAYS of the report date: the SEC data sets date values at a month end (a 10-K for
the year ended 28 Sep has its facts dated 30 Sep), so exact equality would miss 52/53-week years.
When two fact dates fall in that window, the year has no figures (ambiguous, skipped). Revenue is
the first tag present in REVENUE_TAGS, as in FL-M1's ratios.

Templates (counts per split in COUNTS):
- risk_change_metric: how a theme's language changed between two 10-Ks at least MIN_YEAR_GAP
  years apart, and how revenue or operating margin changed;
- metric_margin_risk: revenue, operating income and the implied operating margin in one year,
  and what that year's 10-K says about a theme;
- peer_compare: net margins of two same-sector companies in one year, and each one's theme text;
- growth: revenue growth between consecutive years, and the later 10-K's theme text;
- missing_data (unanswerable by design): revenue that fct_financials doesn't have for that year
  (no revenue tag at the year's fact date), else revenue for EARLY_YEAR, before the data starts.
  The right answer says the figure is unavailable instead of inventing one.

Choices are deterministic: candidates are ordered by sha256(SEED:template:key) and spread over
sectors round-robin; a company appears in at most one question per split.

Outputs: gold/agents/questions.{dev,test}.jsonl (committed before any answer exists) and
bench/agent_questions.json (counts and SHA-256 only, so the test questions' text stays out of
the bench record). Reading the test file needs load_questions("test", allow_test=True). `check`
re-queries every reference figure, recomputes every derived value, re-counts every missing
figure and finds every evidence chunk again.

    uv run python -m filinglens.agents.questions generate
    uv run python -m filinglens.agents.questions check --split dev
    uv run python -m filinglens.agents.questions check --split test --allow-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from filinglens.api.logic import sec_filing_url
from filinglens.retrieval.config import CHUNKS

SEED = 20261128
SPLITS_FILE = Path("gold/splits.csv")
CORPUS = Path("data/interim/risk_paragraphs.parquet")  # FL-M2: one row per paragraph
OUT_DIR = Path("gold/agents")
STATS = Path("bench/agent_questions.json")
SPLITS = ("dev", "test")

REVENUE_TAGS = (  # = features.bench_ratios.REVENUE_TAGS (FL-M1), checked by a test
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
)
OPERATING_TAG = "OperatingIncomeLoss"
NET_TAG = "NetIncomeLoss"
FACT_TAGS = (*REVENUE_TAGS, OPERATING_TAG, NET_TAG)
DATE_TOLERANCE_DAYS = 20
MIN_YEAR_GAP = 3
EARLY_YEAR = 2010  # fct_financials' value dates start in 2012 (FL-M1)
EVIDENCE_CAP = 20  # chunk ids kept per filing (the total is kept too)

TEMPLATES = ("risk_change_metric", "metric_margin_risk", "peer_compare", "growth", "missing_data")
# missing_data first: its candidates are rare, so they get the first pick of companies
SELECTION_ORDER = (
    "missing_data",
    "peer_compare",
    "risk_change_metric",
    "metric_margin_risk",
    "growth",
)
COUNTS = {
    "dev": {
        "risk_change_metric": 3,
        "metric_margin_risk": 2,
        "peer_compare": 2,
        "growth": 2,
        "missing_data": 1,
    },
    "test": {
        "risk_change_metric": 8,
        "metric_margin_risk": 6,
        "peer_compare": 6,
        "growth": 6,
        "missing_data": 4,
    },
}

THEMES: dict[str, tuple[str, str]] = {  # name -> (phrase in the question, pattern)
    "cybersecurity": ("cybersecurity", r"cyber|data breach|ransomware|security (breach|incident)"),
    "pandemic": ("pandemic and public-health", r"covid|coronavirus|pandemic|epidemic"),
    "inflation": ("inflation", r"inflation"),
    "interest_rates": ("interest-rate", r"interest rate"),
    "supply_chain": ("supply-chain", r"supply chain"),
    "energy_prices": (
        "fuel and energy price",
        r"fuel (price|cost)|energy (price|cost)|oil price|price of (oil|fuel|natural gas)",
    ),
    "climate": ("climate-change", r"climate change|greenhouse gas|carbon emission"),
    "trade_tariffs": ("tariff and trade-policy", r"tariff|trade polic|trade war|export control"),
    "artificial_intelligence": (
        "artificial-intelligence",
        r"artificial intelligence|generative ai|\bai\b|machine learning",
    ),
    "labor": (
        "labor and workforce",
        r"labor (shortage|cost)|unioniz|collective bargaining|work stoppage"
        r"|attract and retain|retain and attract",
    ),
}
THEME_PATTERNS = {name: re.compile(rx, re.IGNORECASE) for name, (_, rx) in THEMES.items()}

FACTS_SQL = (
    "SELECT cik, tag, value_date, value, first_reported_value, was_restated,"
    " last_accession_number FROM marts.fct_financials"
    " WHERE duration_quarters = 4 AND unit = 'USD' AND tag = ANY(%(tags)s) AND cik = ANY(%(ciks)s)"
)
COMPANIES_SQL = (
    "SELECT cik, ticker, company_name, sector FROM marts.dim_company WHERE cik = ANY(%(ciks)s)"
)
LOOKUP_SQL = (
    "SELECT value, first_reported_value, was_restated, last_accession_number"
    " FROM marts.fct_financials WHERE cik = %(cik)s AND tag = %(tag)s"
    " AND value_date = %(value_date)s AND duration_quarters = 4 AND unit = 'USD'"
)
MISSING_SQL = (
    "SELECT count(*) FROM marts.fct_financials WHERE cik = %(cik)s AND tag = ANY(%(tags)s)"
    " AND duration_quarters = 4 AND unit = 'USD' AND value_date BETWEEN %(start)s AND %(end)s"
)

Fact = dict[str, Any]
Figures = dict[tuple[int, int], dict[str, Any]]  # (cik, fy) -> {"value_date": date, metric: Fact}
Filings = dict[tuple[int, int], dict[str, Any]]  # (cik, fy) -> {"accession", "report_date"}
Hits = dict[tuple[str, str], list[str]]  # (accession, theme) -> chunk ids, in corpus order


# ---------------------------------------------------------------- data shaping (pure)


def _hash_key(*parts: object) -> str:
    return hashlib.sha256(":".join(str(p) for p in (SEED, *parts)).encode()).hexdigest()


def ordered(items: Iterable[Any], *key: object) -> list[Any]:
    """Items in a fixed pseudo-random order that depends only on the key and the item."""
    return sorted(items, key=lambda item: _hash_key(*key, item))


def as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def filings_table(rows: Iterable[dict]) -> Filings:
    """(cik, fiscal year) -> its one 10-K; a year with two 10-Ks of one company is left out."""
    by_year: dict[tuple[int, int], dict[str, dict]] = defaultdict(dict)
    for r in rows:
        key = (int(r["cik"]), int(r["period_year"]))
        by_year[key][r["accession"]] = {
            "accession": r["accession"],
            "report_date": as_date(r["report_date"]),
        }
    return {key: next(iter(f.values())) for key, f in by_year.items() if len(f) == 1}


def theme_hits(chunks: Iterable[dict]) -> Hits:
    """(accession, theme) -> ids of the chunks whose heading or text matches the theme."""
    hits: Hits = defaultdict(list)
    for c in chunks:
        heading = c["heading"] if isinstance(c.get("heading"), str) else ""
        text = f"{heading}\n{c['text']}"
        for name, pattern in THEME_PATTERNS.items():
            if pattern.search(text):
                hits[(c["accession"], name)].append(c["chunk_id"])
    return dict(hits)


def fact_record(row: dict, metric: str) -> Fact:
    return {
        "metric": metric,
        "tag": row["tag"],
        "value_date": as_date(row["value_date"]).isoformat(),
        "duration_quarters": 4,
        "unit": "USD",
        "value": float(row["value"]),
        "first_reported_value": float(row["first_reported_value"]),
        "was_restated": bool(row["was_restated"]),
        "last_accession": row["last_accession_number"],
    }


def fact_window(report_date: date) -> tuple[date, date]:
    tol = timedelta(days=DATE_TOLERANCE_DAYS)
    return report_date - tol, report_date + tol


def annual_figures(facts: Iterable[dict], filings: Filings) -> tuple[Figures, set]:
    """Figures per (cik, fiscal year), and the (cik, fy) whose window holds no fact date at all.

    A year whose window holds two or more fact dates is in neither: it is ambiguous.
    """
    by_cik: dict[int, dict[date, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for row in facts:
        by_cik[int(row["cik"])][as_date(row["value_date"])][row["tag"]] = row
    figures: Figures = {}
    no_dates: set[tuple[int, int]] = set()
    for (cik, fy), filing in filings.items():
        start, end = fact_window(filing["report_date"])
        dates = [d for d in by_cik.get(cik, {}) if start <= d <= end]
        if not dates:
            no_dates.add((cik, fy))
            continue
        if len(dates) > 1:
            continue
        tags = by_cik[cik][dates[0]]
        entry: dict[str, Any] = {"value_date": dates[0]}
        revenue_tag = next((t for t in REVENUE_TAGS if t in tags), None)
        if revenue_tag:
            entry["revenue"] = fact_record(tags[revenue_tag], "revenue")
        if OPERATING_TAG in tags:
            entry["operating_income"] = fact_record(tags[OPERATING_TAG], "operating_income")
        if NET_TAG in tags:
            entry["net_income"] = fact_record(tags[NET_TAG], "net_income")
        figures[(cik, fy)] = entry
    return figures, no_dates


# ---------------------------------------------------------------- question builders


class Data:
    """Everything the builders read, already shaped (no I/O)."""

    def __init__(
        self,
        companies: dict[int, dict],
        filings: Filings,
        figures: Figures,
        no_dates: set,
        hits: Hits,
    ) -> None:
        self.companies, self.filings, self.figures = companies, filings, figures
        self.no_dates, self.hits = no_dates, hits

    def years(self, cik: int) -> list[int]:
        return sorted(fy for c, fy in self.filings if c == cik)

    def themes_in(self, cik: int, fy: int) -> list[str]:
        filing = self.filings.get((cik, fy))
        if not filing:
            return []
        return [t for t in THEMES if self.hits.get((filing["accession"], t))]

    def metric(self, cik: int, fy: int, name: str) -> Fact | None:
        return self.figures.get((cik, fy), {}).get(name)


def company_ref(c: dict) -> dict:
    return {"cik": c["cik"], "ticker": c["ticker"], "name": c["name"], "sector": c["sector"]}


def who(c: dict) -> str:
    return f"{c['name']} ({c['ticker']})"


def figure(c: dict, fy: int, fact: Fact) -> dict:
    label = f"{c['ticker']} FY{fy} {fact['metric'].replace('_', ' ')}"
    return {"label": label, "cik": c["cik"], "ticker": c["ticker"], "fiscal_year": fy, **fact}


def evidence(data: Data, c: dict, fy: int, theme: str) -> dict:
    filing = data.filings[(c["cik"], fy)]
    ids = data.hits.get((filing["accession"], theme), [])
    return {
        "label": f"{c['ticker']} FY{fy}",
        "cik": c["cik"],
        "fiscal_year": fy,
        "accession": filing["accession"],
        "report_date": filing["report_date"].isoformat(),
        "filing_url": sec_filing_url(c["cik"], filing["accession"]),
        "theme_chunks_total": len(ids),
        "theme_chunk_ids": ids[:EVIDENCE_CAP],
    }


def ratio(label: str, num: dict, den: dict) -> dict:
    return {
        "label": label,
        "kind": "ratio",
        "of": [num["label"], den["label"]],
        "value": num["value"] / den["value"],
    }


def growth_of(label: str, new: dict, old: dict) -> dict:
    return {
        "label": label,
        "kind": "growth",
        "of": [new["label"], old["label"]],
        "value": new["value"] / old["value"] - 1.0,
    }


def difference(label: str, new: dict, old: dict) -> dict:
    return {
        "label": label,
        "kind": "difference",
        "of": [new["label"], old["label"]],
        "value": new["value"] - old["value"],
    }


def question(
    template: str,
    text: str,
    companies: list[dict],
    years: list[int],
    theme: str,
    figures: list[dict],
    derived: list[dict],
    risk: list[dict],
    missing: list[dict] | None = None,
) -> dict:
    return {
        "template": template,
        "question": text,
        "answerable": not missing,
        "companies": [company_ref(c) for c in companies],
        "fiscal_years": years,
        "theme": theme,
        "reference": {
            "figures": figures,
            "derived": derived,
            "missing": missing or [],
            "risk_evidence": risk,
        },
    }


def positive_revenue(fact: Fact | None) -> bool:
    return fact is not None and fact["value"] > 0


def build_risk_change_metric(data: Data, cik: int) -> dict | None:
    c = data.companies[cik]
    years = data.years(cik)
    pairs = [(a, b) for a in years for b in years if b - a >= MIN_YEAR_GAP]
    for y1, y2 in ordered(pairs, "risk_change_metric", cik):
        themes = sorted(set(data.themes_in(cik, y1)) & set(data.themes_in(cik, y2)))
        if not themes:
            continue
        r1, r2 = data.metric(cik, y1, "revenue"), data.metric(cik, y2, "revenue")
        o1, o2 = data.metric(cik, y1, "operating_income"), data.metric(cik, y2, "operating_income")
        metrics = []
        if positive_revenue(r1) and positive_revenue(r2):
            metrics.append("revenue")
            if o1 and o2:
                metrics.append("operating_margin")
        if not metrics:
            continue
        theme = ordered(themes, "risk_change_metric", cik, y1, y2)[0]
        metric = ordered(metrics, "risk_change_metric", cik, y1, y2, theme)[0]
        f_r1, f_r2 = figure(c, y1, r1), figure(c, y2, r2)
        if metric == "revenue":
            figs = [f_r1, f_r2]
            derived = [growth_of(f"{c['ticker']} revenue change FY{y1}-FY{y2}", f_r2, f_r1)]
            phrase = "revenue"
        else:
            f_o1, f_o2 = figure(c, y1, o1), figure(c, y2, o2)
            m1 = ratio(f"{c['ticker']} FY{y1} operating margin", f_o1, f_r1)
            m2 = ratio(f"{c['ticker']} FY{y2} operating margin", f_o2, f_r2)
            figs = [f_r1, f_o1, f_r2, f_o2]
            derived = [m1, m2, difference(f"{c['ticker']} operating margin change", m2, m1)]
            phrase = "operating margin"
        text = (
            f"How did {who(c)}'s {THEMES[theme][0]} risk disclosure change between its 10-Ks "
            f"for the fiscal years that ended in {y1} and {y2}, and how did its {phrase} change "
            "between those two years?"
        )
        risk = [evidence(data, c, y1, theme), evidence(data, c, y2, theme)]
        return question("risk_change_metric", text, [c], [y1, y2], theme, figs, derived, risk)
    return None


def build_metric_margin_risk(data: Data, cik: int) -> dict | None:
    c = data.companies[cik]
    for fy in ordered(data.years(cik), "metric_margin_risk", cik):
        rev, op = data.metric(cik, fy, "revenue"), data.metric(cik, fy, "operating_income")
        themes = data.themes_in(cik, fy)
        if not (positive_revenue(rev) and op and themes):
            continue
        theme = ordered(themes, "metric_margin_risk", cik, fy)[0]
        f_rev, f_op = figure(c, fy, rev), figure(c, fy, op)
        derived = [ratio(f"{c['ticker']} FY{fy} operating margin", f_op, f_rev)]
        text = (
            f"What were {who(c)}'s revenue and operating income for the fiscal year that ended "
            f"in {fy}, what operating margin does that imply, and what does its 10-K for that "
            f"year say about {THEMES[theme][0]} risk?"
        )
        risk = [evidence(data, c, fy, theme)]
        return question("metric_margin_risk", text, [c], [fy], theme, [f_rev, f_op], derived, risk)
    return None


def build_peer_compare(data: Data, cik: int, partners: list[int]) -> dict | None:
    a = data.companies[cik]
    for other in ordered(partners, "peer_compare", cik):
        b = data.companies[other]
        common = sorted(set(data.years(cik)) & set(data.years(other)))
        for fy in ordered(common, "peer_compare", cik, other):
            ra, rb = data.metric(cik, fy, "revenue"), data.metric(other, fy, "revenue")
            na, nb = data.metric(cik, fy, "net_income"), data.metric(other, fy, "net_income")
            themes = sorted(set(data.themes_in(cik, fy)) & set(data.themes_in(other, fy)))
            if not (positive_revenue(ra) and positive_revenue(rb) and na and nb and themes):
                continue
            theme = ordered(themes, "peer_compare", cik, other, fy)[0]
            figs = [figure(a, fy, ra), figure(a, fy, na), figure(b, fy, rb), figure(b, fy, nb)]
            derived = [
                ratio(f"{a['ticker']} FY{fy} net margin", figs[1], figs[0]),
                ratio(f"{b['ticker']} FY{fy} net margin", figs[3], figs[2]),
            ]
            text = (
                f"Compare the net margins of {who(a)} and {who(b)} for their fiscal years that "
                f"ended in {fy}, and how does each company's 10-K for that year describe "
                f"{THEMES[theme][0]} risk?"
            )
            risk = [evidence(data, a, fy, theme), evidence(data, b, fy, theme)]
            return question("peer_compare", text, [a, b], [fy], theme, figs, derived, risk)
    return None


def build_growth(data: Data, cik: int) -> dict | None:
    c = data.companies[cik]
    years = data.years(cik)
    for fy in ordered([y for y in years if y - 1 in years], "growth", cik):
        new, old = data.metric(cik, fy, "revenue"), data.metric(cik, fy - 1, "revenue")
        themes = data.themes_in(cik, fy)
        if not (positive_revenue(new) and positive_revenue(old) and themes):
            continue
        theme = ordered(themes, "growth", cik, fy)[0]
        f_old, f_new = figure(c, fy - 1, old), figure(c, fy, new)
        derived = [growth_of(f"{c['ticker']} revenue growth FY{fy - 1}-FY{fy}", f_new, f_old)]
        text = (
            f"By what percentage did {who(c)}'s revenue grow from the fiscal year that ended in "
            f"{fy - 1} to the one that ended in {fy}, and what does its 10-K for the later year "
            f"say about {THEMES[theme][0]} risk?"
        )
        risk = [evidence(data, c, fy, theme)]
        return question("growth", text, [c], [fy - 1, fy], theme, [f_old, f_new], derived, risk)
    return None


def missing_spec(c: dict, fy: int, start: date, end: date, reason: str) -> dict:
    return {
        "label": f"{c['ticker']} FY{fy} revenue",
        "cik": c["cik"],
        "fiscal_year": fy,
        "metric": "revenue",
        "tags": list(REVENUE_TAGS),
        "window": [start.isoformat(), end.isoformat()],
        "reason": reason,
    }


def build_missing_data(data: Data, cik: int) -> dict | None:
    c = data.companies[cik]
    for fy in ordered(data.years(cik), "missing_data", cik):
        themes = data.themes_in(cik, fy)
        figs = data.figures.get((cik, fy))
        no_revenue = (cik, fy) in data.no_dates or (figs is not None and "revenue" not in figs)
        if not (themes and no_revenue):
            continue
        theme = ordered(themes, "missing_data", cik, fy)[0]
        start, end = fact_window(data.filings[(cik, fy)]["report_date"])
        miss = missing_spec(c, fy, start, end, "no revenue tag in fct_financials for that year")
        text = (
            f"What was {who(c)}'s revenue for the fiscal year that ended in {fy}, and what does "
            f"its 10-K for that year say about {THEMES[theme][0]} risk?"
        )
        risk = [evidence(data, c, fy, theme)]
        return question("missing_data", text, [c], [fy], theme, [], [], risk, [miss])
    return None


def build_missing_early(data: Data, cik: int) -> dict | None:
    """Fallback: revenue for EARLY_YEAR, before the data (and the 10-K corpus) starts."""
    c = data.companies[cik]
    if not data.years(cik) or min(data.years(cik)) <= EARLY_YEAR:
        return None
    start, end = date(EARLY_YEAR, 1, 1), date(EARLY_YEAR, 12, 31)
    miss = missing_spec(c, EARLY_YEAR, start, end, "before fct_financials and the 10-K corpus")
    theme = ordered(THEMES, "missing_early", cik)[0]
    text = (
        f"What was {who(c)}'s revenue for the fiscal year that ended in {EARLY_YEAR}, and what "
        f"did its 10-K for that year say about {THEMES[theme][0]} risk?"
    )
    return question("missing_data", text, [c], [EARLY_YEAR], theme, [], [], [], [miss])


# ---------------------------------------------------------------- selection


def round_robin(ciks: Iterable[int], sector_of: dict[int, str], template: str) -> list[int]:
    """CIKs interleaved across sectors (each sector's own order fixed by the template)."""
    by_sector: dict[str, list[int]] = defaultdict(list)
    for cik in ciks:
        by_sector[sector_of[cik]].append(cik)
    queues = [ordered(by_sector[s], template, s) for s in ordered(by_sector, template)]
    out: list[int] = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def select_split(data: Data, ciks: list[int], counts: dict[str, int], split: str) -> list[dict]:
    sector_of = {cik: data.companies[cik]["sector"] for cik in ciks}
    used: set[int] = set()
    chosen: dict[str, list[dict]] = {t: [] for t in TEMPLATES}
    for template in SELECTION_ORDER:
        need = counts.get(template, 0)
        builders: list[Callable[[int], dict | None]]
        if template == "peer_compare":

            def peer(cik: int) -> dict | None:
                partners = [
                    o for o in ciks if o != cik and o not in used and sector_of[o] == sector_of[cik]
                ]
                return build_peer_compare(data, cik, partners)

            builders = [peer]
        elif template == "missing_data":
            builders = [
                lambda cik: build_missing_data(data, cik),
                lambda cik: build_missing_early(data, cik),
            ]
        else:
            fn = globals()[f"build_{template}"]
            builders = [lambda cik, fn=fn: fn(data, cik)]
        for build in builders:  # a fallback builder only fills what the first one couldn't
            for cik in round_robin(ciks, sector_of, template):
                if len(chosen[template]) == need:
                    break
                if cik in used:
                    continue
                q = build(cik)
                if q is None:
                    continue
                chosen[template].append(q)
                used.update(c["cik"] for c in q["companies"])
        if len(chosen[template]) < need:
            raise SystemExit(f"{split}: only {len(chosen[template])} of {need} {template}")
    out = [q for t in TEMPLATES for q in chosen[t]]
    for i, q in enumerate(out, start=1):
        q["qid"] = f"{split}-{i:03d}"
        q["split"] = split
    return [{"qid": q.pop("qid"), "split": q.pop("split"), **q} for q in out]


def generate(data: Data, split_of: dict[int, str], counts: dict | None = None) -> dict:
    counts = counts or COUNTS
    out = {}
    for split in SPLITS:
        ciks = sorted(c for c, s in split_of.items() if s == split and c in data.companies)
        out[split] = select_split(data, ciks, counts[split], split)
    leaked = {c["cik"] for q in out["dev"] for c in q["companies"]} & {
        c["cik"] for q in out["test"] for c in q["companies"]
    }
    if leaked:
        raise ValueError(f"companies in both splits: {sorted(leaked)}")
    return out


# ---------------------------------------------------------------- checks (lookups injected)


def recompute(derived: dict, by_label: dict[str, dict]) -> float:
    a, b = (by_label[label]["value"] for label in derived["of"])
    if derived["kind"] == "ratio":
        return a / b
    if derived["kind"] == "growth":
        return a / b - 1.0
    if derived["kind"] == "difference":
        return a - b
    raise ValueError(f"unknown derived kind {derived['kind']!r}")


def check_questions(
    questions: list[dict],
    lookup: Callable[[dict], list[tuple]],
    count_missing: Callable[[dict], int],
    chunk_accession: dict[str, str],
) -> dict:
    """Re-query, recompute and re-find every reference. lookup(figure) returns the fct rows
    (value, first_reported_value, was_restated, last_accession_number) for its key."""
    problems: list[str] = []
    n_fig = n_der = n_miss = n_ev = 0
    for q in questions:
        ref = q["reference"]
        for f in ref["figures"]:
            n_fig += 1
            rows = lookup(f)
            if len(rows) != 1:
                problems.append(f"{q['qid']} {f['label']}: {len(rows)} fct rows")
                continue
            value, first, restated, accession = rows[0]
            got = (float(value), float(first), bool(restated), accession)
            want = (f["value"], f["first_reported_value"], f["was_restated"], f["last_accession"])
            if got != want:
                problems.append(f"{q['qid']} {f['label']}: fct has {got}, reference {want}")
        values = {x["label"]: x for x in ref["figures"]}
        for d in ref["derived"]:
            n_der += 1
            values_now = values | {x["label"]: x for x in ref["derived"]}
            if not math.isclose(recompute(d, values_now), d["value"], rel_tol=0, abs_tol=1e-12):
                problems.append(f"{q['qid']} {d['label']}: recomputed value differs")
        for m in ref["missing"]:
            n_miss += 1
            found = count_missing(m)
            if found:
                problems.append(f"{q['qid']} {m['label']}: {found} fct rows, expected none")
        for ev in ref["risk_evidence"]:
            for chunk_id in ev["theme_chunk_ids"]:
                n_ev += 1
                if chunk_accession.get(chunk_id) != ev["accession"]:
                    problems.append(f"{q['qid']} {chunk_id}: not a chunk of {ev['accession']}")
    return {
        "questions": len(questions),
        "figures_checked": n_fig,
        "derived_checked": n_der,
        "missing_checked": n_miss,
        "evidence_chunks_checked": n_ev,
        "problems": problems,
        "ok": not problems,
    }


# ---------------------------------------------------------------- files and I/O


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_questions(split: str, out_dir: Path = OUT_DIR, allow_test: bool = False) -> list[dict]:
    """Questions of one split. Test is for the one scored crew run (after its rule file)."""
    if split not in SPLITS:
        raise ValueError(f"unknown question split {split!r}")
    if split == "test" and not allow_test:
        raise PermissionError("the test questions are for the one scored run: allow_test=True")
    return read_jsonl(out_dir / f"questions.{split}.jsonl")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(questions: dict[str, list[dict]]) -> dict:
    """Counts only: no question text, company or year goes into the bench record."""
    out = {}
    for split, rows in questions.items():
        by = defaultdict(lambda: defaultdict(int))
        for q in rows:
            by["template"][q["template"]] += 1
            by["theme"][q["theme"]] += 1
            for c in q["companies"]:
                by["sector_mentions"][c["sector"]] += 1
        out[split] = {
            "questions": len(rows),
            "answerable": sum(q["answerable"] for q in rows),
            "companies": len({c["cik"] for q in rows for c in q["companies"]}),
            "figures": sum(len(q["reference"]["figures"]) for q in rows),
            "derived": sum(len(q["reference"]["derived"]) for q in rows),
            "restated_figures": sum(
                f["was_restated"] for q in rows for f in q["reference"]["figures"]
            ),
            **{k: dict(sorted(v.items())) for k, v in sorted(by.items())},
        }
    return out


def connect():
    import psycopg

    from filinglens.serve.cost_meter import pg_conninfo

    if os.environ.get("DATABASE_URL"):
        raise SystemExit("unset DATABASE_URL: the marts live in the local compose Postgres")
    return psycopg.connect(pg_conninfo(), autocommit=True)


def fetch_dicts(con, sql: str, params: dict) -> list[dict]:
    with con.cursor() as cur:
        cur.execute(sql, params)
        names = [d.name for d in cur.description]
        return [dict(zip(names, row, strict=True)) for row in cur.fetchall()]


def load_data(con, split_of: dict[int, str], corpus: Path, chunks_path: Path) -> Data:
    import pandas as pd

    ciks = sorted(split_of)
    para = pd.read_parquet(corpus, columns=["cik", "period_year", "accession", "report_date"])
    para = para[para["cik"].isin(ciks)].drop_duplicates("accession")
    filings = filings_table(para.to_dict("records"))
    cols = ["chunk_id", "cik", "accession", "heading", "text"]
    chunks = pd.read_parquet(chunks_path, columns=cols)
    chunks = chunks[chunks["cik"].isin(ciks)]
    hits = theme_hits(chunks.to_dict("records"))
    facts = fetch_dicts(con, FACTS_SQL, {"tags": list(FACT_TAGS), "ciks": ciks})
    figures, no_dates = annual_figures(facts, filings)
    companies = {
        int(r["cik"]): {
            "cik": int(r["cik"]),
            "ticker": r["ticker"],
            "name": r["company_name"],
            "sector": r["sector"],
        }
        for r in fetch_dicts(con, COMPANIES_SQL, {"ciks": ciks})
    }
    return Data(companies, filings, figures, no_dates, hits)


def read_splits(path: Path) -> dict[int, str]:
    import pandas as pd

    df = pd.read_csv(path)
    return {int(r.cik): r.split for r in df.itertuples(index=False) if r.split in SPLITS}


def run_generate(args: argparse.Namespace) -> dict:
    split_of = read_splits(args.splits)
    con = connect()
    data = load_data(con, split_of, args.corpus, args.chunks)
    questions = generate(data, split_of)
    paths = {}
    for split, rows in questions.items():
        path = args.out_dir / f"questions.{split}.jsonl"
        if path.exists() and read_jsonl(path) != rows and not args.overwrite:
            raise SystemExit(f"{path} exists and would change: pass --overwrite on purpose")
        write_jsonl(path, rows)
        paths[split] = path
    return {
        "inputs": {
            "splits": {"path": str(args.splits), "sha256": sha256_file(args.splits)},
            "corpus": {"path": str(args.corpus), "sha256": sha256_file(args.corpus)},
            "chunks": {"path": str(args.chunks), "sha256": sha256_file(args.chunks)},
        },
        "settings": {
            "seed": SEED,
            "counts": COUNTS,
            "date_tolerance_days": DATE_TOLERANCE_DAYS,
            "min_year_gap": MIN_YEAR_GAP,
            "themes": {k: v[1] for k, v in THEMES.items()},
        },
        "outputs": {s: {"path": str(p), "sha256": sha256_file(p)} for s, p in paths.items()},
        "summary": summarize(questions),
    }


def run_check(args: argparse.Namespace) -> dict:
    import pandas as pd

    questions = load_questions(args.split, args.out_dir, allow_test=args.allow_test)
    con = connect()

    def lookup(f: dict) -> list[tuple]:
        with con.cursor() as cur:
            params = {"cik": f["cik"], "tag": f["tag"], "value_date": as_date(f["value_date"])}
            cur.execute(LOOKUP_SQL, params)
            return cur.fetchall()

    def count_missing(m: dict) -> int:
        with con.cursor() as cur:
            start, end = (as_date(d) for d in m["window"])
            cur.execute(
                MISSING_SQL, {"cik": m["cik"], "tags": m["tags"], "start": start, "end": end}
            )
            return int(cur.fetchone()[0])

    chunks = pd.read_parquet(args.chunks, columns=["chunk_id", "accession"])
    accession_of = dict(zip(chunks["chunk_id"], chunks["accession"], strict=True))
    result = check_questions(questions, lookup, count_missing, accession_of)
    path = args.out_dir / f"questions.{args.split}.jsonl"
    return {"split": args.split, "file_sha256": sha256_file(path), **result}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--splits", type=Path, default=SPLITS_FILE)
    gen.add_argument("--corpus", type=Path, default=CORPUS)
    gen.add_argument("--chunks", type=Path, default=CHUNKS)
    gen.add_argument("--out-dir", type=Path, default=OUT_DIR)
    gen.add_argument("--stats", type=Path, default=STATS)
    gen.add_argument("--overwrite", action="store_true")
    chk = sub.add_parser("check")
    chk.add_argument("--split", choices=SPLITS, required=True)
    chk.add_argument("--allow-test", action="store_true")
    chk.add_argument("--chunks", type=Path, default=CHUNKS)
    chk.add_argument("--out-dir", type=Path, default=OUT_DIR)
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    if args.command == "generate":
        record, out = run_generate(args), args.stats
    else:
        record = run_check(args)
        out = Path(f"bench/agent_questions_check_{args.split}.json")
    record = {
        "command": " ".join(
            ["uv run python -m filinglens.agents.questions", *map(shlex.quote, argv)]
        ),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **record,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if record.get("ok") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
