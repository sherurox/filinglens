"""Company entity resolution: a master company table, a name resolver and a dedup report.

Sources (all SEC):
- company_tickers.json: every listed company's CIK, tickers and current name. This is the
  reference list the resolver matches against. Fetched once, then read from the cache.
- data/raw/submissions/CIK##########.json (cached in M1): the universe's names, former
  names, tickers, exchanges and EIN.
- sub.txt inside data/raw/fsds/<quarter>.zip: the name each filer used on each financial
  filing, 2018q1-2026q3. Each (cik, name) pair is a name variant whose true CIK is known.

The resolver goes exact normalized name -> compact key (spaces removed) -> fuzzy
(IDF-weighted token Jaccard over candidates that share a rare token). A key that points to
more than one CIK is "ambiguous" and returns no CIK: a false merge is worse than a miss.
"""

import argparse
import asyncio
import csv
import io
import json
import math
import re
import time
import unicodedata
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Legal forms and filler words carry no identity: "COCA COLA CO" and "Coca-Cola Company".
LEGAL_FORMS = frozenset(
    ["THE", "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED"]
    + ["LLC", "LP", "LLP", "PLC", "NV", "SA", "AG", "SE"]
)
ABBREVIATIONS = {
    "INTL": "INTERNATIONAL",
    "HLDGS": "HOLDINGS",
    "HLDG": "HOLDING",
    "GRP": "GROUP",
    "FINL": "FINANCIAL",
    "MFG": "MANUFACTURING",
    "SVCS": "SERVICES",
    "TECHNOLOGIES": "TECHNOLOGY",
}
# SEC conformed names end in state or status tags: "APPLE INC /CA/", "XYZ CORP\DE\", "/NEW/".
STATE_TAG = re.compile(r"\s*[/\\]\s*[A-Z]{2,4}\s*[/\\]?\s*$")
DELETE = re.compile(r"[.'’]")  # "J.P." -> "JP", "MOODY'S" -> "MOODYS"
NON_ALNUM = re.compile(r"[^A-Z0-9]+")

MAX_DF = 200  # blocking: a token in more reference names than this can't create candidates
FSDS_FIELDS = ["cik", "name", "first_filed", "last_filed", "n_filings"]
MASTER_FIELDS = [
    "cik",
    "name",
    "tickers",
    "exchanges",
    "sic",
    "sector",
    "ein",
    "state_of_incorporation",
    "fiscal_year_end",
    "former_names",
    "fsds_names",
    "n_name_variants",
    "in_ticker_file",
]


def normalize(name: str) -> tuple[str, ...]:
    """Tokens that identify a company: upper case, ASCII, no punctuation or legal forms."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().upper()
    s = s.replace("&", " AND ")
    while (m := STATE_TAG.search(s)) and m.start() > 0:
        s = s[: m.start()]
    s = NON_ALNUM.sub(" ", DELETE.sub("", s))
    words = [ABBREVIATIONS.get(w, w) for w in s.split()]
    tokens = tuple(w for w in words if w not in LEGAL_FORMS)
    return tokens or tuple(words)  # "THE COMPANY" keeps its words rather than becoming empty


def key(tokens: tuple[str, ...]) -> str:
    return " ".join(tokens)


def compact(tokens: tuple[str, ...]) -> str:
    """Spaces removed: "WAL MART" = "WALMART", "J P MORGAN" = "JPMORGAN"."""
    return "".join(tokens)


@dataclass(frozen=True)
class Match:
    """The resolver's answer for one name. cik is None unless method is exact/compact/fuzzy."""

    query: str
    cik: int | None
    name: str | None  # the reference name that matched (or the best candidate, if rejected)
    method: str  # exact | compact | fuzzy | ambiguous | none
    score: float  # 1.0 for exact/compact; weighted Jaccard of the best candidate otherwise
    margin: float  # best score minus the best score of a *different* CIK
    candidates: int  # reference names compared (0 for exact/compact)


class Resolver:
    """Match company names to CIKs from a reference list of (cik, name) aliases."""

    def __init__(self, aliases: Iterable[tuple[int, str]], max_df: int = MAX_DF) -> None:
        self.entries: list[tuple[int, str, tuple[str, ...]]] = []
        self.exact: dict[str, dict[int, str]] = defaultdict(dict)  # key -> {cik: name}
        self.compact: dict[str, dict[int, str]] = defaultdict(dict)
        seen: set[tuple[int, str]] = set()
        for cik, name in aliases:
            tokens = normalize(name)
            if not tokens or (cik, key(tokens)) in seen:
                continue
            seen.add((cik, key(tokens)))
            self.entries.append((cik, name, tokens))
            self.exact[key(tokens)].setdefault(cik, name)
            self.compact[compact(tokens)].setdefault(cik, name)
        df = Counter(t for _, _, tokens in self.entries for t in set(tokens))
        n = len(self.entries)
        self.idf = {t: math.log(1 + n / d) for t, d in df.items()}
        self.unseen_idf = math.log(1 + n)  # a token no reference name has is maximally rare
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, (_, _, tokens) in enumerate(self.entries):
            for t in set(tokens):
                if df[t] <= max_df:
                    self.postings[t].append(i)

    def weight(self, token: str) -> float:
        return self.idf.get(token, self.unseen_idf)

    def similarity(self, a: tuple[str, ...], b: tuple[str, ...]) -> float:
        """IDF-weighted Jaccard: shared rare tokens count a lot, shared common ones little."""
        sa, sb = set(a), set(b)
        union = sum(self.weight(t) for t in sa | sb)
        return sum(self.weight(t) for t in sa & sb) / union if union else 0.0

    def resolve(self, name: str, threshold: float = 0.8, min_margin: float = 0.05) -> Match:
        tokens = normalize(name)
        for method, table, k in (
            ("exact", self.exact, key(tokens)),
            ("compact", self.compact, compact(tokens)),
        ):
            ciks = table.get(k, {})
            if len(ciks) == 1:
                cik, ref = next(iter(ciks.items()))
                return Match(name, cik, ref, method, 1.0, 1.0, 0)
            if len(ciks) > 1:
                return Match(name, None, None, "ambiguous", 1.0, 0.0, 0)
        return self.fuzzy(name, tokens, threshold, min_margin)

    def fuzzy(
        self, name: str, tokens: tuple[str, ...], threshold: float, min_margin: float
    ) -> Match:
        ids = {i for t in set(tokens) for i in self.postings.get(t, ())}
        best: dict[int, tuple[float, str]] = {}  # best alias score per CIK
        for i in ids:
            cik, ref, ref_tokens = self.entries[i]
            score = self.similarity(tokens, ref_tokens)
            if score > best.get(cik, (-1.0, ""))[0]:
                best[cik] = (score, ref)
        if not best:
            return Match(name, None, None, "none", 0.0, 0.0, 0)
        ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))
        cik, (score, ref) = ranked[0]
        margin = score - (ranked[1][1][0] if len(ranked) > 1 else 0.0)
        accepted = score >= threshold and margin >= min_margin
        return Match(
            name,
            cik if accepted else None,
            ref,
            "fuzzy" if accepted else "none",
            round(score, 4),
            round(margin, 4),
            len(ids),
        )


# ---------- loaders ----------


def ticker_rows(data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """company_tickers.json as rows: one per ticker (a CIK can have several share classes)."""
    return [
        {"cik": int(e["cik_str"]), "ticker": e["ticker"], "name": e["title"]}
        for _, e in sorted(data.items(), key=lambda kv: int(kv[0]))
    ]


async def fetch_tickers(path: Path) -> int:
    """One request to SEC through the M1 client (rate limit and User-Agent), saved as-is."""
    from filinglens.ingest.edgar_client import EdgarClient, load_user_agent

    async with EdgarClient(load_user_agent()) as client:
        data = await client.company_tickers()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return len(data)


def read_fsds_names(zip_paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Distinct (cik, name) pairs across sub.txt files, with first/last filing date and count."""
    agg: dict[tuple[int, str], dict[str, Any]] = {}
    for path in sorted(zip_paths):
        with zipfile.ZipFile(path) as zf, zf.open("sub.txt") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
            for row in csv.DictReader(text, delimiter="\t", quoting=csv.QUOTE_NONE):
                name = (row.get("name") or "").strip()
                if not name or not (row.get("cik") or "").isdigit():
                    continue
                filed = row.get("filed") or ""
                k = (int(row["cik"]), name)
                if k not in agg:
                    agg[k] = {
                        "cik": k[0],
                        "name": name,
                        "first_filed": filed,
                        "last_filed": filed,
                        "n_filings": 0,
                    }
                rec = agg[k]
                rec["n_filings"] += 1
                rec["first_filed"] = min(rec["first_filed"], filed)
                rec["last_filed"] = max(rec["last_filed"], filed)
    return sorted(agg.values(), key=lambda r: (r["cik"], r["first_filed"], r["name"]))


def write_csv(rows: list[dict[str, Any]], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


# ---------- master table and dedup report ----------


def master_row(
    company: dict[str, str],
    subs: dict[str, Any],
    tickers: list[str],
    fsds_names: list[str],
) -> dict[str, Any]:
    """The golden record for one universe company: every identifier and name variant we hold."""
    former = [f["name"] for f in subs.get("formerNames") or [] if f.get("name")]
    names = [subs.get("name") or company["name"], *former, *fsds_names]
    return {
        "cik": int(company["cik"]),
        "name": subs.get("name") or company["name"],
        "tickers": "|".join(sorted(set(subs.get("tickers") or []) | set(tickers))),
        "exchanges": "|".join(sorted({e for e in subs.get("exchanges") or [] if e})),
        "sic": subs.get("sic") or company.get("sic", ""),
        "sector": company["sector"],
        "ein": subs.get("ein") or "",
        "state_of_incorporation": subs.get("stateOfIncorporation") or "",
        "fiscal_year_end": subs.get("fiscalYearEnd") or "",
        "former_names": "|".join(former),
        "fsds_names": "|".join(sorted(set(fsds_names))),
        "n_name_variants": len({key(normalize(n)) for n in names}),
        "in_ticker_file": bool(tickers),
    }


def examples(groups: dict[Any, set], n: int = 5) -> list[dict[str, Any]]:
    top = sorted(groups.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))[:n]
    return [{"key": k, "members": sorted(v)} for k, v in top]


def dedup_report(
    tickers: list[dict[str, Any]], fsds: list[dict[str, Any]], master: list[dict[str, Any]]
) -> dict[str, Any]:
    """Where one entity has many records, and where one name points at many entities."""
    by_cik: dict[int, set[str]] = defaultdict(set)
    ref_names: dict[int, str] = {}
    for r in tickers:
        by_cik[r["cik"]].add(r["ticker"])
        ref_names.setdefault(r["cik"], r["name"])
    ref_keys: dict[str, set[int]] = defaultdict(set)
    for cik, name in ref_names.items():
        ref_keys[key(normalize(name))].add(cik)
    fsds_variants: dict[int, set[str]] = defaultdict(set)
    fsds_keys: dict[str, set[int]] = defaultdict(set)
    for r in fsds:
        k = key(normalize(r["name"]))
        fsds_variants[int(r["cik"])].add(k)
        fsds_keys[k].add(int(r["cik"]))
    multi_ticker = {c: t for c, t in by_cik.items() if len(t) > 1}
    ref_collide = {k: c for k, c in ref_keys.items() if len(c) > 1}
    renamed = {c: v for c, v in fsds_variants.items() if len(v) > 1}
    fsds_collide = {k: c for k, c in fsds_keys.items() if len(c) > 1}
    return {
        "ticker_file": {
            "rows": len(tickers),
            "ciks": len(by_cik),
            "ciks_with_multiple_tickers": len(multi_ticker),
            "examples": examples(multi_ticker),
        },
        "ticker_file_name_collisions": {
            "normalized_names_with_multiple_ciks": len(ref_collide),
            "examples": examples(ref_collide),
        },
        "fsds": {
            "cik_name_pairs": len(fsds),
            "ciks": len(fsds_variants),
            "ciks_with_multiple_normalized_names": len(renamed),
            "normalized_names_with_multiple_ciks": len(fsds_collide),
            "rename_examples": examples(renamed),
            "collision_examples": examples(fsds_collide),
        },
        "master": {
            "companies": len(master),
            "in_ticker_file": sum(bool(m["in_ticker_file"]) for m in master),
            "with_former_names": sum(bool(m["former_names"]) for m in master),
            "with_multiple_tickers": sum("|" in m["tickers"] for m in master),
            "name_variants_total": sum(m["n_name_variants"] for m in master),
            "max_name_variants": max((m["n_name_variants"] for m in master), default=0),
        },
    }


# ---------- command line ----------


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if not args.tickers.exists():
        raise SystemExit(f"{args.tickers} missing: run the fetch-tickers subcommand first")
    tickers = ticker_rows(json.loads(args.tickers.read_text()))
    zips = sorted(args.fsds.glob("*.zip"))
    if not zips:
        raise SystemExit(f"no FSDS zips in {args.fsds}")
    fsds = read_fsds_names(zips)
    write_csv(fsds, args.fsds_names, FSDS_FIELDS)

    tickers_by_cik: dict[int, list[str]] = defaultdict(list)
    for r in tickers:
        tickers_by_cik[r["cik"]].append(r["ticker"])
    fsds_by_cik: dict[int, list[str]] = defaultdict(list)
    for r in fsds:
        fsds_by_cik[r["cik"]].append(r["name"])
    master, missing = [], []
    for company in read_csv(args.universe):
        cik = int(company["cik"])
        path = args.submissions / f"CIK{cik:010d}.json"
        if not path.exists():
            missing.append(cik)
            continue
        subs = json.loads(path.read_text())
        master.append(master_row(company, subs, tickers_by_cik.get(cik, []), fsds_by_cik[cik]))
    write_csv(master, args.master, MASTER_FIELDS)

    report = {
        "command": "python -m filinglens.extract.entities build",
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "tickers_file": str(args.tickers),
        "tickers_file_fetched": datetime.fromtimestamp(args.tickers.stat().st_mtime, UTC).isoformat(
            timespec="seconds"
        ),
        "fsds_zips": len(zips),
        "fsds_quarters": [zips[0].stem, zips[-1].stem],
        "submissions_missing": missing,
        **dedup_report(tickers, fsds, master),
        "seconds": round(time.perf_counter() - started, 1),
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch-tickers", help="one request: SEC company_tickers.json")
    fetch.add_argument("--tickers", type=Path, default=Path("data/raw/company_tickers.json"))
    b = sub.add_parser("build", help="master table, FSDS name variants, dedup report")
    b.add_argument("--tickers", type=Path, default=Path("data/raw/company_tickers.json"))
    b.add_argument("--fsds", type=Path, default=Path("data/raw/fsds"))
    b.add_argument("--submissions", type=Path, default=Path("data/raw/submissions"))
    b.add_argument("--universe", type=Path, default=Path("dbt/seeds/universe.csv"))
    b.add_argument("--fsds-names", type=Path, default=Path("data/interim/entities/fsds_names.csv"))
    b.add_argument("--master", type=Path, default=Path("gold/company_master.csv"))
    b.add_argument("--stats", type=Path, default=Path("bench/entities.json"))
    args = parser.parse_args()
    if args.command == "fetch-tickers":
        if args.tickers.exists():
            print(f"{args.tickers} exists: 0 requests (delete it to refetch)")
            return
        n = asyncio.run(fetch_tickers(args.tickers))
        print(f"{n} entries -> {args.tickers} (1 request)")
        return
    report = build(args)
    summary = {k: report[k] for k in ("fsds_zips", "submissions_missing", "seconds")}
    summary |= {
        "ticker_ciks": report["ticker_file"]["ciks"],
        "ticker_ciks_with_multiple_tickers": report["ticker_file"]["ciks_with_multiple_tickers"],
        "ticker_name_collisions": report["ticker_file_name_collisions"][
            "normalized_names_with_multiple_ciks"
        ],
        "fsds_cik_name_pairs": report["fsds"]["cik_name_pairs"],
        "fsds_ciks": report["fsds"]["ciks"],
        "fsds_ciks_renamed": report["fsds"]["ciks_with_multiple_normalized_names"],
        "master": report["master"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
