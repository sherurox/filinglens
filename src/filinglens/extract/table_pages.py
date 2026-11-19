"""Table pages with a ground truth: render real 10-K tables to images, keep their cells (FL-M2 T7).

Each page is one financial table from a 10-K's HTML, rendered with Chromium at letter-page width
(816 CSS px, 2x scale) together with the nearest caption above it. The ground truth comes from the
same HTML the image is drawn from, so it needs no hand labeling: the image can only show what the
HTML holds.

The ground truth is the table's *logical* rows. EDGAR tables use extra cells for layout ("$" in its
own cell, ")" or "%" in the next one, empty spacer columns, colspans), so raw grid positions don't
mean anything. Instead, each row becomes a label (the text before the first number) and its numbers
in reading order: "Net sales | $ | 391,035 | | $ | 383,285" -> label "Net sales", values
["391035", "383285"]. Cell accuracy (T9) then compares (row label, k-th value) pairs, which doesn't
depend on how a model splits columns.

    uv run python -m filinglens.extract.table_pages      # 50 pages: 10 dev, 40 test
"""

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from html import escape
from pathlib import Path

from lxml import html

from filinglens.extract.sections import normalize

SEED = 20261119
PAGES = 50
DEV_PAGES = 10
MIN_ROWS, MAX_ROWS = 4, 40
MIN_NUMBERS = 12
YEARS = range(2019, 2026)

NUMBER = re.compile(r"^-?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?$")
DASH = re.compile(r"^[—–\-]+$")  # "—" means nil in financial tables
PREFIX = {"$", "(", "($", "$("}
SUFFIX = {")", "%", ")%", "%)"}
KINDS = {
    "balance_sheet": r"balance sheet|financial (position|condition)",
    "income_statement": r"statements? of (operations|income|earnings)|income statement",
    "cash_flow": r"cash flows?",
}


def canonical(text: str) -> str | None:
    """A number cell as a canonical string ("(1,234)" -> "-1234", "12.5%" -> "12.5"), or None."""
    t = text.replace(" ", "")
    core = t.strip("$()%")
    if DASH.match(core):  # "—", "—%", "$ —": nil
        return "—"
    if not NUMBER.match(core):
        return None
    negative = t.startswith("(") or t.startswith("$(") or core.startswith("-")
    digits = core.lstrip("-").replace(",", "")
    if "." in digits:
        whole, frac = digits.split(".")
        frac = frac.rstrip("0")
        value = (whole.lstrip("0") or "0") + (f".{frac}" if frac else "")
    else:
        value = digits.lstrip("0") or "0"
    return f"-{value}" if negative and value != "0" else value


def merge_cells(cells: list[str]) -> list[str]:
    """Non-empty cells, with "$", "(" joined to the next cell and ")", "%" to the previous one."""
    merged: list[str] = []
    pending = ""
    for cell in cells:
        if not cell:
            continue
        if cell in PREFIX:
            pending += cell
            continue
        if cell in SUFFIX and merged:
            merged[-1] += cell
            continue
        merged.append(pending + cell)
        pending = ""
    return merged


def row_cells(tr) -> list[str]:
    """Merged cell texts of an HTML table row."""
    return merge_cells([normalize(td.text_content()) for td in tr if td.tag in ("td", "th")])


def to_logical(cells: list[str]) -> dict | None:
    """{"label": cells before the first number, "values": the numbers}, or None if empty.
    Text cells after the first number (notes, "N/A") are ignored."""
    label_parts, values = [], []
    for cell in merge_cells(cells):
        number = canonical(cell)
        if number is None and not values:
            label_parts.append(cell)
        elif number is not None:
            values.append(number)
    label = " ".join(label_parts)
    return {"label": label, "values": values} if label or values else None


def logical_rows(table) -> list[dict]:
    """Rows of a table as {"label", "values"}; rows with neither are dropped."""
    rows = (to_logical(row_cells(tr)) for tr in table.iter("tr"))
    return [r for r in rows if r is not None]


def caption(table, max_chars: int = 140) -> str:
    """The nearest non-empty text above the table (a statement title, usually)."""
    for el in table.itersiblings(preceding=True):
        text = normalize(el.text_content())
        if text:
            return text[:max_chars]
    parent = table.getparent()
    return caption(parent, max_chars) if parent is not None and parent.tag != "body" else ""


def kind_of(text: str) -> str:
    low = text.lower()
    return next((k for k, rx in KINDS.items() if re.search(rx, low)), "other")


def is_candidate(rows: list[dict]) -> bool:
    numeric_rows = [r for r in rows if r["values"]]
    numbers = sum(len(r["values"]) for r in rows)
    return MIN_ROWS <= len(numeric_rows) and len(rows) <= MAX_ROWS and numbers >= MIN_NUMBERS


def pick_table(data: bytes, rng: random.Random) -> tuple[int, object, list[dict]] | None:
    """(index, element, rows) of one random candidate table in a 10-K, or None."""
    root = html.document_fromstring(data)
    candidates = []
    for i, table in enumerate(root.iter("table")):
        rows = logical_rows(table)
        if is_candidate(rows):
            candidates.append((i, table, rows))
    return rng.choice(candidates) if candidates else None


def choose_filings(manifest: Path, rng: random.Random) -> list[dict]:
    """One filing per company, spread over sectors and years (round-robin by sector)."""
    with manifest.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if Path(r["path"]).exists()]
    rows = [r for r in rows if int(r["report_date"][:4]) in YEARS]
    by_sector: dict[str, list[dict]] = {}
    for r in sorted(rows, key=lambda r: r["path"]):
        by_sector.setdefault(r["sector"], []).append(r)
    for group in by_sector.values():
        rng.shuffle(group)
    ordered, seen = [], set()
    while any(by_sector.values()):
        for sector in sorted(by_sector):
            while by_sector[sector]:
                r = by_sector[sector].pop()
                if r["cik"] not in seen:
                    seen.add(r["cik"])
                    ordered.append(r)
                    break
    return ordered


def page_html(table, title: str) -> str:
    body = html.tostring(table, encoding="unicode")
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><style>"
        "body{margin:24px;font-family:'Times New Roman',serif;background:#fff;width:768px}"
        "p.cap{font-weight:bold;margin:0 0 12px 0}</style></head><body>"
        f"<p class='cap'>{escape(title)}</p>"
        f"{body}</body></html>"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(pages: list[dict], out_dir: Path) -> None:
    """Screenshots of each page's HTML with Chromium (letter width, 2x scale, full height)."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 816, "height": 1056}, device_scale_factor=2)
        for item in pages:
            page.set_content(item.pop("_html"), wait_until="load")
            # a table wider than the page would be cut off in the image: record it
            item["overflows"] = page.evaluate("document.documentElement.scrollWidth") > 816
            path = out_dir / f"{item['page_id']}.png"
            page.screenshot(path=str(path), full_page=True)
            item["image"] = str(path)
            item["image_sha256"] = sha256(path)
        browser.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/10k/manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("gold/tables"))
    parser.add_argument("--stats", type=Path, default=Path("bench/table_pages.json"))
    parser.add_argument("--pages", type=int, default=PAGES)
    parser.add_argument("--seed", type=int, default=SEED)
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    pages, scanned = [], 0
    for filing in choose_filings(args.manifest, rng):
        if len(pages) == args.pages:
            break
        scanned += 1
        picked = pick_table(Path(filing["path"]).read_bytes(), rng)
        if picked is None:
            continue
        index, table, rows = picked
        title = caption(table)
        pages.append(
            {
                "page_id": f"{filing['accession']}-t{index}",
                "accession": filing["accession"],
                "cik": int(filing["cik"]),
                "ticker": filing["ticker"],
                "sector": filing["sector"],
                "period_year": int(filing["report_date"][:4]),
                "table_index": index,
                "caption": title,
                "kind": kind_of(title),
                "rows": rows,
                "n_rows": len(rows),
                "n_numbers": sum(len(r["values"]) for r in rows),
                "_html": page_html(table, title),
            }
        )
    order = list(range(len(pages)))
    rng.shuffle(order)
    for rank, i in enumerate(order):
        pages[i]["split"] = "dev" if rank < DEV_PAGES else "test"

    image_dir = args.out / "pages"
    image_dir.mkdir(parents=True, exist_ok=True)
    render(pages, image_dir)
    (args.out / "tables.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pages)
    )
    stats = {
        "command": " ".join(["uv run python -m filinglens.extract.table_pages", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "ground_truth": "cells from the HTML each image is rendered from (no hand labels)",
        "filings_scanned": scanned,
        "pages": len(pages),
        "by_split": dict(Counter(p["split"] for p in pages)),
        "by_kind": dict(Counter(p["kind"] for p in pages)),
        "by_sector": dict(sorted(Counter(p["sector"] for p in pages).items())),
        "by_year": dict(sorted(Counter(p["period_year"] for p in pages).items())),
        "numbers_total": sum(p["n_numbers"] for p in pages),
        "rows_total": sum(p["n_rows"] for p in pages),
        "pages_wider_than_letter": sum(p["overflows"] for p in pages),
    }
    args.stats.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
