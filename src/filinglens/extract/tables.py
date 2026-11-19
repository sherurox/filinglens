"""Table extraction from page images, scored by cell accuracy (FL-M2 T9 OCR baseline, T10 VLM).

OCR baseline: Tesseract gives words with boxes. Words are grouped into rows by their vertical
centers (not Tesseract's own line ids, which split a wide table row into separate blocks), and
within a row into cells wherever the horizontal gap is wider than about one character height.
The cells then go through the same logic as the ground truth (table_pages.to_logical), so OCR and
truth are compared on equal terms.

Dev pages are for choosing settings (e.g. Tesseract's page-segmentation mode); the test pages are
run once per final setting and need --allow-test.

    uv run python -m filinglens.extract.tables ocr --split dev --psm 6
    uv run python -m filinglens.extract.tables ocr --split test --psm 6 --allow-test
    uv run python -m filinglens.extract.tables score --split dev \
        --pred data/interim/tables/tables_vlm_image_dev.jsonl     # T10 predictions
"""

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from filinglens.extract.table_metrics import aggregate, score_table
from filinglens.extract.table_pages import to_logical

TABLES = Path("gold/tables/tables.jsonl")
OUT_DIR = Path("data/interim/tables")
ROW_TOLERANCE = 0.6  # words belong to the same row if centers differ by < 0.6 x median height
CELL_GAP = 1.0  # a gap wider than 1.0 x median height starts a new cell


def load_pages(split: str, path: Path = TABLES, allow_test: bool = False) -> list[dict]:
    if split == "test" and not allow_test:
        raise PermissionError("the test pages are for final runs only: pass --allow-test")
    pages = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [p for p in pages if p["split"] == split]


def ocr_words(image: Path, psm: int) -> list[dict]:
    """Words with boxes from Tesseract (confidence -1 entries are layout, not words)."""
    import pytesseract
    from PIL import Image

    data = pytesseract.image_to_data(
        Image.open(image), config=f"--psm {psm}", output_type=pytesseract.Output.DICT
    )
    keys = ("text", "left", "top", "width", "height", "conf")
    words = [
        dict(zip(keys, values, strict=True))
        for values in zip(*(data[k] for k in keys), strict=True)
    ]
    return [w for w in words if w["text"].strip() and float(w["conf"]) >= 0]


def group_rows(words: list[dict], tolerance: float = ROW_TOLERANCE) -> list[list[dict]]:
    """Words grouped into rows by vertical center, each row sorted left to right."""
    if not words:
        return []
    height = statistics.median(w["height"] for w in words)
    rows: list[list[dict]] = []
    for word in sorted(words, key=lambda w: w["top"] + w["height"] / 2):
        center = word["top"] + word["height"] / 2
        if rows:
            row_center = statistics.mean(w["top"] + w["height"] / 2 for w in rows[-1])
            if abs(center - row_center) <= tolerance * height:
                rows[-1].append(word)
                continue
        rows.append([word])
    return [sorted(row, key=lambda w: w["left"]) for row in rows]


def row_to_cells(row: list[dict], height: float, gap: float = CELL_GAP) -> list[str]:
    """Join neighbouring words into cells; a wide horizontal gap starts a new cell."""
    cells: list[str] = []
    previous_right = None
    for word in row:
        if previous_right is not None and word["left"] - previous_right <= gap * height:
            cells[-1] += " " + word["text"]
        else:
            cells.append(word["text"])
        previous_right = word["left"] + word["width"]
    return cells


def words_to_rows(words: list[dict]) -> list[dict]:
    """Logical rows ({"label", "values"}) from OCR words."""
    if not words:
        return []
    height = statistics.median(w["height"] for w in words)
    rows = (to_logical(row_to_cells(row, height)) for row in group_rows(words))
    return [r for r in rows if r is not None]


def tesseract_version() -> str:
    import pytesseract

    return str(pytesseract.get_tesseract_version())


def run_ocr(pages: list[dict], psm: int) -> tuple[list[dict], float]:
    predictions = []
    started = time.perf_counter()
    for page in pages:
        rows = words_to_rows(ocr_words(Path(page["image"]), psm))
        predictions.append({"page_id": page["page_id"], "rows": rows})
    return predictions, time.perf_counter() - started


def report(pages: list[dict], predictions: list[dict]) -> dict:
    scores = [score_table(p["rows"], q["rows"]) for p, q in zip(pages, predictions, strict=True)]
    per_page = [
        {"page_id": p["page_id"], **s, "accuracy": round(s["correct"] / s["gold_cells"], 4)}
        for p, s in zip(pages, scores, strict=True)
    ]
    return {"overall": aggregate(scores), "per_page": per_page}


def score_predictions(args: argparse.Namespace, pages: list[dict]) -> dict:
    """Score a predictions file (one {"page_id", "rows"} per line); missing pages score 0."""
    lines = args.pred.read_text().splitlines()
    by_id = {r["page_id"]: r for r in (json.loads(line) for line in lines if line)}
    predictions = [by_id.get(p["page_id"], {"page_id": p["page_id"], "rows": []}) for p in pages]
    return {
        "predictions": str(args.pred),
        "pages_missing": sum(p["page_id"] not in by_id for p in pages),
        **report(pages, predictions),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("method", choices=["ocr", "score"])
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--psm", type=int, default=6, help="Tesseract page segmentation mode")
    parser.add_argument("--pred", type=Path, help="predictions to score (method score)")
    parser.add_argument("--tables", type=Path, default=TABLES)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--bench", type=Path, default=Path("bench"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    pages = load_pages(args.split, args.tables, args.allow_test)
    if args.method == "score":
        if args.pred is None:
            parser.error("score needs --pred")
        name = args.pred.stem
        body = score_predictions(args, pages)
    else:
        predictions, elapsed = run_ocr(pages, args.psm)
        name = f"tables_ocr_{args.split}_psm{args.psm}"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions)
        )
        body = {
            "method": f"tesseract {tesseract_version()} --psm {args.psm}, rows by y-center",
            "seconds_total": round(elapsed, 2),
            "seconds_per_page": round(elapsed / max(1, len(pages)), 3),
            **report(pages, predictions),
        }
    result = {
        "command": " ".join(["uv run python -m filinglens.extract.tables", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": args.split,
        **body,
    }
    (args.bench / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_page"}, indent=2))


if __name__ == "__main__":
    main()
