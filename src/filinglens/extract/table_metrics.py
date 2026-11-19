"""Cell-level accuracy of an extracted table against the ground truth (FL-M2 T9).

Both sides are logical rows: {"label": row text before the first number, "values": numbers in
reading order, canonical strings}. A gold cell is (row, k-th value). It counts as correct when the
prediction has a row matched to that gold row and its k-th value is the same canonical string.

Rows are matched by label, in order, so repeated labels ("Total") pair up first-to-first: an exact
match (after normalizing case, spaces, trailing ":" and footnote marks) is tried first, then the
most similar unmatched predicted label if its similarity is at least LABEL_SIMILARITY. Matching by
label, not by row position, means one missed or extra row doesn't shift every row after it.

cell_accuracy = correct gold cells / all gold cells (what the spec calls cell-level accuracy).
Extra predicted cells don't lower it, so precision is reported next to it.
"""

import re
from difflib import SequenceMatcher

from filinglens.extract.table_pages import canonical

LABEL_SIMILARITY = 0.85
FOOTNOTE = re.compile(r"\(\d{1,2}\)|\[\d{1,2}\]|\*+")


def normalize_label(label: str) -> str:
    text = FOOTNOTE.sub("", label.lower())
    text = re.sub(r"[^a-z0-9%&/ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_values(values: list[str]) -> list[str]:
    """Predicted values in canonical form; anything that isn't a number becomes "?"."""
    return [canonical(str(v)) or "?" for v in values]


def match_rows(gold: list[dict], pred: list[dict]) -> dict[int, int]:
    """gold row index -> predicted row index, for gold rows that have numbers."""
    gold_labels = [normalize_label(r["label"]) for r in gold]
    pred_labels = [normalize_label(r["label"]) for r in pred]
    free = list(range(len(pred)))
    pairs: dict[int, int] = {}
    wanted = [i for i, r in enumerate(gold) if r["values"]]
    for i in wanted:  # exact label, in order
        j = next((j for j in free if pred_labels[j] == gold_labels[i]), None)
        if j is not None:
            pairs[i] = j
            free.remove(j)
    for i in wanted:  # then the most similar remaining label
        if i in pairs or not free:
            continue
        scores = [(SequenceMatcher(None, gold_labels[i], pred_labels[j]).ratio(), j) for j in free]
        score, j = max(scores)
        if score >= LABEL_SIMILARITY:
            pairs[i] = j
            free.remove(j)
    return pairs


def score_table(gold: list[dict], pred: list[dict]) -> dict:
    """Counts for one table: gold cells, predicted cells, correct cells, matched rows."""
    pairs = match_rows(gold, pred)
    correct = 0
    for i, j in pairs.items():
        gold_values = gold[i]["values"]
        pred_values = canonical_values(pred[j].get("values", []))
        correct += sum(g == p for g, p in zip(gold_values, pred_values, strict=False))
    gold_cells = sum(len(r["values"]) for r in gold)
    pred_cells = sum(len(r.get("values", [])) for r in pred)
    return {
        "gold_cells": gold_cells,
        "pred_cells": pred_cells,
        "correct": correct,
        "gold_rows": sum(1 for r in gold if r["values"]),
        "matched_rows": len(pairs),
    }


def aggregate(scores: list[dict]) -> dict:
    """Micro (all cells pooled) and macro (mean of per-table) accuracy, plus precision."""
    gold = sum(s["gold_cells"] for s in scores)
    pred = sum(s["pred_cells"] for s in scores)
    correct = sum(s["correct"] for s in scores)
    per_table = [s["correct"] / s["gold_cells"] for s in scores if s["gold_cells"]]
    return {
        "tables": len(scores),
        "gold_cells": gold,
        "correct_cells": correct,
        "cell_accuracy_micro": round(correct / gold, 4) if gold else None,
        "cell_accuracy_macro": round(sum(per_table) / len(per_table), 4) if per_table else None,
        "cell_precision_micro": round(correct / pred, 4) if pred else None,
        "row_match_rate": round(
            sum(s["matched_rows"] for s in scores) / max(1, sum(s["gold_rows"] for s in scores)), 4
        ),
    }
