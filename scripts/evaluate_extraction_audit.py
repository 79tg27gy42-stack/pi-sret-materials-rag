"""Score a completed Phi extraction audit annotation sheet."""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "extraction_gold_audit.csv"
DEFAULT_OUT = ROOT / "results" / "canonical_v3_q1_evidence"
CATEGORIES = ["material", "formula", "numeric_value", "unit", "property", "phase", "structure", "method", "evidence", "qualifier"]


def _normalise(category: str, value: str) -> str:
    text = " ".join(value.strip().lower().split())
    if category == "numeric_value":
        try:
            return format(Decimal(text).normalize(), "f")
        except InvalidOperation:
            return text
    if category == "unit":
        return {"kelvin": "k", "ev per atom": "ev/atom", "ev / atom": "ev/atom"}.get(text, text)
    return text


def _items(category: str, value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {_normalise(category, item) for item in str(value).split(";") if item.strip()}


def _metrics(tp: int, fp: int, fn: int) -> dict[str, float | int | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None and recall is None:
        f1 = None
    elif (precision or 0.0) + (recall or 0.0) == 0:
        f1 = 0.0
    else:
        f1 = 2 * float(precision) * float(recall) / (float(precision) + float(recall))
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate(frame: pd.DataFrame) -> pd.DataFrame:
    complete = frame["annotation_complete"].astype(str).str.strip().str.lower().isin({"yes", "true", "1"})
    if not complete.all():
        raise ValueError(f"Extraction audit is incomplete: {int((~complete).sum())} rows require annotation_complete=yes.")
    rows = []
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for category in CATEGORIES:
        tp = fp = fn = gold_count = predicted_count = 0
        for row in frame.itertuples(index=False):
            predicted_value = getattr(row, f"pred_{category}", getattr(row, f"predicted_{category}", ""))
            gold_value = getattr(row, f"gold_{category}", "")
            predicted = _items(category, predicted_value)
            gold = _items(category, gold_value)
            tp += len(predicted & gold)
            fp += len(predicted - gold)
            fn += len(gold - predicted)
            gold_count += len(gold)
            predicted_count += len(predicted)
        metric = _metrics(tp, fp, fn)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        rows.append({"extraction_type": category, "gold_n": gold_count, "pred_n": predicted_count, **metric})
    rows.append({"extraction_type": "overall_micro", "gold_n": None, "pred_n": None, **_metrics(**totals)})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = evaluate(pd.read_csv(args.input, keep_default_na=False))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "extraction_audit_results.csv", index=False)
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
