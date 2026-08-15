"""Measure semantic-support sensitivity to simple answer segmentation choices."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "ragas_real_canonical_v3_372_qwen_plus_complete_optimized_rules" / "ragas_baseline_scores.csv"
DEFAULT_OUT = ROOT / "results" / "canonical_v3_q1_evidence"

import sys

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sret_materials_rag.evaluation.faithfulness import lexical_faithfulness  # noqa: E402


def _sentence_split(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(text)) if part.strip()]
    return parts or [str(text)]


def _atomic_split(text: str) -> list[str]:
    sentences = _sentence_split(text)
    claims: list[str] = []
    for sentence in sentences:
        clauses = [part.strip() for part in re.split(r"\s*(?:;|,\s+(?:and|but)|\band\b|\bbut\b)\s*", sentence) if part.strip()]
        claims.extend(clauses or [sentence])
    return claims or [str(text)]


def _segmented_score(answer: str, context: str, mode: str) -> float:
    if mode == "whole_answer":
        return lexical_faithfulness(answer, context)
    if mode == "sentence":
        pieces = _sentence_split(answer)
    elif mode == "atomic":
        pieces = _atomic_split(answer)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return float(sum(lexical_faithfulness(piece, context) for piece in pieces) / len(pieces))


def analyze(frame: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rows = []
    for row in frame.itertuples(index=False):
        record = {
            "response_id": row.response_id,
            "constraint_family": row.constraint_family,
            "pi_sret_constraint_score": float(row.pi_sret_constraint_score),
        }
        for mode in ["atomic", "sentence", "whole_answer"]:
            record[f"{mode}_support"] = _segmented_score(row.answer, row.retrieved_context, mode)
            record[f"{mode}_high_support"] = record[f"{mode}_support"] >= threshold
            record[f"{mode}_divergence"] = record[f"{mode}_high_support"] and float(row.pi_sret_constraint_score) < 1.0
        rows.append(record)
    detail = pd.DataFrame(rows)

    summary_rows = []
    divergence_sets: dict[str, set[str]] = {}
    for mode in ["atomic", "sentence", "whole_answer"]:
        high = detail[f"{mode}_high_support"]
        div = detail[f"{mode}_divergence"]
        divergence_sets[mode] = set(detail.loc[div, "response_id"])
        summary_rows.append(
            {
                "segmentation": mode,
                "mean_support": detail[f"{mode}_support"].mean(),
                "high_support_n": int(high.sum()),
                "divergence_n": int(div.sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)

    overlaps = {}
    modes = list(divergence_sets)
    for left in modes:
        for right in modes:
            union = divergence_sets[left] | divergence_sets[right]
            overlaps[f"{left}_vs_{right}"] = len(divergence_sets[left] & divergence_sets[right]) / len(union) if union else 1.0
    metadata = {"threshold": threshold, "divergence_jaccard": overlaps}
    return summary, detail, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    summary, detail, metadata = analyze(frame, args.threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "claim_splitting_sensitivity.csv", index=False)
    detail.to_csv(args.output_dir / "claim_splitting_sensitivity_details.csv", index=False)
    (args.output_dir / "claim_splitting_sensitivity.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
