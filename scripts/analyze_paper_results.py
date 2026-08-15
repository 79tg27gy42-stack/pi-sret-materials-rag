"""Generate revision-only analyses from the frozen canonical_v3 outputs.

The script deliberately does not infer span-level extraction accuracy: the
human-reference files contain response-level adjudications, not span/entity
gold annotations. It reports this boundary alongside the analyses that can be
computed directly from the frozen data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "canonical_v3_q1_evidence"
FIGURES = ROOT / "figures" / "paper"
HUMAN = ROOT / "data" / "annotations" / "expert_review_adjudicated_v3.csv"
SEMANTIC = OUT / "ragas_with_local_semantic_baselines.csv"

HUMAN_REFERENCE = "human_final_violation"
AUTO = "auto_violation"
SEMANTIC_METHODS = {
    "RAGAS": ("ragas_faithfulness", 0.8),
    "Lexical": ("lexical_overlap_score", 2.0 / 3.0),
    "Hashing": ("embedding_similarity_hashing_score", 0.33706122221924645),
    "NLI-style": ("nli_heuristic_score", 0.75),
}


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


def _binary_counts(reference: pd.Series, prediction: pd.Series) -> dict[str, float | int]:
    ref = _as_bool(reference).to_numpy()
    pred = _as_bool(prediction).to_numpy()
    tp = int((ref & pred).sum())
    tn = int((~ref & ~pred).sum())
    fp = int((~ref & pred).sum())
    fn = int((ref & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    elif tp + fp or tp + fn:
        f1 = 0.0
    else:
        f1 = float("nan")
    return {
        "n": len(ref),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _per_family_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in frame.groupby("constraint_family", sort=True):
        row = {"constraint_family": family}
        row.update(_binary_counts(group[HUMAN_REFERENCE], group[AUTO]))
        rows.append(row)
    return pd.DataFrame(rows)


def _semantic_overlap(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    constraint_fail = pd.to_numeric(frame["pi_sret_constraint_score"], errors="raise").lt(1.0)
    case_sets: dict[str, set[str]] = {}
    rows = []
    for name, (column, threshold) in SEMANTIC_METHODS.items():
        scores = pd.to_numeric(frame[column], errors="raise")
        identifiers = set(frame.loc[scores.ge(threshold) & constraint_fail, "response_id"].astype(str))
        case_sets[name] = identifiers
        rows.append(
            {
                "estimator": name,
                "score_column": column,
                "high_support_threshold": threshold,
                "high_support_low_constraint_cases": len(identifiers),
                "pi_sret_violation_set_size": int(constraint_fail.sum()),
                "overlap_with_pi_sret_violation_set": len(identifiers),
                "note": "Divergence cases are defined as high support intersected with PI-SRET constraint failure.",
            }
        )

    names = list(SEMANTIC_METHODS)
    matrix = np.zeros((len(names), len(names)), dtype=float)
    pair_rows = []
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            union = case_sets[left] | case_sets[right]
            value = len(case_sets[left] & case_sets[right]) / len(union) if union else float("nan")
            matrix[i, j] = value
            pair_rows.append({"estimator_a": left, "estimator_b": right, "jaccard": value})

    all_intersection = set.intersection(*case_sets.values())
    union = set.union(*case_sets.values())
    summary = {
        "n_records": int(len(frame)),
        "pi_sret_violation_set_size": int(constraint_fail.sum()),
        "intersection_all_estimators": len(all_intersection),
        "union_any_estimator": len(union),
        "unique_to_estimator": {
            name: len(values - set.union(*(other for other_name, other in case_sets.items() if other_name != name)))
            for name, values in case_sets.items()
        },
    }
    matrix_frame = pd.DataFrame(matrix, index=names, columns=names)
    return pd.DataFrame(rows), pd.DataFrame(pair_rows), {"matrix": matrix_frame, "summary": summary}


def _write_overlap_figure(matrix: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    image = ax.imshow(matrix.to_numpy(), vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            value = matrix.iat[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", color="black", fontsize=9)
    ax.set_xlabel("Semantic-support estimator")
    ax.set_ylabel("Semantic-support estimator")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Jaccard overlap")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig14_semantic_estimator_overlap.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig14_semantic_estimator_overlap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _error_attribution(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = _as_bool(frame[HUMAN_REFERENCE])
    prediction = _as_bool(frame[AUTO])
    disagreements = frame.loc[reference.ne(prediction)].copy()
    disagreements["error_direction"] = np.where(prediction.loc[disagreements.index], "false_positive", "false_negative")
    disagreements["attribution_status"] = "unresolved_without_span_or_claim_gold"
    disagreements["conservative_interpretation"] = np.where(
        disagreements["error_direction"].eq("false_positive"),
        "A rule fired although the human reference judged no violation; root cause cannot be separated into extraction and predicate semantics from response-level labels.",
        "The human reference judged a violation although no rule fired; root cause cannot be separated into missed extraction and uncovered rule logic from response-level labels.",
    )
    keep = [
        "盲标编号",
        "sample_id",
        "response_id",
        "base_task_id",
        "constraint_family",
        "document_status",
        HUMAN_REFERENCE,
        AUTO,
        "auto_constraint_violations_optimized",
        "human_final_type",
        "error_direction",
        "attribution_status",
        "conservative_interpretation",
    ]
    details = disagreements[keep].sort_values(["error_direction", "constraint_family", "response_id"])
    summary = pd.DataFrame(
        [
            {
                "attribution_category": "Confirmed extraction error",
                "fp": 0,
                "fn": 0,
                "total": 0,
                "evidence_status": "not estimable",
                "basis": "No span/entity gold annotations or independently corrected structured fields are present in the human reference.",
            },
            {
                "attribution_category": "Confirmed rule-logic error",
                "fp": 0,
                "fn": 0,
                "total": 0,
                "evidence_status": "not estimable",
                "basis": "Response-level adjudications do not identify whether a matched span was correct before predicate evaluation.",
            },
            {
                "attribution_category": "Confirmed coverage limitation",
                "fp": 0,
                "fn": 0,
                "total": 0,
                "evidence_status": "not estimable",
                "basis": "False negatives may reflect absent rules or missed extraction; the available labels do not distinguish them.",
            },
            {
                "attribution_category": "Observed false-positive disagreements",
                "fp": int((details["error_direction"] == "false_positive").sum()),
                "fn": 0,
                "total": int((details["error_direction"] == "false_positive").sum()),
                "evidence_status": "observed",
                "basis": "PI-SRET violation versus human-adjudicated non-violation; not a confirmed root-cause classification.",
            },
            {
                "attribution_category": "Observed false-negative disagreements",
                "fp": 0,
                "fn": int((details["error_direction"] == "false_negative").sum()),
                "total": int((details["error_direction"] == "false_negative").sum()),
                "evidence_status": "observed",
                "basis": "Human-adjudicated violation versus no PI-SRET trigger; not a confirmed root-cause classification.",
            },
        ]
    )
    return details, summary


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.3f}" if isinstance(value, float) and np.isfinite(value) else ("NA" if isinstance(value, float) else str(value)) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    human = pd.read_csv(HUMAN)
    semantic = pd.read_csv(SEMANTIC)
    if len(human) != 200 or len(semantic) != 372:
        raise ValueError("Expected the frozen 200-record human reference and 372-record RAGAS subset.")

    per_family = _per_family_metrics(human)
    overlap_summary, pairwise, overlap = _semantic_overlap(semantic)
    errors, error_summary = _error_attribution(human)

    per_family.to_csv(OUT / "expert_review_per_family_metrics_v3.csv", index=False)
    overlap_summary.to_csv(OUT / "semantic_estimator_overlap_summary_v3.csv", index=False)
    pairwise.to_csv(OUT / "semantic_estimator_pairwise_jaccard_v3.csv", index=False)
    overlap["matrix"].to_csv(OUT / "semantic_estimator_jaccard_matrix_v3.csv", index_label="estimator")
    errors.to_csv(OUT / "human_reference_error_attribution_details_v3.csv", index=False)
    error_summary.to_csv(OUT / "human_reference_error_attribution_summary_v3.csv", index=False)
    _write_overlap_figure(overlap["matrix"])

    report = [
        "# Revision Evidence Analyses",
        "",
        "## Inputs and scope",
        "",
        "This analysis reads the frozen 200-record human-adjudicated reference and 372-record RAGAS subset. It does not modify annotations, cached model outputs, or PI-SRET labels.",
        "",
        "## Per-family human-reference performance",
        "",
        "Family-level values are diagnostic strata from the sampled human reference, not population-level prevalence estimates. No confidence intervals are reported for small strata.",
        "",
        _markdown_table(per_family),
        "",
        "## Semantic-estimator overlap",
        "",
        "Each divergence set uses the estimator-specific calibrated high-support threshold shown below and the shared condition `pi_sret_constraint_score < 1`. Consequently every listed case is, by construction, in the PI-SRET violation set; the Jaccard values compare which violations each semantic estimator surfaces as high-support.",
        "",
        _markdown_table(overlap_summary.drop(columns=["note"])),
        "",
        _markdown_table(overlap["matrix"].reset_index(names="estimator")),
        "",
        "```json",
        json.dumps(overlap["summary"], indent=2),
        "```",
        "",
        "## Extraction-to-error propagation boundary",
        "",
        "The frozen human-reference files have response-level adjudications and free-text rationales, but no span/entity gold annotations, corrected structured extraction, or claim-level alignment. They therefore support observed FP/FN counts but cannot support a quantitative split into extraction, rule-logic, and coverage causes. The conservative summary below records that limitation rather than assigning unverified causes.",
        "",
        _markdown_table(error_summary),
        "",
        "## Reproduction",
        "",
        "```bash",
        "python3 scripts/analyze_revision_evidence.py",
        "```",
        "",
    ]
    (OUT / "revision_evidence_analyses_v3.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"per_family_rows": len(per_family), **overlap["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
