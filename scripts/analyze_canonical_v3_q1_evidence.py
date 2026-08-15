from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sret_materials_rag.evaluation.faithfulness_evaluators import (  # noqa: E402
    EmbeddingSimilarityFaithfulnessEvaluator,
    LexicalOverlapFaithfulnessEvaluator,
    NLIHeuristicFaithfulnessEvaluator,
)
from sret_materials_rag.evaluation.constraints import (  # noqa: E402
    ConstraintThresholds,
    evaluate_material_qa_constraints,
)


OUT = ROOT / "results" / "canonical_v3_q1_evidence"
RAGAS_DIR = ROOT / "results" / "ragas_real_canonical_v3_372_qwen_plus_complete_optimized_rules"
LLM_DIR = ROOT / "results" / "llm_judge_canonical_v3_full_qwen_plus_optimized_rules"
DUAL_DIR = ROOT / "results" / "llm_judge_canonical_v3_dual_model_agreement"
CANONICAL_V3 = ROOT / "data" / "processed" / "canonical_v3"

BOOTSTRAP_SAMPLES = 5000
SEED = 42
FAITHFULNESS_THRESHOLD = 0.8
CONSTRAINT_PASS_THRESHOLD = 1.0
THRESHOLD_SENSITIVITY_TAU_F = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
THRESHOLD_SENSITIVITY_TAU_G = [0.5, 0.75, 1.0]

L0_VIOLATIONS = {
    "non_negative_band_gap",
    "band_gap_physical_range",
    "valid_chemical_symbols",
    "non_negative_absolute_temperature",
    "non_negative_pressure",
}
L1_VIOLATIONS = {
    "formation_energy_typical_range",
    "elemental_formation_energy_reference_conflict",
    "cautious_stability_claim",
    "positive_formation_energy_stability_conflict",
    "conductivity_band_gap_consistency",
    "crystal_system_space_group_consistency",
    "cautious_superconductivity_claim",
}
FORMATION_ENERGY_VIOLATIONS = {
    "formation_energy_typical_range",
    "elemental_formation_energy_reference_conflict",
    "positive_formation_energy_stability_conflict",
}
BAND_GAP_VIOLATIONS = {
    "non_negative_band_gap",
    "band_gap_physical_range",
    "conductivity_band_gap_consistency",
}

_STRICT_BAND_GAP_RE = re.compile(r"band gap\b[^.\n;:]*?(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)\s*eV", re.I)
_STRICT_EV_BAND_GAP_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*eV\s+band gap", re.I)
_STRICT_FORMATION_ENERGY_RE = re.compile(
    r"formation energy\b[^.\n;:]*?(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)\s*eV(?:/(?:atom|f\.?u\.?)| per atom)?",
    re.I,
)
_STRICT_STANDALONE_EV_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*eV\b", re.I)
_STRICT_TEMPERATURE_K_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:K|Kelvin)\b", re.I)
_STRICT_PRESSURE_RE = re.compile(r"(?:pressure(?:\s+is|\s+of|\s*[:=])?\s*)?(-?\d+(?:\.\d+)?)\s*(?:GPa|MPa|Pa)\b", re.I)
_STRICT_SINGLE_ELEMENT_QUESTION_RE = re.compile(r"\b(?:of|for|is)\s+([A-Z][a-z]?)\??$", re.I)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    ragas = pd.read_csv(RAGAS_DIR / "ragas_baseline_scores.csv")
    divergence = pd.read_csv(RAGAS_DIR / "ragas_high_faithfulness_pi_sret_low_cases.csv")
    llm = pd.read_csv(LLM_DIR / "llm_judge_scores_with_optimized_auto.csv")
    dual = pd.read_csv(DUAL_DIR / "dual_model_judge_scores.csv")
    full_manifest = _load_jsonl(CANONICAL_V3 / "full_response_manifest_v3.jsonl")
    manifest_meta = _manifest_metadata(full_manifest)

    metrics: dict[str, object] = {
        "analysis_scope": "canonical_v3_completed_real_results",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "random_seed": SEED,
        "faithfulness_threshold": FAITHFULNESS_THRESHOLD,
        "constraint_pass_threshold": CONSTRAINT_PASS_THRESHOLD,
    }

    metrics["ragas"] = _analyze_ragas(ragas, divergence)
    metrics["semantic_baselines"] = _analyze_semantic_baselines(ragas)
    metrics["llm_judge"] = _analyze_llm_judge(llm)
    metrics["dual_judge"] = _analyze_dual_judge(dual, manifest_meta)
    metrics["threshold_sensitivity"] = _analyze_threshold_sensitivity(ragas)
    metrics["domain_guard_sensitivity"] = _analyze_domain_guard_sensitivity(ragas, llm)
    metrics["constraint_ablation"] = _analyze_constraint_ablation(ragas, llm)
    metrics["generation_mode_ablation"] = _generation_mode_ablation()

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(metrics)
    return 0


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _manifest_metadata(rows: list[dict]) -> pd.DataFrame:
    keep = [
        "response_id",
        "sample_id",
        "raw_sample_id",
        "base_task_id",
        "source_dataset",
        "dataset_role",
        "model",
        "mode",
        "constraint_family",
        "document_status",
        "source_kind",
    ]
    df = pd.DataFrame([{key: row.get(key, "") for key in keep} for row in rows])
    return df.drop_duplicates(subset=["response_id"], keep="first")


def _analyze_ragas(ragas: pd.DataFrame, divergence: pd.DataFrame) -> dict:
    ragas = ragas.copy()
    clusters = _cluster_ids(ragas)
    low_constraint = ragas["pi_sret_constraint_score"].astype(float) < CONSTRAINT_PASS_THRESHOLD
    high_faith = ragas["ragas_faithfulness"].astype(float) >= FAITHFULNESS_THRESHOLD
    ragas["is_divergence"] = high_faith & low_constraint

    summary = pd.DataFrame(
        [
            _metric_row("ragas_faithfulness_mean", ragas["ragas_faithfulness"], _mean, clusters),
            _metric_row("ragas_answer_relevancy_mean", ragas["ragas_answer_relevancy"], _mean, clusters),
            _metric_row("pi_sret_constraint_mean", ragas["pi_sret_constraint_score"], _mean, clusters),
            _metric_row("pi_sret_violation_rate", low_constraint.astype(float), _mean, clusters),
            _metric_row("divergence_rate", ragas["is_divergence"].astype(float), _mean, clusters),
        ]
    )
    summary.to_csv(OUT / "ragas_bootstrap_summary.csv", index=False)

    family_rows = []
    for family, group in ragas.groupby("constraint_family", dropna=False):
        family_low = group["pi_sret_constraint_score"].astype(float) < CONSTRAINT_PASS_THRESHOLD
        family_div = group["is_divergence"].astype(bool)
        family_rows.append(
            {
                "constraint_family": family,
                "n": int(len(group)),
                "mean_ragas_faithfulness": float(group["ragas_faithfulness"].mean()),
                "mean_answer_relevancy": float(group["ragas_answer_relevancy"].mean()),
                "mean_pi_sret_constraint_score": float(group["pi_sret_constraint_score"].mean()),
                "pi_sret_violation_rate": float(family_low.mean()),
                "divergence_cases": int(family_div.sum()),
                "divergence_rate": float(family_div.mean()),
            }
        )
    family_df = pd.DataFrame(family_rows).sort_values(["divergence_cases", "constraint_family"], ascending=[False, True])
    family_df.to_csv(OUT / "ragas_family_metrics.csv", index=False)

    by_family = (
        divergence["constraint_family"]
        .value_counts(dropna=False)
        .rename_axis("constraint_family")
        .reset_index(name="divergence_cases")
        .sort_values(["divergence_cases", "constraint_family"], ascending=[False, True])
    )
    by_family.to_csv(OUT / "ragas_divergence_by_family.csv", index=False)

    by_status = (
        divergence["document_status"]
        .value_counts(dropna=False)
        .rename_axis("document_status")
        .reset_index(name="divergence_cases")
        .sort_values(["divergence_cases", "document_status"], ascending=[False, True])
    )
    by_status.to_csv(OUT / "ragas_divergence_by_document_status.csv", index=False)

    violations = _explode_semicolon(divergence["pi_sret_violations"])
    by_violation = (
        violations.value_counts(dropna=False)
        .rename_axis("pi_sret_violation")
        .reset_index(name="count")
        .sort_values(["count", "pi_sret_violation"], ascending=[False, True])
    )
    by_violation.to_csv(OUT / "ragas_divergence_by_violation.csv", index=False)

    case_cols = [
        "sample_id",
        "response_id",
        "constraint_family",
        "document_status",
        "model",
        "mode",
        "question",
        "answer",
        "ragas_faithfulness",
        "ragas_answer_relevancy",
        "pi_sret_constraint_score",
        "pi_sret_violations",
    ]
    divergence[case_cols].sort_values(
        ["constraint_family", "pi_sret_violations", "sample_id"]
    ).to_csv(OUT / "ragas_divergence_cases_curated.csv", index=False)

    return {
        "n": int(len(ragas)),
        "divergence_cases": int(ragas["is_divergence"].sum()),
        "divergence_rate": _metric_dict(ragas["is_divergence"].astype(float), _mean, clusters),
        "mean_ragas_faithfulness": _metric_dict(ragas["ragas_faithfulness"], _mean, clusters),
        "mean_answer_relevancy": _metric_dict(ragas["ragas_answer_relevancy"], _mean, clusters),
        "mean_pi_sret_constraint_score": _metric_dict(ragas["pi_sret_constraint_score"], _mean, clusters),
        "pi_sret_violation_rate": _metric_dict(low_constraint.astype(float), _mean, clusters),
        "divergence_by_family": by_family.to_dict("records"),
        "divergence_by_document_status": by_status.to_dict("records"),
        "divergence_by_violation": by_violation.to_dict("records"),
    }


def _analyze_semantic_baselines(ragas: pd.DataFrame) -> dict:
    evaluators = [
        ("ragas_faithfulness", None),
        ("lexical_overlap", LexicalOverlapFaithfulnessEvaluator()),
        ("embedding_similarity_hashing", EmbeddingSimilarityFaithfulnessEvaluator()),
        ("nli_heuristic", NLIHeuristicFaithfulnessEvaluator()),
    ]
    out = ragas.copy()
    clusters = _cluster_ids(out)
    records = []
    constraint = out["pi_sret_constraint_score"].astype(float)
    low_constraint = constraint < CONSTRAINT_PASS_THRESHOLD
    ragas_high_rate = float((out["ragas_faithfulness"].astype(float) >= FAITHFULNESS_THRESHOLD).mean())

    for method, evaluator in evaluators:
        if evaluator is None:
            scores = out["ragas_faithfulness"].astype(float)
        else:
            score_values = []
            for row in out.itertuples(index=False):
                result = evaluator.score(
                    question=str(getattr(row, "question")),
                    answer=str(getattr(row, "answer")),
                    retrieved_context=str(getattr(row, "retrieved_context")),
                )
                score_values.append(result.score)
            scores = pd.Series(score_values, index=out.index, dtype=float)
            out[f"{method}_score"] = scores

        fixed_high = scores >= FAITHFULNESS_THRESHOLD
        quantile_reference_threshold = float(
            np.quantile(scores.to_numpy(dtype=float), max(0.0, 1.0 - ragas_high_rate))
        )
        quantile_reference_high = scores >= quantile_reference_threshold
        records.append(
            {
                "method": method,
                "n": int(len(out)),
                "mean_score": float(scores.mean()),
                "score_ci_low": _bootstrap_ci(scores, _mean, clusters)[0],
                "score_ci_high": _bootstrap_ci(scores, _mean, clusters)[1],
                "pearson_with_constraint": _safe_corr(scores, constraint),
                "spearman_with_constraint": _safe_spearman(scores, constraint),
                "fixed_threshold": FAITHFULNESS_THRESHOLD,
                "fixed_high_support_cases": int(fixed_high.sum()),
                "fixed_high_support_low_constraint_cases": int((fixed_high & low_constraint).sum()),
                "fixed_high_support_low_constraint_rate": _safe_div(int((fixed_high & low_constraint).sum()), int(fixed_high.sum())),
                "quantile_reference_threshold": quantile_reference_threshold,
                "quantile_reference_high_support_cases": int(quantile_reference_high.sum()),
                "quantile_reference_high_support_low_constraint_cases": int(
                    (quantile_reference_high & low_constraint).sum()
                ),
                "quantile_reference_high_support_low_constraint_rate": _safe_div(
                    int((quantile_reference_high & low_constraint).sum()), int(quantile_reference_high.sum())
                ),
            }
        )

    result = pd.DataFrame(records)
    result.to_csv(OUT / "semantic_baseline_robustness.csv", index=False)
    out.to_csv(OUT / "ragas_with_local_semantic_baselines.csv", index=False)
    return {
        "ragas_high_support_rate_quantile_reference": ragas_high_rate,
        "methods": result.to_dict("records"),
    }


def _analyze_threshold_sensitivity(ragas: pd.DataFrame) -> list[dict]:
    df = ragas.copy()
    faith = df["ragas_faithfulness"].astype(float)
    constraint = df["pi_sret_constraint_score"].astype(float)
    clusters = _cluster_ids(df)
    records = []
    for tau_f in THRESHOLD_SENSITIVITY_TAU_F:
        high_support = faith >= tau_f
        for tau_g in THRESHOLD_SENSITIVITY_TAU_G:
            low_constraint = constraint < tau_g
            divergence = high_support & low_constraint
            risk_values = divergence[high_support].astype(float)
            risk_metric = _metric_dict(risk_values, _mean, clusters[high_support]) if int(high_support.sum()) else {
                "value": None,
                "ci_low": None,
                "ci_high": None,
                "n": 0,
            }
            all_metric = _metric_dict(divergence.astype(float), _mean, clusters)
            records.append(
                {
                    "tau_f": tau_f,
                    "tau_g": tau_g,
                    "n": int(len(df)),
                    "high_support_cases": int(high_support.sum()),
                    "low_constraint_cases": int(low_constraint.sum()),
                    "divergence_cases": int(divergence.sum()),
                    "divergence_rate_all": all_metric["value"],
                    "divergence_rate_all_ci_low": all_metric["ci_low"],
                    "divergence_rate_all_ci_high": all_metric["ci_high"],
                    "risk_rate_among_high_support": risk_metric["value"],
                    "risk_rate_among_high_support_ci_low": risk_metric["ci_low"],
                    "risk_rate_among_high_support_ci_high": risk_metric["ci_high"],
                }
            )
    result = pd.DataFrame(records)
    result.to_csv(OUT / "threshold_sensitivity.csv", index=False)
    return result.to_dict("records")


def _analyze_domain_guard_sensitivity(ragas: pd.DataFrame, llm: pd.DataFrame) -> list[dict]:
    """One-at-a-time sensitivity for heuristic materials-domain cutoffs."""
    variants = {
        "default": ConstraintThresholds(),
        "band_gap_upper_10": ConstraintThresholds(band_gap_upper_ev=10.0),
        "band_gap_upper_20": ConstraintThresholds(band_gap_upper_ev=20.0),
        "formation_range_narrow": ConstraintThresholds(
            formation_energy_lower_ev_atom=-3.0, formation_energy_upper_ev_atom=0.5
        ),
        "formation_range_wide": ConstraintThresholds(
            formation_energy_lower_ev_atom=-5.0, formation_energy_upper_ev_atom=1.5
        ),
        "conductivity_gap_0_05": ConstraintThresholds(conductivity_gap_ev=0.05),
        "conductivity_gap_0_20": ConstraintThresholds(conductivity_gap_ev=0.20),
        "element_reference_0_00": ConstraintThresholds(elemental_reference_tolerance_ev_atom=0.0),
        "element_reference_0_10": ConstraintThresholds(elemental_reference_tolerance_ev_atom=0.10),
    }
    rows = []
    judge_violation = _as_bool(llm["judge_violation"])
    for name, thresholds in variants.items():
        ragas_violation = pd.Series(
            [
                bool(evaluate_material_qa_constraints(str(row.question), str(row.answer), thresholds=thresholds).violations)
                for row in ragas.itertuples(index=False)
            ]
        )
        high_support = ragas["ragas_faithfulness"].astype(float).reset_index(drop=True) >= FAITHFULNESS_THRESHOLD
        llm_violation = pd.Series(
            [
                bool(evaluate_material_qa_constraints(str(row.question), str(row.answer), thresholds=thresholds).violations)
                for row in llm.itertuples(index=False)
            ]
        )
        rows.append(
            {
                "variant": name,
                "band_gap_upper_ev": thresholds.band_gap_upper_ev,
                "formation_energy_lower_ev_atom": thresholds.formation_energy_lower_ev_atom,
                "formation_energy_upper_ev_atom": thresholds.formation_energy_upper_ev_atom,
                "conductivity_gap_ev": thresholds.conductivity_gap_ev,
                "elemental_reference_tolerance_ev_atom": thresholds.elemental_reference_tolerance_ev_atom,
                "ragas_violation_rate": float(ragas_violation.mean()),
                "ragas_divergence_cases": int((high_support & ragas_violation).sum()),
                "judge_agreement": float((llm_violation == judge_violation.reset_index(drop=True)).mean()),
                "judge_kappa": _cohen_kappa(llm_violation, judge_violation.reset_index(drop=True)),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "domain_guard_sensitivity.csv", index=False)
    return result.to_dict("records")


def _analyze_constraint_ablation(ragas: pd.DataFrame, llm: pd.DataFrame) -> dict[str, list[dict]]:
    ragas_records = []
    llm_records = []
    variants = [
        ("full_optimized", "All optimized PI-SRET violations"),
        ("l0_only", "Only shallow range/format violations"),
        ("l1_only", "Only domain consistency / evidence violations"),
        ("without_formation_energy_violations", "Ignore formation-energy-specific violation IDs"),
        ("without_band_gap_violations", "Ignore band-gap-specific violation IDs"),
        ("strict_no_source_reliability_exception", "Add strict invalid-value flags without refusal/caveat exceptions"),
        ("drop_formation_energy_records", "Exclude records whose stratification family is formation_energy"),
        ("drop_band_gap_records", "Exclude records whose stratification family is band_gap"),
    ]
    for variant, description in variants:
        ragas_df = _apply_ablation_variant(
            ragas,
            variant=variant,
            violation_col="pi_sret_violations",
        )
        ragas_records.append(_ragas_ablation_row(ragas_df, variant, description))

        llm_df = _apply_ablation_variant(
            llm,
            variant=variant,
            violation_col="auto_constraint_violations_optimized",
        )
        llm_records.append(_llm_ablation_row(llm_df, variant, description))

    ragas_result = pd.DataFrame(ragas_records)
    llm_result = pd.DataFrame(llm_records)
    ragas_result.to_csv(OUT / "constraint_ablation_ragas.csv", index=False)
    llm_result.to_csv(OUT / "constraint_ablation_llm_judge.csv", index=False)
    return {
        "ragas": ragas_result.to_dict("records"),
        "llm_judge": llm_result.to_dict("records"),
    }


def _apply_ablation_variant(df: pd.DataFrame, *, variant: str, violation_col: str) -> pd.DataFrame:
    result = df.copy()
    if variant == "drop_formation_energy_records":
        result = result[result["constraint_family"] != "formation_energy"].copy()
        variant = "full_optimized"
    elif variant == "drop_band_gap_records":
        result = result[result["constraint_family"] != "band_gap"].copy()
        variant = "full_optimized"

    variant_violations = []
    for row in result.itertuples(index=False):
        violations = set(_split_semicolon(getattr(row, violation_col, "")))
        if variant == "l0_only":
            violations = violations & L0_VIOLATIONS
        elif variant == "l1_only":
            violations = violations & L1_VIOLATIONS
        elif variant == "without_formation_energy_violations":
            violations = violations - FORMATION_ENERGY_VIOLATIONS
        elif variant == "without_band_gap_violations":
            violations = violations - BAND_GAP_VIOLATIONS
        elif variant == "strict_no_source_reliability_exception":
            violations = violations | set(
                _strict_invalid_value_violations(
                    question=str(getattr(row, "question", "")),
                    answer=str(getattr(row, "answer", "")),
                )
            )
        elif variant != "full_optimized":
            raise ValueError(f"Unknown ablation variant: {variant}")
        variant_violations.append(sorted(violations))

    result["ablation_violations"] = [";".join(items) for items in variant_violations]
    result["ablation_constraint_score"] = [1.0 if not items else 0.0 for items in variant_violations]
    return result


def _ragas_ablation_row(df: pd.DataFrame, variant: str, description: str) -> dict:
    if df.empty:
        return {
            "variant": variant,
            "description": description,
            "n": 0,
            "mean_constraint_score": None,
            "violation_rate": None,
            "high_support_cases": 0,
            "divergence_cases": 0,
            "divergence_rate_all": None,
            "risk_rate_among_high_support": None,
        }
    constraint = df["ablation_constraint_score"].astype(float)
    low_constraint = constraint < CONSTRAINT_PASS_THRESHOLD
    high_support = df["ragas_faithfulness"].astype(float) >= FAITHFULNESS_THRESHOLD
    divergence = high_support & low_constraint
    return {
        "variant": variant,
        "description": description,
        "n": int(len(df)),
        "mean_constraint_score": float(constraint.mean()),
        "violation_rate": float(low_constraint.mean()),
        "high_support_cases": int(high_support.sum()),
        "divergence_cases": int(divergence.sum()),
        "divergence_rate_all": float(divergence.mean()),
        "risk_rate_among_high_support": _safe_div(int(divergence.sum()), int(high_support.sum())),
    }


def _llm_ablation_row(df: pd.DataFrame, variant: str, description: str) -> dict:
    if df.empty:
        return {
            "variant": variant,
            "description": description,
            "n": 0,
            "auto_violation_rate": None,
            "judge_violation_rate": None,
            "agreement": None,
            "cohen_kappa": None,
            "false_positive_count": None,
            "false_negative_count": None,
        }
    auto_violation = df["ablation_constraint_score"].astype(float) < CONSTRAINT_PASS_THRESHOLD
    judge_violation = _as_bool(df["judge_violation"])
    agreement = auto_violation == judge_violation
    return {
        "variant": variant,
        "description": description,
        "n": int(len(df)),
        "auto_violation_rate": float(auto_violation.mean()),
        "judge_violation_rate": float(judge_violation.mean()),
        "agreement": float(agreement.mean()),
        "cohen_kappa": _cohen_kappa(auto_violation, judge_violation),
        "false_positive_count": int((auto_violation & ~judge_violation).sum()),
        "false_negative_count": int((~auto_violation & judge_violation).sum()),
    }


def _strict_invalid_value_violations(*, question: str, answer: str) -> list[str]:
    """Flag core invalid values without applying refusal/caveat exceptions.

    This is an ablation-only diagnostic. It approximates the effect of removing
    source-reliability-aware exceptions from numeric scientific constraints.
    """
    violations: set[str] = set()
    for match in list(_STRICT_BAND_GAP_RE.finditer(answer)) + list(_STRICT_EV_BAND_GAP_RE.finditer(answer)):
        band_gap = float(match.group(1))
        if band_gap < 0:
            violations.add("non_negative_band_gap")
        if band_gap > 15:
            violations.add("band_gap_physical_range")
    if "band gap" in question.lower():
        for match in _STRICT_STANDALONE_EV_RE.finditer(answer):
            if float(match.group(1)) < 0:
                violations.add("non_negative_band_gap")

    formation_energies = [float(match.group(1)) for match in _STRICT_FORMATION_ENERGY_RE.finditer(answer)]
    for formation_energy in formation_energies:
        if formation_energy < -4.0 or formation_energy > 1.0:
            violations.add("formation_energy_typical_range")
    if "formation energy" in question.lower() and _STRICT_SINGLE_ELEMENT_QUESTION_RE.search(question.strip()):
        for formation_energy in formation_energies:
            if formation_energy > 0.05:
                violations.add("elemental_formation_energy_reference_conflict")

    for match in _STRICT_TEMPERATURE_K_RE.finditer(answer):
        if float(match.group(1)) < 0:
            violations.add("non_negative_absolute_temperature")
    if "pressure" in question.lower() or "pressure" in answer.lower():
        for match in _STRICT_PRESSURE_RE.finditer(answer):
            if float(match.group(1)) < 0:
                violations.add("non_negative_pressure")
    return sorted(violations)


def _analyze_llm_judge(llm: pd.DataFrame) -> dict:
    df = llm.copy()
    clusters = _cluster_ids(df)
    auto_violation = df["auto_constraint_score_optimized"].astype(float) < CONSTRAINT_PASS_THRESHOLD
    judge_violation = _as_bool(df["judge_violation"])
    agreement = auto_violation == judge_violation

    df["auto_violation_optimized"] = auto_violation
    df["judge_violation_bool"] = judge_violation
    df["agreement_optimized"] = agreement
    df["disagreement_type"] = np.where(
        agreement,
        "agreement",
        np.where(auto_violation & ~judge_violation, "auto_false_positive", "auto_false_negative"),
    )

    confusion = _confusion_dataframe(
        row_label="qwen_plus_judge",
        col_label="pi_sret_auto",
        row_values=judge_violation,
        col_values=auto_violation,
    )
    confusion.to_csv(OUT / "llm_judge_confusion_matrix.csv", index=False)

    disagreement = df[~agreement].copy()
    taxonomy = (
        disagreement["disagreement_type"]
        .value_counts()
        .rename_axis("disagreement_type")
        .reset_index(name="count")
    )
    taxonomy["share_of_all"] = taxonomy["count"] / len(df)
    taxonomy["share_of_disagreements"] = taxonomy["count"] / len(disagreement) if len(disagreement) else 0.0
    taxonomy.to_csv(OUT / "llm_judge_disagreement_taxonomy.csv", index=False)

    _group_count(disagreement, ["constraint_family", "disagreement_type"], "count").to_csv(
        OUT / "llm_judge_disagreement_by_family.csv", index=False
    )
    _group_count(disagreement, ["document_status", "disagreement_type"], "count").to_csv(
        OUT / "llm_judge_disagreement_by_document_status.csv", index=False
    )
    violation_tokens = []
    for row in disagreement.itertuples(index=False):
        tokens = _split_semicolon(getattr(row, "auto_constraint_violations_optimized", ""))
        if not tokens:
            tokens = ["no_auto_violation"]
        for token in tokens:
            violation_tokens.append(
                {
                    "disagreement_type": getattr(row, "disagreement_type"),
                    "auto_violation_family": token,
                }
            )
    pd.DataFrame(violation_tokens).value_counts(
        ["disagreement_type", "auto_violation_family"]
    ).rename("count").reset_index().sort_values(
        ["count", "disagreement_type", "auto_violation_family"], ascending=[False, True, True]
    ).to_csv(OUT / "llm_judge_disagreement_by_auto_violation.csv", index=False)

    example_cols = [
        "sample_id",
        "response_id",
        "constraint_family",
        "document_status",
        "model",
        "mode",
        "disagreement_type",
        "question",
        "answer",
        "auto_constraint_score_optimized",
        "auto_constraint_violations_optimized",
        "judge_score",
        "judge_violation",
        "judge_rationale",
    ]
    disagreement[example_cols].head(100).to_csv(OUT / "llm_judge_disagreement_examples.csv", index=False)

    return {
        "n": int(len(df)),
        "agreement_rate": _metric_dict(agreement.astype(float), _mean, clusters),
        "cohen_kappa": _bootstrap_pair_metric(auto_violation, judge_violation, _cohen_kappa, clusters),
        "auto_violation_rate": _metric_dict(auto_violation.astype(float), _mean, clusters),
        "judge_violation_rate": _metric_dict(judge_violation.astype(float), _mean, clusters),
        "disagreements": int((~agreement).sum()),
        "confusion_matrix": confusion.to_dict("records"),
        "disagreement_taxonomy": taxonomy.to_dict("records"),
    }


def _analyze_dual_judge(dual: pd.DataFrame, manifest_meta: pd.DataFrame) -> dict:
    df = dual.copy()
    df = df.merge(
        manifest_meta[["response_id", "constraint_family", "document_status", "dataset_role", "model", "mode"]],
        on="response_id",
        how="left",
    )
    clusters = _cluster_ids(df)
    plus = _as_bool(df["qwen_plus_violation"])
    max_ = _as_bool(df["qwen_max_violation"])
    agreement = plus == max_
    df["qwen_plus_violation_bool"] = plus
    df["qwen_max_violation_bool"] = max_
    df["model_agreement_recomputed"] = agreement
    df["dual_disagreement_type"] = np.where(
        agreement,
        "agreement",
        np.where(plus & ~max_, "plus_only_violation", "max_only_violation"),
    )

    confusion = _confusion_dataframe(
        row_label="qwen_max_judge",
        col_label="qwen_plus_judge",
        row_values=max_,
        col_values=plus,
    )
    confusion.to_csv(OUT / "dual_judge_confusion_matrix.csv", index=False)
    disagreement = df[~agreement].copy()
    _group_count(disagreement, ["constraint_family", "dual_disagreement_type"], "count").to_csv(
        OUT / "dual_judge_disagreement_by_family.csv", index=False
    )
    _group_count(disagreement, ["document_status", "dual_disagreement_type"], "count").to_csv(
        OUT / "dual_judge_disagreement_by_document_status.csv", index=False
    )
    example_cols = [
        "response_id",
        "sample_id",
        "constraint_family",
        "document_status",
        "dual_disagreement_type",
        "qwen_plus_violation",
        "qwen_plus_rationale",
        "qwen_max_violation",
        "qwen_max_rationale",
        "auto_constraint_score",
        "auto_constraint_violations",
    ]
    disagreement[example_cols].to_csv(OUT / "dual_judge_disagreement_examples.csv", index=False)
    df.to_csv(OUT / "dual_judge_scores_with_metadata.csv", index=False)

    taxonomy = (
        disagreement["dual_disagreement_type"]
        .value_counts()
        .rename_axis("dual_disagreement_type")
        .reset_index(name="count")
    )
    taxonomy["share_of_all"] = taxonomy["count"] / len(df)
    taxonomy["share_of_disagreements"] = taxonomy["count"] / len(disagreement) if len(disagreement) else 0.0
    taxonomy.to_csv(OUT / "dual_judge_disagreement_taxonomy.csv", index=False)

    return {
        "n": int(len(df)),
        "agreement_rate": _metric_dict(agreement.astype(float), _mean, clusters),
        "cohen_kappa": _bootstrap_pair_metric(plus, max_, _cohen_kappa, clusters),
        "qwen_plus_violation_rate": _metric_dict(plus.astype(float), _mean, clusters),
        "qwen_max_violation_rate": _metric_dict(max_.astype(float), _mean, clusters),
        "disagreements": int((~agreement).sum()),
        "confusion_matrix": confusion.to_dict("records"),
        "disagreement_taxonomy": taxonomy.to_dict("records"),
    }


def _generation_mode_ablation() -> list[dict]:
    configs = [
        ("qwen-max", "naive_context", ROOT / "results" / "exp_a_qwen_max_naive" / "metrics.json"),
        ("qwen-max", "safety_aware", ROOT / "results" / "exp_a_qwen_max_safety" / "metrics.json"),
        ("deepseek-r1", "naive_context", ROOT / "results" / "exp_a_deepseek_r1_naive" / "metrics.json"),
        ("deepseek-r1", "safety_aware", ROOT / "results" / "exp_a_deepseek_r1_safety" / "metrics.json"),
    ]
    rows = []
    for model, mode, path in configs:
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": model,
                "mode": mode,
                "n": metrics.get("n_samples"),
                "mean_faithfulness": metrics.get("mean_faithfulness"),
                "mean_constraint_score": metrics.get("mean_constraint_score"),
                "violation_rate": metrics.get("violation_rate"),
                "pearson_r": metrics.get("pearson_r"),
                "spearman_rho": metrics.get("spearman_rho"),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "generation_mode_ablation_summary.csv", index=False)
    return rows


def _metric_row(
    name: str,
    values: pd.Series,
    statistic: Callable[[pd.Series], float],
    clusters: pd.Series | None = None,
) -> dict:
    metric = _metric_dict(values, statistic, clusters)
    return {"metric": name, **metric}


def _metric_dict(
    values: pd.Series,
    statistic: Callable[[pd.Series], float],
    clusters: pd.Series | None = None,
) -> dict:
    values, clusters = _aligned_values_and_clusters(values, clusters)
    value = statistic(values)
    ci_low, ci_high = _bootstrap_ci(values, statistic, clusters)
    return {"value": value, "ci_low": ci_low, "ci_high": ci_high, "n": int(len(values))}


def _bootstrap_pair_metric(
    left: pd.Series,
    right: pd.Series,
    statistic: Callable[[pd.Series, pd.Series], float | None],
    clusters: pd.Series | None = None,
) -> dict:
    left = pd.Series(left).reset_index(drop=True)
    right = pd.Series(right).reset_index(drop=True)
    clusters = (
        pd.Series([f"row:{i}" for i in range(len(left))])
        if clusters is None
        else pd.Series(clusters).reset_index(drop=True)
    )
    value = statistic(left, right)
    rng = np.random.default_rng(SEED)
    values = []
    cluster_positions = _cluster_positions(clusters)
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = _sample_cluster_positions(cluster_positions, rng)
        sample_value = statistic(left.iloc[idx].reset_index(drop=True), right.iloc[idx].reset_index(drop=True))
        if sample_value is not None and not math.isnan(sample_value):
            values.append(float(sample_value))
    ci_low, ci_high = _quantile_ci(values)
    return {"value": value, "ci_low": ci_low, "ci_high": ci_high, "n": int(len(left))}


def _bootstrap_ci(
    values: pd.Series,
    statistic: Callable[[pd.Series], float],
    clusters: pd.Series | None = None,
) -> tuple[float | None, float | None]:
    values, clusters = _aligned_values_and_clusters(values, clusters)
    if len(values) == 0:
        return None, None
    rng = np.random.default_rng(SEED)
    boot_values = []
    cluster_positions = _cluster_positions(clusters)
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = _sample_cluster_positions(cluster_positions, rng)
        boot_values.append(statistic(values.iloc[idx]))
    return _quantile_ci(boot_values)


def _cluster_ids(df: pd.DataFrame) -> pd.Series:
    """Prefer base-task clusters; fall back to context and row identity."""
    for column in ("base_task_id", "retrieved_context", "response_id", "sample_id"):
        if column in df.columns:
            values = df[column].fillna("").astype(str).reset_index(drop=True)
            if (values != "").any():
                fallback = pd.Series([f"row:{i}" for i in range(len(df))])
                return values.where(values != "", fallback)
    return pd.Series([f"row:{i}" for i in range(len(df))])


def _aligned_values_and_clusters(
    values: pd.Series,
    clusters: pd.Series | None,
) -> tuple[pd.Series, pd.Series]:
    raw_values = pd.Series(values).reset_index(drop=True)
    mask = raw_values.notna()
    clean_values = raw_values[mask].astype(float).reset_index(drop=True)
    if clusters is None:
        clean_clusters = pd.Series([f"row:{i}" for i in range(len(clean_values))])
    else:
        clean_clusters = pd.Series(clusters).reset_index(drop=True)[mask].astype(str).reset_index(drop=True)
    return clean_values, clean_clusters


def _cluster_positions(clusters: pd.Series) -> list[np.ndarray]:
    positions = pd.Series(np.arange(len(clusters)))
    return [group.to_numpy(dtype=int) for _, group in positions.groupby(clusters, sort=False)]


def _sample_cluster_positions(cluster_positions: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    selected = rng.integers(0, len(cluster_positions), len(cluster_positions))
    return np.concatenate([cluster_positions[index] for index in selected])


def _quantile_ci(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    low, high = np.quantile(np.array(values, dtype=float), [0.025, 0.975])
    return float(low), float(high)


def _mean(values: pd.Series) -> float:
    return float(pd.Series(values).astype(float).mean())


def _safe_corr(left: pd.Series, right: pd.Series) -> float | None:
    left = pd.Series(left).astype(float)
    right = pd.Series(right).astype(float)
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _safe_spearman(left: pd.Series, right: pd.Series) -> float | None:
    return _safe_corr(pd.Series(left).rank(method="average"), pd.Series(right).rank(method="average"))


def _cohen_kappa(left: pd.Series, right: pd.Series) -> float | None:
    left = pd.Series(left).astype(bool)
    right = pd.Series(right).astype(bool)
    if len(left) == 0:
        return None
    observed = float((left == right).mean())
    p_left_true = float(left.mean())
    p_right_true = float(right.mean())
    expected = p_left_true * p_right_true + (1.0 - p_left_true) * (1.0 - p_right_true)
    denominator = 1.0 - expected
    if denominator == 0:
        return None
    return float((observed - expected) / denominator)


def _as_bool(series: pd.Series) -> pd.Series:
    def convert(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not pd.isna(value):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n", "", "nan", "none"}:
            return False
        raise ValueError(f"Cannot parse boolean value: {value!r}")

    return series.map(convert).astype(bool)


def _confusion_dataframe(row_label: str, col_label: str, row_values: pd.Series, col_values: pd.Series) -> pd.DataFrame:
    row_values = pd.Series(row_values).astype(bool)
    col_values = pd.Series(col_values).astype(bool)
    rows = []
    for row_name, row_bool in [("no_violation", False), ("violation", True)]:
        for col_name, col_bool in [("no_violation", False), ("violation", True)]:
            rows.append(
                {
                    row_label: row_name,
                    col_label: col_name,
                    "count": int(((row_values == row_bool) & (col_values == col_bool)).sum()),
                }
            )
    return pd.DataFrame(rows)


def _group_count(df: pd.DataFrame, cols: list[str], name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=cols + [name])
    return (
        df.groupby(cols, dropna=False)
        .size()
        .rename(name)
        .reset_index()
        .sort_values([name] + cols, ascending=[False] + [True] * len(cols))
    )


def _explode_semicolon(series: pd.Series) -> pd.Series:
    tokens = []
    for value in series.fillna(""):
        tokens.extend(_split_semicolon(value))
    return pd.Series(tokens, dtype=str)


def _split_semicolon(value: object) -> list[str]:
    text = "" if pd.isna(value) else str(value)
    return [part.strip() for part in text.split(";") if part.strip()]


def _safe_div(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return float(num / den)


def _write_summary(metrics: dict[str, object]) -> None:
    ragas = metrics["ragas"]
    llm = metrics["llm_judge"]
    dual = metrics["dual_judge"]
    semantic = metrics["semantic_baselines"]["methods"]
    threshold = metrics["threshold_sensitivity"]
    ablation_ragas = metrics["constraint_ablation"]["ragas"]
    ablation_llm = metrics["constraint_ablation"]["llm_judge"]

    lines = [
        "# canonical_v3 Q1 Evidence Enhancement",
        "",
        "This report uses only completed canonical_v3 real-evaluation outputs. It excludes archived canonical_v4 data.",
        "",
        "## Statistical Summary",
        "",
        "| Metric | Value | 95% bootstrap CI | n |",
        "|---|---:|---:|---:|",
        _summary_metric_row("RAGAS faithfulness mean", ragas["mean_ragas_faithfulness"]),
        _summary_metric_row("Answer relevancy mean", ragas["mean_answer_relevancy"]),
        _summary_metric_row("PI-SRET constraint mean", ragas["mean_pi_sret_constraint_score"]),
        _summary_metric_row("PI-SRET violation rate", ragas["pi_sret_violation_rate"]),
        _summary_metric_row("High-faithfulness / low-constraint divergence rate", ragas["divergence_rate"]),
        _summary_metric_row("PI-SRET vs Qwen-plus agreement", llm["agreement_rate"]),
        _summary_metric_row("PI-SRET vs Qwen-plus Cohen's kappa", llm["cohen_kappa"]),
        _summary_metric_row("Qwen-plus vs Qwen-max agreement", dual["agreement_rate"]),
        _summary_metric_row("Qwen-plus vs Qwen-max Cohen's kappa", dual["cohen_kappa"]),
        "",
        "## Semantic Baseline Robustness",
        "",
        "| Method | Mean score | Pearson with G | Fixed high-support low-G cases | Quantile-reference high-support low-G cases |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in semantic:
        pearson = "NA" if row["pearson_with_constraint"] is None else f"{row['pearson_with_constraint']:.3f}"
        lines.append(
            f"| {row['method']} | {row['mean_score']:.3f} | {pearson} | "
            f"{row['fixed_high_support_low_constraint_cases']}/{row['fixed_high_support_cases']} | "
            f"{row['quantile_reference_high_support_low_constraint_cases']}/"
            f"{row['quantile_reference_high_support_cases']} |"
        )
    lines.extend(
        [
            "",
            "## Threshold Sensitivity",
            "",
            "| tau_f | tau_g | High-support cases | Divergence cases | Divergence rate | Risk among high-support |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in threshold:
        if float(row["tau_g"]) != CONSTRAINT_PASS_THRESHOLD:
            continue
        lines.append(
            f"| {row['tau_f']:.2f} | {row['tau_g']:.2f} | {row['high_support_cases']} | "
            f"{row['divergence_cases']} | {row['divergence_rate_all']:.3f} | "
            f"{row['risk_rate_among_high_support']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Constraint Ablation",
            "",
            "| Variant | RAGAS divergence cases | RAGAS divergence rate | LLM agreement | LLM kappa |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    llm_by_variant = {row["variant"]: row for row in ablation_llm}
    for row in ablation_ragas:
        llm_row = llm_by_variant.get(row["variant"], {})
        agreement = llm_row.get("agreement")
        kappa = llm_row.get("cohen_kappa")
        agreement_text = "NA" if agreement is None else f"{agreement:.3f}"
        kappa_text = "NA" if kappa is None else f"{kappa:.3f}"
        rate = "NA" if row["divergence_rate_all"] is None else f"{row['divergence_rate_all']:.3f}"
        lines.append(
            f"| {row['variant']} | {row['divergence_cases']} | {rate} | {agreement_text} | {kappa_text} |"
        )
    lines.extend(
        [
            "",
            "## Key Output Files",
            "",
            "- `ragas_bootstrap_summary.csv`: bootstrap confidence intervals for RAGAS and PI-SRET metrics.",
            "- `semantic_baseline_robustness.csv`: local lexical, embedding, and NLI-style semantic baseline robustness.",
            "- `threshold_sensitivity.csv`: divergence stability across faithfulness and constraint thresholds.",
            "- `constraint_ablation_ragas.csv`: RAGAS divergence under constraint-family ablations.",
            "- `constraint_ablation_llm_judge.csv`: PI-SRET/LLM-judge agreement under constraint-family ablations.",
            "- `ragas_family_metrics.csv`: family-level RAGAS/PI-SRET metrics.",
            "- `ragas_divergence_by_family.csv`: family distribution of divergence cases.",
            "- `ragas_divergence_by_violation.csv`: fine-grained violation distribution among divergence cases.",
            "- `llm_judge_confusion_matrix.csv`: PI-SRET automatic rules vs Qwen-plus judge.",
            "- `llm_judge_disagreement_taxonomy.csv`: false-positive / false-negative disagreement taxonomy.",
            "- `dual_judge_confusion_matrix.csv`: Qwen-plus vs Qwen-max judge matrix.",
            "- `generation_mode_ablation_summary.csv`: naive-context vs safety-aware generation summary.",
            "",
            "## Interpretation Boundary",
            "",
            "These analyses strengthen statistical reliability and baseline coverage, but they do not replace human expert validation.",
            "",
        ]
    )
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _summary_metric_row(label: str, metric: dict) -> str:
    value = metric["value"]
    low = metric["ci_low"]
    high = metric["ci_high"]
    ci = "NA" if low is None or high is None else f"[{low:.3f}, {high:.3f}]"
    val = "NA" if value is None else f"{value:.3f}"
    return f"| {label} | {val} | {ci} | {metric['n']} |"


if __name__ == "__main__":
    raise SystemExit(main())
