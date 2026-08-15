from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from sret_materials_rag.evaluation.constraints import (
    evaluate_material_qa_constraints,
)


class ScientificHallucinationType(str, Enum):
    NO_ERROR = "no_error"
    UNSUPPORTED_HALLUCINATION = "unsupported_hallucination"
    CONTEXT_SUPPORTED_PHYSICALLY_INVALID = "context_supported_physically_invalid"
    SCIENTIFICALLY_UNDERDETERMINED = "scientifically_underdetermined_generation"
    RETRIEVAL_INDUCED_ERROR = "retrieval_induced_scientific_error"


@dataclass(frozen=True)
class PhysicsInformedScore:
    total_score: float
    symbolic_consistency: float
    physical_plausibility: float
    scientific_prior: float
    uncertainty_calibration: float
    violations: list[str] = field(default_factory=list)
    hallucination_type: ScientificHallucinationType = ScientificHallucinationType.NO_ERROR
    rationale: str = ""


_UNCERTAINTY_MARKERS = [
    "cannot determine",
    "cannot be determined",
    "insufficient",
    "not enough",
    "not reliable",
    "uncertain",
    "should not be used",
    "requires",
    "without",
]

_CONTEXT_UNRELIABLE_MARKERS = [
    "outdated_or_incorrect",
    "noisy retrieved",
    "corrupted",
    "human review",
    "marked as outdated",
]

_CONTEXT_INCOMPLETE_MARKERS = [
    "omits",
    "incomplete",
    "insufficient",
    "lacks",
    "without reporting",
]

_NUMERIC_UNIT_RE = re.compile(r"-?\d+(?:\.\d+)?\s*(?:eV|K|GPa|MPa|Pa)\b", re.I)


def score_physics_informed_consistency(
    *,
    question: str,
    answer: str,
    retrieved_context: str,
    document_status: str = "unknown",
) -> PhysicsInformedScore:
    """Score scientific consistency with physics-informed components.

    This is not a neural PINN. It is a deterministic scaffold for the same
    principle: evaluate outputs against scientific invariants and priors that
    are independent of surface-level textual faithfulness.
    """
    constraint = evaluate_material_qa_constraints(question, answer)
    symbolic = 1.0 if not constraint.violations else 0.0
    physical = _physical_plausibility_score(question, answer, constraint.violations)
    prior = _scientific_prior_score(question, answer, retrieved_context, document_status)
    uncertainty = _uncertainty_calibration_score(answer, retrieved_context, document_status, constraint.violations)
    total = _weighted_mean(
        {
            "symbolic": symbolic,
            "physical": physical,
            "prior": prior,
            "uncertainty": uncertainty,
        },
        {
            "symbolic": 0.25,
            "physical": 0.40,
            "prior": 0.20,
            "uncertainty": 0.15,
        },
    )
    hallucination_type = classify_scientific_hallucination(
        question=question,
        answer=answer,
        retrieved_context=retrieved_context,
        document_status=document_status,
        violations=constraint.violations,
        uncertainty_calibration=uncertainty,
    )
    return PhysicsInformedScore(
        total_score=total,
        symbolic_consistency=symbolic,
        physical_plausibility=physical,
        scientific_prior=prior,
        uncertainty_calibration=uncertainty,
        violations=constraint.violations,
        hallucination_type=hallucination_type,
        rationale=_rationale(constraint.violations, document_status, hallucination_type),
    )


def classify_scientific_hallucination(
    *,
    question: str,
    answer: str,
    retrieved_context: str,
    document_status: str = "unknown",
    violations: list[str] | None = None,
    uncertainty_calibration: float | None = None,
) -> ScientificHallucinationType:
    violations = violations if violations is not None else evaluate_material_qa_constraints(question, answer).violations
    uncertainty_calibration = (
        uncertainty_calibration
        if uncertainty_calibration is not None
        else _uncertainty_calibration_score(answer, retrieved_context, document_status, violations)
    )
    if not violations and uncertainty_calibration >= 1.0:
        return ScientificHallucinationType.NO_ERROR
    answer_supported = _answer_is_context_supported(answer, retrieved_context)
    context_unreliable = _context_is_unreliable(retrieved_context, document_status)
    context_incomplete = _context_is_incomplete(retrieved_context, document_status)

    if violations and answer_supported and context_unreliable:
        return ScientificHallucinationType.RETRIEVAL_INDUCED_ERROR
    if violations and answer_supported and context_incomplete:
        return ScientificHallucinationType.SCIENTIFICALLY_UNDERDETERMINED
    if violations and answer_supported:
        return ScientificHallucinationType.CONTEXT_SUPPORTED_PHYSICALLY_INVALID
    if violations:
        return ScientificHallucinationType.UNSUPPORTED_HALLUCINATION
    if context_incomplete and uncertainty_calibration < 1.0:
        return ScientificHallucinationType.SCIENTIFICALLY_UNDERDETERMINED
    if not answer_supported and not _has_uncertainty_marker(answer):
        return ScientificHallucinationType.UNSUPPORTED_HALLUCINATION
    return ScientificHallucinationType.NO_ERROR


def _physical_plausibility_score(question: str, answer: str, violations: list[str]) -> float:
    if not violations:
        return 1.0
    severe = {
        "non_negative_band_gap",
        "band_gap_physical_range",
        "formation_energy_typical_range",
        "non_negative_absolute_temperature",
        "non_negative_pressure",
        "valid_chemical_symbols",
        "positive_formation_energy_stability_conflict",
        "conductivity_band_gap_consistency",
        "crystal_system_space_group_consistency",
    }
    severe_count = sum(1 for violation in violations if violation in severe)
    if severe_count:
        return 0.0
    return 0.25


def _scientific_prior_score(
    question: str,
    answer: str,
    retrieved_context: str,
    document_status: str,
) -> float:
    lower_answer = answer.lower()
    lower_question = question.lower()
    score = 1.0
    if "stable" in lower_answer and not any(
        marker in lower_answer
        for marker in ["formation energy", "hull", "thermodynamic", "relative"]
    ):
        score -= 0.45
    if "superconduct" in lower_answer and not any(
        marker in lower_answer
        for marker in ["tc", "transition temperature", "measured", "reported", "evidence"]
    ):
        score -= 0.45
    if "band gap" in lower_question and _has_negative_quantity(answer):
        score -= 0.55
    if any(marker in lower_answer for marker in ["formation energy", "space group", "metallic", "insulator"]):
        if evaluate_material_qa_constraints(question, answer).violations:
            score -= 0.35
    if _context_is_unreliable(retrieved_context, document_status) and not _has_uncertainty_marker(answer):
        score -= 0.25
    return max(0.0, score)


def _uncertainty_calibration_score(
    answer: str,
    retrieved_context: str,
    document_status: str,
    violations: list[str],
) -> float:
    context_unreliable = _context_is_unreliable(retrieved_context, document_status)
    context_incomplete = _context_is_incomplete(retrieved_context, document_status)
    uncertain = _has_uncertainty_marker(answer)
    if violations and not uncertain:
        return 0.0
    if (context_unreliable or context_incomplete) and uncertain:
        return 1.0
    if (context_unreliable or context_incomplete) and not uncertain:
        return 0.25
    return 1.0


def _weighted_mean(values: dict[str, float], weights: dict[str, float]) -> float:
    return sum(values[key] * weights[key] for key in values)


def _context_is_unreliable(context: str, document_status: str) -> bool:
    lower_context = context.lower()
    return document_status == "outdated_or_incorrect" or any(
        marker in lower_context for marker in _CONTEXT_UNRELIABLE_MARKERS
    )


def _context_is_incomplete(context: str, document_status: str) -> bool:
    lower_context = context.lower()
    return document_status == "incomplete" or any(
        marker in lower_context for marker in _CONTEXT_INCOMPLETE_MARKERS
    )


def _has_uncertainty_marker(answer: str) -> bool:
    lower_answer = answer.lower()
    return any(marker in lower_answer for marker in _UNCERTAINTY_MARKERS)


def _has_negative_quantity(answer: str) -> bool:
    return any(float(match.group(0).split()[0]) < 0 for match in _NUMERIC_UNIT_RE.finditer(answer))


def _answer_is_context_supported(answer: str, context: str) -> bool:
    lower_answer = answer.lower()
    lower_context = context.lower()
    numeric_claims = [match.group(0).lower() for match in _NUMERIC_UNIT_RE.finditer(answer)]
    if numeric_claims:
        return all(claim in lower_context for claim in numeric_claims)
    tokens = {
        token.strip(".+-")
        for token in re.findall(r"[a-z0-9.+-]+", lower_answer)
        if len(token.strip(".+-")) > 2
    }
    if not tokens:
        return False
    context_tokens = {
        token.strip(".+-")
        for token in re.findall(r"[a-z0-9.+-]+", lower_context)
        if len(token.strip(".+-")) > 2
    }
    overlap = len(tokens & context_tokens) / len(tokens)
    return overlap >= 0.35


def _rationale(
    violations: list[str],
    document_status: str,
    hallucination_type: ScientificHallucinationType,
) -> str:
    if hallucination_type == ScientificHallucinationType.NO_ERROR:
        return "No physics-informed inconsistency detected."
    if violations:
        return (
            f"Detected {','.join(violations)} under document_status={document_status}; "
            f"classified as {hallucination_type.value}."
        )
    return f"Classified as {hallucination_type.value} under document_status={document_status}."
