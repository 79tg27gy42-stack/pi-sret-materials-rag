from __future__ import annotations

from dataclasses import dataclass

from sret_materials_rag.evaluation.physics_informed import (
    PhysicsInformedScore,
    ScientificHallucinationType,
    score_physics_informed_consistency,
)


@dataclass(frozen=True)
class PhysicsGuidedRepair:
    repaired_answer: str
    action: str
    score_before: PhysicsInformedScore
    score_after: PhysicsInformedScore


def physics_guided_repair(
    *,
    question: str,
    answer: str,
    retrieved_context: str,
    document_status: str = "unknown",
) -> PhysicsGuidedRepair:
    before = score_physics_informed_consistency(
        question=question,
        answer=answer,
        retrieved_context=retrieved_context,
        document_status=document_status,
    )
    if before.hallucination_type == ScientificHallucinationType.NO_ERROR:
        after = before
        return PhysicsGuidedRepair(answer, "none", before, after)

    repaired, action = _repair_text(question, before)
    after = score_physics_informed_consistency(
        question=question,
        answer=repaired,
        retrieved_context=retrieved_context,
        document_status=document_status,
    )
    return PhysicsGuidedRepair(repaired, action, before, after)


def _repair_text(question: str, score: PhysicsInformedScore) -> tuple[str, str]:
    lower_question = question.lower()
    if score.hallucination_type == ScientificHallucinationType.SCIENTIFICALLY_UNDERDETERMINED:
        return (
            "The retrieved context is insufficient to determine the requested scientific property. "
            "A reliable answer requires the missing thermodynamic or experimental evidence.",
            "uncertainty_aware_generation",
        )
    if score.hallucination_type == ScientificHallucinationType.RETRIEVAL_INDUCED_ERROR:
        return (
            "The retrieved context contains a scientifically inconsistent claim, so the answer "
            "should not copy that value. The relevant property cannot be determined reliably "
            "from this retrieved context without checking a trusted source.",
            "retrieval_correction",
        )
    if "band gap" in lower_question:
        return (
            "A physically valid band gap cannot be negative. The retrieved value should be "
            "treated as inconsistent, and the valid band gap cannot be determined from this context.",
            "physically_plausible_rewrite",
        )
    if "stable" in lower_question or "stability" in lower_question:
        return (
            "The retrieved text does not provide enough thermodynamic evidence to assert stability. "
            "A valid stability answer should cite formation energy, energy above hull, or a comparable thermodynamic criterion.",
            "uncertainty_aware_generation",
        )
    return (
        "The answer should be revised because it violates a scientific consistency constraint; "
        "the corrected value cannot be determined from the retrieved context alone.",
        "consistency_constrained_rewrite",
    )
