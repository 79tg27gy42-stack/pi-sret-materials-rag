from __future__ import annotations

import pytest

from sret_materials_rag.evaluation.physics_informed import (
    ScientificHallucinationType,
    score_physics_informed_consistency,
)
from sret_materials_rag.experiments.physics_guided_repair import physics_guided_repair


def test_context_supported_physically_invalid_claim_is_detected() -> None:
    score = score_physics_informed_consistency(
        question="What is the band gap of Si?",
        retrieved_context=(
            "A noisy retrieved note reports Si with band gap -1.0 eV. "
            "This note is marked as outdated_or_incorrect."
        ),
        answer="The band gap of Si is -1.0 eV.",
        document_status="outdated_or_incorrect",
    )
    assert score.symbolic_consistency == 0.0
    assert "non_negative_band_gap" in score.violations
    assert score.hallucination_type == ScientificHallucinationType.RETRIEVAL_INDUCED_ERROR
    assert score.total_score < 0.5


def test_uncertain_answer_on_incomplete_context_is_not_flagged_as_hallucination() -> None:
    score = score_physics_informed_consistency(
        question="Is Si thermodynamically stable?",
        retrieved_context="A short snippet says Si is stable but omits hull distance.",
        answer="The context is insufficient to determine thermodynamic stability.",
        document_status="incomplete",
    )
    assert score.hallucination_type == ScientificHallucinationType.NO_ERROR
    assert score.uncertainty_calibration == 1.0


def test_constraint_aware_rejection_without_context_overlap_is_not_hallucination() -> None:
    score = score_physics_informed_consistency(
        question="Can an experiment be performed at -50 K?",
        retrieved_context="Absolute temperature on the Kelvin scale cannot be below 0 K.",
        answer="The experiment cannot be performed at -50 K because absolute temperature cannot be below 0 K.",
        document_status="current",
    )
    assert score.hallucination_type == ScientificHallucinationType.NO_ERROR
    assert score.total_score == pytest.approx(1.0)


def test_overconfident_stability_on_incomplete_context_is_underdetermined() -> None:
    score = score_physics_informed_consistency(
        question="Is Si thermodynamically stable?",
        retrieved_context="A short snippet says Si is stable but omits hull distance.",
        answer="Si is stable.",
        document_status="incomplete",
    )
    assert score.hallucination_type == ScientificHallucinationType.SCIENTIFICALLY_UNDERDETERMINED
    assert "cautious_stability_claim" in score.violations


def test_physics_guided_repair_improves_score() -> None:
    repair = physics_guided_repair(
        question="What is the band gap of Si?",
        retrieved_context="A noisy retrieved note reports Si with band gap -1.0 eV.",
        answer="The band gap of Si is -1.0 eV.",
        document_status="outdated_or_incorrect",
    )
    assert repair.action in {"retrieval_correction", "physically_plausible_rewrite"}
    assert repair.score_after.total_score > repair.score_before.total_score
    assert repair.score_after.symbolic_consistency == 1.0
