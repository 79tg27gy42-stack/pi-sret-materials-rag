from __future__ import annotations

from sret_materials_rag.evaluation.chemistry_constraints import evaluate_chemistry_constraints


def test_valid_chemistry_answer_passes():
    result = evaluate_chemistry_constraints("The reported yield is 91% at pH 7.4 and 310 K.")
    assert result.score == 1.0
    assert result.violations == []


def test_invalid_ph_fails():
    result = evaluate_chemistry_constraints("The aqueous reaction has pH 19.")
    assert "aqueous_ph_range" in result.violations


def test_invalid_yield_fails():
    result = evaluate_chemistry_constraints("The isolated yield is 143%.")
    assert "yield_percentage_range" in result.violations


def test_refusal_invalid_value_passes():
    result = evaluate_chemistry_constraints(
        "The retrieved pH 19 should not be used because it is invalid for an aqueous solution."
    )
    assert result.score == 1.0
