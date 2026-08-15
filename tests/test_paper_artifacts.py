from __future__ import annotations

import pandas as pd
import pytest

from scripts.analyze_claim_splitting_sensitivity import analyze
from scripts.evaluate_extraction_audit import evaluate


def test_extraction_audit_requires_completed_gold() -> None:
    frame = pd.DataFrame(
        [
            {
                "predicted_material": "Si",
                "predicted_formula": "Si",
                "predicted_numeric_value": "1.1",
                "predicted_unit": "eV",
                "predicted_property": "band_gap",
                "predicted_phase": "",
                "predicted_method": "",
                "predicted_evidence": "",
                "predicted_qualifier": "",
                "gold_material": "",
                "gold_formula": "",
                "gold_numeric_value": "",
                "predicted_structure": "",
                "gold_structure": "",
                "gold_unit": "",
                "gold_property": "",
                "gold_phase": "",
                "gold_method": "",
                "gold_evidence": "",
                "gold_qualifier": "",
                "annotation_complete": "",
            }
        ]
    )
    with pytest.raises(ValueError, match="incomplete"):
        evaluate(frame)


def test_extraction_audit_scores_micro_metrics() -> None:
    frame = pd.DataFrame(
        [
            {
                "predicted_material": "Si",
                "predicted_formula": "Si",
                "predicted_numeric_value": "1.10",
                "predicted_unit": "Kelvin",
                "predicted_property": "band_gap",
                "predicted_phase": "",
                "predicted_method": "",
                "predicted_evidence": "",
                "predicted_qualifier": "",
                "gold_material": "Si",
                "gold_formula": "Si",
                "gold_numeric_value": "1.1",
                "predicted_structure": "",
                "gold_structure": "",
                "gold_unit": "K",
                "gold_property": "band_gap",
                "gold_phase": "",
                "gold_method": "",
                "gold_evidence": "",
                "gold_qualifier": "",
                "annotation_complete": "yes",
            }
        ]
    )
    result = evaluate(frame)
    overall = result.loc[result["extraction_type"] == "overall_micro"].iloc[0]
    assert overall["tp"] == 5
    assert overall["fp"] == 0
    assert overall["fn"] == 0
    assert overall["f1"] == 1.0


def test_claim_splitting_sensitivity_outputs_modes() -> None:
    frame = pd.DataFrame(
        [
            {
                "response_id": "r1",
                "constraint_family": "band_gap",
                "answer": "Si has a band gap of 1.1 eV. The value is reported by the context.",
                "retrieved_context": "Si has a band gap of 1.1 eV.",
                "pi_sret_constraint_score": 0.0,
            }
        ]
    )
    summary, detail, metadata = analyze(frame, threshold=0.5)
    assert set(summary["segmentation"]) == {"atomic", "sentence", "whole_answer"}
    assert len(detail) == 1
    assert "atomic_vs_sentence" in metadata["divergence_jaccard"]
