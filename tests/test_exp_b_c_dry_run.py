from __future__ import annotations

from pathlib import Path

from sret_materials_rag.evaluation.constraints import evaluate_material_constraints
from sret_materials_rag.experiments.exp_b_h2_attribution import run as run_h2
from sret_materials_rag.experiments.exp_c_h3_repair import repair_answer, run as run_h3


def test_exp_b_dry_run() -> None:
    root = Path(__file__).resolve().parents[1]
    metrics = run_h2(root / "configs/materials_h2.yaml")
    assert metrics["hypothesis"] == "H2"
    assert metrics["n_samples"] > 0
    assert metrics["n_error_samples"] > 0
    assert "correlational_accuracy" in metrics
    assert "causal_intervention_accuracy" in metrics


def test_exp_c_dry_run() -> None:
    root = Path(__file__).resolve().parents[1]
    metrics = run_h3(root / "configs/materials_h3.yaml")
    assert metrics["hypothesis"] == "H3"
    assert metrics["n_samples"] > 0
    assert metrics["n_repaired"] > 0
    assert metrics["mean_constraint_after"] >= metrics["mean_constraint_before"]


def test_repair_answer_removes_known_band_gap_violation() -> None:
    repaired_answer, strategy = repair_answer("The material has a band gap of -1.0 eV.")
    result = evaluate_material_constraints(repaired_answer)
    assert strategy == "non_negative_band_gap"
    assert result.score == 1.0
    assert result.violations == []
