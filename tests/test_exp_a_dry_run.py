from __future__ import annotations

from pathlib import Path

from sret_materials_rag.experiments.exp_a_h1_divergence import run


def test_exp_a_dry_run() -> None:
    root = Path(__file__).resolve().parents[1]
    metrics = run(root / "configs/materials.yaml")
    assert metrics["hypothesis"] == "H1"
    assert metrics["n_samples"] > 0
    assert "pearson_r" in metrics

