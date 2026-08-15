import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "frozen_heldout_rag_v3"


def test_frozen_heldout_design_is_paired_and_disjoint():
    metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    design = metrics["design"]
    assert design["response_records"] == 756
    assert design["paired_prompt_contexts"] == 252
    assert design["base_task_clusters"] == 168
    assert design["base_task_overlap_with_development"] == 0
    assert set(design["systems"]) == {"local_rule_rag", "qwen-max", "deepseek-r1"}

    scores = pd.read_csv(OUT / "scores.csv")
    assert len(scores) == 756
    counts = scores.groupby("raw_sample_id")["model"].agg(["size", "nunique"])
    assert counts["size"].eq(3).all()
    assert counts["nunique"].eq(3).all()


def test_frozen_heldout_summary_matches_expected_direction():
    summary = pd.read_csv(OUT / "system_summary.csv").set_index("system")
    assert summary.loc["deepseek-r1", "mean_nli_support"] > summary.loc["local_rule_rag", "mean_nli_support"]
    assert summary.loc["local_rule_rag", "mean_nli_support"] > summary.loc["qwen-max", "mean_nli_support"]
    assert summary.loc["qwen-max", "mean_pi_sret_constraint_score"] > summary.loc["deepseek-r1", "mean_pi_sret_constraint_score"]
