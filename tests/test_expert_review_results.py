import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "canonical_v3_q1_evidence"


def test_expert_review_outputs_are_complete_and_consistent():
    metrics = json.loads((RESULTS / "expert_review_metrics_v3.json").read_text(encoding="utf-8"))
    assert metrics["protocol"]["n"] == 200
    assert metrics["completion"]["missing_required_fields"] == 0
    assert metrics["expert_labels"]["disagreements"] == 27
    assert metrics["protocol_adjudicated_labels"] == {
        "violation": 85,
        "no_violation": 115,
        "violation_rate": 0.425,
    }
    assert round(metrics["expert_agreement"]["kappa"], 3) == 0.734
    assert metrics["primary_consensus_reference"]["n"] == 173
    assert metrics["primary_consensus_reference"]["tp"] == 80
    assert metrics["primary_consensus_reference"]["tn"] == 80
    assert round(metrics["primary_consensus_reference"]["kappa"], 3) == 0.850
    assert round(metrics["primary_consensus_reference"]["f1"], 3) == 0.925
    assert round(metrics["auto_vs_protocol_adjudicated"]["f1"], 3) == 0.901
    assert metrics["third_expert_adjudication"] == {
        "n_disagreements": 27,
        "violation": 4,
        "no_violation": 23,
        "sides_with_expert_a": 4,
        "sides_with_expert_b": 23,
        "uncertain": 0,
        "agreement_with_protocol_resolution": 1.0,
    }
    assert metrics["third_human_adjudicated_labels"]["violation"] == 85
    assert round(metrics["auto_vs_third_human_adjudicated"]["kappa"], 3) == 0.819
    assert metrics["human_referenced_error_analysis"]["total"] == 18
    assert metrics["human_referenced_error_analysis"]["false_positive"] == 15
    assert metrics["human_referenced_error_analysis"]["false_negative"] == 3

    with (ROOT / "data/annotations/expert_review_adjudicated_v3.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 200
    assert sum(row["protocol_adjudicated_label"] == "是" for row in rows) == 85
    assert sum(row["human_final_label"] == "是" for row in rows) == 85
    assert sum(row["human_final_method"] == "independent_third_human_adjudication" for row in rows) == 27


def test_private_linkage_is_not_released():
    private_linkage = ROOT / "data/processed/canonical_v3/expert_review_stratified_200_v3_private_linkage.csv"
    assert not private_linkage.exists()
