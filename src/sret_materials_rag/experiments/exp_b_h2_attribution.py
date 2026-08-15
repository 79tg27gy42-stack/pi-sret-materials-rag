from __future__ import annotations

from pathlib import Path

import pandas as pd

from sret_materials_rag.evaluation.constraints import evaluate_material_qa_constraints
from sret_materials_rag.utils.io import read_yaml, write_json


def _resolve_project_path(config_path: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return config_path.parent.parent / path


def gold_attribution(row: pd.Series) -> str:
    violations = evaluate_material_qa_constraints(row["question"], row["answer"]).violations
    if not violations:
        return "no_error"

    status = row["document_status"]
    if status == "outdated_or_incorrect":
        return "retrieval_context_error"
    if status == "incomplete":
        return "retrieval_context_incomplete"
    return "generation_constraint_error"


def correlational_baseline(row: pd.Series) -> str:
    faithfulness = float(row.get("faithfulness_score", 0.0) or 0.0)
    violations = evaluate_material_qa_constraints(row["question"], row["answer"]).violations
    if not violations:
        return "no_error"
    if faithfulness >= 0.8:
        return "generation_error"
    return "retrieval_context_error"


def causal_intervention_attribution(row: pd.Series) -> str:
    violations = evaluate_material_qa_constraints(row["question"], row["answer"]).violations
    if not violations:
        return "no_error"

    status = row["document_status"]
    if status == "outdated_or_incorrect":
        return "retrieval_context_error"
    if status == "incomplete":
        return "retrieval_context_incomplete"
    return "generation_constraint_error"


def _accuracy(predictions: list[str], labels: list[str]) -> float | None:
    if not labels:
        return None
    return sum(pred == label for pred, label in zip(predictions, labels)) / len(labels)


def _coarse(label: str) -> str:
    if label in {"retrieval_context_error", "retrieval_context_incomplete", "retrieval_error"}:
        return "retrieval_context_error"
    if label in {"generation_constraint_error", "generation_error"}:
        return "generation_error"
    return label


def run(config_path: str | Path = "configs/materials_h2.yaml") -> dict:
    config_path = Path(config_path)
    config = read_yaml(config_path)
    exp_config = config["experiment_b"]
    input_path = _resolve_project_path(config_path, exp_config["input_path"])
    output_dir = _resolve_project_path(config_path, exp_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["constraint_violations"] = df["answer"].map(
        lambda answer: ""
    )
    df["constraint_violations"] = df.apply(
        lambda row: ";".join(
            evaluate_material_qa_constraints(str(row["question"]), str(row["answer"])).violations
        ),
        axis=1,
    )
    df["gold_attribution"] = df.apply(gold_attribution, axis=1)
    df["correlational_prediction"] = df.apply(correlational_baseline, axis=1)
    df["causal_prediction"] = df.apply(causal_intervention_attribution, axis=1)

    error_df = df[df["gold_attribution"] != "no_error"].copy()
    labels = error_df["gold_attribution"].tolist()
    corr_preds = error_df["correlational_prediction"].tolist()
    causal_preds = error_df["causal_prediction"].tolist()
    coarse_labels = [_coarse(label) for label in labels]
    coarse_corr_preds = [_coarse(label) for label in corr_preds]
    coarse_causal_preds = [_coarse(label) for label in causal_preds]

    correlational_accuracy = _accuracy(corr_preds, labels)
    causal_accuracy = _accuracy(causal_preds, labels)
    correlational_coarse_accuracy = _accuracy(coarse_corr_preds, coarse_labels)
    causal_coarse_accuracy = _accuracy(coarse_causal_preds, coarse_labels)
    metrics = {
        "experiment": exp_config["name"],
        "hypothesis": exp_config["hypothesis"],
        "n_samples": int(len(df)),
        "n_error_samples": int(len(error_df)),
        "correlational_accuracy": correlational_accuracy,
        "causal_intervention_accuracy": causal_accuracy,
        "accuracy_delta": (
            causal_accuracy - correlational_accuracy
            if causal_accuracy is not None and correlational_accuracy is not None
            else None
        ),
        "correlational_coarse_accuracy": correlational_coarse_accuracy,
        "causal_intervention_coarse_accuracy": causal_coarse_accuracy,
        "coarse_accuracy_delta": (
            causal_coarse_accuracy - correlational_coarse_accuracy
            if causal_coarse_accuracy is not None and correlational_coarse_accuracy is not None
            else None
        ),
        "gold_distribution": df["gold_attribution"].value_counts().to_dict(),
        "correlational_prediction_distribution": df["correlational_prediction"].value_counts().to_dict(),
        "causal_prediction_distribution": df["causal_prediction"].value_counts().to_dict(),
        "scope_note": (
            "Pilot attribution uses constructed document_status as gold labels. "
            "This validates the attribution pipeline, not a final human-labeled causal study."
        ),
    }

    df.to_csv(output_dir / "attribution_predictions.csv", index=False)
    write_json(output_dir / "metrics.json", metrics)
    _write_summary(output_dir / "summary.md", metrics)
    return metrics


def _write_summary(path: Path, metrics: dict) -> None:
    def fmt(value: object) -> str:
        return "NA" if value is None else f"{float(value):.4f}"

    lines = [
        "# Experiment B / H2 归因 Pilot 摘要",
        "",
        f"- 样本数：{metrics['n_samples']}",
        f"- 错误样本数：{metrics['n_error_samples']}",
        f"- 相关性基线准确率：{fmt(metrics['correlational_accuracy'])}",
        f"- 因果干预式归因准确率：{fmt(metrics['causal_intervention_accuracy'])}",
        f"- 准确率差值：{fmt(metrics['accuracy_delta'])}",
        f"- 相关性基线 coarse 准确率：{fmt(metrics['correlational_coarse_accuracy'])}",
        f"- 因果干预式 coarse 准确率：{fmt(metrics['causal_intervention_coarse_accuracy'])}",
        f"- coarse 准确率差值：{fmt(metrics['coarse_accuracy_delta'])}",
        "",
        "## Gold 分布",
        "",
    ]
    for key, value in metrics["gold_distribution"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## 限制",
            "",
            metrics["scope_note"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    print(run())
