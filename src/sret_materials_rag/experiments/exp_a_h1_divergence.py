from __future__ import annotations

from pathlib import Path
from dataclasses import fields

import pandas as pd

from sret_materials_rag.evaluation.constraints import evaluate_material_qa_constraints
from sret_materials_rag.evaluation.faithfulness_evaluators import (
    FaithfulnessEvaluator,
    build_faithfulness_evaluator,
)
from sret_materials_rag.evaluation.statistics import correlation_with_ci, pearson_r, spearman_rho
from sret_materials_rag.models import QASample, ScoredSample
from sret_materials_rag.utils.io import read_jsonl, read_yaml, write_json


def load_samples(path: str | Path) -> list[QASample]:
    allowed_fields = {field.name for field in fields(QASample)}
    return [
        QASample(**{key: value for key, value in record.items() if key in allowed_fields})
        for record in read_jsonl(path)
    ]


def score_sample(sample: QASample, faithfulness_evaluator: FaithfulnessEvaluator) -> ScoredSample:
    constraint_result = evaluate_material_qa_constraints(sample.question, sample.answer)
    if sample.faithfulness_score is not None:
        faithfulness_score = float(sample.faithfulness_score)
        faithfulness_method = "provided_column"
    else:
        faithfulness_result = faithfulness_evaluator.score(
            question=sample.question,
            answer=sample.answer,
            retrieved_context=sample.retrieved_context,
        )
        faithfulness_score = faithfulness_result.score
        faithfulness_method = faithfulness_result.method

    return ScoredSample(
        sample_id=sample.sample_id,
        domain=sample.domain,
        document_status=sample.document_status,
        faithfulness_score=faithfulness_score,
        faithfulness_method=faithfulness_method,
        constraint_score=constraint_result.score,
        constraint_violations=constraint_result.violations,
    )


def _resolve_project_path(config_path: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return config_path.parent.parent / path


def _group_metrics(df: pd.DataFrame, *, bootstrap_samples: int, seed: int) -> dict:
    groups: dict[str, dict] = {}
    for name, group in df.groupby("document_status"):
        x = group["faithfulness_score"].tolist()
        y = group["constraint_score"].tolist()
        pearson = correlation_with_ci(x, y, n_bootstrap=bootstrap_samples, seed=seed)
        groups[str(name)] = {
            "n_samples": int(len(group)),
            "pearson_r": pearson.value,
            "pearson_bootstrap_ci": [pearson.ci_low, pearson.ci_high],
            "mean_faithfulness": float(group["faithfulness_score"].mean()),
            "mean_constraint_score": float(group["constraint_score"].mean()),
            "violation_rate": float((group["constraint_score"] < 1.0).mean()),
        }
    return groups


def _load_annotations(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    annotations = pd.read_csv(path)
    if annotations.empty:
        return None
    if "sample_id" not in annotations.columns:
        return None
    return annotations


def _merge_annotations(df: pd.DataFrame, annotation_path: Path) -> pd.DataFrame:
    annotations = _load_annotations(annotation_path)
    if annotations is None:
        return df

    keep_columns = [
        column
        for column in [
            "sample_id",
            "expert_constraint_score",
            "expert_violations",
            "expert_notes",
        ]
        if column in annotations.columns
    ]
    if keep_columns == ["sample_id"]:
        return df
    return df.merge(annotations[keep_columns], on="sample_id", how="left")


def _expert_agreement_metrics(df: pd.DataFrame) -> dict | None:
    if "expert_constraint_score" not in df.columns:
        return None
    labeled = df.dropna(subset=["expert_constraint_score"]).copy()
    if labeled.empty:
        return None

    labeled["expert_constraint_score"] = labeled["expert_constraint_score"].astype(float)
    labeled["auto_constraint_binary"] = (labeled["constraint_score"] >= 1.0).astype(float)
    agreement = (
        labeled["expert_constraint_score"] == labeled["auto_constraint_binary"]
    ).mean()

    return {
        "n_expert_labeled": int(len(labeled)),
        "auto_expert_agreement": float(agreement),
        "expert_mean_constraint_score": float(labeled["expert_constraint_score"].mean()),
    }


def _write_markdown_summary(path: Path, metrics: dict) -> None:
    pearson_ci = metrics["pearson_bootstrap_ci"]
    lines = [
        "# Experiment A 摘要",
        "",
        f"- 假设：{metrics['hypothesis']}",
        f"- 样本数：{metrics['n_samples']}",
        f"- Pearson r: {metrics['pearson_r']}",
        f"- Pearson bootstrap 95% 置信区间：[{pearson_ci[0]}, {pearson_ci[1]}]",
        f"- Spearman rho: {metrics['spearman_rho']}",
        f"- 平均 faithfulness：{metrics['mean_faithfulness']}",
        f"- 平均 constraint score：{metrics['mean_constraint_score']}",
        f"- 约束违反率：{metrics['violation_rate']}",
        "",
        f"- Faithfulness 方法：{metrics['faithfulness_method']}",
        "",
    ]
    if metrics.get("expert_agreement") is not None:
        agreement = metrics["expert_agreement"]
        lines.extend(
            [
                "## 专家一致性",
                "",
                f"- 专家标注样本数：{agreement['n_expert_labeled']}",
                f"- 自动评分/专家评分一致率：{agreement['auto_expert_agreement']}",
                f"- 专家平均 constraint score：{agreement['expert_mean_constraint_score']}",
                "",
            ]
        )
    lines.extend(["## 分组指标", ""])
    for group_name, group in metrics["by_document_status"].items():
        lines.extend(
            [
                f"### {group_name}",
                "",
                f"- 样本数：{group['n_samples']}",
                f"- Pearson r: {group['pearson_r']}",
                f"- Pearson bootstrap 95% 置信区间：{group['pearson_bootstrap_ci']}",
                f"- 平均 faithfulness：{group['mean_faithfulness']}",
                f"- 平均 constraint score：{group['mean_constraint_score']}",
                f"- 约束违反率：{group['violation_rate']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: str | Path = "configs/materials.yaml") -> dict:
    config_path = Path(config_path)
    config = read_yaml(config_path)
    exp_config = config["experiment_a"]
    input_path = _resolve_project_path(config_path, exp_config["input_path"])
    annotation_path = _resolve_project_path(config_path, exp_config["annotation_path"])
    output_dir = _resolve_project_path(config_path, exp_config["output_dir"])
    bootstrap_samples = int(exp_config["bootstrap_samples"])
    seed = int(exp_config["random_seed"])
    faithfulness_method = exp_config.get("faithfulness_method", "lexical_overlap")
    faithfulness_evaluator = build_faithfulness_evaluator(faithfulness_method)

    samples = load_samples(input_path)
    scored = [score_sample(sample, faithfulness_evaluator) for sample in samples]

    df = pd.DataFrame(
        {
            "sample_id": item.sample_id,
            "domain": item.domain,
            "document_status": item.document_status,
            "faithfulness_score": item.faithfulness_score,
            "faithfulness_method": item.faithfulness_method,
            "constraint_score": item.constraint_score,
            "constraint_violations": ";".join(item.constraint_violations),
        }
        for item in scored
    )
    df = _merge_annotations(df, annotation_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "scored_samples.csv", index=False)

    x = df["faithfulness_score"].tolist()
    y = df["constraint_score"].tolist()
    pearson = correlation_with_ci(x, y, n_bootstrap=bootstrap_samples, seed=seed)
    spearman = correlation_with_ci(x, y, statistic=spearman_rho, n_bootstrap=bootstrap_samples, seed=seed)
    metrics = {
        "experiment": exp_config["name"],
        "hypothesis": exp_config["hypothesis"],
        "n_samples": int(len(df)),
        "faithfulness_method": faithfulness_method,
        "pearson_r": pearson.value,
        "pearson_bootstrap_ci": [pearson.ci_low, pearson.ci_high],
        "spearman_rho": spearman.value,
        "spearman_bootstrap_ci": [spearman.ci_low, spearman.ci_high],
        "mean_faithfulness": float(df["faithfulness_score"].mean()),
        "mean_constraint_score": float(df["constraint_score"].mean()),
        "violation_rate": float((df["constraint_score"] < 1.0).mean()),
        "by_document_status": _group_metrics(df, bootstrap_samples=bootstrap_samples, seed=seed),
        "expert_agreement": _expert_agreement_metrics(df),
        "target_pearson_r_upper_bound": exp_config["target_pearson_r_upper_bound"],
        "falsification_pearson_r": exp_config["falsification_pearson_r"],
    }
    write_json(output_dir / "metrics.json", metrics)
    _write_markdown_summary(output_dir / "summary.md", metrics)
    return metrics


if __name__ == "__main__":
    print(run())
