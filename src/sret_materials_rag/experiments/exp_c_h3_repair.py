from __future__ import annotations

from pathlib import Path

import pandas as pd

from sret_materials_rag.evaluation.constraints import (
    evaluate_material_constraints,
    evaluate_material_qa_constraints,
)
from sret_materials_rag.evaluation.faithfulness import lexical_faithfulness
from sret_materials_rag.utils.io import read_yaml, write_json


def _resolve_project_path(config_path: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return config_path.parent.parent / path


def _repair_for_violation(violation: str) -> str:
    repairs = {
        "non_negative_band_gap": (
            "The retrieved value should not be used as a physically valid band gap "
            "because a band gap cannot be negative; the valid band gap cannot be "
            "determined from this context."
        ),
        "non_negative_absolute_temperature": (
            "The retrieved temperature should not be used as physically valid because "
            "absolute temperature cannot be negative; a valid temperature cannot be "
            "determined from this context."
        ),
        "non_negative_pressure": (
            "The retrieved pressure should not be used as physically valid because "
            "pressure cannot be negative in this context; a valid pressure cannot be "
            "determined from this context."
        ),
        "valid_chemical_symbols": (
            "The retrieved formula is not a valid chemical formula, so a valid "
            "composition cannot be determined from this context."
        ),
        "cautious_stability_claim": (
            "The retrieved text is insufficient to establish thermodynamic stability "
            "because it lacks formation energy or energy-above-hull evidence."
        ),
        "cautious_superconductivity_claim": (
            "The material should not be called superconducting without reported Tc, "
            "transition temperature, or measured evidence."
        ),
    }
    return repairs.get(
        violation,
        "The retrieved claim should not be treated as validated without additional domain evidence.",
    )


def repair_answer(answer: str, question: str = "") -> tuple[str, str]:
    before = (
        evaluate_material_qa_constraints(question, answer)
        if question
        else evaluate_material_constraints(answer)
    )
    if not before.violations:
        return answer, "none"

    repaired_parts = [_repair_for_violation(violation) for violation in before.violations]
    return " ".join(repaired_parts), "+".join(before.violations)


def _mean(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.mean())


def answer_usefulness_proxy(question: str, answer: str) -> float:
    """Rule-based proxy for whether the answer still addresses the asked property.

    This is intentionally conservative and does not claim semantic quality. It
    only catches degenerate repairs that remove the target property entirely.
    """
    lower_question = question.lower()
    lower_answer = answer.lower()
    property_markers = {
        "band gap": ["band gap", "gap", "cannot be determined"],
        "stable": ["stable", "stability", "thermodynamic", "cannot be determined", "insufficient"],
        "stability": ["stable", "stability", "thermodynamic", "cannot be determined", "insufficient"],
        "temperature": ["temperature", "absolute temperature", "cannot be determined"],
        " kelvin": ["temperature", "absolute temperature", "cannot be determined"],
        "pressure": ["pressure", "stress", "cannot be determined"],
        "formula": ["formula", "composition", "chemical formula", "not valid"],
        "superconduct": ["superconduct", "tc", "transition temperature", "evidence"],
    }
    for question_marker, answer_markers in property_markers.items():
        if question_marker in lower_question:
            return 1.0 if any(marker in lower_answer for marker in answer_markers) else 0.0
    return 1.0 if answer.strip() else 0.0


def run(config_path: str | Path = "configs/materials_h3.yaml") -> dict:
    config_path = Path(config_path)
    config = read_yaml(config_path)
    exp_config = config["experiment_c"]
    input_path = _resolve_project_path(config_path, exp_config["input_path"])
    output_dir = _resolve_project_path(config_path, exp_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    before_results = df.apply(
        lambda row: evaluate_material_qa_constraints(str(row["question"]), str(row["answer"])),
        axis=1,
    )
    df["constraint_score_before"] = before_results.map(lambda result: result.score)
    df["constraint_violations_before"] = before_results.map(lambda result: ";".join(result.violations))

    repairs = df.apply(
        lambda row: repair_answer(str(row["answer"]), str(row["question"])),
        axis=1,
    )
    df["repaired_answer"] = repairs.map(lambda item: item[0])
    df["repair_strategy"] = repairs.map(lambda item: item[1])

    after_results = df.apply(
        lambda row: evaluate_material_qa_constraints(str(row["question"]), str(row["repaired_answer"])),
        axis=1,
    )
    df["constraint_score_after"] = after_results.map(lambda result: result.score)
    df["constraint_violations_after"] = after_results.map(lambda result: ";".join(result.violations))

    df["lexical_alignment_before"] = df.apply(
        lambda row: lexical_faithfulness(str(row["answer"]), str(row["retrieved_context"])),
        axis=1,
    )
    df["lexical_alignment_after"] = df.apply(
        lambda row: lexical_faithfulness(str(row["repaired_answer"]), str(row["retrieved_context"])),
        axis=1,
    )
    df["lexical_alignment_delta"] = df["lexical_alignment_after"] - df["lexical_alignment_before"]
    df["answer_usefulness_before"] = df.apply(
        lambda row: answer_usefulness_proxy(str(row["question"]), str(row["answer"])),
        axis=1,
    )
    df["answer_usefulness_after"] = df.apply(
        lambda row: answer_usefulness_proxy(str(row["question"]), str(row["repaired_answer"])),
        axis=1,
    )
    df["answer_usefulness_delta"] = df["answer_usefulness_after"] - df["answer_usefulness_before"]

    repaired_df = df[df["repair_strategy"] != "none"].copy()
    successful_repairs = repaired_df[
        (repaired_df["constraint_score_before"] < 1.0)
        & (repaired_df["constraint_score_after"] == 1.0)
    ]

    by_status: dict[str, dict[str, float | int]] = {}
    for status, group in df.groupby("document_status"):
        status_repaired = group[group["repair_strategy"] != "none"]
        by_status[str(status)] = {
            "n_samples": int(len(group)),
            "n_repaired": int(len(status_repaired)),
            "mean_constraint_before": _mean(group["constraint_score_before"]),
            "mean_constraint_after": _mean(group["constraint_score_after"]),
            "repair_success_rate": (
                float((status_repaired["constraint_score_after"] == 1.0).mean())
                if len(status_repaired)
                else None
            ),
            "mean_lexical_alignment_delta": _mean(group["lexical_alignment_delta"]),
            "mean_answer_usefulness_after": _mean(group["answer_usefulness_after"]),
            "mean_answer_usefulness_delta": _mean(group["answer_usefulness_delta"]),
        }

    repair_success_rate = (
        float(len(successful_repairs) / len(repaired_df)) if len(repaired_df) else None
    )
    metrics = {
        "experiment": exp_config["name"],
        "hypothesis": exp_config["hypothesis"],
        "n_samples": int(len(df)),
        "n_repaired": int(len(repaired_df)),
        "repair_success_rate": repair_success_rate,
        "mean_constraint_before": _mean(df["constraint_score_before"]),
        "mean_constraint_after": _mean(df["constraint_score_after"]),
        "mean_constraint_gain": _mean(df["constraint_score_after"])
        - _mean(df["constraint_score_before"]),
        "mean_lexical_alignment_before": _mean(df["lexical_alignment_before"]),
        "mean_lexical_alignment_after": _mean(df["lexical_alignment_after"]),
        "mean_lexical_alignment_delta": _mean(df["lexical_alignment_delta"]),
        "mean_answer_usefulness_before": _mean(df["answer_usefulness_before"]),
        "mean_answer_usefulness_after": _mean(df["answer_usefulness_after"]),
        "mean_answer_usefulness_delta": _mean(df["answer_usefulness_delta"]),
        "repair_strategy_distribution": df["repair_strategy"].value_counts().to_dict(),
        "by_document_status": by_status,
        "scope_note": (
            "Pilot repair uses rule-based minimal safety rewrites over constructed answers. "
            "It tests constraint repair achievability, not final answer correctness."
        ),
    }

    df.to_csv(output_dir / "repair_results.csv", index=False)
    write_json(output_dir / "metrics.json", metrics)
    _write_summary(output_dir / "summary.md", metrics)
    return metrics


def _write_summary(path: Path, metrics: dict) -> None:
    def fmt(value: object) -> str:
        return "NA" if value is None else f"{float(value):.4f}"

    lines = [
        "# Experiment C / H3 最小修复 Pilot 摘要",
        "",
        f"- 样本数：{metrics['n_samples']}",
        f"- 触发修复样本数：{metrics['n_repaired']}",
        f"- 修复成功率：{fmt(metrics['repair_success_rate'])}",
        f"- 修复前平均约束分：{metrics['mean_constraint_before']:.4f}",
        f"- 修复后平均约束分：{metrics['mean_constraint_after']:.4f}",
        f"- 平均约束增益：{metrics['mean_constraint_gain']:.4f}",
        f"- 修复前 lexical alignment：{metrics['mean_lexical_alignment_before']:.4f}",
        f"- 修复后 lexical alignment：{metrics['mean_lexical_alignment_after']:.4f}",
        f"- lexical alignment 变化：{metrics['mean_lexical_alignment_delta']:.4f}",
        f"- 修复前 answer usefulness proxy：{metrics['mean_answer_usefulness_before']:.4f}",
        f"- 修复后 answer usefulness proxy：{metrics['mean_answer_usefulness_after']:.4f}",
        f"- answer usefulness proxy 变化：{metrics['mean_answer_usefulness_delta']:.4f}",
        "",
        "## 按文档状态分组",
        "",
        "| status | n | repaired | constraint_before | constraint_after | success_rate | lexical_delta | usefulness_after | usefulness_delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for status, group in metrics["by_document_status"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    status,
                    str(group["n_samples"]),
                    str(group["n_repaired"]),
                    f"{group['mean_constraint_before']:.4f}",
                    f"{group['mean_constraint_after']:.4f}",
                    fmt(group["repair_success_rate"]),
                    f"{group['mean_lexical_alignment_delta']:.4f}",
                    f"{group['mean_answer_usefulness_after']:.4f}",
                    f"{group['mean_answer_usefulness_delta']:.4f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## 限制", "", metrics["scope_note"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    print(run())
