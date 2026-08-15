from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from sret_materials_rag.evaluation.constraints import evaluate_material_qa_constraints


def _safe_mean(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.mean())


def analyze(input_path: Path, output_dir: Path) -> dict:
    df = pd.read_csv(input_path)
    recomputed = df.apply(
        lambda row: evaluate_material_qa_constraints(str(row["question"]), str(row["qwen_answer"])),
        axis=1,
    )
    df["original_constraint_score"] = df.get("constraint_score", pd.Series([None] * len(df)))
    df["constraint_score"] = recomputed.map(lambda result: result.score)
    df["constraint_violations"] = recomputed.map(lambda result: ";".join(result.violations))
    df["qwen_faithfulness_score"] = df["qwen_faithfulness_score"].astype(float)
    df["generation_tokens"] = df["generation_tokens"].astype(float)
    df["judge_tokens"] = df["judge_tokens"].astype(float)

    by_status: dict[str, dict] = {}
    for status, group in df.groupby("document_status"):
        by_status[str(status)] = {
            "n_samples": int(len(group)),
            "mean_qwen_faithfulness": _safe_mean(group["qwen_faithfulness_score"]),
            "mean_constraint_score": _safe_mean(group["constraint_score"]),
            "violation_rate": float((group["constraint_score"] < 1.0).mean()),
            "mean_generation_tokens": _safe_mean(group["generation_tokens"]),
            "mean_judge_tokens": _safe_mean(group["judge_tokens"]),
        }

    metrics = {
        "input": str(input_path),
        "n_samples": int(len(df)),
        "mean_qwen_faithfulness": _safe_mean(df["qwen_faithfulness_score"]),
        "mean_constraint_score": _safe_mean(df["constraint_score"]),
        "violation_rate": float((df["constraint_score"] < 1.0).mean()),
        "total_generation_tokens": int(df["generation_tokens"].sum()),
        "total_judge_tokens": int(df["judge_tokens"].sum()),
        "total_tokens": int(df["generation_tokens"].sum() + df["judge_tokens"].sum()),
        "by_document_status": by_status,
        "constraint_violation_distribution": (
            df["constraint_violations"].fillna("").replace("", "none").value_counts().to_dict()
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "rescored_outputs.csv", index=False)
    (output_dir / "analysis_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_summary(output_dir / "summary.md", metrics)
    return metrics


def _write_summary(path: Path, metrics: dict) -> None:
    lines = [
        "# Qwen-max 输出分析摘要",
        "",
        f"- 输入：`{metrics['input']}`",
        f"- 样本数：{metrics['n_samples']}",
        f"- 平均 Qwen faithfulness：{metrics['mean_qwen_faithfulness']:.4f}",
        f"- 平均 constraint score：{metrics['mean_constraint_score']:.4f}",
        f"- violation rate：{metrics['violation_rate']:.4f}",
        f"- generation tokens：{metrics['total_generation_tokens']}",
        f"- judge tokens：{metrics['total_judge_tokens']}",
        f"- total tokens：{metrics['total_tokens']}",
        "",
        "## 按 document_status 分组",
        "",
        "| status | n | faithfulness | constraint | violation_rate | gen_tokens | judge_tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for status, group in metrics["by_document_status"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    status,
                    str(group["n_samples"]),
                    f"{group['mean_qwen_faithfulness']:.4f}",
                    f"{group['mean_constraint_score']:.4f}",
                    f"{group['violation_rate']:.4f}",
                    f"{group['mean_generation_tokens']:.1f}",
                    f"{group['mean_judge_tokens']:.1f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## 约束违规分布", ""])
    for key, value in metrics["constraint_violation_distribution"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze qwen-max generation/judge outputs.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "data/llm_outputs/qwen_max_smoke_outputs.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results/qwen_max_smoke_analysis"),
    )
    args = parser.parse_args()
    metrics = analyze(Path(args.input), Path(args.output_dir))
    print("Qwen output analysis complete.")
    for key, value in metrics.items():
        if key != "by_document_status":
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
