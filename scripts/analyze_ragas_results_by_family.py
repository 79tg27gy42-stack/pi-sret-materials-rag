from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze real or proxy RAGAS results by constraint family.")
    parser.add_argument("--scores", required=True, help="Path to ragas_baseline_scores.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--faithfulness-threshold", type=float, default=0.8)
    args = parser.parse_args()

    scores = pd.read_csv(args.scores)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "constraint_family" not in scores.columns:
        raise ValueError("scores CSV must contain constraint_family")
    if "pi_sret_constraint_score" not in scores.columns:
        raise ValueError("scores CSV must contain pi_sret_constraint_score")

    faith_col = _column(scores, ["ragas_faithfulness", "faithfulness"])
    rel_col = _column(scores, ["ragas_answer_relevancy", "answer_relevancy"])
    rows = []
    high_faith_low_constraint = scores[
        (scores[faith_col] >= args.faithfulness_threshold) & (scores["pi_sret_constraint_score"] < 1.0)
    ].copy()

    for family, group in scores.groupby("constraint_family", dropna=False):
        risk = group[(group[faith_col] >= args.faithfulness_threshold) & (group["pi_sret_constraint_score"] < 1.0)]
        rows.append(
            {
                "constraint_family": family,
                "n": int(len(group)),
                "mean_ragas_faithfulness": float(group[faith_col].mean()),
                "mean_answer_relevancy": float(group[rel_col].mean()) if rel_col else None,
                "mean_pi_sret_constraint": float(group["pi_sret_constraint_score"].mean()),
                "pi_sret_violation_rate": float((group["pi_sret_constraint_score"] < 1.0).mean()),
                "high_faithfulness_low_constraint_cases": int(len(risk)),
            }
        )

    per_family = pd.DataFrame(rows).sort_values("constraint_family")
    per_family.to_csv(output_dir / "ragas_per_family_metrics.csv", index=False)
    high_faith_low_constraint.to_csv(output_dir / "ragas_high_faithfulness_low_constraint_cases.csv", index=False)

    metrics = {
        "scores": args.scores,
        "n_samples": int(len(scores)),
        "faithfulness_column": faith_col,
        "answer_relevancy_column": rel_col,
        "mean_ragas_faithfulness": float(scores[faith_col].mean()),
        "mean_answer_relevancy": float(scores[rel_col].mean()) if rel_col else None,
        "mean_pi_sret_constraint": float(scores["pi_sret_constraint_score"].mean()),
        "high_faithfulness_low_constraint_cases": int(len(high_faith_low_constraint)),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(output_dir / "summary.md", metrics, per_family)
    print("RAGAS by-family analysis complete.")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


def _column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _write_summary(path: Path, metrics: dict, per_family: pd.DataFrame) -> None:
    lines = [
        "# RAGAS by Constraint Family",
        "",
        f"- scores: `{metrics['scores']}`",
        f"- n_samples: {metrics['n_samples']}",
        f"- mean_ragas_faithfulness: {metrics['mean_ragas_faithfulness']:.4f}",
        f"- mean_answer_relevancy: {metrics['mean_answer_relevancy']:.4f}" if metrics["mean_answer_relevancy"] is not None else "- mean_answer_relevancy: n/a",
        f"- mean_pi_sret_constraint: {metrics['mean_pi_sret_constraint']:.4f}",
        f"- high_faithfulness_low_constraint_cases: {metrics['high_faithfulness_low_constraint_cases']}",
        "",
        "| Family | n | RAGAS faithfulness | Answer relevancy | PI-SRET constraint | Violation rate | Risk cases |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in per_family.iterrows():
        answer_rel = row["mean_answer_relevancy"]
        answer_rel_text = "n/a" if pd.isna(answer_rel) else f"{answer_rel:.3f}"
        lines.append(
            f"| {row['constraint_family']} | {int(row['n'])} | {row['mean_ragas_faithfulness']:.3f} | "
            f"{answer_rel_text} | {row['mean_pi_sret_constraint']:.3f} | "
            f"{row['pi_sret_violation_rate']:.3f} | {int(row['high_faithfulness_low_constraint_cases'])} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
