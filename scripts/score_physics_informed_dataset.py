from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sret_materials_rag.evaluation.physics_informed import (  # noqa: E402
    score_physics_informed_consistency,
)


def score_dataset(input_path: Path, output_dir: Path) -> dict:
    df = pd.read_csv(input_path)
    records = []
    for _, row in df.iterrows():
        score = score_physics_informed_consistency(
            question=str(row["question"]),
            answer=str(row["answer"]),
            retrieved_context=str(row["retrieved_context"]),
            document_status=str(row.get("document_status", "unknown")),
        )
        records.append(
            {
                "sample_id": row["sample_id"],
                "document_status": row.get("document_status", "unknown"),
                "physics_informed_score": score.total_score,
                "symbolic_consistency": score.symbolic_consistency,
                "physical_plausibility": score.physical_plausibility,
                "scientific_prior": score.scientific_prior,
                "uncertainty_calibration": score.uncertainty_calibration,
                "hallucination_type": score.hallucination_type.value,
                "physics_violations": ";".join(score.violations),
                "physics_rationale": score.rationale,
            }
        )
    scored = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_dir / "physics_informed_scores.csv", index=False)

    by_status = (
        scored.groupby("document_status")
        .agg(
            n=("sample_id", "count"),
            mean_pi_score=("physics_informed_score", "mean"),
            mean_symbolic=("symbolic_consistency", "mean"),
            mean_plausibility=("physical_plausibility", "mean"),
            mean_prior=("scientific_prior", "mean"),
            mean_uncertainty=("uncertainty_calibration", "mean"),
        )
        .reset_index()
    )
    by_status.to_csv(output_dir / "physics_informed_by_status.csv", index=False)
    taxonomy = scored["hallucination_type"].value_counts().to_dict()
    metrics = {
        "n_samples": int(len(scored)),
        "mean_physics_informed_score": float(scored["physics_informed_score"].mean()),
        "mean_symbolic_consistency": float(scored["symbolic_consistency"].mean()),
        "mean_physical_plausibility": float(scored["physical_plausibility"].mean()),
        "mean_scientific_prior": float(scored["scientific_prior"].mean()),
        "mean_uncertainty_calibration": float(scored["uncertainty_calibration"].mean()),
        "taxonomy_distribution": taxonomy,
        "by_document_status": by_status.to_dict(orient="records"),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(output_dir / "summary.md", metrics)
    return metrics


def _write_summary(path: Path, metrics: dict) -> None:
    lines = [
        "# PI-SRET Physics-Informed Scoring Summary",
        "",
        f"- 样本数：{metrics['n_samples']}",
        f"- mean PI score：{metrics['mean_physics_informed_score']:.4f}",
        f"- mean symbolic consistency：{metrics['mean_symbolic_consistency']:.4f}",
        f"- mean physical plausibility：{metrics['mean_physical_plausibility']:.4f}",
        f"- mean scientific prior：{metrics['mean_scientific_prior']:.4f}",
        f"- mean uncertainty calibration：{metrics['mean_uncertainty_calibration']:.4f}",
        "",
        "## Scientific Hallucination Taxonomy",
        "",
    ]
    for key, value in metrics["taxonomy_distribution"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## By Document Status",
            "",
            "| status | n | pi_score | symbolic | plausibility | prior | uncertainty |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in metrics["by_document_status"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["document_status"]),
                    str(int(row["n"])),
                    f"{row['mean_pi_score']:.4f}",
                    f"{row['mean_symbolic']:.4f}",
                    f"{row['mean_plausibility']:.4f}",
                    f"{row['mean_prior']:.4f}",
                    f"{row['mean_uncertainty']:.4f}",
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a dataset with the PI-SRET physics-informed evaluator.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "data/candidates/h1_large_with_faithfulness.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results/pi_sret_large"),
    )
    args = parser.parse_args()
    metrics = score_dataset(Path(args.input), Path(args.output_dir))
    print("PI-SRET scoring complete.")
    for key, value in metrics.items():
        if key != "by_document_status":
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
