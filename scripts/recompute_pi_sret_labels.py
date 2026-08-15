from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sret_materials_rag.evaluation.constraints import evaluate_material_qa_constraints  # noqa: E402


RAGAS_DIR = ROOT / "results" / "ragas_real_canonical_v3_372_qwen_plus_complete_optimized_rules"
RAW_JUDGE_DIR = ROOT / "results" / "llm_judge_canonical_v3_full_qwen_plus"
OPT_JUDGE_DIR = ROOT / "results" / "llm_judge_canonical_v3_full_qwen_plus_optimized_rules"


def _labels(df: pd.DataFrame) -> tuple[list[float], list[str]]:
    results = [
        evaluate_material_qa_constraints(str(row.question), str(row.answer))
        for row in df.itertuples(index=False)
    ]
    return [result.score for result in results], [";".join(result.violations) for result in results]


def _as_bool(series: pd.Series) -> pd.Series:
    return series.map(lambda value: value if isinstance(value, bool) else str(value).lower() in {"1", "true", "yes"})


def _kappa(left: pd.Series, right: pd.Series) -> float | None:
    observed = float((left == right).mean())
    expected = float(left.mean() * right.mean() + (1 - left.mean()) * (1 - right.mean()))
    return None if expected == 1 else (observed - expected) / (1 - expected)


def recompute() -> dict[str, int]:
    ragas_path = RAGAS_DIR / "ragas_baseline_scores.csv"
    ragas = pd.read_csv(ragas_path)
    ragas["pi_sret_constraint_score"], ragas["pi_sret_violations"] = _labels(ragas)
    ragas.to_csv(ragas_path, index=False)
    divergence = ragas[
        (ragas["ragas_faithfulness"].astype(float) >= 0.8)
        & (ragas["pi_sret_constraint_score"].astype(float) < 1.0)
    ].copy()
    divergence.to_csv(RAGAS_DIR / "ragas_high_faithfulness_pi_sret_low_cases.csv", index=False)
    ragas_metrics_path = RAGAS_DIR / "metrics.json"
    ragas_metrics = json.loads(ragas_metrics_path.read_text(encoding="utf-8"))
    ragas_metrics.update(
        {
            "mean_pi_sret_constraint": float(ragas["pi_sret_constraint_score"].mean()),
            "high_faithfulness_low_constraint_cases": int(len(divergence)),
            "labels_recomputed_from_cached_answers": True,
        }
    )
    ragas_metrics_path.write_text(json.dumps(ragas_metrics, indent=2), encoding="utf-8")

    judge = pd.read_csv(RAW_JUDGE_DIR / "llm_judge_scores.csv")
    scores, violations = _labels(judge)
    judge["auto_constraint_score_optimized"] = scores
    judge["auto_constraint_violations_optimized"] = violations
    OPT_JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    judge.to_csv(OPT_JUDGE_DIR / "llm_judge_scores_with_optimized_auto.csv", index=False)

    auto_violation = judge["auto_constraint_score_optimized"].astype(float) < 1.0
    judge_violation = _as_bool(judge["judge_violation"])
    agreement = auto_violation == judge_violation
    disagreements = judge[~agreement].copy()
    disagreements.to_csv(OPT_JUDGE_DIR / "disagreements_optimized_auto.csv", index=False)
    judge_metrics = {
        "n_samples": int(len(judge)),
        "optimized_auto_violation_rate": float(auto_violation.mean()),
        "judge_violation_rate": float(judge_violation.mean()),
        "optimized_agreement_rate": float(agreement.mean()),
        "optimized_cohen_kappa_vs_judge": _kappa(auto_violation, judge_violation),
        "optimized_disagreements": int((~agreement).sum()),
        "labels_recomputed_from_cached_answers": True,
        "llm_judge_scores_reused_without_api_calls": True,
    }
    (OPT_JUDGE_DIR / "metrics.json").write_text(
        json.dumps(judge_metrics, indent=2), encoding="utf-8"
    )
    return {"ragas_records": len(ragas), "ragas_divergence": len(divergence), "judge_records": len(judge)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute PI-SRET labels while preserving cached RAGAS and LLM scores.")
    parser.parse_args()
    print(json.dumps(recompute(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
