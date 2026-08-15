from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    p_a_true = sum(a) / n
    p_b_true = sum(b) / n
    p_a_false = 1.0 - p_a_true
    p_b_false = 1.0 - p_b_true
    expected = p_a_true * p_b_true + p_a_false * p_b_false
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def as_bool_series(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: (
            bool(value)
            if isinstance(value, bool)
            else str(value).strip().lower() in {"1", "true", "yes", "y"}
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two LLM-as-Judge result files.")
    parser.add_argument("--judge-a", required=True)
    parser.add_argument("--judge-b", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    a = pd.read_csv(args.judge_a)
    b = pd.read_csv(args.judge_b)
    key = "response_id" if "response_id" in a.columns and "response_id" in b.columns else "sample_id"
    merged = a.merge(b, on=key, suffixes=("_a", "_b"))
    if len(merged) == 0:
        raise RuntimeError(f"No overlapping records on key {key}")
    judge_a = as_bool_series(merged["judge_violation_a"])
    judge_b = as_bool_series(merged["judge_violation_b"])
    a_values = judge_a.tolist()
    b_values = judge_b.tolist()
    merged["judge_agreement"] = judge_a == judge_b
    disagreements = merged[~merged["judge_agreement"]].copy()

    metrics = {
        "key": key,
        "n_overlap": int(len(merged)),
        "agreement_rate": float(merged["judge_agreement"].mean()),
        "cohen_kappa": float(cohen_kappa(a_values, b_values)),
        "judge_a_violation_rate": float(judge_a.mean()),
        "judge_b_violation_rate": float(judge_b.mean()),
        "n_disagreements": int(len(disagreements)),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "dual_judge_merged.csv", index=False)
    disagreements.to_csv(output_dir / "dual_judge_disagreements.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(output_dir / "summary.md", metrics, disagreements)
    print("Dual judge comparison complete.")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    return 0


def _write_summary(path: Path, metrics: dict, disagreements: pd.DataFrame) -> None:
    lines = [
        "# Dual LLM Judge Comparison",
        "",
        f"- key: `{metrics['key']}`",
        f"- n_overlap: {metrics['n_overlap']}",
        f"- agreement_rate: {metrics['agreement_rate']:.4f}",
        f"- cohen_kappa: {metrics['cohen_kappa']:.4f}",
        f"- judge_a_violation_rate: {metrics['judge_a_violation_rate']:.4f}",
        f"- judge_b_violation_rate: {metrics['judge_b_violation_rate']:.4f}",
        f"- n_disagreements: {metrics['n_disagreements']}",
        "",
        "## Disagreements",
        "",
        "| record | judge_a | judge_b |",
        "| --- | ---: | ---: |",
    ]
    for _, row in disagreements.head(30).iterrows():
        record = row.get(metrics["key"], "")
        lines.append(f"| {record} | {bool(row['judge_violation_a'])} | {bool(row['judge_violation_b'])} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
