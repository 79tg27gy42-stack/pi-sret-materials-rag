from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze PI-SRET vs LLM-as-Judge disagreements.")
    parser.add_argument("--scores", required=True, help="Path to llm_judge_scores.csv")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.scores).fillna("")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "agreement" not in df.columns:
        auto_violation = df["auto_constraint_score"].astype(float) < 1.0
        judge_violation = df["judge_violation"].astype(bool)
        df["agreement"] = ((auto_violation == judge_violation).astype(int))

    disagreements = df[df["agreement"].astype(int) == 0].copy()
    disagreements["disagreement_type"] = disagreements.apply(_classify_disagreement, axis=1)
    disagreements["error_family_note"] = disagreements.apply(_family_note, axis=1)
    disagreements.to_csv(output_dir / "disagreement_cases_classified.csv", index=False)

    by_family = _table(disagreements, "constraint_family", "n_disagreements")
    by_type = _table(disagreements, "disagreement_type", "n_disagreements")
    by_family.to_csv(output_dir / "disagreements_by_family.csv", index=False)
    by_type.to_csv(output_dir / "disagreements_by_type.csv", index=False)

    metrics = {
        "scores": args.scores,
        "n_samples": int(len(df)),
        "n_disagreements": int(len(disagreements)),
        "agreement_rate": float(df["agreement"].astype(int).mean()) if len(df) else 0.0,
        "disagreement_type_distribution": dict(Counter(disagreements["disagreement_type"])),
        "disagreement_family_distribution": dict(Counter(disagreements["constraint_family"])),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(output_dir / "summary.md", metrics, by_family, by_type)
    print("LLM judge disagreement analysis complete.")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


def _classify_disagreement(row: pd.Series) -> str:
    auto_violation = float(row.get("auto_constraint_score", 1.0)) < 1.0
    judge_violation = _as_bool(row.get("judge_violation", False))
    violations = str(row.get("auto_constraint_violations", ""))
    rationale = str(row.get("judge_rationale", "")).lower()
    family = str(row.get("constraint_family", ""))

    if auto_violation and not judge_violation:
        if family == "formation_energy" or "formation_energy" in violations:
            return "auto_stricter_formation_energy_boundary"
        if family == "pressure" or "pressure" in violations:
            return "auto_stricter_pressure_sign_convention"
        if "cautious_stability_claim" in violations:
            return "auto_stricter_stability_evidence_requirement"
        if "valid_chemical_symbols" in violations:
            return "auto_stricter_formula_or_symbol_parsing"
        return "auto_stricter_constraint_interpretation"
    if judge_violation and not auto_violation:
        if "unsupported" in rationale or "context" in rationale:
            return "judge_detects_context_support_issue"
        if family == "natural_claim":
            return "judge_detects_natural_claim_issue"
        return "judge_stricter_unmodeled_scientific_issue"
    return "other"


def _family_note(row: pd.Series) -> str:
    family = str(row.get("constraint_family", ""))
    if family == "formation_energy":
        return "Boundary-sensitive: elemental reference states, MP conventions, and energy-above-hull semantics require careful wording."
    if family == "pressure":
        return "Boundary-sensitive: negative values may indicate invalid pressure reporting or signed stress convention depending on context."
    if family == "stability":
        return "Evidence-sensitive: stable claims should cite thermodynamic, hull, relative-energy, or experimental stability evidence."
    if family == "natural_claim":
        return "Open-ended natural claims are less rule-closed and should be treated as source-reliability rather than property-only checks."
    if family == "superconductivity":
        return "Evidence-sensitive: superconductivity claims should include measured/reporting evidence or transition-temperature context."
    return ""


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _table(df: pd.DataFrame, field: str, count_name: str) -> pd.DataFrame:
    counts = Counter(df[field])
    total = sum(counts.values()) or 1
    return pd.DataFrame(
        [
            {field: key, count_name: value, "share": value / total}
            for key, value in sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))
        ]
    )


def _write_summary(path: Path, metrics: dict, by_family: pd.DataFrame, by_type: pd.DataFrame) -> None:
    lines = [
        "# LLM-as-Judge Disagreement Analysis",
        "",
        f"- scores: `{metrics['scores']}`",
        f"- n_samples: {metrics['n_samples']}",
        f"- agreement_rate: {metrics['agreement_rate']:.4f}",
        f"- n_disagreements: {metrics['n_disagreements']}",
        "",
        "## By Disagreement Type",
        "",
        "| Type | n | Share |",
        "| --- | ---: | ---: |",
    ]
    for _, row in by_type.iterrows():
        lines.append(f"| {row['disagreement_type']} | {int(row['n_disagreements'])} | {row['share']:.3f} |")
    lines.extend(["", "## By Constraint Family", "", "| Family | n | Share |", "| --- | ---: | ---: |"])
    for _, row in by_family.iterrows():
        lines.append(f"| {row['constraint_family']} | {int(row['n_disagreements'])} | {row['share']:.3f} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
