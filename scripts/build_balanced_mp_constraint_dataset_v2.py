from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _context(row: pd.Series) -> str:
    return (
        f"Materials Project reports {row['formula_pretty']} ({row['material_id']}). "
        f"Band gap: {float(row['band_gap'])} eV. "
        f"Formation energy per atom: {float(row['formation_energy_per_atom'])} eV/atom. "
        f"Energy above hull: {float(row['energy_above_hull'])} eV/atom. "
        f"Stable: {bool(row['is_stable'])}."
    )


def _record(row: pd.Series, index: int, suffix: str, question: str, context: str, answer: str, status: str, constraint: str, level: str) -> dict:
    material_id = str(row["material_id"])
    return {
        "sample_id": f"mp_bal_v2_{index:04d}_{suffix}",
        "domain": "materials",
        "source": f"https://materialsproject.org/materials/{material_id}",
        "source_id": material_id,
        "source_url": f"https://materialsproject.org/materials/{material_id}",
        "formula": str(row["formula_pretty"]),
        "question": question,
        "retrieved_context": context,
        "answer": answer,
        "document_status": status,
        "expected_constraints": constraint,
        "constraint_target": constraint,
        "constraint_level": level,
        "faithfulness_score": 1.0 if answer in context or status == "current" else "",
        "faithfulness_method": "mp_property_derived_or_controlled",
        "data_note": "MP-derived controlled diagnostic record from real Materials Project property row.",
    }


def build(mp_input: Path, output: Path, limit_materials: int) -> dict:
    df = pd.read_json(mp_input, lines=True).head(limit_materials).fillna(0)
    rows: list[dict] = []
    for index, row in df.iterrows():
        i = index + 1
        formula = str(row["formula_pretty"])
        band_gap = float(row["band_gap"])
        formation = float(row["formation_energy_per_atom"])
        hull = float(row["energy_above_hull"])
        stable = bool(row["is_stable"])
        true_context = _context(row)
        negative_gap = -max(abs(band_gap), 0.05)
        high_gap = max(abs(band_gap), 16.5)
        noisy_gap_context = (
            f"A controlled noisy note derived from {row['material_id']} reports {formula} with band gap {negative_gap} eV. "
            "This note is marked outdated_or_incorrect for stress testing."
        )
        incomplete_stability_context = (
            f"A short snippet derived from {row['material_id']} says {formula} is stable, "
            "but omits formation energy, hull distance, and thermodynamic qualification."
        )
        noisy_formation_context = (
            f"A controlled noisy note derived from {row['material_id']} reports formation energy 5.5 eV/atom for {formula}. "
            "This note is marked outdated_or_incorrect for stress testing."
        )

        rows.extend(
            [
                _record(
                    row,
                    i,
                    "band_gap_current",
                    f"What is the band gap of {formula}?",
                    true_context,
                    f"The band gap of {formula} is {band_gap} eV according to Materials Project.",
                    "current",
                    "non_negative_band_gap",
                    "L0",
                ),
                _record(
                    row,
                    i,
                    "band_gap_negative",
                    f"What is the band gap of {formula}?",
                    noisy_gap_context,
                    f"The band gap of {formula} is {negative_gap} eV.",
                    "outdated_or_incorrect",
                    "non_negative_band_gap",
                    "L0",
                ),
                _record(
                    row,
                    i,
                    "band_gap_high",
                    f"What is the band gap of {formula}?",
                    f"A controlled noisy note derived from {row['material_id']} reports {formula} with band gap {high_gap} eV.",
                    f"The band gap of {formula} is {high_gap} eV.",
                    "outdated_or_incorrect",
                    "band_gap_physical_range",
                    "L0",
                ),
                _record(
                    row,
                    i,
                    "stability_current",
                    f"Is {formula} thermodynamically stable?",
                    true_context,
                    (
                        f"{formula} has energy above hull {hull} eV/atom and formation energy "
                        f"{formation} eV/atom; Materials Project stable is {stable}."
                    ),
                    "current",
                    "cautious_stability_claim",
                    "L1",
                ),
                _record(
                    row,
                    i,
                    "stability_incomplete",
                    f"Is {formula} thermodynamically stable?",
                    incomplete_stability_context,
                    f"{formula} is stable.",
                    "incomplete",
                    "cautious_stability_claim",
                    "L1",
                ),
                _record(
                    row,
                    i,
                    "formation_current",
                    f"What is the formation energy of {formula}?",
                    true_context,
                    f"The formation energy of {formula} is {formation} eV/atom.",
                    "current",
                    "formation_energy_typical_range",
                    "L0",
                ),
                _record(
                    row,
                    i,
                    "formation_noisy_high",
                    f"What is the formation energy of {formula}?",
                    noisy_formation_context,
                    f"The formation energy of {formula} is 5.5 eV/atom.",
                    "outdated_or_incorrect",
                    "formation_energy_typical_range",
                    "L0",
                ),
                _record(
                    row,
                    i,
                    "conductivity_gap_conflict",
                    f"Is {formula} metallic or insulating based on its band gap?",
                    true_context,
                    (
                        f"{formula} is metallic with a band gap of {max(band_gap, 1.0)} eV."
                        if band_gap >= 0.1
                        else f"{formula} is an insulator with a band gap of {band_gap} eV."
                    ),
                    "outdated_or_incorrect",
                    "conductivity_band_gap_consistency",
                    "L1",
                ),
            ]
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "n_materials": int(len(df)),
        "n_records": len(rows),
        "by_constraint_target": pd.Series([r["constraint_target"] for r in rows]).value_counts().to_dict(),
        "by_document_status": pd.Series([r["document_status"] for r in rows]).value_counts().to_dict(),
        "output": str(output),
    }
    (output.parent / "mp_balanced_constraint_dataset_v2.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MP-derived balanced constraint diagnostic records.")
    parser.add_argument("--mp-input", default=str(ROOT / "data/sources/materials_project_candidates.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data/processed/mp_balanced_constraint_dataset_v2.jsonl"))
    parser.add_argument("--limit-materials", type=int, default=300)
    args = parser.parse_args()
    summary = build(Path(args.mp_input), Path(args.output), args.limit_materials)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
