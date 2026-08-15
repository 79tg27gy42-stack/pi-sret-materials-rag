"""Create a stratified, prefilled annotation sheet for a Phi extraction audit."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sret_materials_rag.evaluation import constraints  # noqa: E402


DEFAULT_OUTPUT = ROOT / "data" / "annotations" / "extraction_audit_annotation.csv"
HUMAN = ROOT / "data" / "annotations" / "expert_review_adjudicated_v3.csv"
MANIFEST = ROOT / "data" / "processed" / "canonical_v3" / "full_response_manifest_v3.jsonl"
SEED = 42

PROPERTY_PATTERNS = {
    "band_gap": r"\bband\s*gap\b",
    "formation_energy": r"\bformation\s+energy\b",
    "stability": r"\b(?:stable|stability|energy above hull|convex hull)\b",
    "temperature": r"\b(?:temperature|kelvin|\bK\b)\b",
    "pressure": r"\b(?:pressure|GPa|MPa|\bPa\b)\b",
    "superconductivity": r"\bsuperconduct(?:ivity|ing|or)?\b",
    "conductivity": r"\b(?:metallic|metal|insulator|insulating|conductivity)\b",
}
PHASE_PATTERN = re.compile(
    r"\b(?:triclinic|monoclinic|orthorhombic|tetragonal|trigonal|rhombohedral|hexagonal|cubic|"
    r"space group(?:\s+number)?\s*(?:is|=|:)?\s*\d{1,3})\b",
    re.I,
)
METHOD_PATTERN = re.compile(
    r"\b(?:DFT|density functional theory|X-ray diffraction|XRD|first-principles|"
    r"molecular dynamics|calculation|measurement|experiment)\b",
    re.I,
)
EVIDENCE_PATTERN = re.compile(
    r"\b(?:formation energy|energy above hull|hull distance|convex hull|thermodynamic analysis|"
    r"transition temperature|superconducting transition|measured|measurement|resistivity|"
    r"zero resistance|magnetic susceptibility|Meissner)\b",
    re.I,
)
QUALIFIER_PATTERN = re.compile(
    r"\b(?:invalid|unreliable|outdated|incorrect|noisy|unsupported|not supported|"
    r"cannot be used|should not be used|metastable|allotrope|relative to|with respect to)\b",
    re.I,
)


def _join(values: list[str]) -> str:
    return "; ".join(dict.fromkeys(value.strip() for value in values if value and value.strip()))


def _predict(question: str, answer: str) -> dict[str, str]:
    formulae = [
        match.group(0)
        for match in constraints._FORMULA_RE.finditer(answer)
        if constraints._looks_like_formula_candidate(answer, match)
    ]
    values: list[str] = []
    units: list[str] = []
    for pattern in [
        constraints._BAND_GAP_RE,
        constraints._EV_BAND_GAP_RE,
        constraints._FORMATION_ENERGY_RE,
        constraints._TEMPERATURE_K_RE,
        constraints._PRESSURE_RE,
    ]:
        for match in pattern.finditer(answer):
            values.append(match.group(1))
            units.append(match.group(0)[match.end(1) - match.start() :].strip())
    properties = [name for name, pattern in PROPERTY_PATTERNS.items() if re.search(pattern, answer, re.I)]
    return {
        "predicted_material": _join(formulae),
        "predicted_formula": _join(formulae),
        "predicted_value": _join(values),
        "predicted_unit": _join(units),
        "predicted_property": _join(properties),
        "predicted_phase": _join([match.group(0) for match in PHASE_PATTERN.finditer(answer)]),
        "predicted_method": _join([match.group(0) for match in METHOD_PATTERN.finditer(answer)]),
        "predicted_evidence": _join([match.group(0) for match in EVIDENCE_PATTERN.finditer(answer)]),
        "predicted_qualifier": _join([match.group(0) for match in QUALIFIER_PATTERN.finditer(answer)]),
    }


def _read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _stratified_sample(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    groups = [(name, group.copy()) for name, group in frame.groupby("constraint_family", sort=True)]
    if n < len(groups):
        raise ValueError("Sample size must cover every evaluation stratum.")
    selected = []
    remaining = n
    for index, (_, group) in enumerate(groups):
        take = min(10, len(group), remaining - (len(groups) - index - 1))
        selected.append(group.sample(n=take, random_state=SEED + index))
        remaining -= take
    pools = []
    for (_, group), initial in zip(groups, selected):
        pools.append(group.loc[~group.index.isin(initial.index)])
    pool = pd.concat(pools).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    if remaining > len(pool):
        raise ValueError(f"Requested {n} records but only {n - remaining + len(pool)} are available.")
    selected.append(pool.head(remaining))
    return pd.concat(selected).sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def build(output: Path, n: int) -> pd.DataFrame:
    human = pd.read_csv(HUMAN)
    manifest = _read_jsonl(MANIFEST)[["response_id", "question", "answer"]]
    frame = human.merge(manifest, on="response_id", validate="one_to_one")
    if len(frame) != 200:
        raise ValueError(f"Expected 200 human-reference records, found {len(frame)}.")
    sample = _stratified_sample(frame, n)
    rows = []
    for row in sample.itertuples(index=False):
        prediction = _predict(str(row.question), str(row.answer))
        rows.append(
            {
                "record_id": row.response_id,
                "blind_id": getattr(row, "盲标编号"),
                "constraint_family": row.constraint_family,
                "document_status_raw": row.document_status,
                "question": row.question,
                "answer": row.answer,
                **prediction,
                "gold_material": "",
                "gold_formula": "",
                "gold_value": "",
                "gold_unit": "",
                "gold_property": "",
                "gold_phase": "",
                "gold_method": "",
                "gold_evidence": "",
                "gold_qualifier": "",
                "annotation_complete": "",
                "annotator_note": "",
            }
        )
    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()
    frame = build(args.output, args.n)
    print(json.dumps({"output": str(args.output), "n": len(frame), "by_stratum": frame["constraint_family"].value_counts().sort_index().to_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
