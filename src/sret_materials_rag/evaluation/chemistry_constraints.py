from __future__ import annotations

import re
from dataclasses import dataclass

from sret_materials_rag.evaluation.constraints import _formula_is_valid


_FORMULA_RE = re.compile(r"\b(?:[A-Z][a-z]?\d*){2,}\b")
_TEMP_K_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:K|Kelvin)\b", re.I)
_PH_RE = re.compile(r"\bpH\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)\b", re.I)
_YIELD_RE = re.compile(r"\b(?:yield|conversion)\b[^.\n;:]*?(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)\s*%", re.I)
_PRESSURE_RE = re.compile(r"(?:pressure(?:\s+is|\s+of|\s*[:=])?\s*)?(-?\d+(?:\.\d+)?)\s*(?:atm|bar|MPa|Pa)\b", re.I)


@dataclass(frozen=True)
class ChemistryConstraintResult:
    score: float
    violations: list[str]


def evaluate_chemistry_constraints(answer: str) -> ChemistryConstraintResult:
    lower = answer.lower()
    rejects = any(marker in lower for marker in ["invalid", "not valid", "cannot", "unreliable", "should not"])
    violations: list[str] = []
    for formula in _FORMULA_RE.findall(answer):
        if not _formula_is_valid(formula) and not rejects:
            violations.append("valid_molecular_formula")
    for match in _TEMP_K_RE.finditer(answer):
        if float(match.group(1)) < 0 and not rejects:
            violations.append("non_negative_absolute_temperature")
    for match in _PH_RE.finditer(answer):
        value = float(match.group(1))
        if (value < 0 or value > 14) and not rejects:
            violations.append("aqueous_ph_range")
    for match in _YIELD_RE.finditer(answer):
        value = float(match.group(1))
        if (value < 0 or value > 100) and not rejects:
            violations.append("yield_percentage_range")
    for match in _PRESSURE_RE.finditer(answer):
        if float(match.group(1)) < 0 and not rejects:
            violations.append("non_negative_pressure")
    return ChemistryConstraintResult(score=1.0 if not violations else 0.0, violations=sorted(set(violations)))
