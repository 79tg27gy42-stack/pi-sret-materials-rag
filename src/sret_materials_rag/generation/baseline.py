from __future__ import annotations

import re


_FORMULA_RE = re.compile(r"\b(?:[A-Z][a-z]?\d*){1,}\b")
_BAND_GAP_PATTERNS = [
    re.compile(r"Band gap:\s*(-?\d+(?:\.\d+)?)\s*eV", re.I),
    re.compile(r"band gap\s+(-?\d+(?:\.\d+)?)\s*eV", re.I),
    re.compile(r"band gap(?:\s+value)?\s+of\s+(-?\d+(?:\.\d+)?)\s*eV", re.I),
    re.compile(r"with band gap\s+(-?\d+(?:\.\d+)?)\s*eV", re.I),
]


def _first_formula(text: str) -> str:
    for match in _FORMULA_RE.finditer(text):
        formula = match.group(0)
        if formula not in {"What", "Is", "Can", "The", "A", "Materials", "Project"}:
            return formula
    return "the material"


def _extract_band_gap(context: str) -> str | None:
    for pattern in _BAND_GAP_PATTERNS:
        match = pattern.search(context)
        if match:
            return match.group(1)
    return None


def _extract_after(label: str, context: str) -> str | None:
    pattern = re.compile(rf"{re.escape(label)}:\s*([^;]+?)(?:\. |$)", re.I)
    match = pattern.search(context)
    if match:
        return match.group(1).strip()
    return None


def generate_answer(question: str, contexts: list[str]) -> str:
    if not contexts:
        return "Insufficient retrieved evidence to answer the question."

    context = contexts[0]
    lower_question = question.lower()
    formula = _first_formula(question)

    if "band gap" in lower_question:
        band_gap = _extract_band_gap(context)
        if band_gap is None:
            return "The retrieved context does not provide a band gap value."
        return f"The retrieved context reports a {band_gap} eV band gap for {formula}."

    if "thermodynamically stable" in lower_question or "stable" in lower_question:
        hull = _extract_after("Energy above hull", context)
        formation = _extract_after("Formation energy per atom", context)
        stable = _extract_after("Stable", context)
        if hull is not None or formation is not None:
            pieces = []
            if hull is not None:
                pieces.append(f"energy above hull {hull}")
            if formation is not None:
                pieces.append(f"formation energy {formation}")
            if stable is not None:
                pieces.append(f"stable flag {stable}")
            return (
                f"For {formula}, the retrieved context gives "
                + ", ".join(pieces)
                + ", so any stability claim should be interpreted thermodynamically."
            )
        if "stable" in context.lower():
            return f"The retrieved context presents {formula} as stable."
        return "The retrieved context is insufficient to establish thermodynamic stability."

    if "formula" in lower_question or "valid" in lower_question:
        return f"The retrieved context treats {_first_formula(context)} as a valid material formula."

    if "temperature" in lower_question or " k" in lower_question:
        match = re.search(r"-?\d+(?:\.\d+)?\s*(?:K|Kelvin)\b", context, re.I)
        if match:
            return f"The retrieved context reports an experimental temperature of {match.group(0)}."
        return "The retrieved context does not provide a temperature."

    if "pressure" in lower_question or "gpa" in lower_question:
        match = re.search(r"-?\d+(?:\.\d+)?\s*(?:GPa|MPa|Pa)\b", context, re.I)
        if match:
            return f"The retrieved context reports a pressure of {match.group(0)}."
        return "The retrieved context does not provide a pressure."

    if "superconduct" in lower_question:
        if "superconduct" in context.lower():
            return f"The retrieved context identifies {formula} as superconducting."
        return "The retrieved context does not provide superconductivity evidence."

    return context
