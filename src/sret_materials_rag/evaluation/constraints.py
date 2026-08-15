from __future__ import annotations

import re
from dataclasses import dataclass


VALID_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu",
}

_BAND_GAP_RE = re.compile(r"band gap\b[^.\n;:]*?(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)\s*eV", re.I)
_EV_BAND_GAP_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*eV\s+band gap", re.I)
_FORMATION_ENERGY_RE = re.compile(
    r"formation energy\b[^.\n;:]*?(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)\s*eV(?:/(?:atom|f\.?u\.?)| per atom)?",
    re.I,
)
_TEMPERATURE_K_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:K|Kelvin)\b", re.I)
_PRESSURE_RE = re.compile(r"(?:pressure(?:\s+is|\s+of|\s*[:=])?\s*)?(-?\d+(?:\.\d+)?)\s*(?:GPa|MPa|Pa)\b", re.I)
_STANDALONE_EV_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*eV\b", re.I)
_STANDALONE_FORMATION_ENERGY_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*eV(?:/(?:atom|f\.?u\.?)| per atom)\b",
    re.I,
)
_STANDALONE_NUMBER_WITH_UNIT_RE = {
    "band_gap": _STANDALONE_EV_RE,
    "formation_energy": _STANDALONE_FORMATION_ENERGY_RE,
    "temperature": _TEMPERATURE_K_RE,
    "pressure": _PRESSURE_RE,
}
_FORMULA_RE = re.compile(r"\b(?:[A-Z][a-z]?\d*){2,}\b")
_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
_SPACE_GROUP_RE = re.compile(r"(?:space group|sg|space-group)\s*(?:number\s*)?(?:is|=|:)?\s*(\d{1,3})", re.I)
_SINGLE_ELEMENT_QUESTION_RE = re.compile(r"\b(?:of|for|is)\s+([A-Z][a-z]?)\??$", re.I)
_UNIT_OR_ACRONYM_TOKENS = {
    "GPa",
    "MPa",
    "Pa",
    "KPa",
    "kPa",
    "PT",
    "KPT",
    "DFT",
    "RAG",
    "RAGAS",
}

_CRYSTAL_SYSTEM_RANGES = {
    "triclinic": range(1, 3),
    "monoclinic": range(3, 16),
    "orthorhombic": range(16, 75),
    "tetragonal": range(75, 143),
    "trigonal": range(143, 168),
    "rhombohedral": {146, 148, 155, 160, 161, 166, 167},
    "hexagonal": range(168, 195),
    "cubic": range(195, 231),
}


@dataclass(frozen=True)
class ConstraintResult:
    score: float
    violations: list[str]


@dataclass(frozen=True)
class ConstraintThresholds:
    band_gap_upper_ev: float = 15.0
    formation_energy_lower_ev_atom: float = -4.0
    formation_energy_upper_ev_atom: float = 1.0
    conductivity_gap_ev: float = 0.1
    elemental_reference_tolerance_ev_atom: float = 0.05


DEFAULT_THRESHOLDS = ConstraintThresholds()


def _formula_is_valid(formula: str) -> bool:
    if formula in _UNIT_OR_ACRONYM_TOKENS:
        return True
    parts = _ELEMENT_RE.findall(formula)
    reconstructed = "".join(element + count for element, count in parts)
    if reconstructed != formula:
        return False
    return all(element == "M" or element in VALID_ELEMENTS for element, _ in parts)


def _looks_like_formula_candidate(answer: str, match: re.Match[str]) -> bool:
    formula = match.group(0)
    if formula in _UNIT_OR_ACRONYM_TOKENS:
        return False
    if formula.isupper() and not any(char.isdigit() for char in formula):
        return False
    if any(char.isdigit() or char.islower() for char in formula):
        return True
    window = answer[max(0, match.start() - 40) : min(len(answer), match.end() + 40)].lower()
    return bool(
        re.search(
            r"\b(?:compound|formula|material|phase|composition|oxide|hydride|carbide|nitride)\b",
            window,
        )
    )


def _single_element_named_in_question(question: str) -> str | None:
    match = _SINGLE_ELEMENT_QUESTION_RE.search(question.strip())
    if not match:
        return None
    element = match.group(1)
    return element if element in VALID_ELEMENTS else None


def _claim_scope(answer: str, match: re.Match[str] | None = None) -> str:
    if match is None:
        return answer
    start, end = match.span()
    boundaries = [
        boundary.start()
        for boundary in re.finditer(r"(?<!\d)[.!?](?!\d)|\n", answer)
    ]
    left = max((index for index in boundaries if index < start), default=-1) + 1
    right_boundary = min((index for index in boundaries if index >= end), default=len(answer) - 1)
    right = min(len(answer), right_boundary + 1)
    return answer[left:right]


def _claim_clause(answer: str, match: re.Match[str]) -> str:
    """Return the local clause containing a matched assertion.

    Sentence scope is too broad for exceptions: an unrelated ``unreliable`` or
    ``without`` later in the same sentence must not cancel an invalid value.
    Coordinating/contrast conjunctions are therefore treated as soft clause
    boundaries in addition to punctuation.
    """
    sentence = _claim_scope(answer, match)
    sentence_start = answer.find(sentence)
    relative_start = max(0, match.start() - sentence_start)
    boundaries = list(
        re.finditer(
            r"[;:]|\b(?:but|however|whereas|while|although|though|yet)\b|,\s+(?:and|but)\b",
            sentence,
            re.I,
        )
    )
    left = max((item.end() for item in boundaries if item.end() <= relative_start), default=0)
    right = min((item.start() for item in boundaries if item.start() >= relative_start), default=len(sentence))
    return sentence[left:right]


def _rejects_invalid_claim(answer: str, match: re.Match[str] | None = None) -> bool:
    lower_answer = (answer if match is None else _claim_clause(answer, match)).lower()
    markers = [
        "not valid",
        "not a valid",
        "cannot be negative",
        "cannot be below",
        "cannot be determined",
        "cannot determine",
        "cannot provide",
        "cannot be used",
        "cannot be physically",
        "does not provide a reliable",
        "insufficient",
        "unreliable",
        "outdated",
        "incorrect",
        "noisy",
        "human review",
        "no verified",
        "not trustworthy",
        "should not",
        "is not valid",
        "is invalid",
        "not physically valid",
        "not supported",
        "unsupported",
    ]
    if any(marker in lower_answer for marker in markers):
        return True
    if match is None:
        return False

    # A following contrast clause may reject the matched value, but only when
    # it explicitly points back to that assertion. This admits "that value is
    # unreliable" while rejecting unrelated text such as "without catalyst".
    sentence = _claim_scope(answer, match)
    sentence_start = answer.find(sentence)
    tail = sentence[max(0, match.end() - sentence_start) :].lower()
    linked_tail = re.search(
        r"\b(?:but|however|yet)\b[^.]{0,100}\b"
        r"(?:that|this|those|these|the reported|the|all)\s+"
        r"(?:value|values|entry|entries|claim|claims|information|data|source|document|report)\b[^.]{0,80}",
        tail,
    )
    return bool(linked_tail and any(marker in linked_tail.group(0) for marker in markers))


def _question_quantity(question: str) -> str | None:
    lower = question.lower()
    patterns = {
        "formation_energy": ("formation energy", "energy of formation"),
        "band_gap": ("band gap", "bandgap"),
        "temperature": ("temperature", "kelvin"),
        "pressure": ("pressure",),
    }
    for quantity, markers in patterns.items():
        if any(marker in lower for marker in markers):
            return quantity
    return None


def _has_pressure_sign_convention(answer: str, match: re.Match[str]) -> bool:
    scope = _claim_scope(answer, match).lower()
    markers = [
        "stress convention",
        "sign convention",
        "tensile loading",
        "tensile stress",
        "under tension",
        "hydrostatic tension",
    ]
    return any(marker in scope for marker in markers)


def _has_stability_evidence(answer: str) -> bool:
    lower_answer = answer.lower()
    markers = [
        "formation energy",
        "energy above hull",
        "above hull",
        "hull distance",
        "convex hull",
        "thermodynamic analysis",
        "thermodynamically",
        "relative stability",
        "stable relative to",
    ]
    return any(marker in lower_answer for marker in markers)


def _is_nonthermodynamic_stability_claim(answer: str, match: re.Match[str]) -> bool:
    scope = _claim_scope(answer, match).lower()
    return bool(
        re.search(
            r"\b(?:dynamically|mechanically|kinetically|electrochemically|operationally)\s+stable\b|"
            r"\b(?:dynamic|mechanical|kinetic|thermal)\s+stability\b|\bstable\s+in\s+air\b",
            scope,
        )
    )


def _has_superconductivity_evidence(answer: str, match: re.Match[str]) -> bool:
    scope = _claim_clause(answer, match).lower()
    scope = re.sub(
        r"\b(?:without|no|lacks?|lacking|insufficient)\b[^.;]{0,30}\b(?:evidence|measurement|transition|tc)\b",
        "",
        scope,
    )
    evidence_markers = [
        "tc",
        "transition temperature",
        "superconducting transition",
        "measured",
        "measurement",
        "experimentally observed",
        "resistivity",
        "resistance transition",
        "zero resistance",
        "magnetic susceptibility",
        "meissner",
        "diamagnetic",
        "experimental evidence",
        "evidence for superconductivity",
    ]
    return any(marker in scope for marker in evidence_markers)


def _has_elemental_phase_qualification(answer: str, match: re.Match[str]) -> bool:
    lower_answer = _claim_clause(answer, match).lower()
    markers = [
        "allotrope",
        "metastable",
        "specific phase",
        "this phase",
        "relative to",
        "with respect to",
        "referenced to",
        "above the ground state",
        "above ground state",
        "nonstandard reference",
        "non-standard reference",
    ]
    return any(marker in lower_answer for marker in markers)


def _negates_stability(answer: str) -> bool:
    return bool(
        re.search(
            r"\b(?:not|isn't|is not|are not|unstable|not\s+thermodynamically\s+stable|not\s+stable)\b",
            answer.lower(),
        )
    )


def evaluate_material_constraints(
    answer: str,
    *,
    thresholds: ConstraintThresholds = DEFAULT_THRESHOLDS,
) -> ConstraintResult:
    violations: list[str] = []

    band_gap_matches = list(_BAND_GAP_RE.finditer(answer)) + list(_EV_BAND_GAP_RE.finditer(answer))
    for match in band_gap_matches:
        band_gap = float(match.group(1))
        if band_gap < 0 and not _rejects_invalid_claim(answer, match):
            violations.append("non_negative_band_gap")
        if band_gap > thresholds.band_gap_upper_ev and not _rejects_invalid_claim(answer, match):
            violations.append("band_gap_physical_range")

    formation_energy_matches = list(_FORMATION_ENERGY_RE.finditer(answer))
    formation_energies = [float(match.group(1)) for match in formation_energy_matches]
    for match, formation_energy in zip(formation_energy_matches, formation_energies):
        if (
            formation_energy < thresholds.formation_energy_lower_ev_atom
            or formation_energy > thresholds.formation_energy_upper_ev_atom
        ) and not _rejects_invalid_claim(answer, match):
            violations.append("formation_energy_typical_range")

    for match in _TEMPERATURE_K_RE.finditer(answer):
        temperature = float(match.group(1))
        if temperature < 0 and not _rejects_invalid_claim(answer, match):
            violations.append("non_negative_absolute_temperature")

    for match in _PRESSURE_RE.finditer(answer):
        pressure = float(match.group(1))
        if pressure < 0 and not _rejects_invalid_claim(answer, match) and not _has_pressure_sign_convention(answer, match):
            violations.append("non_negative_pressure")

    for match in _FORMULA_RE.finditer(answer):
        formula = match.group(0)
        if (
            _looks_like_formula_candidate(answer, match)
            and not _formula_is_valid(formula)
            and not _rejects_invalid_claim(answer, match)
        ):
            violations.append("valid_chemical_symbols")

    lower_answer = answer.lower()
    stability_assertion = re.search(
        r"\b(?:is|are|was|were|be|considered|appears)\s+(?:very\s+|highly\s+|one\s+of\s+the\s+most\s+)?stable\b|"
        r"\b(?:more|most|highly|thermodynamically)\s+stable\b|"
        r"\bstable\s+(?:phase|compound|material|structure)\b|"
        r"\b(?:shows|show|exhibits|exhibit|demonstrates|demonstrate)\b[^.]{0,60}\bstability\b|"
        r"\bstability\s+of\b[^.]{0,80}\b(?:observed|verified)\b|"
        r"\bstability\s+(?:is|was|has been|was confirmed|has been observed|has been verified)\b|"
        r"\blong-term stability\b",
        lower_answer,
    )
    if (
        stability_assertion
        and not _negates_stability(answer)
        and not _is_nonthermodynamic_stability_claim(answer, stability_assertion)
    ):
        if not _has_stability_evidence(answer) and not _rejects_invalid_claim(answer, stability_assertion):
            violations.append("cautious_stability_claim")
        positive_uncaveated_energy = any(
            energy > 0 and not _rejects_invalid_claim(answer, match)
            for match, energy in zip(formation_energy_matches, formation_energies)
        )
        if (
            positive_uncaveated_energy
            and not _rejects_invalid_claim(answer, stability_assertion)
            and not any(
                _has_elemental_phase_qualification(answer, match)
                for match in formation_energy_matches
                if float(match.group(1)) > 0
            )
        ):
            violations.append("positive_formation_energy_stability_conflict")

    if re.search(r"\b(?:metallic|metal)\b", lower_answer):
        for match in band_gap_matches:
            if float(match.group(1)) > thresholds.conductivity_gap_ev and not _rejects_invalid_claim(answer, match):
                violations.append("conductivity_band_gap_consistency")
    if re.search(r"\b(?:insulator|insulating)\b", lower_answer):
        for match in band_gap_matches:
            if float(match.group(1)) < thresholds.conductivity_gap_ev and not _rejects_invalid_claim(answer, match):
                violations.append("conductivity_band_gap_consistency")

    for system, valid_range in _CRYSTAL_SYSTEM_RANGES.items():
        if system in lower_answer:
            for match in _SPACE_GROUP_RE.finditer(answer):
                space_group = int(match.group(1))
                if 1 <= space_group <= 230 and space_group not in valid_range and not _rejects_invalid_claim(answer, match):
                    violations.append("crystal_system_space_group_consistency")

    superconductivity_assertion = re.search(
        r"\b(?:is|are|was|were|be)\s+(?:a\s+)?superconduct(?:or|ing)\b|"
        r"\b(?:believed\s+to\s+)?superconducts?\b|"
        r"\bsuperconduct(?:ing)?\s+(?:behavior|phase)\b|"
        r"\bsuperconductivity\s+(?:was\s+observed|occurs|makes)\b|"
        r"\bnew\s+family\s+of\s+superconductors\b|"
        r"\bpromising\s+superconductor\b",
        lower_answer,
    )
    if superconductivity_assertion and not _rejects_invalid_claim(answer, superconductivity_assertion):
        if not _has_superconductivity_evidence(answer, superconductivity_assertion):
            violations.append("cautious_superconductivity_claim")

    score = 1.0 if not violations else 0.0
    return ConstraintResult(score=score, violations=sorted(set(violations)))


def evaluate_material_qa_constraints(
    question: str,
    answer: str,
    *,
    thresholds: ConstraintThresholds = DEFAULT_THRESHOLDS,
) -> ConstraintResult:
    """Evaluate answer constraints with question context for terse answers.

    LLMs often answer a band-gap question with only "-0.1 eV". The standalone
    answer does not contain the words "band gap", so the answer-only checker
    cannot infer the physical quantity. This wrapper adds that inference only
    when the question explicitly names the quantity.
    """
    result = evaluate_material_constraints(answer, thresholds=thresholds)
    violations = set(result.violations)
    quantity = _question_quantity(question)
    matches = list(_STANDALONE_NUMBER_WITH_UNIT_RE.get(quantity, re.compile(r"(?!x)x")).finditer(answer))

    if quantity == "band_gap":
        for match in matches:
            value = float(match.group(1))
            if value < 0 and not _rejects_invalid_claim(answer, match):
                violations.add("non_negative_band_gap")
            if value > thresholds.band_gap_upper_ev and not _rejects_invalid_claim(answer, match):
                violations.add("band_gap_physical_range")

    if quantity == "temperature":
        for match in matches:
            if float(match.group(1)) < 0 and not _rejects_invalid_claim(answer, match):
                violations.add("non_negative_absolute_temperature")

    if quantity == "pressure":
        for match in matches:
            if (
                float(match.group(1)) < 0
                and not _rejects_invalid_claim(answer, match)
                and not _has_pressure_sign_convention(answer, match)
            ):
                violations.add("non_negative_pressure")

    if quantity == "formation_energy":
        for match in matches:
            formation_energy = float(match.group(1))
            if (
                (
                    formation_energy < thresholds.formation_energy_lower_ev_atom
                    or formation_energy > thresholds.formation_energy_upper_ev_atom
                )
                and not _rejects_invalid_claim(answer, match)
            ):
                violations.add("formation_energy_typical_range")

        element = _single_element_named_in_question(question)
        if element:
            for match in matches:
                formation_energy = float(match.group(1))
                if (
                    formation_energy > thresholds.elemental_reference_tolerance_ev_atom
                    and not _has_elemental_phase_qualification(answer, match)
                    and not _rejects_invalid_claim(answer, match)
                ):
                    violations.add("elemental_formation_energy_reference_conflict")

    return ConstraintResult(score=1.0 if not violations else 0.0, violations=sorted(violations))
