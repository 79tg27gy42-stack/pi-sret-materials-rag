"""Constraint system validation tests.

For each of the 6 material science constraints, provides positive examples
(should pass) and negative examples (should trigger violation), then verifies
that evaluate_material_constraints() matches expectations.
"""
from __future__ import annotations

import pytest

from sret_materials_rag.evaluation.constraints import (
    evaluate_material_constraints,
    evaluate_material_qa_constraints,
)


# ---------------------------------------------------------------------------
# Constraint 1: non_negative_band_gap
# ---------------------------------------------------------------------------

BAND_GAP_POSITIVE = [
    "The band gap of Si is 1.12 eV.",
    "TiO2 has a band gap of 3.2 eV, making it a wide-gap semiconductor.",
    "The computed band gap is 0.5 eV for this material.",
    "GaAs exhibits a direct band gap of 1.42 eV.",
    "The band gap: 0.0 eV, indicating metallic behavior.",
    "A band gap of 4.7 eV was reported for the oxide.",
    "The material shows a band gap is 2.1 eV under ambient conditions.",
    "ZnO has band gap = 3.37 eV.",
    "Perovskite band gap is 1.55 eV, suitable for solar cells.",
    "The indirect band gap is 0.67 eV at room temperature.",
]

BAND_GAP_NEGATIVE = [
    "The band gap of this material is -1.5 eV.",
    "A computed band gap = -0.3 eV was observed.",
    "The band gap is -2.7 eV, which is unphysical.",
    "Band gap: -4.1 eV according to the calculation.",
    "The material exhibits a band gap of -0.01 eV.",
    "The direct band gap is -1.0 eV.",
    "An unrealistic band gap of -3.5 eV was found.",
    "The band gap: -0.8 eV suggests an error in the data.",
    "This hypothetical compound has band gap = -2.0 eV.",
    "The reported band gap is -5.2 eV.",
]


# ---------------------------------------------------------------------------
# Constraint 2: valid_chemical_symbols
# ---------------------------------------------------------------------------

CHEMICAL_POSITIVE = [
    "SiO2 is a common oxide used in semiconductors.",
    "The compound BaTiO3 exhibits ferroelectric behavior.",
    "LiFePO4 is widely used in lithium-ion batteries.",
    "NaCl forms a simple ionic crystal structure.",
    "YBa2Cu3O7 is a well-known high-temperature superconductor.",
    "The stability of CaCO3 depends on temperature and pressure.",
    "CuInSe2 is a promising photovoltaic absorber material.",
    "Fe2O3 and Fe3O4 are common iron oxides.",
    "MgAl2O4 spinel is a refractory ceramic material.",
    "The perovskite CH3NH3PbI3 has attracted much attention.",
]

CHEMICAL_NEGATIVE = [
    "The compound XxYy contains invalid symbols.",
    "ZzO2 is not a real material.",
    "QqFe2 is an impossible formula.",
    "The stability of AbCd needs further study.",
    "Xx2O3 forms a hypothetical crystal structure.",
    "The compound WwVv shows unusual properties.",
    "JjPb3 is not found in nature.",
    "OoCu2 cannot be synthesized.",
    "The formula NnSi contains a non-existent element.",
    "UuFe2O3 is an invalid compound notation.",
]


# ---------------------------------------------------------------------------
# Constraint 3: cautious_stability_claim
# ---------------------------------------------------------------------------

STABILITY_POSITIVE = [
    "SiO2 is thermodynamically stable with a formation energy of -2.5 eV/atom.",
    "The material is stable relative to competing phases (hull distance 0 meV/atom).",
    "Its thermodynamic stability is confirmed by a formation energy of -1.2 eV/atom.",
    "The compound is on the convex hull, indicating stability.",
    "With a hull distance of 0.01 eV, the phase is nearly stable.",
    "The formation energy suggests stability under standard conditions.",
    "This perovskite is stable relative to the binary oxides on the hull.",
    "The stability of Fe2O3 is confirmed by thermodynamic analysis.",
    "Al2O3 is thermodynamically the most stable aluminum oxide.",
    "Phase stability was assessed using the hull distance metric.",
]

STABILITY_NEGATIVE = [
    "SiO2 is stable.",  # no qualifier
    "This material is very stable.",  # no qualifier
    "The compound shows excellent stability.",  # no qualifier
    "It is considered stable for practical applications.",  # no qualifier
    "Stability of the phase has been observed experimentally.",  # no qualifier
    "The structure is stable under ambient conditions.",  # no qualifier
    "This is one of the most stable perovskites known.",  # no qualifier
    "The material demonstrates remarkable stability.",  # no qualifier
    "Long-term stability has been verified.",  # no qualifier
    "The phase exhibits good stability in air.",  # no qualifier
]


# ---------------------------------------------------------------------------
# Constraint 4: non_negative_absolute_temperature
# ---------------------------------------------------------------------------

TEMPERATURE_POSITIVE = [
    "The transition occurs at 300 K.",
    "Measurements were performed at 4.2 K.",
    "The critical temperature is 93 Kelvin for YBCO.",
    "At 298 K, the material shows metallic conductivity.",
    "The experiment was conducted at 0 K in the limit.",
    "Superconductivity was observed below 77 K.",
    "The phase transition happens at 1050 K.",
    "Thermal expansion was measured from 100 K to 500 K.",
    "The Debye temperature is 345 K.",
    "The material melts at 1687 K under standard pressure.",
]

TEMPERATURE_NEGATIVE = [
    "The transition occurs at -50 K.",  # impossible
    "A negative temperature of -10 K was reported.",
    "The measurement was done at -273.15 K.",
    "The critical temperature is -100 Kelvin.",
    "The experiment at -5 K produced unusual results.",
    "Superconductivity was claimed at -200 K.",
    "The phase appears below -1 K.",
    "The measurement at -0.001 K is questionable.",
    "An anomalous signal at -300 K was detected.",
    "The material behaves differently at -40 K.",
]


# ---------------------------------------------------------------------------
# Constraint 5: non_negative_pressure
# ---------------------------------------------------------------------------

PRESSURE_POSITIVE = [
    "The phase transition occurs at 5 GPa.",
    "Synthesis was performed under 10 GPa of pressure.",
    "The material is stable at a pressure of 0.1 GPa.",
    "Pressure: 50 GPa was applied using a diamond anvil cell.",
    "At 2.5 MPa, the structure remains unchanged.",
    "The compressibility was measured up to 100 Pa.",
    "A pressure of 25 GPa induced metallization.",
    "The equation of state was fit to 30 GPa data.",
    "Hydrostatic pressure of 8 GPa was applied.",
    "The material was compressed to 15 GPa.",
]

PRESSURE_NEGATIVE = [
    "The transition occurs at -5 GPa.",
    "A pressure of -10 MPa was applied to the sample.",
    "The material is stable at -0.5 GPa.",
    "Pressure: -2 GPa caused a phase change.",
    "At -100 Pa, the structure collapsed.",
    "Negative pressure of -3 GPa expanded the lattice.",
    "The measurement at -1.5 GPa was inconclusive.",
    "A compressive pressure of -8 GPa was applied.",
    "The phase boundary extends to -20 GPa.",
    "Under -0.01 GPa, the volume increased.",
]


# ---------------------------------------------------------------------------
# Constraint 6: cautious_superconductivity_claim
# ---------------------------------------------------------------------------

SUPERCONDUCTIVITY_POSITIVE = [
    "The material superconducts below a Tc of 93 K.",
    "Superconductivity was measured at a transition temperature of 39 K.",
    "Evidence for superconductivity was reported in MgB2 at 39 K.",
    "The material is reported to superconduct with Tc = 4.2 K.",
    "Superconducting behavior was measured at 77 K in YBCO.",
    "The superconducting transition was reported at 135 K under pressure.",
    "Superconductivity in this compound has been measured by resistivity.",
    "The reported superconducting Tc is 15 K.",
    "Superconducting properties were experimentally observed by Smith et al.",
    "The material exhibits superconductivity with a measured Tc of 26 K.",
]

SUPERCONDUCTIVITY_NEGATIVE = [
    "The material is a superconductor.",  # no Tc or evidence
    "Superconductivity was observed in this compound.",  # no qualifier
    "This compound superconducts under high pressure.",  # no Tc
    "The material shows superconducting behavior.",  # no qualifier
    "It is believed to superconduct at room temperature.",  # no evidence
    "Superconductivity makes this material interesting.",  # no qualifier
    "The superconducting phase is stable.",  # no Tc
    "A new family of superconductors has been discovered.",  # no qualifier
    "This is a promising superconductor for applications.",  # no evidence
    "Superconductivity occurs in the doped sample.",  # no Tc
]


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def _check_positive(text: str, constraint_id: str) -> None:
    """Assert that a positive example does NOT trigger the given constraint."""
    result = evaluate_material_constraints(text)
    assert constraint_id not in result.violations, (
        f"False positive: '{text}' triggered {constraint_id} but should not have. "
        f"All violations: {result.violations}"
    )


def _check_negative(text: str, constraint_id: str) -> None:
    """Assert that a negative example DOES trigger the given constraint."""
    result = evaluate_material_constraints(text)
    assert constraint_id in result.violations, (
        f"False negative: '{text}' did NOT trigger {constraint_id} but should have. "
        f"All violations: {result.violations}"
    )


@pytest.mark.parametrize("text", BAND_GAP_POSITIVE)
def test_band_gap_positive(text):
    _check_positive(text, "non_negative_band_gap")


@pytest.mark.parametrize("text", BAND_GAP_NEGATIVE)
def test_band_gap_negative(text):
    _check_negative(text, "non_negative_band_gap")


@pytest.mark.parametrize("text", CHEMICAL_POSITIVE)
def test_chemical_symbols_positive(text):
    _check_positive(text, "valid_chemical_symbols")


@pytest.mark.parametrize("text", CHEMICAL_NEGATIVE)
def test_chemical_symbols_negative(text):
    _check_negative(text, "valid_chemical_symbols")


@pytest.mark.parametrize("text", STABILITY_POSITIVE)
def test_stability_positive(text):
    _check_positive(text, "cautious_stability_claim")


@pytest.mark.parametrize("text", STABILITY_NEGATIVE)
def test_stability_negative(text):
    _check_negative(text, "cautious_stability_claim")


@pytest.mark.parametrize("text", TEMPERATURE_POSITIVE)
def test_temperature_positive(text):
    _check_positive(text, "non_negative_absolute_temperature")


@pytest.mark.parametrize("text", TEMPERATURE_NEGATIVE)
def test_temperature_negative(text):
    _check_negative(text, "non_negative_absolute_temperature")


@pytest.mark.parametrize("text", PRESSURE_POSITIVE)
def test_pressure_positive(text):
    _check_positive(text, "non_negative_pressure")


@pytest.mark.parametrize("text", PRESSURE_NEGATIVE)
def test_pressure_negative(text):
    _check_negative(text, "non_negative_pressure")


@pytest.mark.parametrize("text", SUPERCONDUCTIVITY_POSITIVE)
def test_superconductivity_positive(text):
    _check_positive(text, "cautious_superconductivity_claim")


@pytest.mark.parametrize("text", SUPERCONDUCTIVITY_NEGATIVE)
def test_superconductivity_negative(text):
    _check_negative(text, "cautious_superconductivity_claim")


def test_no_text_no_violations():
    """Text with no constraint-relevant content should have no violations."""
    result = evaluate_material_constraints(
        "The crystal structure was determined by X-ray diffraction."
    )
    assert result.score == 1.0
    assert result.violations == []


def test_multiple_violations_in_one_answer():
    """An answer can trigger multiple constraints simultaneously."""
    result = evaluate_material_constraints(
        "The compound XxYy has a band gap of -1.5 eV and is stable."
    )
    assert "non_negative_band_gap" in result.violations
    assert "cautious_stability_claim" in result.violations
    assert result.score == 0.0


def test_deeper_band_gap_upper_bound():
    result = evaluate_material_constraints("The band gap of LiF is 21.0 eV.")
    assert "band_gap_physical_range" in result.violations


def test_formation_energy_typical_range():
    result = evaluate_material_constraints("The formation energy is -8.2 eV/atom.")
    assert "formation_energy_typical_range" in result.violations


def test_positive_formation_energy_stability_conflict():
    result = evaluate_material_constraints(
        "The material is thermodynamically stable with a formation energy of 0.6 eV/atom."
    )
    assert "positive_formation_energy_stability_conflict" in result.violations


def test_positive_formation_energy_not_stable_passes():
    result = evaluate_material_constraints(
        "The material is not thermodynamically stable because the formation energy is 0.07 eV/atom."
    )
    assert "positive_formation_energy_stability_conflict" not in result.violations


def test_conductivity_band_gap_consistency_metallic():
    result = evaluate_material_constraints("The compound is metallic with a band gap of 1.2 eV.")
    assert "conductivity_band_gap_consistency" in result.violations


def test_conductivity_band_gap_consistency_insulator():
    result = evaluate_material_constraints("The compound is an insulator with a band gap of 0.02 eV.")
    assert "conductivity_band_gap_consistency" in result.violations


def test_crystal_system_space_group_consistency():
    result = evaluate_material_constraints("The phase is cubic with space group number 62.")
    assert "crystal_system_space_group_consistency" in result.violations


def test_crystal_system_space_group_consistency_valid():
    result = evaluate_material_constraints("The phase is orthorhombic with space group number 62.")
    assert "crystal_system_space_group_consistency" not in result.violations


def test_constraint_score_is_binary():
    """Score should be 1.0 (no violations) or 0.0 (any violation)."""
    for text in BAND_GAP_POSITIVE[:3]:
        result = evaluate_material_constraints(text)
        assert result.score in (0.0, 1.0)
    for text in BAND_GAP_NEGATIVE[:3]:
        result = evaluate_material_constraints(text)
        assert result.score in (0.0, 1.0)


def test_question_aware_negative_band_gap_short_answer():
    result = evaluate_material_qa_constraints(
        "What is the band gap of C?",
        "-0.079 eV",
    )
    assert "non_negative_band_gap" in result.violations
    assert result.score == 0.0


def test_question_aware_negative_band_gap_rejection_passes():
    result = evaluate_material_qa_constraints(
        "What is the band gap of C?",
        "The retrieved value should not be used because a band gap cannot be negative.",
    )
    assert result.violations == []
    assert result.score == 1.0


def test_question_aware_unreliable_negative_band_gap_mention_passes():
    result = evaluate_material_qa_constraints(
        "What is the band gap of O2?",
        (
            "The retrieved context reports band gap values of -1.2773 eV and -1.0933 eV, "
            "but all entries are marked outdated_or_incorrect and the information is "
            "unreliable and insufficient for determining the actual band gap."
        ),
    )
    assert result.violations == []
    assert result.score == 1.0


def test_unstable_phase_is_not_positive_stability_claim():
    result = evaluate_material_qa_constraints(
        "What is the band gap of Te?",
        (
            "Materials Project reports 0.0 eV for an unstable phase. "
            "Due to inconsistent data, I cannot provide a reliable band gap value."
        ),
    )
    assert result.violations == []
    assert result.score == 1.0


def test_missing_stable_phase_information_is_not_stability_assertion():
    result = evaluate_material_qa_constraints(
        "What is the band gap of Te?",
        (
            "Due to inconsistent data and lack of verified stable-phase information, "
            "I cannot provide a reliable band gap value."
        ),
    )
    assert result.violations == []
    assert result.score == 1.0


def test_unrelated_caveat_does_not_cancel_negative_band_gap():
    result = evaluate_material_constraints(
        "The band gap is -0.4 eV, but the synthesis proceeds without a catalyst."
    )
    assert "non_negative_band_gap" in result.violations


def test_caveat_is_bound_to_the_invalid_band_gap_claim():
    result = evaluate_material_constraints(
        "The source reports a band gap of -0.4 eV, but that value is unreliable and should not be used."
    )
    assert "non_negative_band_gap" not in result.violations


def test_generic_source_disclaimer_does_not_suppress_independent_claims():
    result = evaluate_material_constraints(
        "Source X is unreliable. Material A has a band gap of -1.2 eV. "
        "Material B has a temperature of -10 K."
    )
    assert "non_negative_band_gap" in result.violations
    assert "non_negative_absolute_temperature" in result.violations


def test_superconductivity_report_without_measurement_is_not_evidence():
    result = evaluate_material_constraints(
        "The material is reported by Smith et al. to be a superconductor."
    )
    assert "cautious_superconductivity_claim" in result.violations


def test_unreliable_superconductivity_source_is_a_scoped_caveat():
    result = evaluate_material_constraints(
        "The material is reported to be superconducting, but the source is unreliable."
    )
    assert "cautious_superconductivity_claim" not in result.violations


def test_superconductivity_measurement_evidence_passes():
    result = evaluate_material_constraints(
        "The material is superconducting, with zero resistance and a Meissner response below 18 K."
    )
    assert "cautious_superconductivity_claim" not in result.violations


def test_narrow_gap_semiconductor_is_not_a_hard_violation():
    result = evaluate_material_constraints(
        "The material is a narrow-gap semiconductor with a band gap of 0.04 eV."
    )
    assert "conductivity_band_gap_consistency" not in result.violations


def test_question_quantity_resolver_checks_terse_formation_energy():
    result = evaluate_material_qa_constraints(
        "What is the formation energy of MgO?",
        "1.2 eV/atom",
    )
    assert "formation_energy_typical_range" in result.violations


def test_elemental_metastable_allotrope_uses_phase_qualification():
    result = evaluate_material_qa_constraints(
        "What is the formation energy of C?",
        "This metastable allotrope is 0.20 eV/atom above the ground state.",
    )
    assert "elemental_formation_energy_reference_conflict" not in result.violations


def test_unrelated_allotrope_text_does_not_qualify_elemental_reference_claim():
    result = evaluate_material_qa_constraints(
        "What is the formation energy of C?",
        "The formation energy is 0.20 eV/atom. Another material has a metastable allotrope.",
    )
    assert "elemental_formation_energy_reference_conflict" in result.violations


def test_rhombohedral_setting_rejects_non_rhombohedral_trigonal_group():
    result = evaluate_material_constraints(
        "The phase is rhombohedral with space group number 143."
    )
    assert "crystal_system_space_group_consistency" in result.violations


def test_generic_m_formula_does_not_allow_other_invalid_symbols():
    result = evaluate_material_constraints("The generic compound formula MXx2 is proposed.")
    assert "valid_chemical_symbols" in result.violations


@pytest.mark.parametrize(
    ("text", "expected_violation"),
    [
        (
            "The reported band gap is -1.2 eV, but this value is invalid.",
            False,
        ),
        (
            "The band gap is -1.2 eV, but that value is invalid. "
            "The temperature is -5 K, but that value is invalid.",
            False,
        ),
        (
            "Source X is unreliable. Material A has a band gap of -1.2 eV. "
            "Material B has a temperature of -10 K.",
            True,
        ),
        (
            "Material A has a band gap of 1.2 eV. Material B has a band gap of -1.2 eV.",
            True,
        ),
        (
            "If the band gap were -1.2 eV, the reported value would be invalid.",
            True,
        ),
        (
            "Previous work reported a band gap of -1.2 eV; this quoted value is unreliable.",
            True,
        ),
        (
            "If Material A had a band gap of -1.2 eV, it would be invalid.",
            True,
        ),
        (
            "Previous work reported a band gap of -1.2 eV, but the corrected value is 1.2 eV "
            "and the earlier claim is invalid.",
            True,
        ),
    ],
)
def test_complex_exception_discourse_regression_profile(text, expected_violation):
    """Document supported scoped exceptions and known conditional/quotation limits."""
    result = evaluate_material_constraints(text)
    assert ("non_negative_band_gap" in result.violations) is expected_violation
