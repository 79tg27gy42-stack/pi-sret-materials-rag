"""Generate supplementary dataset to cover all 11 constraint types.

This script generates controlled Q&A pairs that trigger each of the 11 constraints
(7 L0 + 4 L1) across all 3 document_status conditions. It uses DashScope qwen-plus
to generate diverse answer phrasings, then validates each answer against the constraint
evaluator.

Usage:
    export OPENAI_API_KEY="sk-..."
    export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
    python scripts/generate_supplementary_dataset.py [--limit-per-type 15] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sret_materials_rag.evaluation.constraints import evaluate_material_qa_constraints

# ---------------------------------------------------------------------------
# Constraint definitions: each entry describes HOW to trigger a violation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConstraintSpec:
    constraint_id: str
    level: str  # L0 or L1
    question_templates: list[str]
    violation_context_templates: list[str]
    clean_context_templates: list[str]
    description: str


CONSTRAINTS: list[ConstraintSpec] = [
    # ── L0: shallow / single-value checks ──────────────────────────────────
    ConstraintSpec(
        constraint_id="non_negative_band_gap",
        level="L0",
        question_templates=[
            "What is the band gap of {material}?",
            "Can you report the band gap value for {material}?",
            "What band gap does {material} have according to calculations?",
            "Provide the computed band gap of {material}.",
            "What is the electronic band gap of {material}?",
        ],
        violation_context_templates=[
            "DFT calculations for {material} (mp-{mp_id}) yield a band gap of {neg_value} eV using the PBE functional.",
            "The band gap of {material} was computed to be {neg_value} eV via GGA-PBE.",
            "A recent computational study reports the band gap of {material} as {neg_value} eV.",
            "HSE06 calculations give a band gap of {neg_value} eV for {material}.",
        ],
        clean_context_templates=[
            "The PBE band gap of {material} (mp-{mp_id}) is {pos_value} eV.",
            "DFT calculations yield a band gap of {pos_value} eV for {material}.",
        ],
        description="Band gap must be non-negative",
    ),
    ConstraintSpec(
        constraint_id="band_gap_physical_range",
        level="L0",
        question_templates=[
            "What is the band gap of {material}?",
            "Report the band gap of {material}.",
            "What band gap does {material} exhibit?",
        ],
        violation_context_templates=[
            "An unusually large band gap of {huge_value} eV was computed for {material} via PBE.",
            "The calculated band gap for {material} reaches {huge_value} eV.",
        ],
        clean_context_templates=[
            "The band gap of {material} is {pos_value} eV, within the typical range.",
        ],
        description="Band gap above ~15 eV is physically suspicious",
    ),
    ConstraintSpec(
        constraint_id="valid_chemical_symbols",
        level="L0",
        question_templates=[
            "What is the chemical formula of {material}?",
            "What compound does {material} refer to?",
            "Give the formula for {material}.",
        ],
        violation_context_templates=[
            "The reported formula for {material} is {invalid_formula}.",
            "A study claims {material} has composition {invalid_formula}.",
        ],
        clean_context_templates=[
            "The chemical formula of {material} is {valid_formula}.",
        ],
        description="Chemical formula must use valid element symbols",
    ),
    ConstraintSpec(
        constraint_id="non_negative_absolute_temperature",
        level="L0",
        question_templates=[
            "At what temperature does {material} undergo this transition?",
            "What is the critical temperature of {material}?",
            "Report the operating temperature for {material} in Kelvin.",
            "What temperature was {material} measured at?",
        ],
        violation_context_templates=[
            "The transition occurs at {neg_temp} K for {material}.",
            "Measurements for {material} were conducted at {neg_temp} K.",
            "The critical temperature of {material} is reported as {neg_temp} K.",
        ],
        clean_context_templates=[
            "The transition temperature for {material} is {pos_temp} K.",
            "Measurements were performed at {pos_temp} K.",
        ],
        description="Absolute temperature in Kelvin must be non-negative",
    ),
    ConstraintSpec(
        constraint_id="non_negative_pressure",
        level="L0",
        question_templates=[
            "What pressure was {material} studied at?",
            "At what pressure does {material} become unstable?",
            "Report the synthesis pressure of {material}.",
        ],
        violation_context_templates=[
            "The pressure condition for {material} was {neg_pressure} GPa.",
            "{material} was synthesized at {neg_pressure} GPa.",
            "A pressure of {neg_pressure} GPa was applied to {material}.",
        ],
        clean_context_templates=[
            "The synthesis pressure for {material} was {pos_pressure} GPa.",
            "{material} was studied at {pos_pressure} GPa.",
        ],
        description="Pressure must be non-negative unless stress convention is stated",
    ),
    ConstraintSpec(
        constraint_id="formation_energy_typical_range",
        level="L0",
        question_templates=[
            "What is the formation energy of {material}?",
            "Report the formation energy per atom for {material}.",
            "What formation energy does {material} have?",
        ],
        violation_context_templates=[
            "The formation energy of {material} is {extreme_value} eV/atom.",
            "DFT gives a formation energy of {extreme_value} eV/atom for {material}.",
        ],
        clean_context_templates=[
            "The formation energy of {material} is {normal_value} eV/atom.",
        ],
        description="Formation energy outside -4 to +1 eV/atom is suspicious",
    ),
    ConstraintSpec(
        constraint_id="cautious_superconductivity_claim",
        level="L0",
        question_templates=[
            "Is {material} a superconductor?",
            "Does {material} exhibit superconductivity?",
            "What superconducting behavior does {material} show?",
        ],
        violation_context_templates=[
            "{material} is a superconductor.",
            "{material} superconducts at ambient pressure.",
            "This material belongs to a new family of superconductors.",
        ],
        clean_context_templates=[
            "{material} exhibits superconductivity with a measured Tc of {tc_value} K.",
            "Superconductivity was observed in {material} at a transition temperature of {tc_value} K.",
        ],
        description="Superconductivity claim needs Tc or evidence",
    ),
    # ── L1: physics-consistency / cross-attribute checks ────────────────────
    ConstraintSpec(
        constraint_id="cautious_stability_claim",
        level="L1",
        question_templates=[
            "Is {material} thermodynamically stable?",
            "How stable is {material}?",
            "Would {material} be stable under ambient conditions?",
            "Assess the thermodynamic stability of {material}.",
        ],
        violation_context_templates=[
            "{material} is very stable.",
            "{material} is one of the most stable compounds in its class.",
            "{material} demonstrates excellent stability.",
            "The material exhibits long-term stability.",
        ],
        clean_context_templates=[
            "{material} has a formation energy of {fe_value} eV/atom and an energy above hull of {hull_value} meV/atom, indicating thermodynamic stability.",
            "{material} is thermodynamically stable with a formation energy of {fe_value} eV/atom.",
        ],
        description="Stability claim needs thermodynamic evidence",
    ),
    ConstraintSpec(
        constraint_id="positive_formation_energy_stability_conflict",
        level="L1",
        question_templates=[
            "Is {material} thermodynamically stable?",
            "Is {material} a stable compound?",
            "Would you consider {material} to be stable?",
        ],
        violation_context_templates=[
            "{material} is stable. Its formation energy is {pos_fe_value} eV/atom.",
            "{material} is a thermodynamically stable compound with a formation energy of {pos_fe_value} eV/atom.",
        ],
        clean_context_templates=[
            "{material} has a formation energy of {neg_fe_value} eV/atom, consistent with its stability.",
            "{material} is unstable, as its formation energy is {pos_fe_value} eV/atom.",
        ],
        description="Stable claim + positive formation energy = contradiction",
    ),
    ConstraintSpec(
        constraint_id="conductivity_band_gap_consistency",
        level="L1",
        question_templates=[
            "What are the electronic properties of {material}?",
            "Describe the conductivity and band gap of {material}.",
            "Is {material} metallic or insulating?",
        ],
        violation_context_templates=[
            "{material} is metallic with a band gap of {large_gap_value} eV.",
            "{material} is an insulator, but its band gap is only 0.02 eV.",
            "{material} shows metallic behavior and has a band gap of 3.5 eV.",
        ],
        clean_context_templates=[
            "{material} is metallic with no band gap, consistent with its conducting behavior.",
            "{material} is an insulator with a band gap of {gap_value} eV.",
        ],
        description="Metallic + large band gap or insulator + near-zero gap is contradictory",
    ),
    ConstraintSpec(
        constraint_id="crystal_system_space_group_consistency",
        level="L1",
        question_templates=[
            "What crystal system does {material} belong to?",
            "Describe the crystal structure of {material}.",
            "What is the space group of {material}?",
        ],
        violation_context_templates=[
            "{material} crystallizes in the cubic system with space group 14.",
            "{material} has a hexagonal structure; its space group is 225.",
            "The crystal system of {material} is tetragonal, space group: 166.",
            "{material} is cubic with space-group number 14.",
            "The space group of {material} is 75, and it belongs to the hexagonal system.",
            "{material} crystallizes in the monoclinic system with space group 200.",
        ],
        clean_context_templates=[
            "{material} crystallizes in the cubic system with space group 225.",
            "{material} has a hexagonal structure with space group 194.",
            "{material} is monoclinic; its space group is 14.",
            "The space group of {material} is 75, consistent with its tetragonal system.",
        ],
        description="Crystal system inconsistent with space group number",
    ),
]

# ---------------------------------------------------------------------------
# Materials for generating diverse samples
# ---------------------------------------------------------------------------

MATERIALS = [
    ("TiO2", "mp-2657"), ("ZnO", "mp-2133"), ("GaN", "mp-804"),
    ("SrTiO3", "mp-5229"), ("La2CuO4", "mp-22392"), ("BaTiO3", "mp-5986"),
    ("Fe2O3", "mp-19770"), ("Cu2O", "mp-36228"), ("SnO2", "mp-856"),
    ("WO3", "mp-1013"), ("Bi2Se3", "mp-541837"), ("MoS2", "mp-1434"),
    ("LiCoO2", "mp-24872"), ("NaCl", "mp-22862"), ("SiC", "mp-8062"),
    ("Al2O3", "mp-1143"), ("MgO", "mp-1265"), ("CdS", "mp-672"),
    ("PbTiO3", "mp-20471"), ("InP", "mp-20351"),
]

# ---------------------------------------------------------------------------
# LLM generation via DashScope
# ---------------------------------------------------------------------------

def call_llm(prompt: str, model: str = "qwen-plus") -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    payload = json.dumps({
        "model": model,
        "temperature": 0.7,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    return raw["choices"][0]["message"]["content"].strip()


def generate_diverse_answers(
    question: str,
    context: str,
    n_variants: int = 3,
    mode: str = "naive",
) -> list[str]:
    """Generate diverse answer phrasings for a given Q&C pair."""
    mode_instruction = (
        "Answer directly based on the retrieved context. Do not question the context's reliability."
        if mode == "naive"
        else "Answer based on the context, but if the context seems unreliable or incomplete, you may express uncertainty."
    )
    prompt = f"""You are a materials-science RAG assistant. Generate {n_variants} DIFFERENT answer phrasings for the given question and context.

Rules:
- {mode_instruction}
- Each answer should use different wording, sentence structure, and level of detail.
- Do NOT copy the context verbatim.
- Paraphrase the key scientific claims in your own words.
- Each answer on its own line, prefixed with "ANSWER: ".

Question: {question}

Retrieved context: {context}

Generate {n_variants} diverse answers:"""

    raw = call_llm(prompt)
    answers = []
    for line in raw.split("\n"):
        line = line.strip()
        if line.upper().startswith("ANSWER:"):
            answers.append(line[len("ANSWER:"):].strip())
        elif line and len(answers) < n_variants and len(line) > 20:
            answers.append(line)
    return answers[:n_variants]


# ---------------------------------------------------------------------------
# Value generators for template filling
# ---------------------------------------------------------------------------

import random

def fill_template(template: str, constraint_id: str, is_violation: bool) -> str:
    material, mp_id = random.choice(MATERIALS)
    neg_values = [-0.08, -0.15, -1.27, -2.65, -0.48, -1.50, -0.72, -2.39]
    pos_values = [0.3, 0.7, 1.12, 1.8, 2.4, 3.2, 5.5, 8.5]
    huge_values = [18.3, 22.7, 31.5, 45.2]
    neg_temps = [-5, -10, -50, -273, -100, -0.5]
    pos_temps = [300, 350, 77, 4.2, 298, 1000]
    neg_pressures = [-0.5, -2.3, -10.0, -0.01]
    pos_pressures = [0.5, 1.0, 5.0, 10.0, 100.0]
    extreme_fe = [-12.5, 8.3, -25.0, 15.7]
    normal_fe = [-1.5, -2.3, -0.8, -3.2, 0.2, -1.0]
    invalid_formulas = ["Xx2Zz5", "Ab3Cd9", "QqRr2", "ZzYy4"]
    valid_formulas = ["TiO2", "ZnO", "BaTiO3", "SrTiO3", "La2CuO4"]
    tc_values = [4.2, 33, 90, 135]
    fe_values = [-1.5, -2.3, -3.8, -0.5, 0.8]
    hull_values = [0, 5, 12, 28]
    gap_values = [0.3, 1.1, 3.2, 5.0]

    replacements = {
        "{material}": material,
        "{mp_id}": mp_id,
        "{neg_value}": str(random.choice(neg_values)),
        "{pos_value}": str(random.choice(pos_values)),
        "{huge_value}": str(random.choice(huge_values)),
        "{neg_temp}": str(random.choice(neg_temps)),
        "{pos_temp}": str(random.choice(pos_temps)),
        "{neg_pressure}": str(random.choice(neg_pressures)),
        "{pos_pressure}": str(random.choice(pos_pressures)),
        "{extreme_value}": str(random.choice(extreme_fe)),
        "{normal_value}": str(random.choice(normal_fe)),
        "{invalid_formula}": random.choice(invalid_formulas),
        "{valid_formula}": random.choice(valid_formulas),
        "{tc_value}": str(random.choice(tc_values)),
        "{fe_value}": str(random.choice(fe_values)),
        "{neg_fe_value}": str(random.choice([-1.5, -2.3, -3.8, -0.5])),
        "{pos_fe_value}": str(random.choice([0.5, 1.2, 2.8, 3.5])),
        "{hull_value}": str(random.choice(hull_values)),
        "{large_gap_value}": str(random.choice([3.2, 5.0, 7.5])),
        "{gap_value}": str(random.choice(gap_values)),
    }
    result = template
    for key, val in replacements.items():
        result = result.replace(key, val)
    return result


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------

def generate_samples_for_constraint(
    spec: ConstraintSpec,
    n_violation: int = 15,
    n_clean: int = 15,
    n_answer_variants: int = 3,
    use_llm: bool = True,
) -> list[dict]:
    """Generate violation + clean samples for one constraint."""
    samples = []
    sample_idx = 0
    statuses = ["current", "outdated_or_incorrect", "incomplete"]

    # Generate violation samples
    violations_generated = 0
    attempts = 0
    max_attempts = n_violation * 4

    while violations_generated < n_violation and attempts < max_attempts:
        attempts += 1
        q_template = random.choice(spec.question_templates)
        ctx_template = random.choice(spec.violation_context_templates)
        status = statuses[violations_generated % 3]

        question = fill_template(q_template, spec.constraint_id, True)
        context = fill_template(ctx_template, spec.constraint_id, True)

        if use_llm and attempts % 3 == 1:
            # Generate diverse answers via LLM every 3rd attempt
            try:
                answers = generate_diverse_answers(question, context, min(n_answer_variants, n_violation - violations_generated), mode="naive")
            except Exception:
                answers = [context]  # Fallback: copy context
        else:
            answers = [context]

        for answer in answers:
            if violations_generated >= n_violation:
                break
            # Validate: the answer should trigger the constraint
            result = evaluate_material_qa_constraints(question, answer)
            target_violation = spec.constraint_id
            triggered = target_violation in result.violations

            sample_id = f"supp_{spec.constraint_id}_v{violations_generated:03d}"
            samples.append({
                "sample_id": sample_id,
                "domain": "materials",
                "source": "supplementary_generated",
                "constraint_target": spec.constraint_id,
                "constraint_level": spec.level,
                "question": question,
                "retrieved_context": context,
                "answer": answer,
                "document_status": status,
                "expected_violation": target_violation,
                "actual_violations": ";".join(result.violations),
                "constraint_score": result.score,
                "violation_triggered": triggered,
                "generation_method": "llm_diverse" if use_llm and attempts % 3 == 1 else "template",
            })
            violations_generated += 1

    # Generate clean (non-violation) samples
    clean_generated = 0
    attempts = 0
    max_attempts = n_clean * 4

    while clean_generated < n_clean and attempts < max_attempts:
        attempts += 1
        q_template = random.choice(spec.question_templates)
        ctx_template = random.choice(spec.clean_context_templates)
        status = statuses[clean_generated % 3]

        question = fill_template(q_template, spec.constraint_id, False)
        context = fill_template(ctx_template, spec.constraint_id, False)

        if use_llm and attempts % 3 == 1:
            try:
                answers = generate_diverse_answers(question, context, min(n_answer_variants, n_clean - clean_generated), mode="naive")
            except Exception:
                answers = [context]
        else:
            answers = [context]

        for answer in answers:
            if clean_generated >= n_clean:
                break
            result = evaluate_material_qa_constraints(question, answer)
            # For clean samples, we want NO violation (or at least not the target one)
            if spec.constraint_id in result.violations:
                continue  # Skip, this accidentally triggered

            sample_id = f"supp_{spec.constraint_id}_c{clean_generated:03d}"
            samples.append({
                "sample_id": sample_id,
                "domain": "materials",
                "source": "supplementary_generated",
                "constraint_target": spec.constraint_id,
                "constraint_level": spec.level,
                "question": question,
                "retrieved_context": context,
                "answer": answer,
                "document_status": status,
                "expected_violation": "none",
                "actual_violations": ";".join(result.violations) if result.violations else "",
                "constraint_score": result.score,
                "violation_triggered": False,
                "generation_method": "llm_diverse" if use_llm and attempts % 3 == 1 else "template",
            })
            clean_generated += 1

    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate supplementary dataset covering all 11 constraints.")
    parser.add_argument("--limit-per-type", type=int, default=15, help="Number of violation + clean samples per constraint type")
    parser.add_argument("--dry-run", action="store_true", help="Only show plan, do not call LLM")
    parser.add_argument("--no-llm", action="store_true", help="Use template-only generation (no LLM API calls)")
    parser.add_argument("--output", default=str(ROOT / "data/processed/supplementary_constraint_coverage.jsonl"))
    args = parser.parse_args()

    n = args.limit_per_type
    use_llm = not args.no_llm

    print(f"Supplementary Dataset Generation Plan")
    print(f"======================================")
    print(f"Samples per constraint type: {n} violation + {n} clean = {2*n}")
    print(f"Total constraints: {len(CONSTRAINTS)}")
    print(f"Expected total samples: {2 * n * len(CONSTRAINTS)}")
    print(f"Use LLM for diverse answers: {use_llm}")
    print(f"Output: {args.output}")
    print()

    if args.dry_run:
        for spec in CONSTRAINTS:
            print(f"  [{spec.level}] {spec.constraint_id}: {spec.description}")
            print(f"    Question templates: {len(spec.question_templates)}")
            print(f"    Violation context templates: {len(spec.violation_context_templates)}")
            print(f"    Clean context templates: {len(spec.clean_context_templates)}")
            print()
        return 0

    all_samples = []
    stats = Counter()

    for i, spec in enumerate(CONSTRAINTS):
        print(f"[{i+1}/{len(CONSTRAINTS)}] Generating {spec.constraint_id} ({spec.level})...")
        try:
            samples = generate_samples_for_constraint(
                spec, n_violation=n, n_clean=n, use_llm=use_llm,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            samples = []

        triggered = sum(1 for s in samples if s.get("violation_triggered"))
        clean = sum(1 for s in samples if not s.get("expected_violation", "").startswith(spec.constraint_id))
        print(f"  Generated: {len(samples)} (violation_triggered={triggered}, clean={clean})")

        all_samples.extend(samples)
        stats[spec.constraint_id] = len(samples)

        # Rate limit: small delay between LLM calls
        if use_llm:
            time.sleep(1)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print()
    print(f"=== Generation Complete ===")
    print(f"Total samples: {len(all_samples)}")
    print(f"Output: {output_path}")
    print()
    print("=== Coverage Summary ===")
    for spec in CONSTRAINTS:
        count = stats.get(spec.constraint_id, 0)
        print(f"  [{spec.level}] {spec.constraint_id}: {count} samples")

    # Validate coverage
    violation_counts = Counter()
    for s in all_samples:
        for v in s.get("actual_violations", "").split(";"):
            if v.strip():
                violation_counts[v.strip()] += 1

    print()
    print("=== Actual Violation Distribution ===")
    for v, c in violation_counts.most_common():
        print(f"  {v}: {c}")

    # Check which constraints have 0 violations in data
    missing = []
    for spec in CONSTRAINTS:
        if violation_counts.get(spec.constraint_id, 0) == 0:
            missing.append(spec.constraint_id)
    if missing:
        print(f"\n⚠️  Constraints with 0 triggered violations: {missing}")
    else:
        print(f"\n✅ All {len(CONSTRAINTS)} constraints have at least some triggered violations!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
