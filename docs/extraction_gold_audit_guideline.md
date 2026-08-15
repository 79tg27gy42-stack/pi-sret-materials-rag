# Phi Extraction Gold Audit Annotation Guideline

Purpose: create independent span/entity gold labels for the deterministic PI-SRET extractor `Phi(q,y)`. The prediction columns are prefilled by code. Gold columns must be completed by a human annotator; do not copy predictions unless the span is actually correct.

## Unit of Annotation

Annotate spans present in the answer text. Use the question only to resolve terse answers, such as an answer of `1.1 eV` to a band-gap question.

Multiple spans in the same field should be separated with semicolons, matching the scorer convention.

## Fields

- `gold_material`: named material or material-like entity in the answer.
- `gold_formula`: chemical formula spans in the answer.
- `gold_numeric_value`: numeric values relevant to materials claims, without units.
- `gold_unit`: units attached to relevant values, such as `eV`, `eV/atom`, `K`, or `GPa`.
- `gold_property`: property being asserted, such as `band_gap`, `formation_energy`, `stability`, `temperature`, `pressure`, `superconductivity`, or `conductivity`.
- `gold_phase`: named phase or crystal system, such as cubic, tetragonal, fcc, or P6/mmm.
- `gold_structure`: structure identifiers, especially space-group numbers or named structural prototypes.
- `gold_method`: method or evidence-source type, such as DFT, XRD, experiment, measurement, or first-principles.
- `gold_evidence`: evidence phrase supporting a scientific claim, such as formation energy, energy above hull, transition temperature, measured, resistivity, zero resistance, magnetic susceptibility, or Meissner.
- `gold_qualifier`: local qualifier that changes rule interpretation, such as invalid, unreliable, unsupported, metastable, relative to, or with respect to.
- `annotator_comment`: short note for ambiguity, span boundary issues, or cross-material/coreference problems.

## Completion Rule

Set `annotation_complete=yes` only after every applicable gold field has been reviewed. Leave genuinely absent fields blank. The evaluation script refuses to compute metrics until every row has `annotation_complete=yes`.

## Do Not

- Do not use LLM output as gold.
- Do not infer spans that are not present in the answer, except for property type implied by a terse answer and explicit question.
- Do not change PI-SRET predictions in the `pred_*` columns.
- Do not alter human validation verdicts in this file.
