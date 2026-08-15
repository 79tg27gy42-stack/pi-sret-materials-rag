# Source Log: H1 Contrastive Dataset

**Auto-generated**: 2026-05-31 — run `python3 scripts/generate_source_log.py` to update.

## Current Dataset Statistics

| Metric | Value |
|--------|-------|
| Total samples | 252 |
| Unique materials | 18 |
| Unique contexts | 252 |

### Status Distribution

- **current**: 84
- **incomplete**: 84
- **outdated_or_incorrect**: 84

### Formula Distribution

| Formula | Samples | Percentage |
|---------|---------|-----------|
| C | 54 | 21.4% ⚠️ OVERREPRESENTED |
| S | 39 | 15.5% ⚠️ OVERREPRESENTED |
| O2 | 36 | 14.3% |
| Si | 15 | 6.0% |
| H2 | 15 | 6.0% |
| Se | 15 | 6.0% |
| P | 12 | 4.8% |
| Hg | 12 | 4.8% |
| F2 | 9 | 3.6% |
| Xe | 9 | 3.6% |
| Bi | 6 | 2.4% |
| B | 6 | 2.4% |
| N2 | 6 | 2.4% |
| Te | 6 | 2.4% |
| Cl2 | 3 | 1.2% |
| Rb | 3 | 1.2% |
| Kr | 3 | 1.2% |
| He | 3 | 1.2% |

### Constraint Coverage

- `cautious_stability_claim`
- `non_negative_band_gap`
- `valid_chemical_symbols`

**Missing constraints (not covered by any sample):**
- `cautious_superconductivity_claim`
- `non_negative_absolute_temperature`
- `non_negative_pressure`

## Annotation Progress

| Metric | Value |
|--------|-------|
| Annotation batch size | 45 |
| Labeled | 0 |
| Progress | 0/45 (0.0%) |

## Data Sources

### Materials Project API
- Source type: Structured database
- Script: `scripts/collect_mp_candidates.py`
- Raw records: `data/sources/materials_project_candidates.jsonl`

### arXiv API
- Source type: Publication metadata
- Script: `scripts/collect_arxiv_sources.py`
- Raw records: `data/sources/arxiv_materials_sources.jsonl`

## Transformations

1. `scripts/build_h1_contrast_set_from_mp.py` → 3 QA pairs per material
2. `scripts/convert_h1_csv_to_jsonl.py` → JSONL format
3. `scripts/create_expert_annotation_batch.py` → stratified sampling

## Status

- [x] Raw data collected (252 samples)
- [x] Constraint validation tests: 123 cases (see tests/test_constraint_validation.py)
- [ ] Human review of candidates pending
- [ ] Author audit: 0/45 completed
- [ ] LLM Judge faithfulness scoring pending