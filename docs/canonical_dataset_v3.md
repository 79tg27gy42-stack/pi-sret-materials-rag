# Canonical Dataset v3

Canonical v3 is the recommended dataset entry point after refreshing the Materials Project API key.

## Main Improvements over v2

- Live Materials Project collection now succeeds: 1000 unique MP material records were collected into `data/sources/materials_project_candidates_v3.jsonl`.
- MP-derived balanced controlled records increased from 2400 to 8000 and are derived from the 1000 live MP rows.
- The primary split increased from 2272 to 2811 response records.
- Unique base tasks increased from 1108 to 1653.
- The full audit manifest increased from 5834 to 11434 response records.

## Files

| File | n | Use |
| --- | ---: | --- |
| `data/processed/canonical_v3/evaluation_manifest_v3.jsonl` | 11434 | Full audit manifest. |
| `data/processed/canonical_v3/full_response_manifest_v3.jsonl` | 11434 | Full response-record dataset. |
| `data/processed/canonical_v3/primary_response_manifest_v3.jsonl` | 2811 | Recommended main paper dataset. |
| `data/processed/canonical_v3/llm_judge_full_pool_v3.jsonl` | 2466 | Full LLM-as-Judge pool. |
| `data/processed/canonical_v3/ragas_stratified_372_v3.jsonl` | 372 | Stratified RAGAS calibration subset. |
| `data/processed/canonical_v3/dual_judge_stratified_200_v3.jsonl` | 200 | Stratified dual-judge subset. |

## Primary Split

| statistic | value |
| --- | ---: |
| response records | 2811 |
| unique base tasks | 1653 |
| unique contexts | 1592 |
| unique questions | 628 |
| natural source-linked arXiv records | 500 |
| unique public arXiv sources | 254 |
| live Materials Project rows | 1000 |

## Constraint Balance

| constraint family | n |
| --- | ---: |
| band_gap | 500 |
| formation_energy | 500 |
| stability | 500 |
| temperature | 439 |
| chemical_formula | 301 |
| superconductivity | 226 |
| pressure | 177 |
| natural_claim | 168 |

The primary split keeps band-gap records below 20% of the dataset while allowing the refreshed MP data to increase coverage.

## Rebuild

```bash
python3 scripts/collect_mp_candidates.py \
  --limit 1000 \
  --output-stem data/sources/materials_project_candidates_v3

python3 scripts/build_balanced_mp_constraint_dataset_v2.py \
  --mp-input data/sources/materials_project_candidates_v3.jsonl \
  --output data/processed/mp_balanced_constraint_dataset_v3.jsonl \
  --limit-materials 1000

python3 scripts/build_canonical_evaluation_dataset_v3.py
python3 scripts/validate_canonical_evaluation_dataset_v3.py
```

## Recommended Paper Wording

> We evaluate PI-SRET on a canonical v3 primary split containing 2811 response records derived from 1653 unique materials-science base tasks. The split includes 500 source-linked arXiv records from 254 public arXiv sources and MP-derived balanced diagnostic records built from 1000 live Materials Project rows collected with the refreshed API key. For external evaluation, we use a 2466-record LLM-as-Judge pool, a 372-record stratified RAGAS calibration subset, and a 200-record dual-judge agreement subset.

Report response-record counts separately from unique base-task counts.
