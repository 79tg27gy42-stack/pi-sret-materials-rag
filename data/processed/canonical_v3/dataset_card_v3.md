# Canonical PI-SRET Evaluation Dataset v3

This dataset separates base scientific tasks from evaluated response records.
A single base task can have multiple response records from different generators or modes.

## Sizes

| split | n |
| --- | ---: |
| evaluation_manifest_all_records | 11434 |
| primary_response_manifest | 2811 |
| llm_judge_full_pool | 2466 |
| ragas_stratified_372 | 372 |
| dual_judge_stratified_200 | 200 |
| unique_base_tasks_primary | 1653 |
| unique_contexts_primary | 1592 |
| unique_questions_primary | 628 |
| full_response_manifest | 11434 |

## Recommended Usage

- Use `primary_response_manifest_v3.jsonl` for paper-level response-record statistics.
- Use `llm_judge_full_pool_v3.jsonl` for full-coverage LLM-as-Judge scoring.
- Use `ragas_stratified_372_v3.jsonl` for real RAGAS calibration.
- Use `dual_judge_stratified_200_v3.jsonl` for two-model agreement and Cohen's kappa.
- Report both response-record counts and unique base-task counts.

## primary_breakdown


### by_dataset_role

| value | n |
| --- | ---: |
| mp_derived_balanced_controlled | 1387 |
| natural_source_linked | 500 |
| controlled_candidate | 105 |
| llm_generation | 480 |
| local_rag | 120 |
| official_only_rag | 120 |
| supplementary_constraint_probe | 99 |

### by_source_dataset

| value | n |
| --- | ---: |
| mp_balanced_constraints_v3 | 1387 |
| natural_noisy_arxiv_v2 | 500 |
| controlled_diverse_prefilled | 105 |
| deepseek_r1_naive | 120 |
| deepseek_r1_safety | 120 |
| qwen_max_naive | 120 |
| qwen_max_safety | 120 |
| local_rag_naive | 120 |
| local_rag_official | 120 |
| supplementary_constraint_coverage | 99 |

### by_model

| value | n |
| --- | ---: |
| mp_property_derived | 1387 |
| extractive_real_arxiv | 500 |
| prefilled_context_support | 105 |
| deepseek-r1 | 240 |
| qwen-max | 240 |
| local_rule_rag | 240 |
| llm_diverse_probe | 99 |

### by_mode

| value | n |
| --- | ---: |
| controlled_balanced | 1387 |
| matched_and_mismatched | 500 |
| diverse_prefilled | 105 |
| naive_context | 360 |
| safety_aware | 240 |
| official_only | 120 |
| constraint_coverage | 99 |

### by_document_status

| value | n |
| --- | ---: |
| outdated_or_incorrect | 891 |
| current | 1217 |
| incomplete | 703 |

### by_constraint_family

| value | n |
| --- | ---: |
| band_gap | 500 |
| chemical_formula | 301 |
| formation_energy | 500 |
| natural_claim | 168 |
| pressure | 177 |
| stability | 500 |
| superconductivity | 226 |
| temperature | 439 |

### by_source_kind

| value | n |
| --- | ---: |
| materials_project_derived | 1567 |
| real_arxiv_abstract | 590 |
| synthetic_constraint_probe | 195 |
| derived_response | 459 |

## llm_pool_breakdown


### by_dataset_role

| value | n |
| --- | ---: |
| mp_derived_balanced_controlled | 1387 |
| natural_source_linked | 500 |
| llm_generation | 480 |
| supplementary_constraint_probe | 99 |

### by_source_dataset

| value | n |
| --- | ---: |
| mp_balanced_constraints_v3 | 1387 |
| natural_noisy_arxiv_v2 | 500 |
| deepseek_r1_naive | 120 |
| deepseek_r1_safety | 120 |
| qwen_max_naive | 120 |
| qwen_max_safety | 120 |
| supplementary_constraint_coverage | 99 |

### by_model

| value | n |
| --- | ---: |
| mp_property_derived | 1387 |
| extractive_real_arxiv | 500 |
| deepseek-r1 | 240 |
| qwen-max | 240 |
| llm_diverse_probe | 99 |

### by_mode

| value | n |
| --- | ---: |
| controlled_balanced | 1387 |
| matched_and_mismatched | 500 |
| naive_context | 240 |
| safety_aware | 240 |
| constraint_coverage | 99 |

### by_document_status

| value | n |
| --- | ---: |
| outdated_or_incorrect | 786 |
| current | 977 |
| incomplete | 703 |

### by_constraint_family

| value | n |
| --- | ---: |
| band_gap | 500 |
| chemical_formula | 166 |
| formation_energy | 500 |
| natural_claim | 168 |
| pressure | 147 |
| stability | 500 |
| superconductivity | 151 |
| temperature | 334 |

### by_source_kind

| value | n |
| --- | ---: |
| materials_project_derived | 1507 |
| real_arxiv_abstract | 500 |
| derived_response | 459 |

## ragas_372_breakdown


### by_dataset_role

| value | n |
| --- | ---: |
| llm_generation | 79 |
| mp_derived_balanced_controlled | 204 |
| natural_source_linked | 72 |
| supplementary_constraint_probe | 17 |

### by_source_dataset

| value | n |
| --- | ---: |
| deepseek_r1_naive | 20 |
| deepseek_r1_safety | 20 |
| qwen_max_naive | 20 |
| qwen_max_safety | 19 |
| mp_balanced_constraints_v3 | 204 |
| natural_noisy_arxiv_v2 | 72 |
| supplementary_constraint_coverage | 17 |

### by_model

| value | n |
| --- | ---: |
| deepseek-r1 | 40 |
| qwen-max | 39 |
| mp_property_derived | 204 |
| extractive_real_arxiv | 72 |
| llm_diverse_probe | 17 |

### by_mode

| value | n |
| --- | ---: |
| naive_context | 40 |
| safety_aware | 39 |
| controlled_balanced | 204 |
| matched_and_mismatched | 72 |
| constraint_coverage | 17 |

### by_document_status

| value | n |
| --- | ---: |
| current | 147 |
| outdated_or_incorrect | 121 |
| incomplete | 104 |

### by_constraint_family

| value | n |
| --- | ---: |
| pressure | 25 |
| chemical_formula | 28 |
| temperature | 53 |
| stability | 72 |
| formation_energy | 74 |
| band_gap | 74 |
| superconductivity | 22 |
| natural_claim | 24 |

### by_source_kind

| value | n |
| --- | ---: |
| derived_response | 77 |
| materials_project_derived | 223 |
| real_arxiv_abstract | 72 |

## dual_200_breakdown


### by_dataset_role

| value | n |
| --- | ---: |
| llm_generation | 32 |
| mp_derived_balanced_controlled | 113 |
| natural_source_linked | 43 |
| supplementary_constraint_probe | 12 |

### by_source_dataset

| value | n |
| --- | ---: |
| deepseek_r1_naive | 8 |
| deepseek_r1_safety | 8 |
| qwen_max_naive | 8 |
| qwen_max_safety | 8 |
| mp_balanced_constraints_v3 | 113 |
| natural_noisy_arxiv_v2 | 43 |
| supplementary_constraint_coverage | 12 |

### by_model

| value | n |
| --- | ---: |
| deepseek-r1 | 16 |
| qwen-max | 16 |
| mp_property_derived | 113 |
| extractive_real_arxiv | 43 |
| llm_diverse_probe | 12 |

### by_mode

| value | n |
| --- | ---: |
| naive_context | 16 |
| safety_aware | 16 |
| controlled_balanced | 113 |
| matched_and_mismatched | 43 |
| constraint_coverage | 12 |

### by_document_status

| value | n |
| --- | ---: |
| current | 79 |
| outdated_or_incorrect | 62 |
| incomplete | 59 |

### by_constraint_family

| value | n |
| --- | ---: |
| pressure | 13 |
| chemical_formula | 13 |
| temperature | 23 |
| stability | 40 |
| formation_energy | 41 |
| band_gap | 42 |
| superconductivity | 14 |
| natural_claim | 14 |

### by_source_kind

| value | n |
| --- | ---: |
| derived_response | 36 |
| materials_project_derived | 121 |
| real_arxiv_abstract | 43 |

## Quality Notes

- Counts distinguish base tasks from evaluated response records; model/mode variants are not treated as independent base tasks.
- The RAGAS and dual-judge subsets are deterministic stratified samples over dataset role, model, mode, document status, and constraint family.
- The natural arXiv split is source-linked to real arXiv metadata, while controlled stress splits remain diagnostic rather than natural benchmark data.
