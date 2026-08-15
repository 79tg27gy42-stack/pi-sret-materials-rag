# Canonical Dataset Schema v3

| field | meaning |
| --- | --- |
| `response_id` | Stable ID for one evaluated answer. |
| `base_task_id` | Stable ID for the underlying scientific QA task. |
| `raw_sample_id` | Original source-file sample ID. |
| `source_dataset` | Original dataset/source split name. |
| `dataset_role` | Controlled, local RAG, LLM generation, or natural source-linked role. |
| `source_kind` | Real source or controlled/synthetic derivation class. |
| `model` | Generator or response source. |
| `mode` | Generation/retrieval mode. |
| `question` | Question text. |
| `retrieved_context` | Retrieved evidence text. |
| `answer` | Evaluated answer. |
| `document_status` | current/outdated_or_incorrect/incomplete/unknown. |
| `constraint_family` | Primary targeted constraint family inferred for stratification. |
| `provided_faithfulness_score` | Existing score, if supplied by original file. |
| `include_llm_judge_pool` | Whether this response belongs to the full LLM judge pool. |
| `recommended_for_ragas` | Whether this response is eligible for RAGAS stratified sampling. |
