# Experiment B / H2 归因 Pilot 摘要

- 样本数：372
- 错误样本数：172
- 相关性基线准确率：0.0000
- 因果干预式归因准确率：1.0000
- 准确率差值：1.0000
- 相关性基线 coarse 准确率：0.2500
- 因果干预式 coarse 准确率：1.0000
- coarse 准确率差值：0.7500

## Gold 分布

- no_error: 200
- retrieval_context_error: 129
- retrieval_context_incomplete: 42
- generation_constraint_error: 1

## 限制

Pilot attribution uses constructed document_status as gold labels. This validates the attribution pipeline, not a final human-labeled causal study.
