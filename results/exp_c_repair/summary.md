# Experiment C / H3 最小修复 Pilot 摘要

- 样本数：372
- 触发修复样本数：172
- 修复成功率：1.0000
- 修复前平均约束分：0.5376
- 修复后平均约束分：1.0000
- 平均约束增益：0.4624
- 修复前 lexical alignment：0.5083
- 修复后 lexical alignment：0.3842
- lexical alignment 变化：-0.1241
- 修复前 answer usefulness proxy：1.0000
- 修复后 answer usefulness proxy：0.9973
- answer usefulness proxy 变化：-0.0027

## 按文档状态分组

| status | n | repaired | constraint_before | constraint_after | success_rate | lexical_delta | usefulness_after | usefulness_delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 144 | 1 | 0.9931 | 1.0000 | 1.0000 | -0.0032 | 0.9931 | -0.0069 |
| incomplete | 84 | 42 | 0.5000 | 1.0000 | 1.0000 | -0.0118 | 1.0000 | 0.0000 |
| outdated_or_incorrect | 144 | 129 | 0.1042 | 1.0000 | 1.0000 | -0.3103 | 1.0000 | 0.0000 |

## 限制

Pilot repair uses rule-based minimal safety rewrites over constructed answers. It tests constraint repair achievability, not final answer correctness.
