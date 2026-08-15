# 数据格式说明

H1 原始样本使用 JSONL。每一行是一条 QA 样本。

## 必需字段

```json
{
  "sample_id": "mat_0001",
  "question": "What is the band gap of metallic copper?",
  "retrieved_context": "Copper is a metallic conductor...",
  "answer": "Copper is metallic, so its band gap is approximately 0 eV.",
  "document_status": "current"
}
```

字段说明：

- `sample_id`：唯一样本 ID。
- `question`：材料科学问题。
- `retrieved_context`：RAG 检索到的文本。
- `answer`：RAG 或候选流程生成的回答。
- `document_status`：检索文本状态。

## 可选字段

```json
{
  "domain": "materials",
  "source": "materials_project",
  "expected_constraints": ["non_negative_band_gap"],
  "faithfulness_score": 0.92
}
```

如果提供 `faithfulness_score`，Experiment A 会直接使用这个分数。否则实验会使用配置中的 evaluator 自动计算。

## `document_status`

取值必须是：

- `current`：检索文本对问题而言足够新且科学上可接受。
- `outdated_or_incorrect`：检索文本包含过时、噪声或错误科学声明。
- `incomplete`：检索文本相关，但缺少必要科学约束或条件。
- `unknown`：尚未标注。

## 专家标注

专家审核使用：

```text
data/annotations/h1_annotation_template.csv
```

关键字段：

- `expert_constraint_score`：如果 answer 满足材料约束，填 `1`；否则填 `0`。
- `expert_violations`：违反的约束 ID，多个用英文分号 `;` 分隔。
- `expert_notes`：一句简短判断理由。

模板文件刻意保持为空。`data/annotations/h1_annotation_example.csv` 只是格式示例，不能当作真实专家证据。

## CSV 转 JSONL

如果更习惯用表格准备数据，填写：

```text
data/raw/h1_samples_template.csv
```

然后转换：

```bash
python3 scripts/convert_h1_csv_to_jsonl.py \
  data/raw/h1_samples_template.csv \
  data/raw/materials_qa_seed.jsonl
```

