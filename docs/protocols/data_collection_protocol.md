# 数据采集协议

## 来源优先级

1. Materials Project 属性记录。
2. arXiv 公开元数据和摘要。
3. 官方公开数据库页面。
4. 你自己的 RAG 输出和 retrieved chunks。

不要使用需要付费或登录的全文，除非你本地提供文件并确认可以用于研究。

## Materials Project 流程

API key 可以放在本地 `.env` 中：

```text
MP_API_KEY=...
```

也可以在终端临时设置：

```bash
export MP_API_KEY="..."
```

然后运行：

```bash
python3 scripts/collect_mp_candidates.py --limit 300 --backend rest
python3 scripts/build_h1_candidates_from_mp.py
python3 scripts/build_h1_contrast_set_from_mp.py
python3 scripts/make_source_log.py
```

输出：

```text
data/sources/materials_project_candidates.csv
data/sources/materials_project_candidates.jsonl
data/candidates/h1_mp_candidates.csv
data/candidates/h1_contrastive_candidates.csv
```

`h1_mp_candidates.csv` 是基于 MP 真值生成的正常候选 QA。

`h1_contrastive_candidates.csv` 包含有意扰动和不完整上下文，用于 H1 压力测试。它必须先被标为 candidate data，经过审核后才能进入正式论文数据。

## 审核后提升为正式数据

不要求 `review_decision` 时：

```bash
python3 scripts/promote_reviewed_candidates.py \
  data/candidates/h1_contrastive_candidates.csv \
  data/processed/h1_reviewed.jsonl
```

只提升 `review_decision` 为 approve/yes/1 的样本：

```bash
python3 scripts/promote_reviewed_candidates.py \
  data/candidates/h1_contrastive_candidates.csv \
  data/processed/h1_reviewed.jsonl \
  --require-approval
```

## arXiv 流程

```bash
python3 scripts/collect_arxiv_sources.py --max-results 50
python3 scripts/build_arxiv_review_queue.py
python3 scripts/make_source_log.py
```

输出：

```text
data/sources/arxiv_materials_sources.csv
data/sources/arxiv_materials_sources.jsonl
data/candidates/arxiv_review_queue.csv
```

arXiv 记录主要用于公开来源追踪和候选材料科学声明发现。

## 密钥处理

不要提交 API key。`.env` 已经被 `.gitignore` 忽略。

如果 key 曾经出现在聊天或共享文档里，建议完成数据拉取后去 Materials Project 后台轮换。

