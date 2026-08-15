from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from sret_materials_rag.evaluation.constraints import evaluate_material_qa_constraints
from sret_materials_rag.evaluation.faithfulness_evaluators import (
    EmbeddingSimilarityFaithfulnessEvaluator,
    LexicalOverlapFaithfulnessEvaluator,
)


@dataclass(frozen=True)
class RagasBaselineResult:
    scores: pd.DataFrame
    metrics: dict
    backend: str


def load_rag_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix == ".csv":
        records = pd.read_csv(path).to_dict("records")
    else:
        raise ValueError(f"Unsupported input format: {path}")
    return [_normalize_record(record) for record in records]


def run_ragas_baseline(
    *,
    input_path: Path,
    output_dir: Path,
    backend: str = "proxy",
    limit: int | None = None,
    allow_proxy_fallback: bool = False,
) -> RagasBaselineResult:
    records = load_rag_records(input_path)
    if limit is not None:
        records = records[:limit]
    if backend == "ragas":
        try:
            scored = _run_real_ragas(records)
            actual_backend = "ragas"
        except Exception as exc:
            if not allow_proxy_fallback:
                raise RuntimeError(
                    "Real RAGAS failed. Refusing to fall back to proxy because this run "
                    "was requested as a paid/real evaluation."
                ) from exc
            scored = _run_proxy(records)
            scored["ragas_error"] = str(exc)
            actual_backend = "proxy_after_ragas_failure"
    elif backend == "proxy":
        scored = _run_proxy(records)
        actual_backend = "ragas_compatible_proxy"
    else:
        raise ValueError(f"Unknown backend: {backend}")

    scored["pi_sret_constraint_score"] = scored.apply(
        lambda row: evaluate_material_qa_constraints(str(row["question"]), str(row["answer"])).score,
        axis=1,
    )
    scored["pi_sret_violations"] = scored.apply(
        lambda row: ";".join(evaluate_material_qa_constraints(str(row["question"]), str(row["answer"])).violations),
        axis=1,
    )

    metrics = _summarize(scored, input_path, actual_backend)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_dir / "ragas_baseline_scores.csv", index=False)
    cases = scored[(scored["ragas_faithfulness"] >= 0.8) & (scored["pi_sret_constraint_score"] < 1.0)].copy()
    cases.to_csv(output_dir / "ragas_high_faithfulness_pi_sret_low_cases.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(output_dir / "summary.md", metrics, cases)
    return RagasBaselineResult(scores=scored, metrics=metrics, backend=actual_backend)


def _normalize_record(record: dict) -> dict:
    answer = record.get("answer", record.get("qwen_answer", record.get("response", "")))
    return {
        "sample_id": record.get("sample_id", record.get("raw_sample_id", record.get("id", ""))),
        "response_id": record.get("response_id", ""),
        "base_task_id": record.get("base_task_id", ""),
        "source_dataset": record.get("source_dataset", ""),
        "dataset_role": record.get("dataset_role", ""),
        "model": record.get("model", ""),
        "mode": record.get("mode", ""),
        "constraint_family": record.get("constraint_family", ""),
        "question": record.get("question", ""),
        "answer": answer,
        "retrieved_context": record.get("retrieved_context", record.get("context", "")),
        "ground_truth": record.get("ground_truth", record.get("reference_answer", record.get("expected_answer", ""))),
        "document_status": record.get("document_status", ""),
    }


def _run_proxy(records: Iterable[dict]) -> pd.DataFrame:
    lexical = LexicalOverlapFaithfulnessEvaluator()
    embedding = EmbeddingSimilarityFaithfulnessEvaluator()
    rows = []
    for record in records:
        question = str(record["question"])
        answer = str(record["answer"])
        context = str(record["retrieved_context"])
        lexical_score = lexical.score(question=question, answer=answer, retrieved_context=context).score
        embedding_score = embedding.score(question=question, answer=answer, retrieved_context=context).score
        rows.append(
            {
                **record,
                "ragas_faithfulness": max(lexical_score, embedding_score),
                "ragas_answer_relevancy": _answer_relevancy(question, answer),
                "ragas_context_precision": _context_precision(question, context),
                "ragas_context_recall": _context_recall(answer, context),
                "ragas_backend": "ragas_compatible_proxy",
            }
        )
    return pd.DataFrame(rows)


def _run_real_ragas(records: list[dict]) -> pd.DataFrame:
    import os

    from datasets import Dataset
    from openai import OpenAI
    from ragas import evaluate
    from ragas.llms import llm_factory
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._faithfulness import Faithfulness

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        api_key = os.environ.get("QWEN_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get(
        "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model_name = os.environ.get("SRET_RAGAS_MODEL", "qwen-plus")
    if not api_key:
        raise RuntimeError("Real RAGAS requires OPENAI_API_KEY or QWEN_API_KEY.")

    client = OpenAI(api_key=api_key, base_url=base_url)
    llm = llm_factory(model_name, client=client)

    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception:
        from ragas.embeddings import embedding_factory

        embeddings = embedding_factory(model="text-embedding-3-small", client=client)

    dataset = Dataset.from_list(
        [
            {
                "question": record["question"],
                "answer": record["answer"],
                "contexts": [record["retrieved_context"]],
                "ground_truth": record["ground_truth"] or record["retrieved_context"],
            }
            for record in records
        ]
    )
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
    ]
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        show_progress=True,
        batch_size=5,
    )
    ragas_df = result.to_pandas()
    base = pd.DataFrame(records)
    scored = pd.concat([base.reset_index(drop=True), ragas_df.reset_index(drop=True)], axis=1)
    rename = {
        "faithfulness": "ragas_faithfulness",
        "answer_relevancy": "ragas_answer_relevancy",
    }
    for old, new in rename.items():
        if old in scored.columns:
            scored = scored.rename(columns={old: new})
    if "ragas_faithfulness" not in scored.columns:
        scored["ragas_faithfulness"] = float("nan")
    if "ragas_answer_relevancy" not in scored.columns:
        scored["ragas_answer_relevancy"] = float("nan")
    scored["ragas_context_precision"] = float("nan")
    scored["ragas_context_recall"] = float("nan")
    scored["ragas_backend"] = "ragas"
    scored["ragas_llm_model"] = model_name
    return scored


def _answer_relevancy(question: str, answer: str) -> float:
    return _token_overlap(answer, question)


def _context_precision(question: str, context: str) -> float:
    return _token_overlap(context, question)


def _context_recall(answer: str, context: str) -> float:
    return _token_overlap(answer, context)


def _token_overlap(a: str, b: str) -> float:
    left = _tokens(a)
    right = _tokens(b)
    if not left:
        return 0.0
    return len(left & right) / len(left)


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in str(text).replace("_", " ").split() if len(token) > 2}


def _summarize(df: pd.DataFrame, input_path: Path, backend: str) -> dict:
    high_faith_low_constraint = df[(df["ragas_faithfulness"] >= 0.8) & (df["pi_sret_constraint_score"] < 1.0)]
    return {
        "input": str(input_path),
        "backend": backend,
        "n_samples": int(len(df)),
        "valid_ragas_faithfulness": int(df["ragas_faithfulness"].notna().sum()) if "ragas_faithfulness" in df else 0,
        "valid_answer_relevancy": int(df["ragas_answer_relevancy"].notna().sum()) if "ragas_answer_relevancy" in df else 0,
        "mean_ragas_faithfulness": float(df["ragas_faithfulness"].mean()) if len(df) else 0.0,
        "mean_answer_relevancy": float(df["ragas_answer_relevancy"].mean()) if len(df) else 0.0,
        "mean_context_precision": float(df["ragas_context_precision"].mean()) if len(df) else 0.0,
        "mean_context_recall": float(df["ragas_context_recall"].mean()) if len(df) else 0.0,
        "mean_pi_sret_constraint": float(df["pi_sret_constraint_score"].mean()) if len(df) else 0.0,
        "high_faithfulness_low_constraint_cases": int(len(high_faith_low_constraint)),
    }


def _write_summary(path: Path, metrics: dict, cases: pd.DataFrame) -> None:
    lines = [
        "# RAGAS Baseline Summary",
        "",
        f"- input: `{metrics['input']}`",
        f"- backend: `{metrics['backend']}`",
        f"- n_samples: {metrics['n_samples']}",
        f"- mean_ragas_faithfulness: {metrics['mean_ragas_faithfulness']:.4f}",
        f"- mean_answer_relevancy: {metrics['mean_answer_relevancy']:.4f}",
        f"- mean_context_precision: {metrics['mean_context_precision']:.4f}",
        f"- mean_context_recall: {metrics['mean_context_recall']:.4f}",
        f"- mean_pi_sret_constraint: {metrics['mean_pi_sret_constraint']:.4f}",
        f"- high faithfulness but low PI-SRET constraint cases: {metrics['high_faithfulness_low_constraint_cases']}",
        "",
        "## High-Faithfulness / Low-Constraint Cases",
        "",
        "| sample_id | ragas_faithfulness | pi_sret_constraint | violations |",
        "| --- | ---: | ---: | --- |",
    ]
    for _, row in cases.head(20).iterrows():
        lines.append(
            f"| {row['sample_id']} | {row['ragas_faithfulness']:.4f} | "
            f"{row['pi_sret_constraint_score']:.4f} | {row['pi_sret_violations']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
