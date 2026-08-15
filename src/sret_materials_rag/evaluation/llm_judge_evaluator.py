from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sret_materials_rag.evaluation.constraints import evaluate_material_qa_constraints
from sret_materials_rag.evaluation.ragas_baseline import load_rag_records


JUDGE_PROMPT = """You are a materials-science reviewer evaluating a RAG answer.
Assess the answer's scientific validity using your materials-domain knowledge, rather than reproducing a supplied checklist or trusting the retrieved text by default. Distinguish an asserted conclusion from a value quoted only for criticism. Consider quantities, units, reference states, phase qualifications, mutually dependent properties, and whether categorical experimental claims have property-appropriate evidence. Explain the concrete scientific reason for your decision.

Return JSON only: {"violation": true/false, "score": 0 or 1, "rationale": "..."}.
"""


@dataclass(frozen=True)
class JudgeResult:
    violation: bool
    score: float
    rationale: str
    backend: str
    model: str = ""


def judge_answer(
    *,
    question: str,
    answer: str,
    retrieved_context: str,
    backend: str = "heuristic",
) -> JudgeResult:
    if backend == "heuristic":
        result = evaluate_material_qa_constraints(question, answer)
        return JudgeResult(
            violation=bool(result.violations),
            score=result.score,
            rationale=";".join(result.violations) if result.violations else "no automatic constraint violation",
            backend="heuristic",
            model="heuristic",
        )
    if backend == "llm":
        return _judge_with_llm(question=question, answer=answer, retrieved_context=retrieved_context)
    raise ValueError(f"Unknown judge backend: {backend}")


def run_llm_judge_evaluation(
    *,
    input_path: Path,
    output_dir: Path,
    backend: str = "heuristic",
    limit: int | None = None,
) -> dict:
    records = load_rag_records(input_path)
    if limit is not None:
        records = records[:limit]
    rows = []
    for record in records:
        auto = evaluate_material_qa_constraints(record["question"], record["answer"])
        judge = judge_answer(
            question=record["question"],
            answer=record["answer"],
            retrieved_context=record["retrieved_context"],
            backend=backend,
        )
        rows.append(
            {
                **record,
                "auto_constraint_score": auto.score,
                "auto_constraint_violations": ";".join(auto.violations),
                "judge_score": judge.score,
                "judge_violation": judge.violation,
                "judge_backend": judge.backend,
                "judge_model": judge.model,
                "judge_rationale": judge.rationale,
                "agreement": int((auto.score == 1.0) == (judge.score == 1.0)),
            }
        )
    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "llm_judge_scores.csv", index=False)
    disagreements = df[df["agreement"] == 0].copy()
    disagreements.to_csv(output_dir / "disagreements.csv", index=False)
    metrics = {
        "input": str(input_path),
        "backend": backend,
        "n_samples": int(len(df)),
        "agreement_rate": float(df["agreement"].mean()) if len(df) else 0.0,
        "auto_violation_rate": float((df["auto_constraint_score"] < 1.0).mean()) if len(df) else 0.0,
        "judge_violation_rate": float(df["judge_violation"].mean()) if len(df) else 0.0,
        "n_disagreements": int(len(disagreements)),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(output_dir / "summary.md", metrics, disagreements)
    return metrics


def _judge_with_llm(*, question: str, answer: str, retrieved_context: str) -> JudgeResult:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("QWEN_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("QWEN_BASE_URL")
    model = os.environ.get("SRET_LLM_JUDGE_MODEL", os.environ.get("OPENAI_MODEL", "qwen-max"))
    if not api_key or not base_url:
        raise RuntimeError("LLM judge requires OPENAI_API_KEY/OPENAI_BASE_URL or QWEN_API_KEY/QWEN_BASE_URL.")
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nRetrieved context:\n{retrieved_context}\n\nAnswer:\n{answer}",
            },
        ],
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        raw = json.loads(response.read().decode("utf-8"))
    content = raw["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", content, re.S)
    parsed = json.loads(match.group(0) if match else content)
    violation = _parse_bool(parsed.get("violation"))
    score = float(parsed.get("score", 0 if violation else 1))
    return JudgeResult(
        violation=violation,
        score=max(0.0, min(1.0, score)),
        rationale=str(parsed.get("rationale", "")),
        backend="llm",
        model=model,
    )


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0"}:
        return False
    raise ValueError(f"Cannot parse boolean judge value: {value!r}")


def _write_summary(path: Path, metrics: dict, disagreements: pd.DataFrame) -> None:
    lines = [
        "# LLM-as-Judge Constraint Evaluation",
        "",
        f"- input: `{metrics['input']}`",
        f"- backend: `{metrics['backend']}`",
        f"- n_samples: {metrics['n_samples']}",
        f"- agreement_rate: {metrics['agreement_rate']:.4f}",
        f"- auto_violation_rate: {metrics['auto_violation_rate']:.4f}",
        f"- judge_violation_rate: {metrics['judge_violation_rate']:.4f}",
        f"- n_disagreements: {metrics['n_disagreements']}",
        "",
        "## Disagreements",
        "",
        "| sample_id | auto_violations | judge_score | judge_rationale |",
        "| --- | --- | ---: | --- |",
    ]
    for _, row in disagreements.head(20).iterrows():
        rationale = str(row["judge_rationale"]).replace("|", "/")[:180]
        lines.append(
            f"| {row['sample_id']} | {row['auto_constraint_violations']} | "
            f"{row['judge_score']:.1f} | {rationale} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
