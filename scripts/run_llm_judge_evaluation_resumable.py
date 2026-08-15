from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sret_materials_rag.evaluation.constraints import evaluate_material_qa_constraints
from sret_materials_rag.evaluation.llm_judge_evaluator import judge_answer
from sret_materials_rag.evaluation.ragas_baseline import load_rag_records


def _record_key(record: dict, index: int) -> str:
    return str(record.get("response_id") or record.get("sample_id") or index)


def _load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done.add(str(row["judge_record_key"]))
    return done


def _cohen_kappa(auto_clean: pd.Series, judge_clean: pd.Series) -> float:
    if len(auto_clean) == 0:
        return 0.0
    observed = float((auto_clean == judge_clean).mean())
    auto_yes = float(auto_clean.mean())
    judge_yes = float(judge_clean.mean())
    expected = auto_yes * judge_yes + (1 - auto_yes) * (1 - judge_yes)
    if expected == 1.0:
        return 0.0
    return (observed - expected) / (1 - expected)


def _as_bool_series(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: (
            value
            if isinstance(value, bool)
            else str(value).strip().lower() in {"1", "true", "yes", "y"}
        )
    ).astype(bool)


def _judge_model_for_run(backend: str) -> str:
    if backend == "heuristic":
        return "heuristic"
    return os.environ.get("SRET_LLM_JUDGE_MODEL", os.environ.get("OPENAI_MODEL", "qwen-max"))


def _require_llm_environment(backend: str) -> None:
    if backend != "llm":
        return
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("QWEN_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("QWEN_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("LLM judge requires OPENAI_API_KEY/OPENAI_BASE_URL or QWEN_API_KEY/QWEN_BASE_URL.")


def _check_or_write_run_config(output_dir: Path, *, input_path: Path, backend: str, judge_model: str) -> None:
    config_path = output_dir / "run_config.json"
    config = {
        "input": str(input_path),
        "backend": backend,
        "judge_model": judge_model,
    }
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(
                "Existing judge output directory has a different run_config.json. "
                f"existing={existing}, requested={config}. Use a new output directory."
            )
    else:
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def _finalize(output_dir: Path, input_path: Path, backend: str) -> dict:
    rows_path = output_dir / "llm_judge_scores.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "llm_judge_scores.csv", index=False)
    disagreements = df[df["agreement"] == 0].copy()
    disagreements.to_csv(output_dir / "disagreements.csv", index=False)
    auto_violation = df["auto_constraint_score"] < 1.0
    judge_violation = _as_bool_series(df["judge_violation"])
    metrics = {
        "input": str(input_path),
        "backend": backend,
        "n_samples": int(len(df)),
        "agreement_rate": float(df["agreement"].mean()) if len(df) else 0.0,
        "auto_violation_rate": float(auto_violation.mean()) if len(df) else 0.0,
        "judge_violation_rate": float(judge_violation.mean()) if len(df) else 0.0,
        "n_disagreements": int(len(disagreements)),
        "cohen_kappa_vs_auto_constraint": float(_cohen_kappa(auto_violation, judge_violation)),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Resumable LLM-as-Judge Constraint Evaluation",
        "",
        f"- input: `{metrics['input']}`",
        f"- backend: `{metrics['backend']}`",
        f"- n_samples: {metrics['n_samples']}",
        f"- agreement_rate: {metrics['agreement_rate']:.4f}",
        f"- auto_violation_rate: {metrics['auto_violation_rate']:.4f}",
        f"- judge_violation_rate: {metrics['judge_violation_rate']:.4f}",
        f"- cohen_kappa_vs_auto_constraint: {metrics['cohen_kappa_vs_auto_constraint']:.4f}",
        f"- n_disagreements: {metrics['n_disagreements']}",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run resumable LLM-as-Judge evaluation with per-row checkpointing.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=["heuristic", "llm"], default="llm")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    _require_llm_environment(args.backend)
    output_dir.mkdir(parents=True, exist_ok=True)
    judge_model = _judge_model_for_run(args.backend)
    _check_or_write_run_config(output_dir, input_path=input_path, backend=args.backend, judge_model=judge_model)
    rows_path = output_dir / "llm_judge_scores.jsonl"

    records = load_rag_records(input_path)
    if args.limit is not None:
        records = records[: args.limit]

    done = _load_done(rows_path)
    total = len(records)
    current_keys = {_record_key(record, index) for index, record in enumerate(records)}
    completed = len(done & current_keys)
    print(f"Loaded {total} records; already completed {completed}.", flush=True)

    with rows_path.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            key = _record_key(record, index)
            if key in done:
                continue
            last_error = ""
            for attempt in range(1, args.max_retries + 1):
                try:
                    auto = evaluate_material_qa_constraints(record["question"], record["answer"])
                    judge = judge_answer(
                        question=record["question"],
                        answer=record["answer"],
                        retrieved_context=record["retrieved_context"],
                        backend=args.backend,
                    )
                    row = {
                        **record,
                        "judge_record_key": key,
                        "auto_constraint_score": auto.score,
                        "auto_constraint_violations": ";".join(auto.violations),
                        "judge_score": judge.score,
                        "judge_violation": judge.violation,
                        "judge_backend": judge.backend,
                        "judge_model": judge.model,
                        "judge_rationale": judge.rationale,
                        "agreement": int((auto.score == 1.0) == (judge.score == 1.0)),
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    done.add(key)
                    completed += 1
                    if completed % 25 == 0 or completed == total:
                        print(f"Completed {completed}/{total}", flush=True)
                    time.sleep(args.sleep)
                    break
                except Exception as exc:  # noqa: BLE001 - keep paid API run resumable.
                    last_error = repr(exc)
                    wait = min(60.0, 2.0**attempt)
                    print(f"Record {index} attempt {attempt} failed: {last_error}; sleeping {wait}s", flush=True)
                    time.sleep(wait)
            else:
                raise RuntimeError(f"Failed record {index} after {args.max_retries} attempts: {last_error}")

    metrics = _finalize(output_dir, input_path, args.backend)
    print("Resumable LLM judge evaluation complete.")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
