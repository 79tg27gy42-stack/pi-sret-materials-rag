from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sret_materials_rag.evaluation.llm_judge_evaluator import run_llm_judge_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run independent scientific-constraint LLM-as-Judge evaluation.")
    parser.add_argument("--input", default=str(ROOT / "data/processed/deepseek_r1_naive_full_standard.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/llm_judge_deepseek_r1_naive"))
    parser.add_argument("--backend", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    metrics = run_llm_judge_evaluation(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        backend=args.backend,
        limit=args.limit,
    )
    print("LLM judge evaluation complete.")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
