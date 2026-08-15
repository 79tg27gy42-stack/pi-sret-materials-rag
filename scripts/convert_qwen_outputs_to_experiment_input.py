from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def convert(input_path: Path, output_csv: Path, output_jsonl: Path) -> None:
    df = pd.read_csv(input_path)
    records: list[dict] = []
    for _, row in df.iterrows():
        records.append(
            {
                "sample_id": row["sample_id"],
                "domain": "materials",
                "source": row.get("source", "qwen_output"),
                "question": row["question"],
                "retrieved_context": row["retrieved_context"],
                "answer": row["qwen_answer"],
                "document_status": row.get("document_status", "unknown"),
                "expected_constraints": row.get("expected_constraints", ""),
                "faithfulness_score": float(row["qwen_faithfulness_score"]),
                "faithfulness_method": "qwen_judge",
                "faithfulness_notes": row.get("qwen_judge_rationale", ""),
                "generation_tokens": int(row.get("generation_tokens", 0) or 0),
                "judge_tokens": int(row.get("judge_tokens", 0) or 0),
                "generation_mode": row.get("generation_mode", "safety_aware"),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_csv, index=False)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert qwen output CSV to standard H1/H2/H3 input.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()
    convert(Path(args.input), Path(args.output_csv), Path(args.output_jsonl))
    print(args.output_csv)
    print(args.output_jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
