from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _sentences(text: str) -> list[str]:
    text = " ".join(str(text).split())
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if len(part.strip()) >= 40]


def _claim_sentence(summary: str) -> str:
    sentences = _sentences(summary)
    if not sentences:
        return " ".join(str(summary).split())[:500]
    priority_terms = [
        "band gap",
        "formation energy",
        "stability",
        "stable",
        "superconduct",
        "transition temperature",
        "pressure",
        "synthesis",
    ]
    for sentence in sentences:
        lower = sentence.lower()
        if any(term in lower for term in priority_terms):
            return sentence
    return sentences[0]


def _make_question(title: str) -> str:
    return f"What is the main materials-science claim reported in the source titled '{title}'?"


def _context(row: pd.Series) -> str:
    return (
        f"arXiv source: {row['source_id']}\n"
        f"Title: {row['title']}\n"
        f"Published: {row['published']}\n"
        f"URL: {row['url']}\n"
        f"Abstract: {row['summary']}"
    )


def build(input_path: Path, output_csv: Path, output_jsonl: Path, *, limit: int) -> dict:
    sources = pd.read_csv(input_path).dropna(subset=["source_id", "title", "summary", "url"]).copy()
    sources = sources[sources["summary"].map(lambda value: len(str(value).split()) >= 30)].reset_index(drop=True)
    if len(sources) < 2:
        raise ValueError(f"Need at least two usable sources, got {len(sources)}")

    n_current = min(limit // 2, len(sources))
    n_noisy = min(limit - n_current, len(sources))
    records: list[dict] = []

    for idx, row in sources.head(n_current).iterrows():
        records.append(
            {
                "sample_id": f"natural_arxiv_current_{idx + 1:04d}",
                "domain": "materials",
                "source": "arxiv",
                "source_id": row["source_id"],
                "source_url": row["url"],
                "source_title": row["title"],
                "question": _make_question(row["title"]),
                "retrieved_context": _context(row),
                "answer": _claim_sentence(row["summary"]),
                "document_status": "current",
                "expected_constraints": "source_reliability_aware_material_constraints",
                "faithfulness_score": 1.0,
                "faithfulness_method": "extractive_from_same_arxiv_abstract",
                "natural_noise_type": "matched_real_abstract",
                "retrieved_source_id": row["source_id"],
                "target_source_id": row["source_id"],
            }
        )

    for idx, row in sources.head(n_noisy).iterrows():
        retrieved = sources.iloc[(idx + max(3, len(sources) // 3)) % len(sources)]
        records.append(
            {
                "sample_id": f"natural_arxiv_noisy_{idx + 1:04d}",
                "domain": "materials",
                "source": "arxiv",
                "source_id": row["source_id"],
                "source_url": row["url"],
                "source_title": row["title"],
                "question": _make_question(row["title"]),
                "retrieved_context": _context(retrieved),
                "answer": _claim_sentence(retrieved["summary"]),
                "document_status": "incomplete",
                "expected_constraints": "source_reliability_aware_material_constraints",
                "faithfulness_score": 1.0,
                "faithfulness_method": "extractive_from_mismatched_real_arxiv_abstract",
                "natural_noise_type": "retrieval_topic_mismatch",
                "retrieved_source_id": retrieved["source_id"],
                "target_source_id": row["source_id"],
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_csv, index=False)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    source_log = ROOT / "docs/natural_noisy_arxiv_source_log.md"
    _write_source_log(records, source_log)

    return {
        "n_sources": int(len(sources)),
        "n_samples": len(records),
        "n_current": n_current,
        "n_noisy": n_noisy,
        "output_csv": str(output_csv),
        "output_jsonl": str(output_jsonl),
        "source_log": str(source_log),
    }


def _write_source_log(records: list[dict], output: Path) -> None:
    seen: dict[str, dict] = {}
    for record in records:
        for key in ["target_source_id", "retrieved_source_id"]:
            source_id = record[key]
            if source_id not in seen:
                seen[source_id] = {
                    "source_id": source_id,
                    "url": record["source_url"] if source_id == record["target_source_id"] else "",
                }
    by_source: dict[str, dict] = {}
    for record in records:
        by_source.setdefault(
            record["target_source_id"],
            {
                "title": record["source_title"],
                "url": record["source_url"],
                "sample_ids": [],
            },
        )["sample_ids"].append(record["sample_id"])

    lines = [
        "# Natural Noisy arXiv Source Log",
        "",
        "All records in the natural noisy set are derived from public arXiv metadata returned by the arXiv API.",
        "",
    ]
    for source_id, item in by_source.items():
        lines.extend(
            [
                f"## {source_id}",
                "",
                f"- Title: {item['title']}",
                f"- URL: {item['url']}",
                f"- Samples: {', '.join(item['sample_ids'])}",
                "",
            ]
        )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a real-source natural noisy RAG set from arXiv metadata.")
    parser.add_argument("--input", default=str(ROOT / "data/sources/arxiv_materials_sources.csv"))
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument(
        "--output-csv",
        default=str(ROOT / "data/rag_outputs/natural_noisy_arxiv_outputs.csv"),
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(ROOT / "data/processed/natural_noisy_arxiv_outputs.jsonl"),
    )
    args = parser.parse_args()
    metrics = build(Path(args.input), Path(args.output_csv), Path(args.output_jsonl), limit=args.limit)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
