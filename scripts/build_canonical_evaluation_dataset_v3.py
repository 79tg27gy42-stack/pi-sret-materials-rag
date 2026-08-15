from __future__ import annotations

import json
from pathlib import Path

from build_canonical_evaluation_dataset import (
    ROOT,
    _assign_base_task_ids,
    _breakdown,
    _build_report,
    _load_source_records,
    _stratified_sample,
    _write_jsonl,
    _write_markdown_report,
    _write_schema,
)

OUTPUT_DIR = ROOT / "data/processed/canonical_v3"

SOURCE_FILES_V2 = [
    {
        "path": "data/processed/h1_expanded_diverse_with_faithfulness.jsonl",
        "source_dataset": "controlled_diverse_prefilled",
        "dataset_role": "controlled_candidate",
        "model": "prefilled_context_support",
        "mode": "diverse_prefilled",
        "include_primary": True,
        "include_llm_pool": False,
    },
    {
        "path": "data/processed/mp_balanced_constraint_dataset_v3.jsonl",
        "source_dataset": "mp_balanced_constraints_v3",
        "dataset_role": "mp_derived_balanced_controlled",
        "model": "mp_property_derived",
        "mode": "controlled_balanced",
        "include_primary": True,
        "include_llm_pool": True,
    },
    {
        "path": "data/processed/supplementary_constraint_coverage.jsonl",
        "source_dataset": "supplementary_constraint_coverage",
        "dataset_role": "supplementary_constraint_probe",
        "model": "llm_diverse_probe",
        "mode": "constraint_coverage",
        "include_primary": True,
        "include_llm_pool": True,
    },
    {
        "path": "data/processed/local_rag_baseline_outputs.jsonl",
        "source_dataset": "local_rag_naive",
        "dataset_role": "local_rag",
        "model": "local_rule_rag",
        "mode": "naive_context",
        "include_primary": True,
        "include_llm_pool": False,
    },
    {
        "path": "data/processed/local_rag_official_outputs.jsonl",
        "source_dataset": "local_rag_official",
        "dataset_role": "official_only_rag",
        "model": "local_rule_rag",
        "mode": "official_only",
        "include_primary": True,
        "include_llm_pool": False,
    },
    {
        "path": "data/processed/qwen_max_naive_full_standard.jsonl",
        "source_dataset": "qwen_max_naive",
        "dataset_role": "llm_generation",
        "model": "qwen-max",
        "mode": "naive_context",
        "include_primary": True,
        "include_llm_pool": True,
    },
    {
        "path": "data/processed/qwen_max_full_standard.jsonl",
        "source_dataset": "qwen_max_safety",
        "dataset_role": "llm_generation",
        "model": "qwen-max",
        "mode": "safety_aware",
        "include_primary": True,
        "include_llm_pool": True,
    },
    {
        "path": "data/processed/deepseek_r1_naive_full_standard.jsonl",
        "source_dataset": "deepseek_r1_naive",
        "dataset_role": "llm_generation",
        "model": "deepseek-r1",
        "mode": "naive_context",
        "include_primary": True,
        "include_llm_pool": True,
    },
    {
        "path": "data/processed/deepseek_r1_safety_full_standard.jsonl",
        "source_dataset": "deepseek_r1_safety",
        "dataset_role": "llm_generation",
        "model": "deepseek-r1",
        "mode": "safety_aware",
        "include_primary": True,
        "include_llm_pool": True,
    },
    {
        "path": "data/processed/natural_noisy_arxiv_outputs_v2.jsonl",
        "source_dataset": "natural_noisy_arxiv_v2",
        "dataset_role": "natural_source_linked",
        "model": "extractive_real_arxiv",
        "mode": "matched_and_mismatched",
        "include_primary": True,
        "include_llm_pool": True,
    },
]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in SOURCE_FILES_V2:
        records.extend(_load_source_records(spec))
    records = _assign_base_task_ids(records)
    full_primary = [r for r in records if r["include_primary_manifest"]]
    primary = _balanced_primary_split(full_primary, max_per_constraint_family=500)
    llm_pool = [r for r in primary if r["include_llm_judge_pool"]]
    ragas_372 = _stratified_sample(
        llm_pool,
        n=372,
        key_fields=["dataset_role", "source_dataset", "model", "mode", "document_status", "constraint_family"],
    )
    dual_200 = _stratified_sample(
        llm_pool,
        n=200,
        key_fields=["dataset_role", "source_dataset", "model", "mode", "document_status", "constraint_family"],
    )

    _write_jsonl(OUTPUT_DIR / "evaluation_manifest_v3.jsonl", records)
    _write_jsonl(OUTPUT_DIR / "full_response_manifest_v3.jsonl", full_primary)
    _write_jsonl(OUTPUT_DIR / "primary_response_manifest_v3.jsonl", primary)
    _write_jsonl(OUTPUT_DIR / "llm_judge_full_pool_v3.jsonl", llm_pool)
    _write_jsonl(OUTPUT_DIR / "ragas_stratified_372_v3.jsonl", ragas_372)
    _write_jsonl(OUTPUT_DIR / "dual_judge_stratified_200_v3.jsonl", dual_200)

    report = _build_report(records, primary, llm_pool, ragas_372, dual_200)
    report["version"] = "canonical_v3"
    report["full_response_breakdown"] = _breakdown(full_primary)
    report["sizes"]["full_response_manifest"] = len(full_primary)
    report["v3_changes"] = [
        "Expanded natural arXiv source-linked records from 100 to 500 using 254 unique public arXiv sources.",
        "Re-collected 1000 live Materials Project rows with the updated API key.",
        "Added 8000 MP-derived balanced controlled records from the 1000 live Materials Project rows.",
        "Added 330 supplementary constraint coverage probe records for low-frequency constraint families.",
        "RAGAS and dual-judge subsets are re-stratified over source_dataset in addition to role/model/mode/status/constraint.",
        "Primary split uses a 500-record cap per constraint family to let the expanded MP data enter the main dataset while keeping band_gap below 20%.",
    ]
    (OUTPUT_DIR / "dataset_quality_report_v3.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_markdown_report(OUTPUT_DIR / "dataset_card_v3.md", report)
    _write_schema(OUTPUT_DIR / "schema_v3.md")

    print("Canonical evaluation dataset v3 complete.")
    for key, value in report["sizes"].items():
        print(f"{key}: {value}")
    print("primary constraint breakdown:", report["primary_breakdown"]["by_constraint_family"])
    return 0


def _balanced_primary_split(records: list[dict], *, max_per_constraint_family: int) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for record in records:
        family = record.get("constraint_family", "none")
        if family == "none":
            continue
        buckets.setdefault(family, []).append(record)
    selected: list[dict] = []
    for family, values in sorted(buckets.items()):
        values.sort(
            key=lambda r: (
                r["dataset_role"] != "natural_source_linked",
                r["dataset_role"] != "mp_derived_balanced_controlled",
                r["source_dataset"],
                r["model"],
                r["mode"],
                r["response_id"],
            )
        )
        selected.extend(values[:max_per_constraint_family])
    return sorted(selected, key=lambda r: (r["constraint_family"], r["dataset_role"], r["source_dataset"], r["response_id"]))


if __name__ == "__main__":
    raise SystemExit(main())
