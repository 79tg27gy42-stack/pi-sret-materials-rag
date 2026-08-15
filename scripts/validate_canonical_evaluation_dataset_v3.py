from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data/processed/canonical_v3"

FILES = {
    "evaluation_manifest_v3.jsonl": None,
    "full_response_manifest_v3.jsonl": None,
    "primary_response_manifest_v3.jsonl": None,
    "llm_judge_full_pool_v3.jsonl": None,
    "ragas_stratified_372_v3.jsonl": 372,
    "dual_judge_stratified_200_v3.jsonl": 200,
}

REQUIRED_FIELDS = {
    "response_id",
    "raw_sample_id",
    "base_task_id",
    "domain",
    "source_dataset",
    "dataset_role",
    "source_kind",
    "model",
    "mode",
    "question",
    "retrieved_context",
    "answer",
    "document_status",
    "constraint_family",
}


def main() -> int:
    errors: list[str] = []
    data = {name: _load(DATASET_DIR / name) for name in FILES}
    for name, expected in FILES.items():
        rows = data[name]
        if expected is not None and len(rows) != expected:
            errors.append(f"{name}: expected {expected}, got {len(rows)}")
        errors.extend(_validate_rows(name, rows))

    eval_ids = {r["response_id"] for r in data["evaluation_manifest_v3.jsonl"]}
    full_ids = {r["response_id"] for r in data["full_response_manifest_v3.jsonl"]}
    primary_ids = {r["response_id"] for r in data["primary_response_manifest_v3.jsonl"]}
    llm_ids = {r["response_id"] for r in data["llm_judge_full_pool_v3.jsonl"]}
    ragas_ids = {r["response_id"] for r in data["ragas_stratified_372_v3.jsonl"]}
    dual_ids = {r["response_id"] for r in data["dual_judge_stratified_200_v3.jsonl"]}

    for child_name, child_ids, parent_name, parent_ids in [
        ("full", full_ids, "evaluation", eval_ids),
        ("primary", primary_ids, "full", full_ids),
        ("llm", llm_ids, "primary", primary_ids),
        ("ragas", ragas_ids, "llm", llm_ids),
        ("dual", dual_ids, "llm", llm_ids),
    ]:
        if not child_ids <= parent_ids:
            errors.append(f"{child_name} ids are not a subset of {parent_name} ids")

    primary = data["primary_response_manifest_v3.jsonl"]
    fam = Counter(r["constraint_family"] for r in primary)
    if fam["band_gap"] > 0.20 * len(primary):
        errors.append(f"band_gap is too dominant in primary split: {fam['band_gap']} / {len(primary)}")
    if len(fam) < 8:
        errors.append(f"primary split covers only {len(fam)} constraint families")
    if len({r["base_task_id"] for r in primary}) < 1000:
        errors.append("primary split has fewer than 1000 unique base tasks")
    if Counter(r["dataset_role"] for r in primary)["natural_source_linked"] < 400:
        errors.append("primary split has fewer than 400 natural source-linked records")

    report = {
        "valid": not errors,
        "errors": errors,
        "sizes": {name: len(rows) for name, rows in data.items()},
        "primary_breakdown": _breakdown(primary),
        "ragas_breakdown": _breakdown(data["ragas_stratified_372_v3.jsonl"]),
        "dual_breakdown": _breakdown(data["dual_judge_stratified_200_v3.jsonl"]),
    }
    (DATASET_DIR / "validation_report_v3.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if errors:
        print("Canonical v3 validation failed.")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Canonical v3 validation passed.")
    for name, size in report["sizes"].items():
        print(f"{name}: {size}")
    print("primary constraint families:", report["primary_breakdown"]["by_constraint_family"])
    return 0


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _validate_rows(name: str, rows: list[dict]) -> list[str]:
    errors = []
    ids = [r.get("response_id", "") for r in rows]
    dupes = [value for value, count in Counter(ids).items() if count > 1]
    if dupes:
        errors.append(f"{name}: duplicate response_id count={len(dupes)}")
    for i, row in enumerate(rows):
        missing = sorted(field for field in REQUIRED_FIELDS if not row.get(field))
        if missing:
            errors.append(f"{name}: row {i} missing {missing}")
            break
    return errors


def _breakdown(rows: list[dict]) -> dict:
    return {
        "by_dataset_role": dict(Counter(r["dataset_role"] for r in rows)),
        "by_source_dataset": dict(Counter(r["source_dataset"] for r in rows)),
        "by_model": dict(Counter(r["model"] for r in rows)),
        "by_mode": dict(Counter(r["mode"] for r in rows)),
        "by_document_status": dict(Counter(r["document_status"] for r in rows)),
        "by_constraint_family": dict(Counter(r["constraint_family"] for r in rows)),
        "by_source_kind": dict(Counter(r["source_kind"] for r in rows)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
