from __future__ import annotations

from collections import Counter


ALLOWED_DOCUMENT_STATUS = {
    "current",
    "outdated_or_incorrect",
    "incomplete",
    "unknown",
}

REQUIRED_SAMPLE_FIELDS = {
    "sample_id",
    "question",
    "retrieved_context",
    "answer",
    "document_status",
}


def validate_h1_records(records: list[dict]) -> tuple[list[str], dict]:
    errors: list[str] = []
    sample_ids: list[str] = []
    statuses: list[str] = []

    for index, record in enumerate(records, start=1):
        missing = sorted(REQUIRED_SAMPLE_FIELDS - set(record))
        if missing:
            errors.append(f"line {index}: missing required fields {missing}")

        sample_id = record.get("sample_id")
        if sample_id:
            sample_ids.append(str(sample_id))

        status = record.get("document_status")
        if status:
            statuses.append(str(status))
            if status not in ALLOWED_DOCUMENT_STATUS:
                errors.append(
                    f"line {index}: invalid document_status {status!r}; "
                    f"expected one of {sorted(ALLOWED_DOCUMENT_STATUS)}"
                )

        for field in ["question", "retrieved_context", "answer"]:
            value = record.get(field)
            if value is not None and not str(value).strip():
                errors.append(f"line {index}: field {field!r} is empty")

        faithfulness_score = record.get("faithfulness_score")
        if faithfulness_score is not None:
            try:
                numeric_score = float(faithfulness_score)
            except (TypeError, ValueError):
                errors.append(f"line {index}: faithfulness_score must be numeric")
            else:
                if not 0.0 <= numeric_score <= 1.0:
                    errors.append(f"line {index}: faithfulness_score must be in [0, 1]")

    duplicate_ids = sorted(
        sample_id for sample_id, count in Counter(sample_ids).items() if count > 1
    )
    if duplicate_ids:
        errors.append(f"duplicate sample_id values: {duplicate_ids}")

    summary = {
        "n_records": len(records),
        "status_counts": dict(Counter(statuses)),
        "duplicate_sample_ids": duplicate_ids,
    }
    return errors, summary
