"""Tests for H1 data validation logic."""
from __future__ import annotations

from sret_materials_rag.utils.validation import (
    ALLOWED_DOCUMENT_STATUS,
    REQUIRED_SAMPLE_FIELDS,
    validate_h1_records,
)


def test_valid_record():
    records = [
        {
            "sample_id": "test_001",
            "question": "What is the band gap of Si?",
            "retrieved_context": "Si has a band gap of 1.12 eV.",
            "answer": "The band gap of Si is 1.12 eV.",
            "document_status": "current",
        }
    ]
    errors, summary = validate_h1_records(records)
    assert errors == []
    assert summary["n_records"] == 1
    assert summary["status_counts"]["current"] == 1


def test_missing_required_field():
    records = [
        {
            "sample_id": "test_002",
            "question": "What is the band gap?",
            # missing retrieved_context, answer, document_status
        }
    ]
    errors, summary = validate_h1_records(records)
    assert len(errors) > 0
    assert any("missing required fields" in e for e in errors)


def test_invalid_document_status():
    records = [
        {
            "sample_id": "test_003",
            "question": "Q",
            "retrieved_context": "C",
            "answer": "A",
            "document_status": "invalid_status",
        }
    ]
    errors, summary = validate_h1_records(records)
    assert any("invalid document_status" in e for e in errors)


def test_all_valid_statuses():
    for status in ALLOWED_DOCUMENT_STATUS:
        records = [
            {
                "sample_id": f"test_{status}",
                "question": "Q",
                "retrieved_context": "C",
                "answer": "A",
                "document_status": status,
            }
        ]
        errors, _ = validate_h1_records(records)
        status_errors = [e for e in errors if "invalid document_status" in e]
        assert status_errors == [], f"Status {status} should be valid"


def test_duplicate_sample_ids():
    records = [
        {
            "sample_id": "dup_001",
            "question": "Q1",
            "retrieved_context": "C1",
            "answer": "A1",
            "document_status": "current",
        },
        {
            "sample_id": "dup_001",
            "question": "Q2",
            "retrieved_context": "C2",
            "answer": "A2",
            "document_status": "current",
        },
    ]
    errors, summary = validate_h1_records(records)
    assert any("duplicate" in e.lower() for e in errors)
    assert summary["duplicate_sample_ids"] == ["dup_001"]


def test_empty_text_field():
    records = [
        {
            "sample_id": "test_empty",
            "question": "   ",
            "retrieved_context": "C",
            "answer": "A",
            "document_status": "current",
        }
    ]
    errors, _ = validate_h1_records(records)
    assert any("empty" in e for e in errors)


def test_faithfulness_score_out_of_range():
    records = [
        {
            "sample_id": "test_oob",
            "question": "Q",
            "retrieved_context": "C",
            "answer": "A",
            "document_status": "current",
            "faithfulness_score": 1.5,
        }
    ]
    errors, _ = validate_h1_records(records)
    assert any("[0, 1]" in e for e in errors)
