from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QASample:
    sample_id: str
    question: str
    retrieved_context: str
    answer: str
    document_status: str = "unknown"
    domain: str = "materials"
    source: str = "seed"
    expected_constraints: list[str] | None = None
    faithfulness_score: float | None = None


@dataclass(frozen=True)
class ScoredSample:
    sample_id: str
    domain: str
    document_status: str
    faithfulness_score: float
    faithfulness_method: str
    constraint_score: float
    constraint_violations: list[str]
