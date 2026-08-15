from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


_FORMULA_RE = re.compile(r"\b(?:[A-Z][a-z]?\d*){1,}\b")


def _formula_from_question(question: str) -> str:
    blocked = {"What", "Is", "Can", "The", "A", "An", "Does"}
    for match in _FORMULA_RE.finditer(str(question)):
        formula = match.group(0)
        if formula not in blocked:
            return formula
    return ""


def _dedupe_documents(documents: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for document in documents:
        key = (str(document.get("source", "")), str(document.get("text", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(document)
    return deduped


def build_materials_project_corpus(path: str | Path) -> list[dict]:
    df = pd.read_csv(path)
    documents: list[dict] = []
    for index, row in df.iterrows():
        material_id = row["material_id"]
        formula = row["formula_pretty"]
        text = (
            f"Materials Project reports {formula} ({material_id}). "
            f"Band gap: {row.get('band_gap')} eV. "
            f"Formation energy per atom: {row.get('formation_energy_per_atom')} eV/atom. "
            f"Energy above hull: {row.get('energy_above_hull')} eV/atom. "
            f"Stable: {row.get('is_stable')}."
        )
        documents.append(
            {
                "doc_id": f"mp_{index + 1:04d}",
                "source": row.get("source_url", ""),
                "source_type": "materials_project",
                "document_status": "current",
                "priority": 3,
                "formula": formula,
                "text": text,
            }
        )
    return documents


def build_arxiv_corpus(path: str | Path) -> list[dict]:
    df = pd.read_csv(path)
    documents: list[dict] = []
    for index, row in df.iterrows():
        documents.append(
            {
                "doc_id": f"arxiv_{index + 1:04d}",
                "source": row.get("url", ""),
                "source_type": "arxiv",
                "document_status": "current",
                "priority": 2,
                "formula": "",
                "text": f"{row.get('title', '')}. {row.get('summary', '')}",
            }
        )
    return documents


def build_stress_context_corpus(path: str | Path) -> list[dict]:
    df = pd.read_csv(path)
    documents: list[dict] = []
    for index, row in df.iterrows():
        documents.append(
            {
                "doc_id": f"h1ctx_{index + 1:04d}_{row['sample_id']}",
                "source": row.get("source", ""),
                "source_type": "h1_stress_context",
                "document_status": row.get("document_status", "unknown"),
                "priority": {
                    "outdated_or_incorrect": 4,
                    "incomplete": 3,
                    "current": 2,
                }.get(row.get("document_status", "unknown"), 1),
                "formula": _formula_from_question(row.get("question", "")),
                "text": row.get("retrieved_context", ""),
            }
        )
    return _dedupe_documents(documents)


def build_rag_corpus(
    *,
    materials_project_path: str | Path,
    arxiv_path: str | Path,
    stress_context_path: str | Path | None = None,
    include_stress_contexts: bool = True,
) -> list[dict]:
    documents: list[dict] = []
    documents.extend(build_materials_project_corpus(materials_project_path))
    documents.extend(build_arxiv_corpus(arxiv_path))
    if include_stress_contexts and stress_context_path is not None:
        documents.extend(build_stress_context_corpus(stress_context_path))
    return _dedupe_documents(documents)
