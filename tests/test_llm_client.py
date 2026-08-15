from __future__ import annotations

from sret_materials_rag.utils.llm_client import parse_json_object


def test_parse_json_object_plain() -> None:
    assert parse_json_object('{"answer": "ok"}') == {"answer": "ok"}


def test_parse_json_object_fenced() -> None:
    assert parse_json_object('```json\n{"answer": "ok"}\n```') == {"answer": "ok"}
