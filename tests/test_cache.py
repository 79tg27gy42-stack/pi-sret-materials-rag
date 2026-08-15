"""Tests for the score caching utility."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from sret_materials_rag.utils.cache import (
    load_cache,
    make_cache_key,
    save_cache_entry,
)


def test_make_cache_key_deterministic():
    k1 = make_cache_key("q", "a", "c", "llm_judge")
    k2 = make_cache_key("q", "a", "c", "llm_judge")
    assert k1 == k2
    assert len(k1) == 16


def test_make_cache_key_differs_on_input():
    k1 = make_cache_key("q1", "a", "c", "llm_judge")
    k2 = make_cache_key("q2", "a", "c", "llm_judge")
    assert k1 != k2


def test_make_cache_key_differs_on_method():
    k1 = make_cache_key("q", "a", "c", "llm_judge")
    k2 = make_cache_key("q", "a", "c", "lexical_overlap")
    assert k1 != k2


def test_load_cache_empty(tmp_path):
    cache = load_cache(tmp_path / "nonexistent.jsonl")
    assert cache == {}


def test_save_and_load(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    record = {
        "_cache_key": "abc123",
        "score": 0.85,
        "method": "llm_judge",
    }
    save_cache_entry(cache_path, record)
    cache = load_cache(cache_path)
    assert "abc123" in cache
    assert cache["abc123"]["score"] == 0.85


def test_multiple_entries(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    for i in range(5):
        save_cache_entry(cache_path, {"_cache_key": f"key_{i}", "score": i / 5.0})
    cache = load_cache(cache_path)
    assert len(cache) == 5
    assert cache["key_3"]["score"] == 0.6
