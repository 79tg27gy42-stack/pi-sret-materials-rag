"""Utility for caching LLM Judge scores to avoid redundant API calls."""
from __future__ import annotations

import json
from pathlib import Path


def load_cache(path: str | Path) -> dict[str, dict]:
    """Load cached scores from a JSONL file.

    Returns a dict keyed by a cache key string.
    """
    path = Path(path)
    if not path.exists():
        return {}
    cache: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        key = record.get("_cache_key", "")
        if key:
            cache[key] = record
    return cache


def save_cache_entry(path: str | Path, record: dict) -> None:
    """Append a single record to the cache JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_cache_key(question: str, answer: str, context: str, method: str) -> str:
    """Create a deterministic cache key from inputs and method."""
    import hashlib
    raw = f"{method}||{question}||{answer}||{context}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
