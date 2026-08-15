from __future__ import annotations

import re


_TOKEN_RE = re.compile(r"[A-Za-z0-9.+-]+")


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}


def lexical_faithfulness(answer: str, retrieved_context: str) -> float:
    """Toy baseline: fraction of answer tokens supported by retrieved context."""
    answer_tokens = tokenize(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = tokenize(retrieved_context)
    return len(answer_tokens & context_tokens) / len(answer_tokens)

