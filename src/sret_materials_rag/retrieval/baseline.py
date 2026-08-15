from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


_TOKEN_RE = re.compile(r"[A-Za-z0-9.+-]+")
_FORMULA_RE = re.compile(r"\b(?:[A-Z][a-z]?\d*){1,}\b")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _query_formulas(query: str) -> set[str]:
    blocked = {"What", "Is", "Can", "The", "A", "An", "Does"}
    return {match.group(0) for match in _FORMULA_RE.finditer(query) if match.group(0) not in blocked}


@dataclass(frozen=True)
class RetrievedDocument:
    document: dict
    score: float
    rank: int


class BM25Retriever:
    def __init__(self, corpus: list[dict], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(str(item.get("text", ""))) for item in corpus]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )
        self.doc_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))

    def _idf(self, term: str) -> float:
        n_docs = len(self.corpus)
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def _score(self, query_terms: list[str], doc_index: int) -> float:
        if not query_terms:
            return 0.0
        term_counts = Counter(self.doc_tokens[doc_index])
        doc_length = self.doc_lengths[doc_index] or 1
        denominator_norm = 1 - self.b + self.b * doc_length / (self.avg_doc_length or 1)
        score = 0.0
        for term in query_terms:
            frequency = term_counts.get(term, 0)
            if frequency == 0:
                continue
            numerator = frequency * (self.k1 + 1)
            denominator = frequency + self.k1 * denominator_norm
            score += self._idf(term) * numerator / denominator
        return score

    def _metadata_boost(self, query: str, document: dict) -> float:
        formulas = _query_formulas(query)
        doc_formula = str(document.get("formula", ""))
        if doc_formula and doc_formula in formulas:
            return 20.0
        return 0.0

    def retrieve(self, query: str, *, top_k: int = 3) -> list[RetrievedDocument]:
        query_terms = tokenize(query)
        scored = [
            (
                self._score(query_terms, index) + self._metadata_boost(query, document),
                index,
                document,
            )
            for index, document in enumerate(self.corpus)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            RetrievedDocument(document=document, score=score, rank=rank)
            for rank, (score, _, document) in enumerate(scored[:top_k], start=1)
        ]


def retrieve(query: str, corpus: list[dict], top_k: int = 3) -> list[dict]:
    """Compatibility wrapper returning only document dictionaries."""
    return [item.document for item in BM25Retriever(corpus).retrieve(query, top_k=top_k)]
