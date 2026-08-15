from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sret_materials_rag.evaluation.faithfulness import lexical_faithfulness


@dataclass(frozen=True)
class FaithfulnessResult:
    score: float
    method: str
    rationale: str = ""


class FaithfulnessEvaluator(ABC):
    method: str

    @abstractmethod
    def score(self, *, question: str, answer: str, retrieved_context: str) -> FaithfulnessResult:
        raise NotImplementedError


class LexicalOverlapFaithfulnessEvaluator(FaithfulnessEvaluator):
    method = "lexical_overlap"

    def score(self, *, question: str, answer: str, retrieved_context: str) -> FaithfulnessResult:
        score = lexical_faithfulness(answer, retrieved_context)
        return FaithfulnessResult(
            score=score,
            method=self.method,
            rationale="Fraction of answer tokens appearing in retrieved context.",
        )


class ManualFaithfulnessEvaluator(FaithfulnessEvaluator):
    method = "manual_column"

    def score(self, *, question: str, answer: str, retrieved_context: str) -> FaithfulnessResult:
        raise RuntimeError(
            "manual_column faithfulness is loaded from data, not computed from text."
        )


class EmbeddingSimilarityFaithfulnessEvaluator(FaithfulnessEvaluator):
    method = "embedding_similarity"

    def __init__(self) -> None:
        self._model = None
        self._vectorizer = HashingVectorizer(
            n_features=4096,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
        )
        if os.environ.get("SRET_ENABLE_SENTENCE_TRANSFORMER") == "1":
            try:
                from sentence_transformers import SentenceTransformer

                model_name = os.environ.get("SRET_SENTENCE_TRANSFORMER", "sentence-transformers/all-MiniLM-L6-v2")
                self._model = SentenceTransformer(model_name)
            except Exception:
                self._model = None

    def score(self, *, question: str, answer: str, retrieved_context: str) -> FaithfulnessResult:
        if not answer.strip() or not retrieved_context.strip():
            return FaithfulnessResult(score=0.0, method=self.method, rationale="Empty answer or context.")
        if self._model is not None:
            vectors = self._model.encode([answer, retrieved_context], normalize_embeddings=True)
            score = float(np.dot(vectors[0], vectors[1]))
            return FaithfulnessResult(
                score=max(0.0, min(1.0, score)),
                method=self.method,
                rationale="Sentence-transformer cosine similarity between answer and context.",
            )
        vectors = self._vectorizer.transform([answer, retrieved_context])
        score = float(cosine_similarity(vectors[0], vectors[1])[0][0])
        return FaithfulnessResult(
            score=max(0.0, min(1.0, score)),
            method="embedding_similarity_hashing",
            rationale="Hashing-vectorizer cosine similarity fallback.",
        )


class NLIHeuristicFaithfulnessEvaluator(FaithfulnessEvaluator):
    method = "nli_heuristic"

    _NEGATION_RE = re.compile(r"\b(no|not|never|without|cannot|can't|does not|do not|isn't|aren't)\b", re.I)
    _NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

    def score(self, *, question: str, answer: str, retrieved_context: str) -> FaithfulnessResult:
        lexical = lexical_faithfulness(answer, retrieved_context)
        answer_numbers = set(self._NUMBER_RE.findall(answer))
        context_numbers = set(self._NUMBER_RE.findall(retrieved_context))
        if answer_numbers and not answer_numbers.issubset(context_numbers):
            return FaithfulnessResult(
                score=min(lexical, 0.35),
                method=self.method,
                rationale="Numeric claim in answer is not present in retrieved context.",
            )
        answer_negated = bool(self._NEGATION_RE.search(answer))
        context_negated = bool(self._NEGATION_RE.search(retrieved_context))
        if answer_negated != context_negated and lexical < 0.8:
            return FaithfulnessResult(
                score=min(lexical, 0.55),
                method=self.method,
                rationale="Potential polarity mismatch under lexical support threshold.",
            )
        if lexical >= 0.80:
            score = 1.0
        elif lexical >= 0.55:
            score = 0.75
        elif lexical >= 0.30:
            score = 0.45
        else:
            score = lexical
        return FaithfulnessResult(
            score=float(score),
            method=self.method,
            rationale="Rule-based NLI proxy using lexical support, numeric entailment, and polarity checks.",
        )


class LLMJudgeFaithfulnessEvaluator(FaithfulnessEvaluator):
    method = "llm_judge"

    def score(self, *, question: str, answer: str, retrieved_context: str) -> FaithfulnessResult:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("QWEN_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("QWEN_BASE_URL")
        model = os.environ.get("SRET_LLM_JUDGE_MODEL", os.environ.get("OPENAI_MODEL", "qwen-max"))
        if not api_key or not base_url:
            raise RuntimeError("LLM judge requires OPENAI_API_KEY/OPENAI_BASE_URL or QWEN_API_KEY/QWEN_BASE_URL.")
        payload = {
            "model": model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You judge answer faithfulness for RAG. Return JSON only with keys "
                        "score (0, 0.5, or 1) and rationale. Score 1 only if every factual "
                        "claim in the answer is directly supported by the context."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\nContext:\n{retrieved_context}\n\n"
                        f"Answer:\n{answer}\n\nReturn JSON."
                    ),
                },
            ],
        }
        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM judge request failed: {exc}") from exc
        content = raw["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", content, re.S)
        parsed = json.loads(match.group(0) if match else content)
        score = max(0.0, min(1.0, float(parsed["score"])))
        return FaithfulnessResult(
            score=score,
            method=self.method,
            rationale=str(parsed.get("rationale", "")),
        )


def build_faithfulness_evaluator(method: str) -> FaithfulnessEvaluator:
    if method == "lexical_overlap":
        return LexicalOverlapFaithfulnessEvaluator()
    if method in {"embedding_similarity", "embedding_similarity_hashing"}:
        return EmbeddingSimilarityFaithfulnessEvaluator()
    if method in {"nli", "nli_heuristic"}:
        return NLIHeuristicFaithfulnessEvaluator()
    if method == "llm_judge":
        return LLMJudgeFaithfulnessEvaluator()
    if method == "manual_column":
        return ManualFaithfulnessEvaluator()
    raise ValueError(f"Unknown faithfulness evaluator: {method}")
