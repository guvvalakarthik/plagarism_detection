"""Hybrid lexical/dense retrieval with optional cross-encoder reranking."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class SourcePassage:
    passage_id: str
    document_id: str
    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    passage: SourcePassage
    score: float
    lexical_rank: int | None
    dense_rank: int | None
    reranker_score: float | None
    method: str


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


def chunk_document(
    document_id: str,
    text: str,
    *,
    max_tokens: int = 120,
    overlap_tokens: int = 24,
) -> list[SourcePassage]:
    if not document_id.strip():
        raise ValueError("document_id must not be empty")
    if max_tokens < 8:
        raise ValueError("max_tokens must be at least 8")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must satisfy 0 <= overlap < max_tokens")

    matches = list(_TOKEN_RE.finditer(text))
    if not matches:
        return []
    step = max_tokens - overlap_tokens
    passages: list[SourcePassage] = []
    for index in range(0, len(matches), step):
        window = matches[index : index + max_tokens]
        if not window:
            break
        start = window[0].start()
        end = window[-1].end()
        passages.append(
            SourcePassage(
                passage_id=f"{document_id}:{start}:{end}",
                document_id=document_id,
                text=text[start:end],
                start=start,
                end=end,
            )
        )
        if index + max_tokens >= len(matches):
            break
    return passages


class HybridRetriever:
    """In-memory retrieval baseline with explicit model injection."""

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        lexical_weight: float = 1.0,
        dense_weight: float = 1.0,
        rrf_constant: int = 60,
    ) -> None:
        if lexical_weight < 0 or dense_weight < 0:
            raise ValueError("retrieval weights must be non-negative")
        if lexical_weight == dense_weight == 0:
            raise ValueError("at least one retrieval channel must be enabled")
        self.embedder = embedder
        self.reranker = reranker
        self.lexical_weight = lexical_weight
        self.dense_weight = dense_weight
        self.rrf_constant = rrf_constant
        self.passages: list[SourcePassage] = []
        self._tokens: list[tuple[str, ...]] = []
        self._document_frequency: Counter[str] = Counter()
        self._average_length = 0.0
        self._embeddings: list[list[float]] = []

    def index(self, passages: Sequence[SourcePassage]) -> None:
        identifiers = [passage.passage_id for passage in passages]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("passage IDs must be unique")
        self.passages = list(passages)
        self._tokens = [_tokens(passage.text) for passage in self.passages]
        self._document_frequency = Counter(
            token for tokens in self._tokens for token in set(tokens)
        )
        self._average_length = (
            sum(len(tokens) for tokens in self._tokens) / len(self._tokens)
            if self._tokens
            else 0.0
        )
        self._embeddings = (
            self.embedder.encode([passage.text for passage in self.passages])
            if self.embedder and self.passages
            else []
        )
        if self._embeddings and len(self._embeddings) != len(self.passages):
            raise ValueError("embedder returned an unexpected number of vectors")

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        candidate_k: int = 50,
    ) -> list[RetrievalHit]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1 or candidate_k < top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        if not self.passages:
            return []

        lexical_scores = self._bm25(query)
        lexical_order = _rank(lexical_scores)
        dense_order: list[int] = []
        if self.embedder:
            query_vector = self.embedder.encode([query])[0]
            dense_order = _rank(
                [_cosine(query_vector, vector) for vector in self._embeddings]
            )

        lexical_ranks = {index: rank for rank, index in enumerate(lexical_order, 1)}
        dense_ranks = {index: rank for rank, index in enumerate(dense_order, 1)}
        fused: dict[int, float] = {}
        for index, rank in lexical_ranks.items():
            fused[index] = fused.get(index, 0.0) + self.lexical_weight / (
                self.rrf_constant + rank
            )
        for index, rank in dense_ranks.items():
            fused[index] = fused.get(index, 0.0) + self.dense_weight / (
                self.rrf_constant + rank
            )

        candidates = sorted(fused, key=lambda index: (-fused[index], index))[
            :candidate_k
        ]
        reranker_scores: dict[int, float] = {}
        if self.reranker and candidates:
            scores = self.reranker.score(
                query, [self.passages[index].text for index in candidates]
            )
            if len(scores) != len(candidates):
                raise ValueError("reranker returned an unexpected number of scores")
            reranker_scores = dict(zip(candidates, scores, strict=True))
            normalized_reranker = _min_max(reranker_scores)
            normalized_fused = _min_max({index: fused[index] for index in candidates})
            candidates.sort(
                key=lambda index: (
                    -(0.35 * normalized_fused[index] + 0.65 * normalized_reranker[index]),
                    index,
                )
            )

        method = self.method
        return [
            RetrievalHit(
                passage=self.passages[index],
                score=round(
                    (
                        reranker_scores[index]
                        if index in reranker_scores
                        else fused[index]
                    ),
                    6,
                ),
                lexical_rank=lexical_ranks.get(index),
                dense_rank=dense_ranks.get(index),
                reranker_score=(
                    round(reranker_scores[index], 6)
                    if index in reranker_scores
                    else None
                ),
                method=method,
            )
            for index in candidates[:top_k]
        ]

    @property
    def method(self) -> str:
        components = ["bm25"]
        if self.embedder:
            components.append(self.embedder.name)
        if self.reranker:
            components.append(self.reranker.name)
        return "+".join(components)

    def _bm25(self, query: str, k1: float = 1.5, b: float = 0.75) -> list[float]:
        query_tokens = set(_tokens(query))
        documents = len(self._tokens)
        scores: list[float] = []
        for tokens in self._tokens:
            frequencies = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                document_frequency = self._document_frequency.get(token, 0)
                inverse_frequency = math.log(
                    1 + (documents - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                frequency = frequencies.get(token, 0)
                denominator = frequency + k1 * (
                    1 - b + b * len(tokens) / max(1.0, self._average_length)
                )
                score += (
                    inverse_frequency * frequency * (k1 + 1) / denominator
                    if denominator
                    else 0.0
                )
            scores.append(score)
        return scores


class HashingNgramEmbedder:
    """Small deterministic dense baseline; not a semantic language model."""

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 16:
            raise ValueError("dimensions must be at least 16")
        self.dimensions = dimensions

    @property
    def name(self) -> str:
        return f"hashing-ngram-{self.dimensions}"

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        normalized = " ".join(_tokens(text))
        for size in (3, 4, 5):
            for index in range(max(0, len(normalized) - size + 1)):
                feature = normalized[index : index + size]
                digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
                position = int.from_bytes(digest, "big") % self.dimensions
                vector[position] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class SentenceTransformerEmbedder:
    """Lazy optional adapter for a real sentence-transformer bi-encoder."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    @property
    def name(self) -> str:
        return f"sentence-transformer:{self.model_name}"

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "install SourceLens with the 'semantic' extra"
                ) from error
            self._model = SentenceTransformer(self.model_name)
        encoded = self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        return encoded.tolist()


class CrossEncoderReranker:
    """Lazy optional adapter for cross-encoder pair scoring."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    @property
    def name(self) -> str:
        return f"cross-encoder:{self.model_name}"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as error:
                raise RuntimeError(
                    "install SourceLens with the 'semantic' extra"
                ) from error
            self._model = CrossEncoder(self.model_name)
        predictions = self._model.predict([(query, passage) for passage in passages])
        return [float(value) for value in predictions]


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(text))


def _rank(scores: Sequence[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )


def _min_max(scores: dict[int, float]) -> dict[int, float]:
    minimum = min(scores.values())
    maximum = max(scores.values())
    if maximum == minimum:
        return {key: 1.0 for key in scores}
    return {
        key: (value - minimum) / (maximum - minimum)
        for key, value in scores.items()
    }
