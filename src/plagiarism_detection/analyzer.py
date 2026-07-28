"""Transparent pairwise similarity with human-reviewable evidence.

This is deliberately a similarity system, not an authorship or misconduct
classifier. A score can prioritize review; only a person with source and
citation context can determine plagiarism.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from .models import AnalysisResult, EvidenceMatch, Verdict

_TOKEN_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
_SEGMENT_RE = re.compile(r"[^\n.!?]+(?:[.!?]+|$)|[^\n]+", re.MULTILINE)
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Versioned decision policy, calibrated by ``evaluate.py``."""

    review_threshold: float = 0.36
    high_overlap_threshold: float = 0.64
    evidence_threshold: float = 0.30
    max_evidence: int = 8
    min_document_tokens: int = 3
    min_segment_tokens: int = 4
    max_segment_tokens: int = 80

    def __post_init__(self) -> None:
        if not 0 <= self.review_threshold < self.high_overlap_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= review < high <= 1")
        if not 0 <= self.evidence_threshold <= 1:
            raise ValueError("evidence_threshold must be between 0 and 1")
        if self.max_evidence < 1:
            raise ValueError("max_evidence must be positive")


@dataclass(frozen=True, slots=True)
class _Segment:
    text: str
    start: int
    end: int
    tokens: tuple[str, ...]


class PairwiseAnalyzer:
    """Compare one candidate against one source using symmetric pair features."""

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        self.config = config or AnalysisConfig()

    def analyze(self, source: str, candidate: str) -> AnalysisResult:
        source_tokens = _tokens(source)
        candidate_tokens = _tokens(candidate)
        self._validate(source, candidate, source_tokens, candidate_tokens)

        lexical = _lexical_similarity(source_tokens, candidate_tokens)
        character = _character_similarity(source, candidate)
        evidence = self._evidence(source, candidate)
        coverage = self._candidate_coverage(candidate_tokens, evidence)

        global_score = 0.68 * lexical + 0.32 * character
        evidence_signal = evidence[0].similarity if evidence else 0.0
        score = _clamp(max(global_score, 0.72 * evidence_signal + 0.28 * coverage))
        verdict = self._verdict(score)

        return AnalysisResult(
            similarity_score=_rounded(score),
            verdict=verdict,
            lexical_similarity=_rounded(lexical),
            character_similarity=_rounded(character),
            candidate_coverage=_rounded(coverage),
            evidence=tuple(evidence),
        )

    def _validate(
        self,
        source: str,
        candidate: str,
        source_tokens: tuple[str, ...],
        candidate_tokens: tuple[str, ...],
    ) -> None:
        if not isinstance(source, str) or not isinstance(candidate, str):
            raise TypeError("source and candidate must be strings")
        if not source.strip() or not candidate.strip():
            raise ValueError("source and candidate must not be empty")
        minimum = self.config.min_document_tokens
        if len(source_tokens) < minimum or len(candidate_tokens) < minimum:
            raise ValueError(f"each document must contain at least {minimum} words")

    def _verdict(self, score: float) -> Verdict:
        if score >= self.config.high_overlap_threshold:
            return "high_overlap"
        if score >= self.config.review_threshold:
            return "review_recommended"
        return "low_overlap"

    def _evidence(self, source: str, candidate: str) -> list[EvidenceMatch]:
        source_segments = _segments(source, self.config)
        candidate_segments = _segments(candidate, self.config)
        matches: list[EvidenceMatch] = []

        for candidate_segment in candidate_segments:
            best: tuple[float, _Segment] | None = None
            for source_segment in source_segments:
                score = _segment_similarity(
                    source_segment.tokens,
                    candidate_segment.tokens,
                    source_segment.text,
                    candidate_segment.text,
                )
                if best is None or score > best[0]:
                    best = (score, source_segment)

            if best is None or best[0] < self.config.evidence_threshold:
                continue
            similarity, source_segment = best
            matches.append(
                EvidenceMatch(
                    source_text=source_segment.text,
                    candidate_text=candidate_segment.text,
                    source_start=source_segment.start,
                    source_end=source_segment.end,
                    candidate_start=candidate_segment.start,
                    candidate_end=candidate_segment.end,
                    similarity=_rounded(similarity),
                    match_type=_match_type(
                        similarity, source_segment.tokens, candidate_segment.tokens
                    ),
                )
            )

        matches.sort(
            key=lambda match: (
                -match.similarity,
                match.candidate_start,
                match.source_start,
            )
        )
        return _non_overlapping(matches, self.config.max_evidence)

    @staticmethod
    def _candidate_coverage(
        candidate_tokens: tuple[str, ...], evidence: list[EvidenceMatch]
    ) -> float:
        if not evidence:
            return 0.0
        matched = {
            token for match in evidence for token in _tokens(match.candidate_text) if len(token) > 2
        }
        meaningful = {token for token in candidate_tokens if len(token) > 2}
        return len(matched) / len(meaningful) if meaningful else 0.0


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _SPACE_RE.sub(" ", normalized).strip()


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(_normalize(text)))


def _features(tokens: tuple[str, ...]) -> Counter[str]:
    features: Counter[str] = Counter(f"w:{token}" for token in tokens)
    features.update(f"b:{left}_{right}" for left, right in zip(tokens, tokens[1:], strict=False))
    return features


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _lexical_similarity(source_tokens: tuple[str, ...], candidate_tokens: tuple[str, ...]) -> float:
    return _cosine(_features(source_tokens), _features(candidate_tokens))


def _char_features(text: str) -> Counter[str]:
    compact = re.sub(r"[^\w]+", " ", _normalize(text))
    features: Counter[str] = Counter()
    for size in (3, 4, 5):
        features.update(
            f"c{size}:{compact[index : index + size]}"
            for index in range(max(0, len(compact) - size + 1))
        )
    return features


def _character_similarity(source: str, candidate: str) -> float:
    return _cosine(_char_features(source), _char_features(candidate))


def _containment(source_tokens: tuple[str, ...], candidate_tokens: tuple[str, ...]) -> float:
    source = Counter(token for token in source_tokens if len(token) > 2)
    candidate = Counter(token for token in candidate_tokens if len(token) > 2)
    denominator = sum(candidate.values())
    if not denominator:
        return 0.0
    return sum(min(count, source.get(token, 0)) for token, count in candidate.items()) / denominator


def _segment_similarity(
    source_tokens: tuple[str, ...],
    candidate_tokens: tuple[str, ...],
    source_text: str,
    candidate_text: str,
) -> float:
    lexical = _lexical_similarity(source_tokens, candidate_tokens)
    character = _character_similarity(source_text, candidate_text)
    containment = _containment(source_tokens, candidate_tokens)
    return _clamp(0.55 * lexical + 0.30 * character + 0.15 * containment)


def _segments(text: str, config: AnalysisConfig) -> list[_Segment]:
    result: list[_Segment] = []
    for match in _SEGMENT_RE.finditer(text):
        raw = match.group(0)
        left_trim = len(raw) - len(raw.lstrip())
        value = raw.strip()
        if not value:
            continue
        start = match.start() + left_trim
        tokens = _tokens(value)
        if len(tokens) < config.min_segment_tokens:
            continue
        if len(tokens) <= config.max_segment_tokens:
            result.append(_Segment(value, start, start + len(value), tokens))
            continue
        result.extend(_window_segment(value, start, tokens, config.max_segment_tokens))

    if not result:
        normalized_tokens = _tokens(text)
        if normalized_tokens:
            stripped = text.strip()
            start = text.find(stripped)
            result.append(_Segment(stripped, start, start + len(stripped), normalized_tokens))
    return result


def _window_segment(
    text: str, start: int, tokens: tuple[str, ...], size: int
) -> Iterable[_Segment]:
    token_matches = list(_TOKEN_RE.finditer(text))
    step = max(1, size // 2)
    for index in range(0, len(token_matches), step):
        window = token_matches[index : index + size]
        if not window:
            break
        window_start = window[0].start()
        window_end = window[-1].end()
        value = text[window_start:window_end]
        yield _Segment(
            value,
            start + window_start,
            start + window_end,
            tokens[index : index + size],
        )
        if index + size >= len(token_matches):
            break


def _match_type(
    similarity: float,
    source_tokens: tuple[str, ...],
    candidate_tokens: tuple[str, ...],
) -> str:
    if source_tokens == candidate_tokens:
        return "exact"
    if similarity >= 0.72:
        return "near_verbatim"
    return "lexical"


def _non_overlapping(matches: list[EvidenceMatch], limit: int) -> list[EvidenceMatch]:
    selected: list[EvidenceMatch] = []
    for match in matches:
        overlaps = any(
            match.candidate_start < existing.candidate_end
            and existing.candidate_start < match.candidate_end
            for existing in selected
        )
        if not overlaps:
            selected.append(match)
        if len(selected) == limit:
            break
    return selected


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _rounded(value: float) -> float:
    return round(_clamp(value), 4)
