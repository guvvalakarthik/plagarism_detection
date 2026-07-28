"""Dependency-free domain models used by the engine and delivery layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Verdict = Literal["low_overlap", "review_recommended", "high_overlap"]


@dataclass(frozen=True, slots=True)
class EvidenceMatch:
    source_text: str
    candidate_text: str
    source_start: int
    source_end: int
    candidate_start: int
    candidate_end: int
    similarity: float
    match_type: Literal["exact", "near_verbatim", "lexical"]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    similarity_score: float
    verdict: Verdict
    lexical_similarity: float
    character_similarity: float
    candidate_coverage: float
    evidence: tuple[EvidenceMatch, ...]
    method: str = "pairwise_tfidf_style_v2"
    score_interpretation: str = (
        "Similarity score, not a probability or a determination of plagiarism."
    )

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["evidence"] = [match.to_dict() for match in self.evidence]
        return result
