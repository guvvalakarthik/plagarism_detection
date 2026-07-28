"""Passage-level and adversarial evaluation primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

from .models import AnalysisResult


@dataclass(frozen=True, slots=True)
class PassageAnnotation:
    candidate_start: int
    candidate_end: int
    source_start: int | None = None
    source_end: int | None = None

    def __post_init__(self) -> None:
        if self.candidate_start < 0 or self.candidate_end <= self.candidate_start:
            raise ValueError("candidate annotation must be a positive half-open interval")
        source_values = (self.source_start, self.source_end)
        if (source_values[0] is None) != (source_values[1] is None):
            raise ValueError("source interval must provide both start and end")
        if (
            self.source_start is not None
            and self.source_end is not None
            and (self.source_start < 0 or self.source_end <= self.source_start)
        ):
            raise ValueError("source annotation must be a positive half-open interval")


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    example_id: str
    group_id: str
    source: str
    candidate: str
    annotations: tuple[PassageAnnotation, ...]
    category: str = "unspecified"

    @property
    def has_reuse(self) -> bool:
        return bool(self.annotations)


class Analyzer(Protocol):
    def analyze(self, source: str, candidate: str) -> AnalysisResult: ...


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


@dataclass(frozen=True, slots=True)
class PassageMetrics:
    precision: float
    recall: float
    f1: float
    predicted_characters: int
    annotated_characters: int
    overlapping_characters: int


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    examples: int
    unique_groups: int
    binary: BinaryMetrics
    candidate_passages: PassageMetrics
    categories: dict[str, BinaryMetrics]
    method_versions: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate(analyzer: Analyzer, examples: list[EvaluationExample]) -> BenchmarkReport:
    if not examples:
        raise ValueError("at least one evaluation example is required")
    _validate_unique_ids(examples)

    outcomes: list[tuple[EvaluationExample, AnalysisResult]] = [
        (example, analyzer.analyze(example.source, example.candidate))
        for example in examples
    ]
    binary = _binary_metrics(outcomes)
    passages = _passage_metrics(outcomes)
    categories = {
        category: _binary_metrics(
            [(example, result) for example, result in outcomes if example.category == category]
        )
        for category in sorted({example.category for example in examples})
    }
    return BenchmarkReport(
        examples=len(examples),
        unique_groups=len({example.group_id for example in examples}),
        binary=binary,
        candidate_passages=passages,
        categories=categories,
        method_versions=tuple(sorted({result.method for _, result in outcomes})),
    )


def _validate_unique_ids(examples: list[EvaluationExample]) -> None:
    ids = [example.example_id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation example IDs must be unique")


def _binary_metrics(
    outcomes: list[tuple[EvaluationExample, AnalysisResult]],
) -> BinaryMetrics:
    true_positive = sum(
        example.has_reuse and result.verdict != "low_overlap"
        for example, result in outcomes
    )
    false_positive = sum(
        not example.has_reuse and result.verdict != "low_overlap"
        for example, result in outcomes
    )
    true_negative = sum(
        not example.has_reuse and result.verdict == "low_overlap"
        for example, result in outcomes
    )
    false_negative = sum(
        example.has_reuse and result.verdict == "low_overlap"
        for example, result in outcomes
    )
    return _classification_result(
        true_positive, false_positive, true_negative, false_negative
    )


def _classification_result(
    true_positive: int,
    false_positive: int,
    true_negative: int,
    false_negative: int,
) -> BinaryMetrics:
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryMetrics(
        precision=_rounded(precision),
        recall=_rounded(recall),
        f1=_rounded(f1),
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
    )


def _passage_metrics(
    outcomes: list[tuple[EvaluationExample, AnalysisResult]],
) -> PassageMetrics:
    predicted_characters = 0
    annotated_characters = 0
    overlapping_characters = 0

    for example, result in outcomes:
        predicted = _merge_intervals(
            [(match.candidate_start, match.candidate_end) for match in result.evidence]
        )
        annotated = _merge_intervals(
            [
                (annotation.candidate_start, annotation.candidate_end)
                for annotation in example.annotations
            ]
        )
        predicted_characters += _interval_length(predicted)
        annotated_characters += _interval_length(annotated)
        overlapping_characters += _intersection_length(predicted, annotated)

    precision = (
        overlapping_characters / predicted_characters if predicted_characters else 0.0
    )
    recall = (
        overlapping_characters / annotated_characters if annotated_characters else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PassageMetrics(
        precision=_rounded(precision),
        recall=_rounded(recall),
        f1=_rounded(f1),
        predicted_characters=predicted_characters,
        annotated_characters=annotated_characters,
        overlapping_characters=overlapping_characters,
    )


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _interval_length(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def _intersection_length(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> int:
    left_index = 0
    right_index = 0
    overlap = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        overlap += max(0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return overlap


def _rounded(value: float) -> float:
    return round(value, 4)
