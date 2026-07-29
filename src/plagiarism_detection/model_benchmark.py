"""Leakage-aware pair-model comparison with latency and ranking metrics."""

from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Protocol

from .analyzer import PairwiseAnalyzer
from .retrieval import Embedder, Reranker


@dataclass(frozen=True, slots=True)
class BenchmarkPair:
    pair_id: str
    group_id: str
    source: str
    candidate: str
    label: int
    category: str = "unspecified"

    def __post_init__(self) -> None:
        if not self.pair_id.strip() or not self.group_id.strip():
            raise ValueError("pair and group IDs must not be empty")
        if not self.source.strip() or not self.candidate.strip():
            raise ValueError("benchmark text must not be empty")
        if self.label not in {0, 1}:
            raise ValueError("benchmark labels must be 0 or 1")


class PairScorer(Protocol):
    @property
    def name(self) -> str: ...

    def score(self, source: str, candidate: str) -> float: ...


class AnalyzerPairScorer:
    def __init__(self, analyzer: PairwiseAnalyzer | None = None) -> None:
        self.analyzer = analyzer or PairwiseAnalyzer()

    @property
    def name(self) -> str:
        return "pairwise-analyzer-v2"

    def score(self, source: str, candidate: str) -> float:
        return self.analyzer.analyze(source, candidate).similarity_score


class EmbeddingPairScorer:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    @property
    def name(self) -> str:
        return f"cosine:{self.embedder.name}"

    def score(self, source: str, candidate: str) -> float:
        source_vector, candidate_vector = self.embedder.encode([source, candidate])
        return _cosine(source_vector, candidate_vector)


class RerankerPairScorer:
    def __init__(self, reranker: Reranker) -> None:
        self.reranker = reranker

    @property
    def name(self) -> str:
        return self.reranker.name

    def score(self, source: str, candidate: str) -> float:
        return self.reranker.score(candidate, [source])[0]


@dataclass(frozen=True, slots=True)
class ScoredPair:
    pair: BenchmarkPair
    score: float
    latency_ms: float


def run_model_benchmark(
    pairs: list[BenchmarkPair],
    scorers: list[PairScorer],
    *,
    dataset_name: str,
    seed: int = 42,
    calibration_fraction: float = 0.75,
    limitations: list[str] | None = None,
) -> dict[str, object]:
    if not pairs:
        raise ValueError("at least one benchmark pair is required")
    if not scorers:
        raise ValueError("at least one model scorer is required")
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise ValueError("benchmark pair IDs must be unique")
    if not 0.1 <= calibration_fraction <= 0.9:
        raise ValueError("calibration fraction must be between 0.1 and 0.9")

    calibration, test = grouped_split(
        pairs,
        calibration_fraction=calibration_fraction,
        seed=seed,
    )
    calibration_groups = {pair.group_id for pair in calibration}
    test_groups = {pair.group_id for pair in test}
    overlap = calibration_groups & test_groups
    if overlap:
        raise ValueError("source groups leaked across benchmark partitions")

    model_reports = []
    for scorer in scorers:
        scorer.score(calibration[0].source, calibration[0].candidate)
        calibration_scores = score_pairs(scorer, calibration)
        threshold = choose_threshold(calibration_scores)
        test_scores = score_pairs(scorer, test)
        model_reports.append(
            {
                "model": scorer.name,
                "threshold": round(threshold, 6),
                "calibration": classification_metrics(
                    calibration_scores,
                    threshold,
                ),
                "held_out_test": classification_metrics(test_scores, threshold),
                "average_precision": average_precision(test_scores),
                "latency_ms": latency_summary(
                    calibration_scores + test_scores
                ),
                "score_summary": score_summary(test_scores),
                "categories": {
                    category: classification_metrics(
                        [
                            scored
                            for scored in test_scores
                            if scored.pair.category == category
                        ],
                        threshold,
                    )
                    for category in sorted(
                        {scored.pair.category for scored in test_scores}
                    )
                },
            }
        )

    return {
        "schema_version": 1,
        "dataset": dataset_name,
        "dataset_fingerprint": dataset_fingerprint(pairs),
        "seed": seed,
        "pairs": len(pairs),
        "positive_pairs": sum(pair.label for pair in pairs),
        "negative_pairs": sum(1 - pair.label for pair in pairs),
        "unique_source_groups": len({pair.group_id for pair in pairs}),
        "calibration_pairs": len(calibration),
        "held_out_test_pairs": len(test),
        "source_group_leakage": len(overlap),
        "models": model_reports,
        "limitations": limitations or [],
    }


def grouped_split(
    pairs: list[BenchmarkPair],
    *,
    calibration_fraction: float,
    seed: int,
) -> tuple[list[BenchmarkPair], list[BenchmarkPair]]:
    groups = sorted({pair.group_id for pair in pairs}, key=lambda group: _split_key(group, seed))
    if len(groups) < 2:
        raise ValueError("at least two source groups are required")
    calibration_count = round(len(groups) * calibration_fraction)
    calibration_count = min(max(calibration_count, 1), len(groups) - 1)
    calibration_groups = set(groups[:calibration_count])
    calibration = [pair for pair in pairs if pair.group_id in calibration_groups]
    test = [pair for pair in pairs if pair.group_id not in calibration_groups]
    return calibration, test


def score_pairs(scorer: PairScorer, pairs: list[BenchmarkPair]) -> list[ScoredPair]:
    scored = []
    for pair in pairs:
        started = perf_counter()
        score = float(scorer.score(pair.source, pair.candidate))
        elapsed_ms = (perf_counter() - started) * 1_000
        if not math.isfinite(score):
            raise ValueError(f"{scorer.name} produced a non-finite score")
        scored.append(
            ScoredPair(
                pair=pair,
                score=score,
                latency_ms=elapsed_ms,
            )
        )
    return scored


def choose_threshold(scored: list[ScoredPair]) -> float:
    if not scored:
        raise ValueError("calibration scores must not be empty")
    values = sorted({item.score for item in scored})
    candidates = values + [math.nextafter(values[-1], math.inf)]
    return max(
        candidates,
        key=lambda threshold: (
            classification_metrics(scored, threshold)["f1"],
            classification_metrics(scored, threshold)["precision"],
            classification_metrics(scored, threshold)["recall"],
            threshold,
        ),
    )


def classification_metrics(
    scored: list[ScoredPair],
    threshold: float,
) -> dict[str, float | int]:
    true_positive = sum(
        item.score >= threshold and item.pair.label == 1 for item in scored
    )
    false_positive = sum(
        item.score >= threshold and item.pair.label == 0 for item in scored
    )
    true_negative = sum(
        item.score < threshold and item.pair.label == 0 for item in scored
    )
    false_negative = sum(
        item.score < threshold and item.pair.label == 1 for item in scored
    )
    predicted_positive = true_positive + false_positive
    actual_positive = true_positive + false_negative
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (
        (true_positive + true_negative) / len(scored) if scored else 0.0
    )
    return {
        "examples": len(scored),
        "precision": _rounded(precision),
        "recall": _rounded(recall),
        "f1": _rounded(f1),
        "accuracy": _rounded(accuracy),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def average_precision(scored: list[ScoredPair]) -> float:
    positives = sum(item.pair.label for item in scored)
    if positives == 0:
        return 0.0
    ordered = sorted(scored, key=lambda item: (-item.score, item.pair.pair_id))
    found = 0
    precision_sum = 0.0
    for rank, item in enumerate(ordered, start=1):
        if item.pair.label == 1:
            found += 1
            precision_sum += found / rank
    return _rounded(precision_sum / positives)


def latency_summary(scored: list[ScoredPair]) -> dict[str, float]:
    values = sorted(item.latency_ms for item in scored)
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": round(statistics.fmean(values), 3),
        "p50": round(_percentile(values, 0.50), 3),
        "p95": round(_percentile(values, 0.95), 3),
    }


def score_summary(scored: list[ScoredPair]) -> dict[str, float]:
    values = [item.score for item in scored]
    if not values:
        return {"minimum": 0.0, "maximum": 0.0, "mean": 0.0}
    return {
        "minimum": round(min(values), 6),
        "maximum": round(max(values), 6),
        "mean": round(statistics.fmean(values), 6),
    }


def dataset_fingerprint(pairs: list[BenchmarkPair]) -> str:
    digest = hashlib.sha256()
    for pair in sorted(pairs, key=lambda item: item.pair_id):
        values = (
            pair.pair_id,
            pair.group_id,
            str(pair.label),
            pair.category,
            hashlib.sha256(pair.source.encode("utf-8")).hexdigest(),
            hashlib.sha256(pair.candidate.encode("utf-8")).hexdigest(),
        )
        digest.update("\0".join(values).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _split_key(group_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    index = max(0, math.ceil(percentile * len(values)) - 1)
    return values[index]


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _rounded(value: float) -> float:
    return round(value, 4)


def metrics_to_dict(metrics) -> dict[str, object]:
    return asdict(metrics)
