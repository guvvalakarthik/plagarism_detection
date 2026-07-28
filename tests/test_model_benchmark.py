from plagiarism_detection.model_benchmark import (
    BenchmarkPair,
    ScoredPair,
    average_precision,
    dataset_fingerprint,
    grouped_split,
    run_model_benchmark,
)


class TokenOverlapScorer:
    @property
    def name(self) -> str:
        return "token-overlap-test"

    def score(self, source: str, candidate: str) -> float:
        source_tokens = set(source.casefold().split())
        candidate_tokens = set(candidate.casefold().split())
        return len(source_tokens & candidate_tokens) / len(source_tokens | candidate_tokens)


def benchmark_pairs() -> list[BenchmarkPair]:
    pairs = []
    for group in range(6):
        source = f"group {group} machine learning evaluation evidence"
        pairs.extend(
            [
                BenchmarkPair(
                    pair_id=f"{group}-positive",
                    group_id=f"group-{group}",
                    source=source,
                    candidate=f"machine learning evaluation evidence group {group}",
                    label=1,
                    category="paraphrase",
                ),
                BenchmarkPair(
                    pair_id=f"{group}-negative",
                    group_id=f"group-{group}",
                    source=source,
                    candidate=f"garden soil watering vegetables example {group}",
                    label=0,
                    category="hard-negative",
                ),
            ]
        )
    return pairs


def test_model_benchmark_is_group_isolated_and_reproducible() -> None:
    pairs = benchmark_pairs()

    first = run_model_benchmark(
        pairs,
        [TokenOverlapScorer()],
        dataset_name="unit-test",
        seed=17,
        calibration_fraction=0.67,
    )
    second = run_model_benchmark(
        pairs,
        [TokenOverlapScorer()],
        dataset_name="unit-test",
        seed=17,
        calibration_fraction=0.67,
    )

    assert first["source_group_leakage"] == 0
    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]
    model = first["models"][0]
    assert model["held_out_test"]["f1"] == 1.0
    assert model["average_precision"] == 1.0
    assert model["categories"]["paraphrase"]["recall"] == 1.0


def test_group_split_never_places_one_source_in_both_partitions() -> None:
    calibration, test = grouped_split(
        benchmark_pairs(),
        calibration_fraction=0.5,
        seed=42,
    )

    assert {pair.group_id for pair in calibration}.isdisjoint(
        pair.group_id for pair in test
    )


def test_average_precision_uses_ranked_positive_positions() -> None:
    pairs = benchmark_pairs()[:3]
    scored = [
        ScoredPair(pair=pairs[0], score=0.9, latency_ms=1.0),
        ScoredPair(pair=pairs[1], score=0.8, latency_ms=1.0),
        ScoredPair(pair=pairs[2], score=0.7, latency_ms=1.0),
    ]

    assert average_precision(scored) == 0.8333


def test_dataset_fingerprint_changes_with_content() -> None:
    pairs = benchmark_pairs()
    changed = list(pairs)
    original = changed[0]
    changed[0] = BenchmarkPair(
        pair_id=original.pair_id,
        group_id=original.group_id,
        source=original.source,
        candidate=original.candidate + " changed",
        label=original.label,
        category=original.category,
    )

    assert dataset_fingerprint(pairs) != dataset_fingerprint(changed)
