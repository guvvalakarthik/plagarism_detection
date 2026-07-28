import pytest

from plagiarism_detection import PairwiseAnalyzer
from plagiarism_detection.adversarial import generate_adversarial_suite
from plagiarism_detection.evaluation import (
    EvaluationExample,
    PassageAnnotation,
    evaluate,
)


def test_passage_evaluation_reports_perfect_exact_match() -> None:
    text = "Reliable evaluation needs representative documents and passage annotations."
    example = EvaluationExample(
        example_id="exact-1",
        group_id="source-1",
        source=text,
        candidate=text,
        annotations=(
            PassageAnnotation(
                candidate_start=0,
                candidate_end=len(text),
                source_start=0,
                source_end=len(text),
            ),
        ),
        category="exact",
    )

    report = evaluate(PairwiseAnalyzer(), [example])

    assert report.binary.f1 == 1.0
    assert report.candidate_passages.f1 == 1.0
    assert report.categories["exact"].recall == 1.0
    assert report.method_versions == ("pairwise_tfidf_style_v2",)


def test_evaluation_includes_true_negative() -> None:
    example = EvaluationExample(
        example_id="negative-1",
        group_id="source-1",
        source="Machine learning systems need monitoring and evaluation evidence.",
        candidate="Gardeners watered tomatoes before preparing dinner for their guests.",
        annotations=(),
        category="hard_negative",
    )

    report = evaluate(PairwiseAnalyzer(), [example])

    assert report.binary.true_negative == 1
    assert report.binary.f1 == 0.0
    assert report.candidate_passages.predicted_characters == 0


def test_duplicate_ids_are_rejected() -> None:
    example = EvaluationExample(
        example_id="duplicate",
        group_id="source",
        source="One sufficiently long source sentence for this evaluation case.",
        candidate="One sufficiently long candidate sentence for this evaluation case.",
        annotations=(),
    )

    with pytest.raises(ValueError, match="unique"):
        evaluate(PairwiseAnalyzer(), [example, example])


@pytest.mark.parametrize(
    "arguments",
    [
        {"candidate_start": -1, "candidate_end": 2},
        {"candidate_start": 2, "candidate_end": 2},
        {
            "candidate_start": 0,
            "candidate_end": 2,
            "source_start": 1,
            "source_end": None,
        },
    ],
)
def test_invalid_annotations_are_rejected(arguments) -> None:
    with pytest.raises(ValueError):
        PassageAnnotation(**arguments)


def test_adversarial_suite_is_deterministic_and_grouped() -> None:
    source = (
        "Reliable systems require evidence and monitoring. "
        "Evaluation must include representative documents."
    )

    first = generate_adversarial_suite(source, "robustness")
    second = generate_adversarial_suite(source, "robustness")

    assert first == second
    assert len(first) == 5
    assert {example.group_id for example in first} == {"robustness"}
    assert {example.category for example in first} == {
        "case_change",
        "punctuation_noise",
        "whitespace_noise",
        "sentence_reorder",
        "filler_insertion",
    }
