import pytest

from plagiarism_detection import AnalysisConfig, PairwiseAnalyzer

SOURCE = (
    "Reliable machine learning systems require representative evaluation data. "
    "They also need monitoring after deployment and evidence for every prediction."
)


def test_exact_copy_is_high_overlap_with_offsets() -> None:
    result = PairwiseAnalyzer().analyze(SOURCE, SOURCE)

    assert result.verdict == "high_overlap"
    assert result.similarity_score == 1.0
    assert result.candidate_coverage == 1.0
    assert result.evidence[0].match_type == "exact"
    assert result.evidence[0].source_text.endswith("evaluation data.")
    assert (
        SOURCE[result.evidence[0].source_start : result.evidence[0].source_end]
        == result.evidence[0].source_text
    )


def test_unrelated_text_is_low_overlap() -> None:
    candidate = "The chef roasted aubergines with garlic. Guests enjoyed dessert beside the garden."
    result = PairwiseAnalyzer().analyze(SOURCE, candidate)

    assert result.verdict == "low_overlap"
    assert result.similarity_score < 0.2
    assert result.evidence == ()


def test_copied_passage_in_longer_candidate_returns_evidence() -> None:
    candidate = (
        "This report introduces our deployment process. "
        "Reliable machine learning systems require representative evaluation data. "
        "The final section discusses ownership."
    )
    result = PairwiseAnalyzer().analyze(SOURCE, candidate)

    assert result.verdict == "high_overlap"
    assert result.evidence
    match = result.evidence[0]
    assert match.match_type == "exact"
    assert candidate[match.candidate_start : match.candidate_end] == match.candidate_text


def test_score_is_explicitly_not_a_probability() -> None:
    result = PairwiseAnalyzer().analyze(SOURCE, SOURCE)

    assert "not a probability" in result.score_interpretation


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("", "three valid words"),
        ("three valid words", " "),
        ("only two", "three valid words"),
    ],
)
def test_rejects_empty_or_too_short_documents(source: str, candidate: str) -> None:
    with pytest.raises(ValueError):
        PairwiseAnalyzer().analyze(source, candidate)


def test_config_rejects_invalid_threshold_order() -> None:
    with pytest.raises(ValueError):
        AnalysisConfig(review_threshold=0.8, high_overlap_threshold=0.5)
