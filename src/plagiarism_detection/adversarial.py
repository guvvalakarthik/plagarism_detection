"""Deterministic robustness cases that require no external model or dataset."""

from __future__ import annotations

import re

from .evaluation import EvaluationExample, PassageAnnotation


def generate_adversarial_suite(source: str, group_id: str = "synthetic") -> list[EvaluationExample]:
    if len(source.split()) < 8:
        raise ValueError("source must contain at least eight words")

    transformations = {
        "case_change": source.swapcase(),
        "punctuation_noise": re.sub(r"\s+", " , ", source),
        "whitespace_noise": re.sub(r"\s+", " \n\t ", source),
        "sentence_reorder": _reverse_sentences(source),
        "filler_insertion": f"Context before the reused material. {source} Additional context.",
    }
    return [
        EvaluationExample(
            example_id=f"{group_id}:{category}",
            group_id=group_id,
            source=source,
            candidate=candidate,
            annotations=(
                PassageAnnotation(
                    candidate_start=0,
                    candidate_end=len(candidate),
                    source_start=0,
                    source_end=len(source),
                ),
            ),
            category=category,
        )
        for category, candidate in transformations.items()
    ]


def _reverse_sentences(text: str) -> str:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text)]
    return " ".join(reversed([sentence for sentence in sentences if sentence]))
