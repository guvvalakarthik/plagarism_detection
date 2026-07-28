"""Leakage-aware evaluation for the transparent baseline.

Rows are grouped by normalized source hash before splitting, so duplicate source
documents cannot appear in both calibration and test partitions. The supplied
toy dataset is useful for a smoke benchmark only; see MODEL_CARD.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plagiarism_detection import AnalysisConfig, PairwiseAnalyzer  # noqa: E402


@dataclass(frozen=True)
class Row:
    source: str
    candidate: str
    label: int
    group: str


def normalized_hash(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"\W+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_rows(path: Path) -> list[Row]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"source_text", "plagiarized_text", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain: {', '.join(sorted(required))}")
        return [
            Row(
                source=row["source_text"],
                candidate=row["plagiarized_text"],
                label=int(row["label"]),
                group=normalized_hash(row["source_text"]),
            )
            for row in reader
        ]


def grouped_split(
    rows: list[Row], test_fraction: float = 0.25, seed: int = 42
) -> tuple[list[Row], list[Row]]:
    groups: dict[str, list[Row]] = {}
    for row in rows:
        groups.setdefault(row.group, []).append(row)
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    test_count = max(1, round(len(group_ids) * test_fraction))
    test_groups = set(group_ids[:test_count])
    calibration = [row for row in rows if row.group not in test_groups]
    test = [row for row in rows if row.group in test_groups]
    return calibration, test


def score_rows(rows: list[Row]) -> list[tuple[float, int]]:
    analyzer = PairwiseAnalyzer()
    return [
        (analyzer.analyze(row.source, row.candidate).similarity_score, row.label) for row in rows
    ]


def classification_metrics(
    scored: list[tuple[float, int]], threshold: float
) -> dict[str, float | int]:
    true_positive = sum(score >= threshold and label == 1 for score, label in scored)
    false_positive = sum(score >= threshold and label == 0 for score, label in scored)
    false_negative = sum(score < threshold and label == 1 for score, label in scored)
    true_negative = sum(score < threshold and label == 0 for score, label in scored)
    predicted_positive = true_positive + false_positive
    actual_positive = true_positive + false_negative
    precision = (
        true_positive / predicted_positive if predicted_positive else 0
    )
    recall = (
        true_positive / actual_positive if actual_positive else 0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {
        "threshold": round(threshold, 2),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def choose_threshold(scored: list[tuple[float, int]]) -> float:
    candidates = [value / 100 for value in range(15, 86)]
    return max(
        candidates,
        key=lambda threshold: (
            classification_metrics(scored, threshold)["f1"],
            classification_metrics(scored, threshold)["precision"],
            threshold,
        ),
    )


def duplicate_leakage(calibration: list[Row], test: list[Row]) -> int:
    calibration_groups = {row.group for row in calibration}
    return len(calibration_groups.intersection(row.group for row in test))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    rows = load_rows(arguments.dataset)
    calibration, test = grouped_split(rows)
    calibration_scores = score_rows(calibration)
    threshold = choose_threshold(calibration_scores)
    report = {
        "method": "deterministic_source_group_split",
        "seed": 42,
        "rows": len(rows),
        "unique_source_groups": len({row.group for row in rows}),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "source_group_leakage": duplicate_leakage(calibration, test),
        "calibration_metrics": classification_metrics(calibration_scores, threshold),
        "held_out_test_metrics": classification_metrics(score_rows(test), threshold),
        "production_policy": asdict(AnalysisConfig()),
        "limitations": [
            "The included 370-row dataset contains short synthetic sentences.",
            "Labels do not establish plagiarism without citation and authorship context.",
            "Metrics are a baseline smoke benchmark, not production validation.",
        ],
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
