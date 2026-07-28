"""Evaluate SourceLens against a path-based JSONL manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plagiarism_detection import PairwiseAnalyzer  # noqa: E402
from plagiarism_detection.evaluation import (  # noqa: E402
    EvaluationExample,
    PassageAnnotation,
    evaluate,
)


def safe_document_path(dataset_root: Path, relative_path: str) -> Path:
    resolved_root = dataset_root.resolve()
    resolved = (resolved_root / relative_path).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"manifest path escapes dataset root: {relative_path}")
    return resolved


def load_manifest(manifest: Path, dataset_root: Path) -> list[EvaluationExample]:
    examples: list[EvaluationExample] = []
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            source_path = safe_document_path(dataset_root, record["source_path"])
            candidate_path = safe_document_path(dataset_root, record["candidate_path"])
            annotations = tuple(
                PassageAnnotation(
                    candidate_start=item["candidate_start"],
                    candidate_end=item["candidate_end"],
                    source_start=item.get("source_start"),
                    source_end=item.get("source_end"),
                )
                for item in record.get("annotations", [])
            )
            examples.append(
                EvaluationExample(
                    example_id=record["example_id"],
                    group_id=record["group_id"],
                    source=source_path.read_text(encoding="utf-8"),
                    candidate=candidate_path.read_text(encoding="utf-8"),
                    annotations=annotations,
                    category=record.get("category", "unspecified"),
                )
            )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid manifest line {line_number}: {error}") from error
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    examples = load_manifest(arguments.manifest, arguments.dataset_root)
    report = evaluate(PairwiseAnalyzer(), examples).to_dict()
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
