"""Command-line interface for local and CI use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyzer import PairwiseAnalyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sourcelens",
        description="Compare a candidate text with a source and return evidence.",
    )
    parser.add_argument("source", type=Path, help="UTF-8 source document")
    parser.add_argument("candidate", type=Path, help="UTF-8 candidate document")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    source = arguments.source.read_text(encoding="utf-8")
    candidate = arguments.candidate.read_text(encoding="utf-8")
    result = PairwiseAnalyzer().analyze(source, candidate)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
