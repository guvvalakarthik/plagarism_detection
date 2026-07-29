"""Compare transparent, hashing, and optional transformer pair scorers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plagiarism_detection.model_benchmark import (  # noqa: E402
    AnalyzerPairScorer,
    BenchmarkPair,
    EmbeddingPairScorer,
    PairScorer,
    RerankerPairScorer,
    run_model_benchmark,
)
from plagiarism_detection.retrieval import (  # noqa: E402
    CrossEncoderReranker,
    HashingNgramEmbedder,
    SentenceTransformerEmbedder,
)

DEFAULT_SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SMOKE_LIMITATIONS = [
    "The bundled CSV contains short synthetic sentences.",
    "Its labels do not establish plagiarism, permission, or missing attribution.",
    "Use a representative licensed passage corpus before making quality claims.",
]


def normalized_hash(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"\W+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_csv_pairs(path: Path) -> list[BenchmarkPair]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"source_text", "plagiarized_text", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain: {', '.join(sorted(required))}")
        pairs = []
        for row_number, row in enumerate(reader, start=1):
            source = row["source_text"]
            candidate = row["plagiarized_text"]
            label = int(row["label"])
            pairs.append(
                BenchmarkPair(
                    pair_id=f"row-{row_number}",
                    group_id=normalized_hash(source),
                    source=source,
                    candidate=candidate,
                    label=label,
                    category="positive" if label == 1 else "negative",
                )
            )
    return pairs


def build_scorers(
    model_names: list[str],
    *,
    enable_transformers: bool,
    sentence_model: str,
    cross_encoder: str,
) -> list[PairScorer]:
    unknown = sorted(
        set(model_names) - {"baseline", "hashing", "semantic", "cross-encoder"}
    )
    if unknown:
        raise ValueError(f"unknown model names: {', '.join(unknown)}")
    transformer_names = {"semantic", "cross-encoder"} & set(model_names)
    if transformer_names and not enable_transformers:
        raise ValueError(
            "transformer models require --enable-transformers to allow model loading"
        )

    scorers: list[PairScorer] = []
    for name in model_names:
        if name == "baseline":
            scorers.append(AnalyzerPairScorer())
        elif name == "hashing":
            scorers.append(EmbeddingPairScorer(HashingNgramEmbedder(256)))
        elif name == "semantic":
            scorers.append(
                EmbeddingPairScorer(SentenceTransformerEmbedder(sentence_model))
            )
        elif name == "cross-encoder":
            scorers.append(
                RerankerPairScorer(CrossEncoderReranker(cross_encoder))
            )
    return scorers


def parse_models(value: str) -> list[str]:
    models = [item.strip() for item in value.split(",") if item.strip()]
    if not models:
        raise argparse.ArgumentTypeError("at least one model is required")
    return models


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a leakage-aware SourceLens pair-model benchmark."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--models",
        type=parse_models,
        default=["baseline", "hashing"],
        help="Comma-separated: baseline,hashing,semantic,cross-encoder",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-fraction", type=float, default=0.75)
    parser.add_argument("--enable-transformers", action="store_true")
    parser.add_argument("--sentence-model", default=DEFAULT_SENTENCE_MODEL)
    parser.add_argument("--cross-encoder", default=DEFAULT_CROSS_ENCODER)
    arguments = parser.parse_args()

    pairs = load_csv_pairs(arguments.dataset)
    scorers = build_scorers(
        arguments.models,
        enable_transformers=arguments.enable_transformers,
        sentence_model=arguments.sentence_model,
        cross_encoder=arguments.cross_encoder,
    )
    report = run_model_benchmark(
        pairs,
        scorers,
        dataset_name=arguments.dataset.name,
        seed=arguments.seed,
        calibration_fraction=arguments.calibration_fraction,
        limitations=SMOKE_LIMITATIONS,
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
