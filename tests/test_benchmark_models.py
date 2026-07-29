from pathlib import Path

import pytest

from scripts.benchmark_models import build_scorers, load_csv_pairs, parse_models


def test_csv_loader_groups_duplicate_sources(tmp_path: Path) -> None:
    dataset = tmp_path / "pairs.csv"
    dataset.write_text(
        "source_text,plagiarized_text,label\n"
        '"A shared source sentence","A related candidate sentence",1\n'
        '"A shared source sentence","An unrelated candidate sentence",0\n',
        encoding="utf-8",
    )

    pairs = load_csv_pairs(dataset)

    assert len(pairs) == 2
    assert pairs[0].group_id == pairs[1].group_id
    assert pairs[0].category == "positive"
    assert pairs[1].category == "negative"


def test_default_scorers_do_not_require_model_downloads() -> None:
    scorers = build_scorers(
        ["baseline", "hashing"],
        enable_transformers=False,
        sentence_model="unused",
        cross_encoder="unused",
    )

    assert [scorer.name for scorer in scorers] == [
        "pairwise-analyzer-v2",
        "cosine:hashing-ngram-256",
    ]


def test_transformer_models_require_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="enable-transformers"):
        build_scorers(
            ["semantic"],
            enable_transformers=False,
            sentence_model="model",
            cross_encoder="reranker",
        )


def test_model_list_parser_and_unknown_model_validation() -> None:
    assert parse_models("baseline, hashing") == ["baseline", "hashing"]
    with pytest.raises(ValueError, match="unknown model"):
        build_scorers(
            ["mystery"],
            enable_transformers=False,
            sentence_model="unused",
            cross_encoder="unused",
        )
