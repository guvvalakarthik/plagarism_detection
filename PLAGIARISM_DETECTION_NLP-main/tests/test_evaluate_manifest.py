from pathlib import Path

import pytest

from scripts.evaluate_manifest import load_manifest, safe_document_path


def test_manifest_loads_documents_and_annotations(tmp_path: Path) -> None:
    (tmp_path / "source.txt").write_text(
        "A source document with enough words for reliable analysis.",
        encoding="utf-8",
    )
    (tmp_path / "candidate.txt").write_text(
        "A source document with enough words for reliable analysis.",
        encoding="utf-8",
    )
    (tmp_path / "manifest.jsonl").write_text(
        '{"example_id":"one","group_id":"source","source_path":"source.txt",'
        '"candidate_path":"candidate.txt","annotations":'
        '[{"candidate_start":0,"candidate_end":58}]}\n',
        encoding="utf-8",
    )

    examples = load_manifest(tmp_path / "manifest.jsonl", tmp_path)

    assert len(examples) == 1
    assert examples[0].annotations[0].candidate_start == 0


def test_manifest_path_cannot_escape_dataset_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        safe_document_path(tmp_path, "../secret.txt")
