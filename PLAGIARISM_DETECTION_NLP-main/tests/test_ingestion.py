import uuid

import pytest

from plagiarism_detection.ingestion import CorpusIngestionService
from plagiarism_detection.retrieval import HashingNgramEmbedder
from plagiarism_detection.storage import MemoryCorpusRepository


def workspace() -> str:
    return str(uuid.uuid4())


def test_ingestion_is_idempotent_per_workspace() -> None:
    repository = MemoryCorpusRepository()
    service = CorpusIngestionService(
        repository,
        HashingNgramEmbedder(32),
        max_tokens=8,
        overlap_tokens=2,
    )
    workspace_id = workspace()
    text = "One two three four five six seven eight nine ten eleven twelve."

    first = service.ingest(
        workspace_id=workspace_id, filename="document.txt", text=text
    )
    second = service.ingest(
        workspace_id=workspace_id, filename="renamed.txt", text=text
    )

    assert first.duplicate is False
    assert first.passages == 2
    assert second.duplicate is True
    assert second.document_id == first.document_id
    assert len(repository.documents) == 1


def test_same_content_is_isolated_between_workspaces() -> None:
    repository = MemoryCorpusRepository()
    service = CorpusIngestionService(repository, HashingNgramEmbedder(32))
    text = "A sufficiently long shared document for testing tenant isolation."

    first = service.ingest(workspace_id=workspace(), filename="one.txt", text=text)
    second = service.ingest(workspace_id=workspace(), filename="two.txt", text=text)

    assert first.duplicate is False
    assert second.duplicate is False
    assert first.document_id != second.document_id
    assert len(repository.documents) == 2


def test_search_never_returns_another_workspace() -> None:
    repository = MemoryCorpusRepository()
    embedder = HashingNgramEmbedder(32)
    service = CorpusIngestionService(repository, embedder)
    first_workspace = workspace()
    second_workspace = workspace()
    first_text = "Machine learning systems require representative evaluation evidence."
    second_text = "Garden vegetables require healthy soil and regular watering."
    service.ingest(
        workspace_id=first_workspace, filename="ml.txt", text=first_text
    )
    service.ingest(
        workspace_id=second_workspace, filename="garden.txt", text=second_text
    )

    hits = repository.search(
        first_workspace, embedder.encode([first_text])[0], limit=10
    )

    assert hits
    assert {hit.passage.workspace_id for hit in hits} == {first_workspace}


def test_ingestion_rejects_empty_identity_or_content() -> None:
    service = CorpusIngestionService(
        MemoryCorpusRepository(), HashingNgramEmbedder(32)
    )

    with pytest.raises(ValueError):
        service.ingest(workspace_id="", filename="file.txt", text="valid text")
    with pytest.raises(ValueError):
        service.ingest(workspace_id=workspace(), filename="", text="valid text")
    with pytest.raises(ValueError):
        service.ingest(workspace_id=workspace(), filename="file.txt", text=" ")
