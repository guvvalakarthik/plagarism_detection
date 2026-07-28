"""Idempotent document-to-passage ingestion."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass

from .retrieval import Embedder, chunk_document
from .storage import (
    CorpusRepository,
    StoredDocument,
    StoredPassage,
    utc_now,
)


@dataclass(frozen=True, slots=True)
class IngestionResult:
    document_id: str
    content_sha256: str
    passages: int
    duplicate: bool
    embedding_method: str


class CorpusIngestionService:
    def __init__(
        self,
        repository: CorpusRepository,
        embedder: Embedder,
        *,
        max_tokens: int = 120,
        overlap_tokens: int = 24,
    ) -> None:
        self.repository = repository
        self.embedder = embedder
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def ingest(
        self,
        *,
        workspace_id: str,
        filename: str,
        text: str,
    ) -> IngestionResult:
        if not workspace_id.strip():
            raise ValueError("workspace_id must not be empty")
        if not filename.strip():
            raise ValueError("filename must not be empty")
        if not text.strip():
            raise ValueError("text must not be empty")

        content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.repository.ensure_workspace(workspace_id)
        existing = self.repository.find_document_by_hash(
            workspace_id, content_sha256
        )
        if existing:
            return IngestionResult(
                document_id=existing.document_id,
                content_sha256=content_sha256,
                passages=0,
                duplicate=True,
                embedding_method=self.embedder.name,
            )

        document_id = str(uuid.uuid4())
        chunks = chunk_document(
            document_id,
            text,
            max_tokens=self.max_tokens,
            overlap_tokens=self.overlap_tokens,
        )
        embeddings = self.embedder.encode([chunk.text for chunk in chunks])
        if len(embeddings) != len(chunks):
            raise ValueError("embedder returned an unexpected number of vectors")

        document = StoredDocument(
            workspace_id=workspace_id,
            document_id=document_id,
            content_sha256=content_sha256,
            original_filename=filename,
            character_count=len(text),
            created_at=utc_now(),
        )
        passages = [
            StoredPassage(
                workspace_id=workspace_id,
                passage_id=chunk.passage_id,
                document_id=document_id,
                content=chunk.text,
                start_offset=chunk.start,
                end_offset=chunk.end,
                embedding=tuple(embedding),
                embedding_method=self.embedder.name,
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self.repository.save_document(document)
        self.repository.replace_passages(workspace_id, document_id, passages)
        return IngestionResult(
            document_id=document_id,
            content_sha256=content_sha256,
            passages=len(passages),
            duplicate=False,
            embedding_method=self.embedder.name,
        )
