"""Tenant-scoped corpus repository interfaces and implementations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class StoredDocument:
    workspace_id: str
    document_id: str
    content_sha256: str
    original_filename: str
    character_count: int
    status: str = "ready"
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class StoredPassage:
    workspace_id: str
    passage_id: str
    document_id: str
    content: str
    start_offset: int
    end_offset: int
    embedding: tuple[float, ...]
    embedding_method: str


@dataclass(frozen=True, slots=True)
class StoredSearchHit:
    passage: StoredPassage
    score: float


class CorpusRepository(Protocol):
    def ensure_workspace(self, workspace_id: str) -> None: ...

    def find_document_by_hash(
        self, workspace_id: str, content_sha256: str
    ) -> StoredDocument | None: ...

    def save_document(self, document: StoredDocument) -> None: ...

    def replace_passages(
        self, workspace_id: str, document_id: str, passages: Sequence[StoredPassage]
    ) -> None: ...

    def search(
        self, workspace_id: str, query_embedding: Sequence[float], limit: int
    ) -> list[StoredSearchHit]: ...


class MemoryCorpusRepository:
    """Deterministic repository used by unit tests and local prototypes."""

    def __init__(self) -> None:
        self.workspaces: set[str] = set()
        self.documents: dict[tuple[str, str], StoredDocument] = {}
        self.passages: dict[tuple[str, str], StoredPassage] = {}

    def ensure_workspace(self, workspace_id: str) -> None:
        self.workspaces.add(workspace_id)

    def find_document_by_hash(
        self, workspace_id: str, content_sha256: str
    ) -> StoredDocument | None:
        return next(
            (
                document
                for (stored_workspace, _), document in self.documents.items()
                if stored_workspace == workspace_id
                and document.content_sha256 == content_sha256
            ),
            None,
        )

    def save_document(self, document: StoredDocument) -> None:
        if document.workspace_id not in self.workspaces:
            raise ValueError("workspace must exist before saving a document")
        self.documents[(document.workspace_id, document.document_id)] = document

    def replace_passages(
        self, workspace_id: str, document_id: str, passages: Sequence[StoredPassage]
    ) -> None:
        if (workspace_id, document_id) not in self.documents:
            raise ValueError("document must exist before saving passages")
        self.passages = {
            key: passage
            for key, passage in self.passages.items()
            if not (
                passage.workspace_id == workspace_id
                and passage.document_id == document_id
            )
        }
        for passage in passages:
            if (
                passage.workspace_id != workspace_id
                or passage.document_id != document_id
            ):
                raise ValueError("passage scope must match document scope")
            self.passages[(workspace_id, passage.passage_id)] = passage

    def search(
        self, workspace_id: str, query_embedding: Sequence[float], limit: int
    ) -> list[StoredSearchHit]:
        if limit < 1:
            raise ValueError("limit must be positive")
        hits = [
            StoredSearchHit(passage=passage, score=_cosine(query_embedding, passage.embedding))
            for (stored_workspace, _), passage in self.passages.items()
            if stored_workspace == workspace_id
        ]
        return sorted(
            hits,
            key=lambda hit: (-hit.score, hit.passage.passage_id),
        )[:limit]


class PostgresCorpusRepository:
    """PostgreSQL/pgvector repository with explicit workspace predicates."""

    def __init__(self, dsn: str, embedding_dimensions: int = 256) -> None:
        if not dsn:
            raise ValueError("database DSN must not be empty")
        self.dsn = dsn
        self.embedding_dimensions = embedding_dimensions

    def ensure_workspace(self, workspace_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO workspaces (workspace_id)
                VALUES (%s)
                ON CONFLICT (workspace_id) DO NOTHING
                """,
                (workspace_id,),
            )

    def find_document_by_hash(
        self, workspace_id: str, content_sha256: str
    ) -> StoredDocument | None:
        with self._connect(workspace_id=workspace_id) as connection:
            row = connection.execute(
                """
                SELECT workspace_id, document_id, content_sha256,
                       original_filename, character_count, status, created_at
                  FROM documents
                 WHERE workspace_id = %s AND content_sha256 = %s
                """,
                (workspace_id, content_sha256),
            ).fetchone()
        return _document_from_row(row) if row else None

    def save_document(self, document: StoredDocument) -> None:
        with self._connect(workspace_id=document.workspace_id) as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    workspace_id, document_id, content_sha256,
                    original_filename, character_count, status
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (workspace_id, document_id) DO UPDATE SET
                    original_filename = EXCLUDED.original_filename,
                    character_count = EXCLUDED.character_count,
                    status = EXCLUDED.status
                """,
                (
                    document.workspace_id,
                    document.document_id,
                    document.content_sha256,
                    document.original_filename,
                    document.character_count,
                    document.status,
                ),
            )

    def replace_passages(
        self, workspace_id: str, document_id: str, passages: Sequence[StoredPassage]
    ) -> None:
        for passage in passages:
            if (
                passage.workspace_id != workspace_id
                or passage.document_id != document_id
            ):
                raise ValueError("passage scope must match document scope")
            if len(passage.embedding) != self.embedding_dimensions:
                raise ValueError("passage embedding has unexpected dimensions")

        with self._connect(workspace_id=workspace_id) as connection:
            connection.execute(
                "DELETE FROM passages WHERE workspace_id = %s AND document_id = %s",
                (workspace_id, document_id),
            )
            for passage in passages:
                connection.execute(
                    """
                    INSERT INTO passages (
                        workspace_id, passage_id, document_id, content,
                        start_offset, end_offset, embedding, embedding_method
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        passage.workspace_id,
                        passage.passage_id,
                        passage.document_id,
                        passage.content,
                        passage.start_offset,
                        passage.end_offset,
                        _as_vector(passage.embedding),
                        passage.embedding_method,
                    ),
                )

    def search(
        self, workspace_id: str, query_embedding: Sequence[float], limit: int
    ) -> list[StoredSearchHit]:
        if len(query_embedding) != self.embedding_dimensions:
            raise ValueError("query embedding has unexpected dimensions")
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._connect(workspace_id=workspace_id) as connection:
            rows = connection.execute(
                """
                SELECT workspace_id, passage_id, document_id, content,
                       start_offset, end_offset, embedding, embedding_method,
                       1 - (embedding <=> %s) AS score
                  FROM passages
                 WHERE workspace_id = %s
                 ORDER BY embedding <=> %s
                 LIMIT %s
                """,
                (
                    _as_vector(query_embedding),
                    workspace_id,
                    _as_vector(query_embedding),
                    limit,
                ),
            ).fetchall()
        return [_search_hit_from_row(row) for row in rows]

    def apply_migrations(self, migrations_directory: Path) -> None:
        migrations = sorted(migrations_directory.glob("*.sql"))
        if not migrations:
            raise ValueError("no SQL migrations found")
        with self._connect(register_vectors=False) as connection:
            for migration in migrations:
                connection.execute(migration.read_text(encoding="utf-8"))

    def _connect(
        self,
        *,
        register_vectors: bool = True,
        workspace_id: str | None = None,
    ):
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as error:
            raise RuntimeError("install SourceLens with the 'storage' extra") from error
        connection = psycopg.connect(self.dsn)
        if register_vectors:
            register_vector(connection)
        if workspace_id:
            connection.execute(
                "SELECT set_config('app.workspace_id', %s, true)",
                (workspace_id,),
            )
        return connection


def _document_from_row(row) -> StoredDocument:
    return StoredDocument(
        workspace_id=str(row[0]),
        document_id=row[1],
        content_sha256=row[2],
        original_filename=row[3],
        character_count=row[4],
        status=row[5],
        created_at=row[6],
    )


def _search_hit_from_row(row) -> StoredSearchHit:
    raw_embedding = row[6].to_list() if hasattr(row[6], "to_list") else row[6]
    embedding = tuple(float(value) for value in raw_embedding)
    passage = StoredPassage(
        workspace_id=str(row[0]),
        passage_id=row[1],
        document_id=row[2],
        content=row[3],
        start_offset=row[4],
        end_offset=row[5],
        embedding=embedding,
        embedding_method=row[7],
    )
    return StoredSearchHit(passage=passage, score=round(float(row[8]), 6))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _as_vector(values: Sequence[float]):
    try:
        from pgvector import Vector
    except ImportError as error:
        raise RuntimeError("install SourceLens with the 'storage' extra") from error
    return Vector(list(values))


def utc_now() -> datetime:
    return datetime.now(UTC)
