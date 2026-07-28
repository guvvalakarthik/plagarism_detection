# PostgreSQL and pgvector corpus storage

The storage layer keeps document metadata, offset-preserving passages, and
embeddings behind a tenant-scoped repository contract.

## Invariants

- Every document and passage belongs to a workspace.
- Every repository read, write, update, and delete includes `workspace_id`.
- Composite primary and foreign keys prevent cross-workspace references.
- Content SHA-256 is unique per workspace, making ingestion idempotent.
- Passage offsets are half-open character intervals into the original text.
- Embedding dimensions and method identity are validated before persistence.
- Row-level-security policies provide a database-level second boundary for
  non-owner application roles.

## Local database

```bash
docker compose up --build
```

The compose file binds PostgreSQL to `127.0.0.1:5433`, applies migrations, and
starts the API only after the database is healthy and migrations succeed.

The checked-in password is explicitly local-development-only. Production
credentials must come from a secret manager.

## Migration

```bash
pip install -e ".[storage]"
set DATABASE_URL=postgresql://sourcelens:password@localhost:5432/sourcelens
python scripts/migrate.py
```

Migration `001_corpus.sql` enables pgvector, creates composite tenant keys,
creates a cosine HNSW index, and installs workspace row-level-security policies.

## Repository boundary

`CorpusIngestionService` depends on `CorpusRepository`, not PostgreSQL directly.
Unit tests use `MemoryCorpusRepository`; deployment uses
`PostgresCorpusRepository`. This keeps chunking, hashing, embedding, and
idempotency behavior testable without weakening production query scoping.
