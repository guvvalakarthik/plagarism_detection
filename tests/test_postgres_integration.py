import os
import uuid
from pathlib import Path

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from plagiarism_detection.ingestion import CorpusIngestionService
from plagiarism_detection.retrieval import HashingNgramEmbedder
from plagiarism_detection.storage import PostgresCorpusRepository

DATABASE_URL = os.getenv("SOURCELENS_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="SOURCELENS_TEST_DATABASE_URL is not configured",
)
APP_ROLE = "sourcelens_test_app"
APP_PASSWORD = "integration-test-only"


def restricted_database_url() -> str:
    parameters = conninfo_to_dict(DATABASE_URL)
    parameters.update(user=APP_ROLE, password=APP_PASSWORD)
    return make_conninfo(**parameters)


def provision_restricted_role() -> None:
    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        connection.execute(
            f"""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
                    CREATE ROLE {APP_ROLE} LOGIN PASSWORD '{APP_PASSWORD}';
                END IF;
            END
            $$;
            """
        )
        connection.execute(f"GRANT USAGE ON SCHEMA public TO {APP_ROLE}")
        connection.execute(
            f"""
            GRANT SELECT, INSERT, UPDATE, DELETE
            ON workspaces, documents, passages
            TO {APP_ROLE}
            """
        )


def test_postgres_ingestion_search_and_workspace_isolation() -> None:
    admin_repository = PostgresCorpusRepository(
        DATABASE_URL, embedding_dimensions=256
    )
    migrations = Path(__file__).resolve().parents[1] / "migrations"
    admin_repository.apply_migrations(migrations)
    provision_restricted_role()

    repository = PostgresCorpusRepository(
        restricted_database_url(),
        embedding_dimensions=256,
    )
    service = CorpusIngestionService(repository, HashingNgramEmbedder(256))
    first_workspace = str(uuid.uuid4())
    second_workspace = str(uuid.uuid4())
    first_text = "Machine learning systems require representative evaluation evidence."
    second_text = "Garden vegetables require healthy soil and careful watering."

    first = service.ingest(
        workspace_id=first_workspace,
        filename="ml.txt",
        text=first_text,
    )
    repeated = service.ingest(
        workspace_id=first_workspace,
        filename="renamed.txt",
        text=first_text,
    )
    service.ingest(
        workspace_id=second_workspace,
        filename="garden.txt",
        text=second_text,
    )

    hits = repository.search(
        first_workspace,
        HashingNgramEmbedder(256).encode([first_text])[0],
        limit=10,
    )

    assert repeated.duplicate is True
    assert repeated.document_id == first.document_id
    assert hits
    assert {hit.passage.workspace_id for hit in hits} == {first_workspace}
