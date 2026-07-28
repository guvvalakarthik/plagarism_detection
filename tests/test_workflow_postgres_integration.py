import os
import uuid
from pathlib import Path

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from plagiarism_detection.ingestion import CorpusIngestionService
from plagiarism_detection.retrieval import HashingNgramEmbedder
from plagiarism_detection.storage import PostgresCorpusRepository, utc_now
from plagiarism_detection.workflow import (
    IngestionJob,
    PostgresWorkflowRepository,
    ReviewFeedback,
)

DATABASE_URL = os.getenv("SOURCELENS_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="SOURCELENS_TEST_DATABASE_URL is not configured",
)
APP_ROLE = "sourcelens_workflow_test_app"
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
            ON workspaces, documents, passages, ingestion_jobs, review_feedback
            TO {APP_ROLE}
            """
        )


def test_workflow_jobs_and_feedback_obey_database_workspace_policy() -> None:
    admin = PostgresCorpusRepository(DATABASE_URL, embedding_dimensions=256)
    admin.apply_migrations(Path(__file__).resolve().parents[1] / "migrations")
    provision_restricted_role()
    dsn = restricted_database_url()
    corpus = PostgresCorpusRepository(dsn, embedding_dimensions=256)
    workflow = PostgresWorkflowRepository(dsn)
    workspace_id = str(uuid.uuid4())
    other_workspace = str(uuid.uuid4())
    ingestion = CorpusIngestionService(corpus, HashingNgramEmbedder(256))
    result = ingestion.ingest(
        workspace_id=workspace_id,
        filename="evidence.txt",
        text="A production workflow needs durable jobs and reviewer evidence.",
    )
    corpus.ensure_workspace(other_workspace)
    now = utc_now()
    queued = IngestionJob(
        workspace_id=workspace_id,
        job_id=str(uuid.uuid4()),
        filename="evidence.txt",
        status="queued",
        created_at=now,
        updated_at=now,
    )
    workflow.create_job(queued)
    ready = IngestionJob(
        workspace_id=workspace_id,
        job_id=queued.job_id,
        filename=queued.filename,
        status="ready",
        document_id=result.document_id,
        created_at=now,
        updated_at=utc_now(),
    )
    workflow.update_job(ready)
    workflow.add_feedback(
        ReviewFeedback(
            workspace_id=workspace_id,
            feedback_id=str(uuid.uuid4()),
            document_id=result.document_id,
            evidence_id="candidate:0:30",
            decision="accepted_match",
            note="Confirmed during integration testing.",
            created_at=utc_now(),
        )
    )

    stored = workflow.get_job(workspace_id, queued.job_id)

    assert stored.status == "ready"
    assert stored.document_id == result.document_id
    assert workflow.get_job(other_workspace, queued.job_id) is None
