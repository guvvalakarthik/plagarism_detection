"""Asynchronous ingestion jobs and persisted reviewer feedback."""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal, Protocol

from .extractors import DocumentExtractionError, extract_document
from .ingestion import CorpusIngestionService
from .observability import MetricsRegistry
from .storage import CorpusRepository, utc_now

JobStatus = Literal["queued", "processing", "ready", "failed"]
ReviewDecision = Literal[
    "accepted_match", "dismissed", "properly_cited", "common_phrase"
]


@dataclass(frozen=True, slots=True)
class IngestionJob:
    workspace_id: str
    job_id: str
    filename: str
    status: JobStatus
    document_id: str | None = None
    error_code: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ReviewFeedback:
    workspace_id: str
    feedback_id: str
    document_id: str
    evidence_id: str
    decision: ReviewDecision
    note: str | None
    created_at: datetime | None = None


class WorkflowRepository(Protocol):
    def create_job(self, job: IngestionJob) -> None: ...

    def update_job(self, job: IngestionJob) -> None: ...

    def get_job(self, workspace_id: str, job_id: str) -> IngestionJob | None: ...

    def add_feedback(self, feedback: ReviewFeedback) -> None: ...


class MemoryWorkflowRepository:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.jobs: dict[tuple[str, str], IngestionJob] = {}
        self.feedback: dict[tuple[str, str], ReviewFeedback] = {}

    def create_job(self, job: IngestionJob) -> None:
        with self._lock:
            key = (job.workspace_id, job.job_id)
            if key in self.jobs:
                raise ValueError("job already exists")
            self.jobs[key] = job

    def update_job(self, job: IngestionJob) -> None:
        with self._lock:
            key = (job.workspace_id, job.job_id)
            if key not in self.jobs:
                raise ValueError("job does not exist")
            self.jobs[key] = job

    def get_job(self, workspace_id: str, job_id: str) -> IngestionJob | None:
        with self._lock:
            return self.jobs.get((workspace_id, job_id))

    def add_feedback(self, feedback: ReviewFeedback) -> None:
        with self._lock:
            self.feedback[(feedback.workspace_id, feedback.feedback_id)] = feedback


class PostgresWorkflowRepository:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def create_job(self, job: IngestionJob) -> None:
        with self._connect(job.workspace_id) as connection:
            connection.execute(
                """
                INSERT INTO ingestion_jobs (
                    workspace_id, job_id, filename, status,
                    document_id, error_code
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    job.workspace_id,
                    job.job_id,
                    job.filename,
                    job.status,
                    job.document_id,
                    job.error_code,
                ),
            )

    def update_job(self, job: IngestionJob) -> None:
        with self._connect(job.workspace_id) as connection:
            cursor = connection.execute(
                """
                UPDATE ingestion_jobs
                   SET status = %s, document_id = %s, error_code = %s,
                       updated_at = now()
                 WHERE workspace_id = %s AND job_id = %s
                """,
                (
                    job.status,
                    job.document_id,
                    job.error_code,
                    job.workspace_id,
                    job.job_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("job does not exist in workspace")

    def get_job(self, workspace_id: str, job_id: str) -> IngestionJob | None:
        with self._connect(workspace_id) as connection:
            row = connection.execute(
                """
                SELECT workspace_id, job_id, filename, status, document_id,
                       error_code, created_at, updated_at
                  FROM ingestion_jobs
                 WHERE workspace_id = %s AND job_id = %s
                """,
                (workspace_id, job_id),
            ).fetchone()
        return _job_from_row(row) if row else None

    def add_feedback(self, feedback: ReviewFeedback) -> None:
        with self._connect(feedback.workspace_id) as connection:
            connection.execute(
                """
                INSERT INTO review_feedback (
                    workspace_id, feedback_id, document_id,
                    evidence_id, decision, note
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    feedback.workspace_id,
                    feedback.feedback_id,
                    feedback.document_id,
                    feedback.evidence_id,
                    feedback.decision,
                    feedback.note,
                ),
            )

    def _connect(self, workspace_id: str):
        try:
            import psycopg
        except ImportError as error:
            raise RuntimeError("install SourceLens with the 'storage' extra") from error
        connection = psycopg.connect(self.dsn)
        connection.execute(
            "SELECT set_config('app.workspace_id', %s, true)",
            (workspace_id,),
        )
        return connection


class AsyncIngestionWorkflow:
    def __init__(
        self,
        *,
        corpus_repository: CorpusRepository,
        ingestion: CorpusIngestionService,
        workflow_repository: WorkflowRepository,
        metrics: MetricsRegistry,
        workers: int = 2,
    ) -> None:
        self.corpus_repository = corpus_repository
        self.ingestion = ingestion
        self.workflow_repository = workflow_repository
        self.metrics = metrics
        self.executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="sourcelens-ingest"
        )

    def submit(self, workspace_id: str, filename: str, content: bytes) -> IngestionJob:
        self.corpus_repository.ensure_workspace(workspace_id)
        now = utc_now()
        job = IngestionJob(
            workspace_id=workspace_id,
            job_id=str(uuid.uuid4()),
            filename=filename,
            status="queued",
            created_at=now,
            updated_at=now,
        )
        self.workflow_repository.create_job(job)
        self.executor.submit(self._run, job, content)
        self.metrics.increment("ingestion_jobs_total", status="queued")
        return job

    def _run(self, job: IngestionJob, content: bytes) -> None:
        processing = replace(job, status="processing", updated_at=utc_now())
        self.workflow_repository.update_job(processing)
        try:
            text = extract_document(job.filename, content)
            result = self.ingestion.ingest(
                workspace_id=job.workspace_id,
                filename=job.filename,
                text=text,
            )
            ready = replace(
                processing,
                status="ready",
                document_id=result.document_id,
                updated_at=utc_now(),
            )
            self.workflow_repository.update_job(ready)
            self.metrics.increment("ingestion_jobs_total", status="ready")
        except DocumentExtractionError:
            failed = replace(
                processing,
                status="failed",
                error_code="document_extraction_failed",
                updated_at=utc_now(),
            )
            self.workflow_repository.update_job(failed)
            self.metrics.increment("ingestion_jobs_total", status="failed")
        except Exception:
            failed = replace(
                processing,
                status="failed",
                error_code="internal_processing_error",
                updated_at=utc_now(),
            )
            self.workflow_repository.update_job(failed)
            self.metrics.increment("ingestion_jobs_total", status="failed")

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)


def _job_from_row(row) -> IngestionJob:
    return IngestionJob(
        workspace_id=str(row[0]),
        job_id=row[1],
        filename=row[2],
        status=row[3],
        document_id=row[4],
        error_code=row[5],
        created_at=row[6],
        updated_at=row[7],
    )
