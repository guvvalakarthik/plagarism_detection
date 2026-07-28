"""Authenticated asynchronous ingestion and review API."""

from __future__ import annotations

import os
import uuid
from dataclasses import asdict, dataclass
from typing import Annotated

from fastapi import (
    APIRouter,
    File,
    Header,
    HTTPException,
    Path,
    Response,
    UploadFile,
    status,
)
from pydantic import BaseModel, ConfigDict, Field

from .auth import ApiKeyAuthenticator
from .extractors import MAX_UPLOAD_BYTES
from .ingestion import CorpusIngestionService
from .observability import MetricsRegistry, SlidingWindowRateLimiter
from .retrieval import Embedder, HashingNgramEmbedder
from .storage import CorpusRepository, PostgresCorpusRepository
from .workflow import (
    AsyncIngestionWorkflow,
    PostgresWorkflowRepository,
    ReviewFeedback,
)


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    document_id: str = Field(min_length=1, max_length=200)
    evidence_id: str = Field(min_length=1, max_length=300)
    decision: str = Field(
        pattern="^(accepted_match|dismissed|properly_cited|common_phrase)$"
    )
    note: str | None = Field(default=None, max_length=2_000)


class SearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(min_length=3, max_length=5_000)
    limit: int = Field(default=10, ge=1, le=50)


@dataclass(slots=True)
class WorkflowRuntime:
    authenticator: ApiKeyAuthenticator
    jobs: AsyncIngestionWorkflow
    repository: PostgresWorkflowRepository
    corpus_repository: CorpusRepository
    embedder: Embedder
    metrics: MetricsRegistry
    limiter: SlidingWindowRateLimiter


_runtime: WorkflowRuntime | None = None


def configure_runtime(runtime: WorkflowRuntime | None) -> None:
    global _runtime
    _runtime = runtime


def runtime_from_environment() -> WorkflowRuntime:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is required")
    authenticator = ApiKeyAuthenticator.from_json(
        os.getenv("SOURCELENS_API_KEYS_JSON")
    )
    corpus_repository = PostgresCorpusRepository(database_url)
    embedder = HashingNgramEmbedder(256)
    metrics = MetricsRegistry()
    workflow_repository = PostgresWorkflowRepository(database_url)
    ingestion = CorpusIngestionService(corpus_repository, embedder)
    jobs = AsyncIngestionWorkflow(
        corpus_repository=corpus_repository,
        ingestion=ingestion,
        workflow_repository=workflow_repository,
        metrics=metrics,
    )
    return WorkflowRuntime(
        authenticator=authenticator,
        jobs=jobs,
        repository=workflow_repository,
        corpus_repository=corpus_repository,
        embedder=embedder,
        metrics=metrics,
        limiter=SlidingWindowRateLimiter(),
    )


def get_runtime() -> WorkflowRuntime:
    global _runtime
    if _runtime is None:
        try:
            _runtime = runtime_from_environment()
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="production workflow is not configured",
            ) from error
    return _runtime


def authorize(
    workspace_id: str,
    api_key: str | None,
) -> WorkflowRuntime:
    runtime = get_runtime()
    principal = runtime.authenticator.authenticate(api_key)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing API key",
        )
    if principal.workspace_id != workspace_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key cannot access this workspace",
        )
    if not runtime.limiter.allow(workspace_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="workspace rate limit exceeded",
        )
    return runtime


router = APIRouter(prefix="/v1/workspaces", tags=["production workflow"])


@router.post(
    "/{workspace_id}/documents",
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document(
    workspace_id: Annotated[uuid.UUID, Path()],
    file: Annotated[UploadFile, File()],
    x_api_key: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    workspace_value = str(workspace_id)
    runtime = authorize(workspace_value, x_api_key)
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="document exceeds the 10 MB limit",
        )
    job = runtime.jobs.submit(
        workspace_value,
        file.filename or "upload.txt",
        content,
    )
    return asdict(job)


@router.get("/{workspace_id}/jobs/{job_id}")
async def get_job(
    workspace_id: uuid.UUID,
    job_id: str,
    x_api_key: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    workspace_value = str(workspace_id)
    runtime = authorize(workspace_value, x_api_key)
    job = runtime.repository.get_job(workspace_value, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return asdict(job)


@router.post("/{workspace_id}/search")
async def search_workspace(
    workspace_id: uuid.UUID,
    payload: SearchRequest,
    x_api_key: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    workspace_value = str(workspace_id)
    runtime = authorize(workspace_value, x_api_key)
    query_embedding = runtime.embedder.encode([payload.query])[0]
    hits = runtime.corpus_repository.search(
        workspace_value,
        query_embedding,
        payload.limit,
    )
    runtime.metrics.increment("workspace_search_total", outcome="success")
    return {
        "query": payload.query,
        "method": runtime.embedder.name,
        "hits": [
            {
                "passage_id": hit.passage.passage_id,
                "document_id": hit.passage.document_id,
                "content": hit.passage.content,
                "start_offset": hit.passage.start_offset,
                "end_offset": hit.passage.end_offset,
                "score": hit.score,
                "embedding_method": hit.passage.embedding_method,
            }
            for hit in hits
        ],
    }


@router.post(
    "/{workspace_id}/feedback",
    status_code=status.HTTP_201_CREATED,
)
async def submit_feedback(
    workspace_id: uuid.UUID,
    payload: FeedbackRequest,
    x_api_key: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    workspace_value = str(workspace_id)
    runtime = authorize(workspace_value, x_api_key)
    feedback = ReviewFeedback(
        workspace_id=workspace_value,
        feedback_id=str(uuid.uuid4()),
        document_id=payload.document_id,
        evidence_id=payload.evidence_id,
        decision=payload.decision,
        note=payload.note,
        created_at=None,
    )
    runtime.repository.add_feedback(feedback)
    runtime.metrics.increment("review_feedback_total", decision=payload.decision)
    return asdict(feedback)


@router.get("/-/metrics", include_in_schema=False)
async def metrics() -> Response:
    runtime = get_runtime()
    return Response(
        runtime.metrics.render_prometheus(),
        media_type="text/plain; version=0.0.4",
    )
