import time
import uuid

from fastapi.testclient import TestClient

from plagiarism_detection.api import app
from plagiarism_detection.auth import ApiKeyAuthenticator
from plagiarism_detection.ingestion import CorpusIngestionService
from plagiarism_detection.observability import (
    MetricsRegistry,
    SlidingWindowRateLimiter,
)
from plagiarism_detection.retrieval import HashingNgramEmbedder
from plagiarism_detection.storage import MemoryCorpusRepository
from plagiarism_detection.workflow import (
    AsyncIngestionWorkflow,
    MemoryWorkflowRepository,
)
from plagiarism_detection.workflow_api import WorkflowRuntime, configure_runtime

client = TestClient(app)
API_KEY = "test-api-key-with-enough-entropy"


def runtime(workspace_id: str) -> WorkflowRuntime:
    corpus = MemoryCorpusRepository()
    repository = MemoryWorkflowRepository()
    metrics = MetricsRegistry()
    jobs = AsyncIngestionWorkflow(
        corpus_repository=corpus,
        ingestion=CorpusIngestionService(corpus, HashingNgramEmbedder(32)),
        workflow_repository=repository,
        metrics=metrics,
        workers=1,
    )
    return WorkflowRuntime(
        authenticator=ApiKeyAuthenticator({API_KEY: workspace_id}),
        jobs=jobs,
        repository=repository,
        metrics=metrics,
        limiter=SlidingWindowRateLimiter(requests=100),
    )


def test_authenticated_upload_job_and_feedback() -> None:
    workspace_id = str(uuid.uuid4())
    configured = runtime(workspace_id)
    configure_runtime(configured)
    headers = {"x-api-key": API_KEY}

    upload = client.post(
        f"/v1/workspaces/{workspace_id}/documents",
        headers=headers,
        files={
            "file": (
                "source.txt",
                b"Machine learning systems need representative evaluation evidence.",
                "text/plain",
            )
        },
    )

    assert upload.status_code == 202
    job_id = upload.json()["job_id"]
    job = None
    for _ in range(100):
        response = client.get(
            f"/v1/workspaces/{workspace_id}/jobs/{job_id}",
            headers=headers,
        )
        job = response.json()
        if job["status"] in {"ready", "failed"}:
            break
        time.sleep(0.01)
    assert job["status"] == "ready"

    feedback = client.post(
        f"/v1/workspaces/{workspace_id}/feedback",
        headers=headers,
        json={
            "document_id": job["document_id"],
            "evidence_id": "candidate:0:20",
            "decision": "properly_cited",
            "note": "Quotation includes a source citation.",
        },
    )
    assert feedback.status_code == 201
    assert feedback.json()["decision"] == "properly_cited"

    metrics = client.get("/v1/workspaces/-/metrics")
    assert metrics.status_code == 200
    assert "sourcelens_ingestion_jobs_total" in metrics.text
    configured.jobs.shutdown()
    configure_runtime(None)


def test_workflow_rejects_missing_and_cross_workspace_credentials() -> None:
    workspace_id = str(uuid.uuid4())
    configured = runtime(workspace_id)
    configure_runtime(configured)

    missing = client.post(
        f"/v1/workspaces/{workspace_id}/documents",
        files={"file": ("source.txt", b"enough text for a document", "text/plain")},
    )
    forbidden = client.post(
        f"/v1/workspaces/{uuid.uuid4()}/documents",
        headers={"x-api-key": API_KEY},
        files={"file": ("source.txt", b"enough text for a document", "text/plain")},
    )

    assert missing.status_code == 401
    assert forbidden.status_code == 403
    configured.jobs.shutdown()
    configure_runtime(None)


def test_request_id_is_propagated() -> None:
    response = client.get("/health", headers={"x-request-id": "candidate-request-1"})

    assert response.headers["x-request-id"] == "candidate-request-1"
