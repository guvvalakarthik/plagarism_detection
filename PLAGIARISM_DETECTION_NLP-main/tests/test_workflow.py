import time
import uuid

from plagiarism_detection.ingestion import CorpusIngestionService
from plagiarism_detection.observability import MetricsRegistry
from plagiarism_detection.retrieval import HashingNgramEmbedder
from plagiarism_detection.storage import MemoryCorpusRepository
from plagiarism_detection.workflow import (
    AsyncIngestionWorkflow,
    MemoryWorkflowRepository,
)


def wait_for_terminal(repository, workspace_id, job_id):
    for _ in range(100):
        job = repository.get_job(workspace_id, job_id)
        if job and job.status in {"ready", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError("job did not reach a terminal state")


def test_async_ingestion_reaches_ready_without_storing_bytes() -> None:
    corpus = MemoryCorpusRepository()
    repository = MemoryWorkflowRepository()
    metrics = MetricsRegistry()
    workflow = AsyncIngestionWorkflow(
        corpus_repository=corpus,
        ingestion=CorpusIngestionService(corpus, HashingNgramEmbedder(32)),
        workflow_repository=repository,
        metrics=metrics,
        workers=1,
    )
    workspace_id = str(uuid.uuid4())

    queued = workflow.submit(
        workspace_id,
        "source.txt",
        b"Machine learning systems need representative evaluation evidence.",
    )
    ready = wait_for_terminal(repository, workspace_id, queued.job_id)
    workflow.shutdown()

    assert ready.status == "ready"
    assert ready.document_id
    assert not hasattr(ready, "content")
    assert 'status="ready"' in metrics.render_prometheus()


def test_extraction_failure_uses_stable_error_code() -> None:
    corpus = MemoryCorpusRepository()
    repository = MemoryWorkflowRepository()
    workflow = AsyncIngestionWorkflow(
        corpus_repository=corpus,
        ingestion=CorpusIngestionService(corpus, HashingNgramEmbedder(32)),
        workflow_repository=repository,
        metrics=MetricsRegistry(),
        workers=1,
    )
    workspace_id = str(uuid.uuid4())

    queued = workflow.submit(workspace_id, "malware.exe", b"unsupported")
    failed = wait_for_terminal(repository, workspace_id, queued.job_id)
    workflow.shutdown()

    assert failed.status == "failed"
    assert failed.error_code == "document_extraction_failed"
