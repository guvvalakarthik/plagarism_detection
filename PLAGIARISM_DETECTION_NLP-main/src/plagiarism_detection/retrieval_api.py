"""In-memory corpus search API used for local evaluation and demonstrations."""

from __future__ import annotations

import threading

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from .retrieval import HashingNgramEmbedder, HybridRetriever, chunk_document


class CorpusDocument(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    document_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=10, max_length=500_000)


class IndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[CorpusDocument] = Field(min_length=1, max_length=1_000)
    max_tokens: int = Field(default=120, ge=8, le=1_000)
    overlap_tokens: int = Field(default=24, ge=0, le=500)


class SearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(min_length=3, max_length=100_000)
    top_k: int = Field(default=10, ge=1, le=100)
    candidate_k: int = Field(default=50, ge=1, le=500)


class InMemoryCorpusService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._retriever = HybridRetriever(embedder=HashingNgramEmbedder())

    def index(self, request: IndexRequest) -> dict[str, object]:
        passages = [
            passage
            for document in request.documents
            for passage in chunk_document(
                document.document_id,
                document.text,
                max_tokens=request.max_tokens,
                overlap_tokens=request.overlap_tokens,
            )
        ]
        with self._lock:
            self._retriever.index(passages)
        return {
            "documents": len(request.documents),
            "passages": len(passages),
            "method": self._retriever.method,
            "persistence": "memory",
        }

    def search(self, request: SearchRequest) -> dict[str, object]:
        with self._lock:
            hits = self._retriever.search(
                request.query,
                top_k=request.top_k,
                candidate_k=max(request.top_k, request.candidate_k),
            )
        return {
            "method": self._retriever.method,
            "hits": [
                {
                    "passage_id": hit.passage.passage_id,
                    "document_id": hit.passage.document_id,
                    "text": hit.passage.text,
                    "start": hit.passage.start,
                    "end": hit.passage.end,
                    "score": hit.score,
                    "lexical_rank": hit.lexical_rank,
                    "dense_rank": hit.dense_rank,
                    "reranker_score": hit.reranker_score,
                }
                for hit in hits
            ],
        }


router = APIRouter(prefix="/v1/corpus", tags=["corpus retrieval"])
service = InMemoryCorpusService()


@router.post("/index")
async def index_corpus(payload: IndexRequest) -> dict[str, object]:
    return service.index(payload)


@router.post("/search")
async def search_corpus(payload: SearchRequest) -> dict[str, object]:
    return service.search(payload)
