"""FastAPI delivery layer."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .analyzer import PairwiseAnalyzer
from .retrieval_api import router as retrieval_router

MAX_DOCUMENT_LENGTH = 100_000
PACKAGE_WEB_DIRECTORY = Path(__file__).resolve().parent / "web"
PROJECT_WEB_DIRECTORY = Path(__file__).resolve().parents[2] / "web"
WEB_DIRECTORY = (
    PACKAGE_WEB_DIRECTORY if PACKAGE_WEB_DIRECTORY.is_dir() else PROJECT_WEB_DIRECTORY
)


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    source: str = Field(min_length=10, max_length=MAX_DOCUMENT_LENGTH)
    candidate: str = Field(min_length=10, max_length=MAX_DOCUMENT_LENGTH)


class EvidenceResponse(BaseModel):
    source_text: str
    candidate_text: str
    source_start: int
    source_end: int
    candidate_start: int
    candidate_end: int
    similarity: float
    match_type: str


class AnalysisResponse(BaseModel):
    similarity_score: float
    verdict: str
    lexical_similarity: float
    character_similarity: float
    candidate_coverage: float
    evidence: list[EvidenceResponse]
    method: str
    score_interpretation: str


app = FastAPI(
    title="SourceLens",
    description="Pairwise text similarity with human-reviewable evidence.",
    version=__version__,
)
analyzer = PairwiseAnalyzer()
app.mount("/static", StaticFiles(directory=WEB_DIRECTORY), name="static")
app.include_router(retrieval_router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request) -> HTMLResponse:
    del request
    with (WEB_DIRECTORY / "index.html").open(encoding="utf-8") as page:
        return HTMLResponse(page.read())


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/v1/analyze", response_model=AnalysisResponse)
async def analyze(payload: AnalysisRequest) -> dict[str, object]:
    return analyzer.analyze(payload.source, payload.candidate).to_dict()
