from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from graphrag.orchestration.graph import GraphRAGPipeline

_pipeline: GraphRAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eager init at startup, not on first request — the pipeline loads two
    # sentence-transformer models (embedding + cross-encoder) and opens a DB
    # connection; deferring that to the first /query call would misattribute a
    # multi-second model-load cost to that request's latency.
    global _pipeline
    _pipeline = GraphRAGPipeline()

    # Warm-up call: observed once that the very first real inference through a
    # freshly-loaded cross-encoder produced a lower relevance score than every
    # subsequent call on the identical input (4/4 repeats in the same process were
    # consistent; only the first call after a fresh model load differed) — enough
    # to occasionally trip the orchestration's relevance gate on a genuinely
    # answerable question. Running one throwaway query during startup absorbs
    # that first-call variance before real traffic arrives, instead of the first
    # user's request risking an incorrect "insufficient evidence" response.
    _pipeline.answer("warm-up query, response is discarded")
    yield


app = FastAPI(title="GraphRAG Research Assistant", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str


class CitationResponse(BaseModel):
    claim: str
    chunk_ids: list[str]


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    retrieved_chunk_ids: list[str]
    latency_ms: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    assert _pipeline is not None, "pipeline not initialized"
    result = _pipeline.answer(request.question)
    return QueryResponse(
        answer=result.predicted_answer,
        citations=[CitationResponse(claim=c.claim, chunk_ids=c.chunk_ids) for c in result.citations],
        retrieved_chunk_ids=result.retrieved_chunk_ids,
        latency_ms=result.latency_ms,
    )
