from __future__ import annotations

import time

from groq import Groq

from graphrag.config import settings
from graphrag.eval.schema import PredictionResult
from graphrag.systems.reranked_hybrid import RerankedHybridRetrieval
from graphrag.systems.synthesis import SourceCandidate, synthesize_answer


class GroundedHybridSystem:
    """Full pipeline: hybrid retrieval -> cross-encoder rerank -> grounded LLM
    synthesis with per-claim citations. This is the system that F6 (grounded
    answers with citations) actually measures — the other systems in
    scripts/run_eval.py exist to isolate which stage (retrieval, graph hop,
    rerank) contributes what.
    """

    name = "hybrid_reranked_grounded"

    def __init__(
        self,
        reranked: RerankedHybridRetrieval | None = None,
        synthesis_top_n: int = 5,
        groq_client: Groq | None = None,
    ):
        self.reranked = reranked or RerankedHybridRetrieval()
        self.synthesis_top_n = synthesis_top_n
        self.client = groq_client or Groq(api_key=settings.groq_api_key)
        self.conn = self.reranked.conn

    def answer(self, question: str) -> PredictionResult:
        start = time.perf_counter()
        base_result = self.reranked.answer(question)
        top_ids = base_result.retrieved_chunk_ids[: self.synthesis_top_n]

        rows = self.conn.execute(
            """
            SELECT c.chunk_id, p.title, c.text
            FROM chunks c JOIN papers p ON p.arxiv_id = c.arxiv_id
            WHERE c.chunk_id = ANY(%s)
            """,
            (top_ids,),
        ).fetchall()
        by_id = {chunk_id: SourceCandidate(chunk_id, title, text) for chunk_id, title, text in rows}
        candidates = [by_id[cid] for cid in top_ids if cid in by_id]

        answer_text, citations = synthesize_answer(question, candidates, client=self.client)
        latency_ms = (time.perf_counter() - start) * 1000
        return PredictionResult(
            question_id="unused",
            predicted_answer=answer_text,
            retrieved_chunk_ids=base_result.retrieved_chunk_ids,
            latency_ms=latency_ms,
            citations=citations,
        )
