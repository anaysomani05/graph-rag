from __future__ import annotations

import time

from graphrag.eval.schema import PredictionResult
from graphrag.systems.hybrid_retrieval import HybridRetrieval
from graphrag.systems.reranker import CrossEncoderReranker


class RerankedHybridRetrieval:
    """Hybrid retrieval's merged candidates, reranked by a cross-encoder before
    being returned. Exists to isolate the reranker's effect: same retrieval
    candidates as `HybridRetrieval`, only the ordering changes, so a precision@5
    delta between the two systems is attributable to reranking alone.
    """

    name = "hybrid_reranked"

    def __init__(
        self,
        base: HybridRetrieval | None = None,
        reranker: CrossEncoderReranker | None = None,
        rerank_top_n: int = 20,
    ):
        self.base = base or HybridRetrieval()
        self.reranker = reranker or CrossEncoderReranker()
        self.rerank_top_n = rerank_top_n
        self.conn = self.base.conn

    def answer(self, question: str) -> PredictionResult:
        start = time.perf_counter()
        base_result = self.base.answer(question)
        candidate_ids = base_result.retrieved_chunk_ids[: self.rerank_top_n]

        rows = self.conn.execute(
            "SELECT chunk_id, text FROM chunks WHERE chunk_id = ANY(%s)", (candidate_ids,)
        ).fetchall()
        text_by_id = dict(rows)
        # preserve hybrid's original order for any id whose text lookup somehow misses
        candidates = [(cid, text_by_id[cid]) for cid in candidate_ids if cid in text_by_id]

        reranked_ids = self.reranker.rerank(question, candidates)
        predicted_answer = (text_by_id[reranked_ids[0]].split(". ")[0] + ".") if reranked_ids else ""
        latency_ms = (time.perf_counter() - start) * 1000
        return PredictionResult(
            question_id="unused",
            predicted_answer=predicted_answer,
            retrieved_chunk_ids=reranked_ids,
            latency_ms=latency_ms,
        )
