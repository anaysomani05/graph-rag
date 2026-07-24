from __future__ import annotations

import time

from sentence_transformers import SentenceTransformer

from graphrag.config import settings
from graphrag.eval.schema import PredictionResult
from graphrag.ingestion.store import get_connection


class FlatVectorBaseline:
    """The control-group system: single-vector cosine search over chunk embeddings in
    Postgres+pgvector, no graph hops, no reranker, no LLM synthesis. Exists to produce
    a real, honest "bad" number that hybrid retrieval has to beat — see eval/README.md.

    Predicted answer is just the top-1 chunk's first sentence, deliberately crude:
    this system isn't meant to answer well, it isolates retrieval quality from
    generation quality (LLM synthesis is added later, once hybrid retrieval exists).
    """

    name = "flat_baseline"

    def __init__(self, model_name: str = settings.embedding_model, top_k: int = 20):
        # device="cpu": see hybrid_retrieval.py's HybridRetrieval.__init__ for why.
        self.model = SentenceTransformer(model_name, device="cpu")
        self.conn = get_connection()
        self.top_k = top_k

    def answer(self, question: str) -> PredictionResult:
        start = time.perf_counter()
        query_embedding = self.model.encode(question, normalize_embeddings=True)
        rows = self.conn.execute(
            """
            SELECT chunk_id, text
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query_embedding, self.top_k),
        ).fetchall()
        retrieved_chunk_ids = [r[0] for r in rows]
        predicted_answer = (rows[0][1].split(". ")[0] + ".") if rows else ""
        latency_ms = (time.perf_counter() - start) * 1000
        return PredictionResult(
            question_id="unused",
            predicted_answer=predicted_answer,
            retrieved_chunk_ids=retrieved_chunk_ids,
            latency_ms=latency_ms,
        )
