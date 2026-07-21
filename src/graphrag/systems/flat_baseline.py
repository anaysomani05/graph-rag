from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from graphrag.eval.schema import PredictionResult

DEFAULT_ABSTRACTS_PATH = Path(__file__).resolve().parents[3] / "eval" / "source_abstracts.jsonl"


def _load_chunks(path: Path) -> list[dict]:
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                chunks.append(
                    {
                        "chunk_id": f"{row['arxiv_id']}#abstract",
                        "text": row["abstract"],
                        "title": row["title"],
                    }
                )
    return chunks


class FlatVectorBaseline:
    """The control-group system: single-vector cosine search over chunk embeddings,
    no graph hops, no reranker, no LLM synthesis. Exists to produce a real, honest
    "bad" number before any hybrid-retrieval work starts — see eval/README.md.

    Predicted answer is just the top-1 chunk's raw text truncated to one sentence,
    which is deliberately crude: this system isn't meant to answer well, it's meant
    to be the thing hybrid retrieval has to beat.
    """

    name = "flat_baseline"

    def __init__(
        self,
        chunks_path: Path = DEFAULT_ABSTRACTS_PATH,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.chunks = _load_chunks(chunks_path)
        self.model = SentenceTransformer(model_name)
        self.chunk_embeddings = self.model.encode(
            [c["text"] for c in self.chunks], normalize_embeddings=True
        )

    def answer(self, question: str) -> PredictionResult:
        start = time.perf_counter()
        query_embedding = self.model.encode([question], normalize_embeddings=True)[0]
        scores = self.chunk_embeddings @ query_embedding
        ranked = np.argsort(-scores)
        retrieved_chunk_ids = [self.chunks[i]["chunk_id"] for i in ranked]
        top_chunk = self.chunks[ranked[0]]
        predicted_answer = top_chunk["text"].split(". ")[0] + "."
        latency_ms = (time.perf_counter() - start) * 1000
        return PredictionResult(
            question_id="unused",
            predicted_answer=predicted_answer,
            retrieved_chunk_ids=retrieved_chunk_ids,
            latency_ms=latency_ms,
        )
