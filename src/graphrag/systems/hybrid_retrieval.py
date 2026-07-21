from __future__ import annotations

import time

from sentence_transformers import SentenceTransformer

from graphrag.config import settings
from graphrag.eval.schema import PredictionResult
from graphrag.graph.store import neighbor_chunk_ids, subtopic_bridge_chunk_ids
from graphrag.ingestion.store import get_connection


class HybridRetrieval:
    """Vector top-k merged with graph-hop neighbors of the top vector hits.

    For each of the top `seed_k` vector hits, in rank order, two kinds of graph
    expansion are inserted immediately after it:
    - discrete entity-graph neighbors (shared/near-duplicate entities, hub-degree
      capped — see neighbor_chunk_ids)
    - subtopic-embedding bridges (semantically related but differently-worded
      sub-problems, e.g. "dense retrieval latency" vs. "in-storage vector search
      acceleration" — see subtopic_bridge_chunk_ids)

    Both exist because exact entity dedup is deliberately strict (it's meant to
    merge surface-form duplicates, not cluster related concepts), so a second
    paper that's thematically relevant but textually dissimilar to the question
    would otherwise never surface from vector search alone or from the discrete
    graph hop.

    Graph expansion is anchored on each seed's *paper-level* abstract chunk, not
    the literal seed chunk id. Extraction only ran on abstracts (see extract.py),
    so no edge is ever keyed on a body chunk id like "{arxiv_id}#chunk21" — but
    most vector-search seeds are body chunks once full-text chunking exists (they
    tend to out-rank abstracts for specific factual questions). Looking up graph
    neighbors by the literal seed id would silently find nothing for almost every
    seed; looking up by "this chunk's paper's abstract" is what the graph
    construction actually supports.

    Ranking: all `seed_k` pure-vector hits are placed first, unmodified, THEN graph
    expansions fill the remaining slots up to final_k. An earlier version inserted
    each seed's expansions immediately after it — with seed_k equal to the harness's
    scoring k, that let expansions from the very first seed push seeds 2..k (often
    the actually-correct second paper) out of the scored window entirely, making
    hybrid strictly worse than flat vector search rather than additive. Keeping the
    top seed_k pure-vector ranks untouched means hybrid can only add value in the
    marginal slots, never remove a hit flat search would have found on its own.
    """

    name = "hybrid"

    def __init__(
        self,
        model_name: str = settings.embedding_model,
        seed_k: int = 3,
        hops: int = 1,
        final_k: int = 20,
        max_expansions_per_seed: int = 3,
    ):
        self.model = SentenceTransformer(model_name)
        self.conn = get_connection()
        self.seed_k = seed_k
        self.hops = hops
        self.final_k = final_k
        self.max_expansions_per_seed = max_expansions_per_seed

    def answer(self, question: str) -> PredictionResult:
        start = time.perf_counter()
        query_embedding = self.model.encode(question, normalize_embeddings=True)

        seed_rows = self.conn.execute(
            "SELECT chunk_id, text FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (query_embedding, self.seed_k),
        ).fetchall()

        merged: list[str] = []
        seen: set[str] = set()

        def _add(chunk_id: str) -> None:
            if chunk_id not in seen:
                merged.append(chunk_id)
                seen.add(chunk_id)

        for chunk_id, _ in seed_rows:
            _add(chunk_id)

        for chunk_id, _ in seed_rows:
            abstract_chunk_id = f"{chunk_id.split('#')[0]}#abstract"
            expansions = list(neighbor_chunk_ids(self.conn, abstract_chunk_id, hops=self.hops))
            expansions += subtopic_bridge_chunk_ids(self.conn, abstract_chunk_id)
            for candidate in expansions[: self.max_expansions_per_seed]:
                _add(candidate)

        if len(merged) < self.final_k:
            extra_rows = self.conn.execute(
                "SELECT chunk_id FROM chunks ORDER BY embedding <=> %s LIMIT %s",
                (query_embedding, self.final_k * 3),
            ).fetchall()
            for (chunk_id,) in extra_rows:
                if len(merged) >= self.final_k:
                    break
                _add(chunk_id)

        predicted_answer = (seed_rows[0][1].split(". ")[0] + ".") if seed_rows else ""
        latency_ms = (time.perf_counter() - start) * 1000
        return PredictionResult(
            question_id="unused",
            predicted_answer=predicted_answer,
            retrieved_chunk_ids=merged[: self.final_k],
            latency_ms=latency_ms,
        )
