from __future__ import annotations

from sentence_transformers import CrossEncoder


def apply_diversity_cap(ranked_chunk_ids: list[str], max_per_paper: int) -> list[str]:
    """Given chunk ids already sorted best-first, returns them reordered so no more
    than `max_per_paper` chunks from the same paper occupy the front — excess
    same-paper chunks are pushed after all other papers' candidates, not dropped.

    Split out from CrossEncoderReranker.rerank so this policy (the actual fix for
    the redundancy-collapse bug, see class docstring) is testable without loading
    the cross-encoder model.
    """
    selected: list[str] = []
    deferred: list[str] = []
    per_paper_count: dict[str, int] = {}
    for chunk_id in ranked_chunk_ids:
        paper = chunk_id.split("#")[0]
        if per_paper_count.get(paper, 0) < max_per_paper:
            selected.append(chunk_id)
            per_paper_count[paper] = per_paper_count.get(paper, 0) + 1
        else:
            deferred.append(chunk_id)
    return selected + deferred


class CrossEncoderReranker:
    """Off-the-shelf pretrained cross-encoder, no fine-tuning — F5's v1 scope per
    the build plan. Reranks a set of retrieved candidates against the question by
    running each (question, chunk_text) pair jointly through the cross-encoder,
    which scores relevance far more precisely than the bi-encoder cosine similarity
    used for initial retrieval (at the cost of being too slow to run over the whole
    corpus, hence rerank-after-retrieve rather than rerank-instead-of-retrieve).

    max_per_paper caps how many chunks from the same paper can occupy the front of
    the ranking (see apply_diversity_cap). Plain pointwise reranking has no notion
    of "this chunk is redundant with one I already ranked highly from the same
    paper" — measured directly on this project's eval set, unconstrained reranking
    let one paper's chunks fill all 5 top slots, pushing the second paper a genuine
    multi-hop question needs out entirely (q1/q2 recall dropped from 1.0 to 0.5
    despite aggregate precision improving). This is the standard fix for that
    redundancy-collapse failure mode.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", max_per_paper: int = 2):
        self.model = CrossEncoder(model_name)
        self.max_per_paper = max_per_paper

    def rerank(self, question: str, candidates: list[tuple[str, str]]) -> list[str]:
        """candidates: list of (chunk_id, text). Returns chunk_ids ranked best-first,
        with same-paper chunks beyond max_per_paper deprioritized to the tail."""
        if not candidates:
            return []
        pairs = [(question, text) for _, text in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda item: -item[1])
        ranked_chunk_ids = [chunk_id for (chunk_id, _text), _score in ranked]
        return apply_diversity_cap(ranked_chunk_ids, self.max_per_paper)
