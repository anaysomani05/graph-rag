from __future__ import annotations


def _paper_id(chunk_id: str) -> str:
    return chunk_id.split("#")[0]


def precision_at_k(retrieved_chunk_ids: list[str], gold_chunk_ids: list[str], k: int) -> float:
    """Paper-level precision: a retrieved chunk counts as a hit if it belongs to a
    paper referenced by any gold chunk id, not only on exact chunk-id match.

    Gold labels were written pinning "#abstract" as the supporting chunk (see
    eval/README.md), but Day 1 full-text chunking means a paper can be correctly
    retrieved via one of its ~20 body chunks without ever surfacing the literal
    abstract chunk in top-k. Scoring at chunk granularity would count that as a
    miss even though the system found the right source paper — which is what a
    multi-hop question is actually asking for. Chunking granularity is a
    retrieval implementation detail, not the thing the label is testing.
    """
    top_k = retrieved_chunk_ids[:k]
    if not top_k:
        return 0.0
    gold_papers = {_paper_id(c) for c in gold_chunk_ids}
    hits = sum(1 for c in top_k if _paper_id(c) in gold_papers)
    return hits / len(top_k)


def recall_at_k(retrieved_chunk_ids: list[str], gold_chunk_ids: list[str], k: int) -> float:
    """Paper-level recall — see precision_at_k for why matching is by paper, not
    exact chunk id."""
    if not gold_chunk_ids:
        return 1.0
    gold_papers = {_paper_id(c) for c in gold_chunk_ids}
    retrieved_papers = {_paper_id(c) for c in retrieved_chunk_ids[:k]}
    hits = sum(1 for p in gold_papers if p in retrieved_papers)
    return hits / len(gold_papers)
