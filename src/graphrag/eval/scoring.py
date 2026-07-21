from __future__ import annotations


def precision_at_k(retrieved_chunk_ids: list[str], gold_chunk_ids: list[str], k: int) -> float:
    top_k = retrieved_chunk_ids[:k]
    if not top_k:
        return 0.0
    gold = set(gold_chunk_ids)
    hits = sum(1 for c in top_k if c in gold)
    return hits / len(top_k)


def recall_at_k(retrieved_chunk_ids: list[str], gold_chunk_ids: list[str], k: int) -> float:
    if not gold_chunk_ids:
        return 1.0
    top_k = set(retrieved_chunk_ids[:k])
    hits = sum(1 for c in gold_chunk_ids if c in top_k)
    return hits / len(gold_chunk_ids)
