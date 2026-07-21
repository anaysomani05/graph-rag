from __future__ import annotations


def chunk_text(text: str, chunk_size_words: int = 500, overlap_words: int = 50) -> list[str]:
    """Word-count chunking as an approximation for ~500 tokens (real tokenizers run
    ~0.75 words/token on English text, so this slightly over-sizes chunks; close
    enough for v1 and avoids adding a tokenizer dependency just for chunk boundaries).
    """
    words = text.split()
    if not words:
        return []

    step = chunk_size_words - overlap_words
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + chunk_size_words]))
        if start + chunk_size_words >= len(words):
            break
        start += step
    return chunks
