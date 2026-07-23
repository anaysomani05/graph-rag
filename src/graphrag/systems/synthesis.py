from __future__ import annotations

import json
import re
from typing import NamedTuple

from groq import Groq

from graphrag.config import settings
from graphrag.eval.schema import Citation

_SYNTHESIS_PROMPT = """Answer the question using ONLY the numbered source passages below. \
Every claim in your answer must be grounded in at least one source — do not use outside \
knowledge, and do not restate facts that only appear in the question text itself as if they \
were findings from the sources. Name the specific paper/system each claim comes from — the \
source paper's title is given alongside its number, use it instead of saying "source 3". If \
the sources don't contain enough information to answer, say so honestly instead of guessing.

Question: {question}

Sources:
{sources_block}

Break your answer into 2-5 discrete factual claims. Respond with ONLY a JSON object:
{{"claims": [{{"text": "one factual claim, a sentence or two, naming the paper it's from", \
"source_numbers": [1, 3]}}, ...]}}
Each claim's "source_numbers" must list which source(s) above support it — every claim must \
cite at least one."""

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for",
    "and", "or", "by", "at", "as", "this", "that", "it", "its", "be", "with",
}


class SourceCandidate(NamedTuple):
    chunk_id: str
    title: str
    text: str


def _lexical_overlap(claim: str, source_texts: list[str]) -> float:
    """Cheap grounding proxy: fraction of the claim's content words that appear in
    at least one cited source. Not a real entailment check (see LexicalOverlapJudge
    in eval/judge.py for the same tradeoff) — but it catches the concrete failure
    mode observed in practice: the model restating a fact given in the question
    text itself (which it can always do, question-leaked facts need no retrieval)
    and citing an unrelated real chunk id for it, since "cites a real chunk" alone
    doesn't verify the chunk's content actually supports the claim.
    """
    claim_words = {w for w in re.findall(r"[a-z0-9]+", claim.lower()) if w not in _STOPWORDS}
    if not claim_words:
        return 1.0
    source_words = set()
    for text in source_texts:
        source_words |= {w for w in re.findall(r"[a-z0-9]+", text.lower())}
    return len(claim_words & source_words) / len(claim_words)


def _build_sources_block(candidates: list[SourceCandidate], max_chars_per_source: int = 500) -> str:
    lines = []
    for i, c in enumerate(candidates, 1):
        excerpt = c.text[:max_chars_per_source]
        lines.append(f'[{i}] (from "{c.title}") {excerpt}')
    return "\n\n".join(lines)


def synthesize_answer(
    question: str,
    candidates: list[SourceCandidate],
    client: Groq | None = None,
    min_overlap: float = 0.4,
) -> tuple[str, list[Citation]]:
    """candidates: reranked source passages, best first, each carrying its paper's
    title so the model can name the specific system rather than saying "source 3"
    (raw chunk text alone often doesn't repeat the paper's title, especially for
    body chunks deep in the paper — without the title attached, the synthesizer
    couldn't answer "which paper" even when it had the right facts in hand).
    Returns (answer_text, citations) where each citation maps one claim to the
    chunk ids that support it.

    Grounding has two layers, not one:
    1. By construction — the model can only reference sources by number, and every
       source number is mapped back to a real chunk_id from `candidates`, so it
       cannot cite a chunk that wasn't actually retrieved.
    2. By content check — (1) alone doesn't verify the cited chunk's *text* actually
       supports the claim. Observed in practice: the model restated a fact given in
       the question itself (needs no retrieval) and cited an unrelated real chunk
       for it. `_lexical_overlap` below is a cheap proxy that catches this specific
       failure; claims below `min_overlap` are dropped rather than shown as if
       verified.
    """
    if not candidates:
        return "", []

    client = client or Groq(api_key=settings.groq_api_key)
    sources_block = _build_sources_block(candidates)
    prompt = _SYNTHESIS_PROMPT.format(question=question, sources_block=sources_block)

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024,
        )
        content = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"WARN: synthesis call failed: {e}")
        fallback = candidates[0].text.split(". ")[0] + "."
        return fallback, []

    if content.startswith("```"):
        content = "\n".join(line for line in content.splitlines() if not line.strip().startswith("```"))

    try:
        parsed, _ = json.JSONDecoder().raw_decode(content.strip())
        claims = parsed["claims"]
    except (json.JSONDecodeError, KeyError, TypeError):
        fallback = candidates[0].text.split(". ")[0] + "."
        return fallback, []

    citations: list[Citation] = []
    answer_parts: list[str] = []
    for claim in claims:
        try:
            text = str(claim["text"])
            source_numbers = claim["source_numbers"]
            chunk_ids = [
                candidates[n - 1].chunk_id
                for n in source_numbers
                if isinstance(n, int) and 1 <= n <= len(candidates)
            ]
        except (KeyError, TypeError):
            continue
        if not chunk_ids:
            continue  # drop ungrounded claims rather than let them slip through uncited

        cited_texts = [c.text for c in candidates if c.chunk_id in chunk_ids]
        if _lexical_overlap(text, cited_texts) < min_overlap:
            continue  # cites a real chunk, but that chunk's content doesn't actually support the claim

        answer_parts.append(text)
        citations.append(Citation(claim=text, chunk_ids=chunk_ids))

    answer = " ".join(answer_parts)
    return answer, citations
