from __future__ import annotations

import json
from dataclasses import dataclass

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from graphrag.config import settings

_EXTRACTION_PROMPT = """Extract key entity relationships from this paper's title and abstract, \
for building a knowledge graph that connects papers addressing similar sub-problems, so a \
multi-hop question spanning two papers can be answered by traversing shared entities.

Title: {title}
Abstract: {abstract}

Extract two kinds of triples:

1. 5-9 triples about this paper's own contribution: methods/systems named in this paper, what \
they do, what they improve on or compare against, datasets/benchmarks used, key claimed results. \
Use short entity names (proper nouns, method/system names, dataset names), not full sentences. \
Every "source" and "target" value MUST be under 8 words — never a full sentence or clause. If a \
claimed result needs more detail than that, shorten it (e.g. "83.3% attack success rate", not a \
sentence explaining the attack).

2. 2-3 triples of the form ("{title_short}", "addresses_subtopic", "<specific sub-topic>"), where \
<specific sub-topic> names the specific sub-problem or technique category this paper belongs to \
at a granularity narrow enough that only a handful of related papers would share it — e.g. \
"dense retrieval latency", "knowledge-graph-based multi-hop reasoning", "hallucination detection \
via evidence grounding", "multi-agent RAG orchestration". Do NOT use bare umbrella terms like \
"RAG", "LLM", "retrieval-augmented generation", or "large language models" alone as the sub-topic \
— those are too generic to distinguish this paper from hundreds of others.

Respond with ONLY a JSON array, no other text: \
[{{"source": "...", "relation": "...", "target": "..."}}, ...]"""


@dataclass
class RawTriple:
    source: str
    relation: str
    target: str
    source_chunk_id: str


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
    return "\n".join(lines)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=5, max=65))
def _call_llm(client: Groq, prompt: str) -> str:
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=3072,
    )
    return response.choices[0].message.content.strip()


def extract_triples(
    arxiv_id: str, title: str, abstract: str, client: Groq | None = None
) -> list[RawTriple]:
    """Extracts (subject, relation, object) triples from a paper's abstract.

    Scoped to abstracts only, not the ~2000 full-text chunks: that's where the
    hand-labeled eval questions get their gold facts from, and it keeps this at
    90 LLM calls instead of ~2000. Extracting from full text is a stretch goal
    alongside growing the eval set, not a v1 requirement.
    """
    client = client or Groq(api_key=settings.groq_api_key)
    title_short = title if len(title) <= 60 else title[:57] + "..."
    prompt = _EXTRACTION_PROMPT.format(title=title, title_short=title_short, abstract=abstract)

    try:
        content = _call_llm(client, prompt)
    except Exception as e:
        print(f"WARN: extraction call failed for {arxiv_id}: {e}")
        return []

    content = _strip_code_fence(content)
    try:
        # Some responses append a second, repeated JSON array after a valid first
        # one. raw_decode parses just the first valid value and ignores trailing
        # data, instead of failing the whole response over it.
        raw, _ = json.JSONDecoder().raw_decode(content.strip())
    except json.JSONDecodeError:
        print(f"WARN: unparseable extraction response for {arxiv_id}: {content[:200]!r}")
        return []

    chunk_id = f"{arxiv_id}#abstract"
    triples = []
    for item in raw:
        try:
            triples.append(
                RawTriple(
                    source=str(item["source"]),
                    relation=str(item["relation"]),
                    target=str(item["target"]),
                    source_chunk_id=chunk_id,
                )
            )
        except (KeyError, TypeError):
            continue
    return triples
