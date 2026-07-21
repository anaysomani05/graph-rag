# Hand-labeled eval set

`questions.jsonl` holds the hand-labeled multi-hop questions used by the eval harness
(`src/graphrag/eval/harness.py`). `source_papers.jsonl` lists every arXiv paper those
questions reference — the corpus built during Day 1 ingestion must include all of these
so the gold labels resolve to real chunks.

## Chunk id convention

Every gold question currently references `{arxiv_id}#abstract` as its gold chunk id.
That id is a placeholder standing in for "the abstract of this paper," since Day 1
ingestion (chunking the full paper text) hasn't run yet.

**Reconciliation required on Day 1:** whatever chunking scheme ingestion uses, it must
either (a) preserve `{arxiv_id}#abstract` as the literal id of the chunk containing the
abstract, or (b) these gold_chunk_ids need to be remapped to the real chunk ids the
ingestion pipeline actually produces. Don't skip this silently — if the ids don't match,
every precision@k / recall@k score will read as 0 even when retrieval is working, which
will look like a bug in the retriever when it's actually a labeling/id-mismatch bug.

## Current status

8 questions, all two-hop, all grounded in real abstracts pulled from arXiv cs.CL
(retrieval / RAG subfield) on 2026-07-21 — no fabricated facts, every gold answer traces
to specific sentences in the two cited abstracts.

Per the build plan: grow this to 15-20 by the end of Phase 0 / start of Weekend 1, then
to 25-30 during Weekend 2 Day 3, once full-text chunks (not just abstracts) exist to
write richer, deeper-than-abstract multi-hop questions against.

## Labeling a new question

Each line in `questions.jsonl` is a `LabeledQuestion` (see
`src/graphrag/eval/schema.py`):

```json
{"id": "q9", "question": "...", "gold_answer": "...", "gold_chunk_ids": ["<arxiv_id>#<chunk>", "..."], "hop_count": 2, "notes": "..."}
```

Rules for a good label:
- The question must require combining facts from 2+ distinct papers (or, after Day 1,
  2+ distinct chunks that are not adjacent in the same paper) — otherwise it's not
  testing the thing this project exists to prove.
- The gold answer must be traceable to specific text in the cited chunks, not a
  subjective judgment call — needed for the LLM judge (or human) to grade it
  consistently.
