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

**Revised 2026-07-22 (Day 2):** the original v1 of these questions named both papers'
systems explicitly in the question text (e.g. "Between D-NOVA's hardware accelerator and
C2KV's KV-cache reuse framework..."). That's a real methodology bug — it means flat
vector search can trivially find both papers just by matching the proper nouns against
each paper's own text, since papers reference themselves by name throughout. It measured
nothing about the multi-hop hypothesis. Genuine multi-hop questions (HotpotQA-style)
don't name the bridge document; the question paraphrases each paper's contribution
without using its system name/title, so retrieval has to work from the underlying
semantic content instead of a lexical shortcut. All 8 questions were rewritten under this
rule; gold answers still name the actual system so grading remains checkable.

## Scoring granularity

`precision_at_k`/`recall_at_k` (`src/graphrag/eval/scoring.py`) score at the **paper**
level, not exact chunk id: a retrieved chunk counts as a hit if it belongs to a paper
referenced by any gold chunk id. Gold labels pin `#abstract` as the nominal supporting
chunk, but Day 1 full-text chunking means a paper can be correctly retrieved via one of
its ~20 body chunks without the literal abstract chunk ever appearing in top-k. Scoring
at exact-chunk granularity would count that as a miss even though the system found the
right source paper — chunking granularity is a retrieval implementation detail, not what
these labels are testing.

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
- **Do not name either paper's system/method by name in the question text.** Paraphrase
  each contribution using different vocabulary than the paper's own abstract. Naming the
  system lets flat vector search find it by lexical/proper-noun match, defeating the
  point of a multi-hop test. Gold answers should still name the actual system, since
  that's what makes them checkable.
- The gold answer must be traceable to specific text in the cited chunks, not a
  subjective judgment call — needed for the LLM judge (or human) to grade it
  consistently.
