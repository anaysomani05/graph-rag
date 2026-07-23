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

**16 questions** (grown from 8 on 2026-07-22, closing out the Day 3 plan item), all
two-hop, all grounded in real abstracts pulled from arXiv cs.CL (retrieval / RAG
subfield) — no fabricated facts, every gold answer traces to specific sentences in the
two cited abstracts. q9-q16 span 14 additional corpus papers not used by q1-q8 (dataset
staleness/knowledge conflicts, hallucination detection, GraphRAG construction, reranking,
retrieval-budget allocation — see `source_papers.jsonl` for the full list), so the set
now covers more of the corpus's actual thematic diversity instead of repeatedly
re-pairing the same 8 papers.

Full-corpus result on all 16 questions (see scripts/run_eval.py):

```
system                  n    accuracy   precision@k    recall@k   latency(ms)
flat_baseline           16     18.75%        67.50%      65.62%          89.3
hybrid                  16     18.75%        47.50%      65.62%         102.4
hybrid_reranked         16      6.25%        57.50%      78.12%         277.7
hybrid_reranked_grounded 16    37.50%        57.50%      78.12%       13100.1
```

This confirms the Day 2/3 findings hold on a larger, more diverse sample, not just the
original 8 hand-picked pairs: hybrid+rerank's recall (78.12%) clearly beats flat baseline
(65.62%), and the full grounded pipeline's end-to-end accuracy (37.5%) matches what was
seen on the smaller 8-question set almost exactly — a good sign the number isn't an
artifact of a small sample. Precision on this larger set is actually lower for the hybrid
systems than flat (57.5% vs 67.5%) even though it was higher on the original 8 — recall
is the metric this project is actually about (did retrieval find the needed evidence at
all), and that one is unambiguously better across both sample sizes.

Per the build plan: grow further to 25-30 once there's time; 16 was chosen as a
pragmatic stopping point inside the 15-20 target range for this build session.

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

## Day 2 graph density (resolved 2026-07-22)

First hybrid-vs-flat run: identical numbers on every question. Root cause was
abstract-only extraction (~9 triples/paper, 1578 edges) — too sparse to guarantee a
graph or subtopic-embedding bridge existed between any given correct pair. Confirmed by
direct check: neither `neighbor_chunk_ids` nor `subtopic_bridge_chunk_ids` connected
D-NOVA to Debate-on-Graph (q7) or DeLIVeR to C2KV (q8) at all.

Fix: extraction now includes each paper's first 2 body chunks alongside the abstract
(see `extract.py`, `pipeline.py:_body_excerpt`), producing ~13-23 triples/paper. Re-ran
the full corpus (80/90 papers with edges, 1483 edges, 10 papers hit persistent
LLM over-generation failures even at max_tokens=4096 — accepted as a known small gap,
same as Day 1/2's earlier extraction runs).

Result: **hybrid recall (56.25%) now beats flat baseline (43.75%)**, with q1 and q2 each
going from 0.50 to 1.00 recall — hybrid found both gold papers where flat found only one
— and zero regressions on any question. Precision is still lower for hybrid (37.5% vs
50%) since expansion candidates that aren't hits dilute it; recall is the metric that
matters for "did retrieval find the needed evidence at all," which is what multi-hop QA
is actually testing.

Not fully solved: q4-q8 remain ties (q7/q8 still find neither gold paper for either
system). More body-chunk coverage per paper, or extracting from full papers rather than
abstract + 2 chunks, would likely close more of these — noted as a stretch item.

**Groq model/quota notes for future extraction runs:** `llama-3.3-70b-versatile` has a
much lower daily token budget (100,000 TPD) than its per-minute limit (12,000 TPM)
suggests — the richer abstract+body-chunk prompt (~3000-3500 tokens/call) exhausts the
whole daily budget after ~33 calls, not from bursty pacing but from cumulative daily
usage. `llama-3.1-8b-instant` has a separate, much larger quota bucket (14,400 RPD) and
a smaller per-minute cap (6,000 TPM vs 12,000) — with this richer prompt it needs ~40s
between calls, not 10s (10s was fine for the abstract-only prompt, which used ~4x fewer
tokens/call). A handful of papers cause either model to loop/over-generate regardless of
token ceiling — not fixable by raising max_tokens further; shortening the input excerpt
sometimes helps (worked for Debate-on-Graph specifically).

## Day 3: reranker + grounded synthesis (2026-07-22)

**Reranker (F5).** Off-the-shelf `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks hybrid
retrieval's top-20 candidates. First version regressed q1/q2 recall from 1.0 to 0.5 —
plain pointwise reranking has no notion of "this chunk is redundant with one I already
ranked highly from the same paper," so it let one paper's chunks fill all 5 top slots,
pushing the second paper a multi-hop question needs out entirely. Fixed with a
`max_per_paper=2` cap in `CrossEncoderReranker.rerank` (excess same-paper chunks are
deprioritized to the tail, not dropped). After the fix: **precision@5 52.5% vs hybrid's
35% / flat's 50%, recall@5 68.75% vs hybrid's 50% / flat's 43.75%** — a clean win on both
axes over both baselines, reproducible across repeated runs.

**Reproducibility bug found and fixed along the way:** `neighbor_chunk_ids` returned a
Python `set`, and set iteration order is randomized per-process (`PYTHONHASHSEED`) —
running the identical eval script twice in separate processes gave different aggregate
numbers (56.25% vs 50.00% recall) because which candidates survived truncation to
`max_expansions_per_seed` depended on hash order, not relevance. Fixed by returning a
sorted list. Worth remembering: any function whose output gets sliced/truncated
downstream needs a deterministic order, not just "correct contents."

**Grounded synthesis (F6).** Real LLM synthesis (Groq) over the reranked top-5, broken
into discrete claims each citing specific chunk ids — replaces the earlier placeholder
"first sentence of the top chunk." Two grounding layers: (1) by construction, the model
can only cite source numbers that map to chunks that were actually retrieved; (2) by
content check, `_lexical_overlap` in `synthesis.py` drops any claim whose text doesn't
actually overlap with its cited source's content. (2) exists because (1) alone doesn't
verify the cited chunk's *content* supports the claim — caught in practice: the model
restated a fact given directly in the question text itself (needs no retrieval) and
cited an unrelated real chunk for it. A real chunk id being cited doesn't mean the
citation is honest.

**Paper-title attribution mattered more than expected.** Source excerpts handed to the
synthesizer initially carried no paper attribution, just raw chunk text — body chunks
deep in a paper often don't repeat the paper's own title, so the model could describe a
fact correctly but could only call it "source 5," never naming the actual system. Once
each source was prefixed with its paper's title (`grounded_hybrid.py` joins `chunks` to
`papers` for this), end-to-end accuracy (LLM-judged against gold answers) jumped from
12.5% to **37.5%** with no other change. Lesson: what the synthesizer can *say* is
bounded by what's literally in front of it — anonymized excerpts silently cap answer
quality even when retrieval and grounding are both working correctly.

**Latency: initial ~11.5s reading was a measurement artifact, corrected on Day 4.**
Originally reported as a real gap against the <5s target. Root cause: each ad-hoc test
script launched a fresh Python process, paying sentence-transformer + cross-encoder
model-load cost (several seconds) on every single query, and that cost got misattributed
to "synthesis latency." Running the same system inside one long-lived process across all
16 questions (as `scripts/run_eval.py` and a real server both do) gives **903ms/question**
for the grounded pipeline — comfortably under the target. Lesson: always benchmark
latency in a warm process that matches real deployment, not a fresh script per call.

## Day 4: LangGraph orchestration + API (2026-07-23)

**F3 orchestration.** `src/graphrag/orchestration/graph.py` is a real LangGraph
`StateGraph` (planner → retriever → synthesizer → verifier), deliberately linear (no
retry/loop edges) per the build plan — the point of using LangGraph even for a
single-pass pipeline is that a real iterative verify/retry loop later is an incremental
conditional edge, not a rewrite of the hand-rolled function-call chain that
`systems/grounded_hybrid.py` is.

- **Planner** decomposes the question into 1-2 additional search queries, so a question
  mixing two sub-topics into one embedding gives hybrid retrieval more/better seeds.
- **Retriever** runs hybrid retrieval for the original question and every sub-question,
  merges candidates, reranks the merged set against the *original* question (verified
  deterministic across repeated calls — sub-question generation itself was checked for
  the Groq non-determinism seen elsewhere in this project and found stable here).
- **Synthesizer** reuses the existing grounded synthesis unchanged.
- **Verifier** — found a real gap while testing this manually: an out-of-corpus question
  ("what's the capital of France...") still returned *some* top-k chunks (cosine
  similarity always ranks something highest) and the synthesizer built a plausible but
  off-topic answer from them, with citations that passed the lexical-overlap check
  (they were verbatim restatements of the wrong chunks). The overlap check alone verifies
  claim-matches-source, not source-relevant-to-question. Fix: `CrossEncoderReranker`
  now also exposes raw scores (`rerank_with_scores`), and the retriever gates on the top
  reranked score — empirically, genuinely relevant pairs on this corpus score roughly
  +2 to +4, irrelevant ones -6 to -11, so a threshold of -2.0 cleanly separates them.
  Below it, retrieval returns no candidates and the verifier reports "insufficient
  evidence" honestly instead of a confabulated answer.
- **Cold-start flakiness found and fixed:** the very first inference call through a
  freshly-loaded cross-encoder in a new process scored differently than every
  subsequent identical call (4/4 repeats in the same warm process were consistent).
  Server startup now runs one throwaway query to absorb that variance before real
  traffic arrives (see `api/main.py`'s lifespan).

**F8 API.** `src/graphrag/api/main.py` — single `POST /query` (question in, grounded
answer + citations + retrieved chunk ids + latency out) plus `GET /health`, backed by
the orchestrated pipeline, eagerly initialized (with warm-up) at server startup rather
than on first request. Verified with real `curl` requests against a running server, not
just in-process calls.

**Result on the full 16-question set** (all 5 systems, same run):

```
system                    n   accuracy  precision@k  recall@k  latency(ms)
flat_baseline            16     18.75%      67.50%     65.62%       131.6
hybrid                   16     18.75%      47.50%     65.62%       152.5
hybrid_reranked          16      6.25%      57.50%     78.12%       310.6
hybrid_reranked_grounded 16     37.50%      57.50%     78.12%       903.3
langgraph_orchestrated   16     31.25%      55.00%     75.00%      2702.3
```

Honest reading: the orchestrated pipeline does **not** beat the simpler grounded system
on this eval set (31.25% vs 37.5% accuracy, slightly lower precision/recall too) — the
relevance gate correctly declines to answer on some borderline cases where the simpler
system still ventures a guess that the judge happens to credit as correct. That's a real
accuracy-vs-honesty tradeoff, not a bug: the gate exists specifically to stop confident
wrong answers on evidence that doesn't actually support them, and it's doing that job.
Latency is higher too (2.7s vs 0.9s) from the extra planner call and per-sub-question
retrieval — still under the 5s target, but the added machinery has a real cost that
isn't paid back in accuracy on this particular question set. Worth remembering before
assuming "more pipeline stages" is automatically better.

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
