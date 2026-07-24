---
title: GraphRAG Research Assistant
emoji: 🕸️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
---

# GraphRAG Research Assistant

Flat vector RAG fails on multi-hop questions because the answer spans documents that no
single chunk contains. This project builds a hybrid graph+vector retrieval system over a
real corpus of ~90 arXiv papers and measures — with a hand-labeled eval set, not vibes —
whether it actually beats flat RAG at answering questions that require combining facts
from two different papers.

**Live demo:** https://huggingface.co/spaces/anaysomani05/graph-rag (Gradio UI, backed
by Neon Postgres + pgvector and Groq — the first query after an idle period is slow while
the Space wakes and loads models)

## Results

Measured on 16 hand-labeled two-hop questions, each requiring facts from two different
papers, phrased so neither paper is named in the question (naming them would let flat
vector search find both papers by lexical match alone — see "What I got wrong" below).

| System | Accuracy | Precision@5 | Recall@5 | Latency |
|---|---|---|---|---|
| Flat vector baseline | 18.75% | 67.50% | 65.62% | 164ms |
| + Graph-hybrid retrieval | 18.75% | 47.50% | 65.62% | 101ms |
| + Cross-encoder reranking | 6.25% | 57.50% | **78.12%** | 304ms |
| + Grounded synthesis & citations | **37.50%** | 57.50% | **78.12%** | 1.4s |
| + LangGraph orchestration | 31.25% | 55.00% | 75.00% | 2.5s |

(Latency measured in a warm, long-running process — see "What I got wrong" below for why
that distinction matters. LangGraph orchestration's higher latency and slightly lower
accuracy than the simpler grounded pipeline is a real, honest tradeoff — its relevance
gate correctly declines to answer on some borderline cases the simpler system guesses
right on; see `eval/README.md`'s Day 4 notes.)

**Recall is the metric this project is actually about** — did retrieval find the
evidence a multi-hop question needs at all. Graph-hybrid retrieval measurably beats flat
vector search on it; the rest of the pipeline (reranking, grounded synthesis) turns that
retrieved evidence into an actually-correct, citable answer.

## Architecture

```
arXiv API ──▶ ingest (parse/chunk/embed) ──▶ Postgres + pgvector
                                                     │
                              LLM entity/relation extraction
                                                     │
                                              knowledge graph
                                                     │
question ──▶ planner ──▶ hybrid retrieval (vector + graph-hop) ──▶ cross-encoder rerank
                                                                          │
                                                              grounded synthesis + citations
                                                                          │
                                                                     verifier
                                                                          │
                                                                  answer + citations
```

- **Ingestion** — async arXiv fetch, PDF parsing, chunking, embedding (`sentence-transformers`), stored in Postgres+pgvector.
- **Knowledge graph** — LLM (Groq) extracts entity/relation triples per paper; a crude embedding-similarity dedup merges near-duplicate entities across papers.
- **Hybrid retrieval** — vector top-k merged with 1-hop graph neighbors and a subtopic-embedding bridge for thematically related but differently-worded papers.
- **Reranking** — off-the-shelf cross-encoder (`ms-marco-MiniLM-L-6-v2`), diversity-capped so one paper's chunks can't crowd out the second paper a multi-hop question needs.
- **Synthesis** — LLM answer broken into discrete claims, each cited to a real retrieved chunk, with a lexical-overlap check that drops claims whose citation doesn't actually support them.
- **Orchestration** — a real LangGraph `StateGraph` (planner → retriever → synthesizer → verifier), with a relevance gate that returns an honest "insufficient evidence" instead of a confident wrong answer on out-of-corpus questions.
- **API** — FastAPI `/query` endpoint, backed by the orchestrated pipeline.

## What I got wrong (and fixed)

This project's value is as much in the debugging as the architecture — a few of the real
bugs found by actually running the system, not just building it:

- **The eval questions leaked the answer.** The first version named both papers by
  system/title in the question text, so flat vector search could find both trivially by
  matching proper nouns. Rewrote all questions to paraphrase without naming either
  system — the same discipline HotpotQA uses.
- **The graph was too sparse to matter.** Abstract-only extraction produced ~9
  triples/paper — not enough for a bridge to reliably exist between any given correct
  pair. Extending extraction to include body-chunk excerpts fixed it.
- **Reranking silently hurt recall.** A plain cross-encoder rerank let one paper's
  chunks fill the entire top-5, pushing the second paper a multi-hop question needs out
  entirely — recall dropped even as aggregate precision improved. Fixed with a
  diversity cap (max chunks per paper in the final ranking).
- **A cited source isn't the same as a correct citation.** The synthesizer could
  restate a fact given directly in the question and cite an unrelated real chunk for
  it — a real chunk id being cited doesn't mean the citation is honest. Added a
  lexical-overlap check between claim and cited source content.
- **N+1 queries are invisible on localhost.** The graph-hop retrieval logic issued
  5-8+ sequential DB round trips per call — fine at ~1ms/round-trip locally, but it
  turned into 6-7 seconds per retrieval once the DB moved to a network-latency-bound
  managed host (Neon). Rewrote the hot-path queries as single statements (CTEs and a
  LATERAL join) instead of Python loops issuing one query per graph hop/subtopic.
- **Ties need an explicit tie-break.** Two separate non-determinism bugs, same root
  cause: Python's `set()` iteration order is randomized per process, and Postgres
  doesn't guarantee row order among exact-similarity ties under `LIMIT` — both made
  eval scores vary run-to-run on identical code with no data change. Fixed by sorting
  explicitly and adding tie-breakers to `ORDER BY`.

Full write-up of each finding, with before/after numbers, in
[`eval/README.md`](eval/README.md).

## Interfaces

Two interfaces over the same `GraphRAGPipeline` orchestration code — nothing about the
retrieval/reranking/synthesis logic differs between them:

- **FastAPI** (`src/graphrag/api/main.py`) — `POST /query`, the "real" API for
  programmatic use. Run with `uvicorn graphrag.api.main:app`, or via `Dockerfile`.
- **Gradio** (`app.py`) — the deployed demo UI (type a question, see the grounded
  answer + citations). What's actually running at the live demo link above.

## Running locally

```bash
cp .env.example .env   # fill in GROQ_API_KEY and DATABASE_URL
docker compose up -d   # or: brew install postgresql@18 pgvector
python scripts/ingest.py
python -m graphrag.graph.pipeline
uvicorn graphrag.api.main:app --reload   # or: python app.py for the Gradio UI
```

Eval harness: `python scripts/run_eval.py`. Tests: `pytest`.

## Stack

LangGraph · Postgres + pgvector · FastAPI · Gradio · sentence-transformers (bi-encoder +
cross-encoder) · Groq (Llama 3.1/3.3) · Docker
