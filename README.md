# GraphRAG Research Assistant

Plain vector search struggles with questions whose answer is spread across two different
papers, because no single chunk contains the whole answer. This project builds a
**graph + vector** retrieval system over ~90 real arXiv papers and checks, with a
hand-labeled test set, whether it actually answers these multi-hop questions better than
plain vector search.

**Live demo:** https://huggingface.co/spaces/anaysomani05/graph-rag
(The first question after the demo has been idle is slow while it wakes up and loads the
models. After that it's fast.)

## Results

Tested on 16 hand-written questions, each needing facts from two different papers. The
questions never name the papers directly, so the system has to actually find them by
meaning, not by keyword matching.

| System | Accuracy | Precision@5 | Recall@5 | Latency |
|---|---|---|---|---|
| Plain vector search | 18.75% | 67.50% | 65.62% | 164ms |
| + Graph-hybrid retrieval | 18.75% | 47.50% | 65.62% | 101ms |
| + Reranking | 6.25% | 57.50% | **78.12%** | 304ms |
| + Grounded answers with citations | **37.50%** | 57.50% | **78.12%** | 1.4s |
| + LangGraph pipeline | 31.25% | 55.00% | 75.00% | 2.5s |

**The number that matters here is recall** — did the system find the evidence the
question needs? Graph-hybrid retrieval plus reranking pushes recall from 65% to 78%, and
grounded synthesis turns that evidence into a correct, cited answer (accuracy 18% → 37%).

The full LangGraph pipeline scores slightly lower on accuracy on purpose: it declines to
answer when the evidence is weak, instead of guessing. That's a deliberate trade-off, not
a regression.

## How it works

```
arXiv papers ──▶ parse, chunk, embed ──▶ Postgres + pgvector
                                                │
                                     LLM extracts entities/relations
                                                │
                                         knowledge graph
                                                │
question ──▶ plan ──▶ retrieve (vector + graph) ──▶ rerank ──▶ write answer + citations
```

- **Ingestion** — pulls papers from the arXiv API, parses the PDFs, splits them into
  chunks, embeds them, and stores everything in Postgres with pgvector.
- **Knowledge graph** — an LLM reads each paper and pulls out entity/relation facts. A
  simple similarity check merges duplicate entities across papers.
- **Hybrid retrieval** — combines plain vector search with the graph: for a top vector
  hit, it also pulls in papers connected to it through shared entities and topics.
- **Reranking** — a cross-encoder re-scores the candidates, with a cap so one paper can't
  fill every slot and crowd out the second paper the question needs.
- **Grounded answers** — an LLM writes the answer as separate claims, each linked to a
  real source chunk. A check drops any claim that isn't actually supported by its source.
- **Pipeline** — a LangGraph flow (plan → retrieve → write → verify) that also refuses to
  answer when nothing relevant was found, instead of making something up.

## What broke along the way

The honest engineering story is in the bugs. A few of the ones worth knowing:

- **The test questions leaked their own answers.** The first version named both papers in
  the question, so plain keyword search found them for free. Rewrote every question to
  describe the papers without naming them.
- **The graph was too thin to help.** Pulling facts from abstracts only gave too few
  connections. Including the start of each paper's body text fixed it.
- **Reranking quietly hurt recall.** Without a per-paper cap, one paper's chunks took over
  the whole top-5 and pushed the second needed paper out. Added a cap.
- **A citation isn't proof.** The model would sometimes cite a real chunk that didn't
  actually support the claim. Added a check that the cited text really overlaps the claim.
- **Slow queries hid on localhost.** Retrieval made 5-8 separate database calls per
  question — instant locally, but 6-7 seconds against a cloud database. Rewrote them as
  single queries.
- **Random ordering broke reproducibility.** Python sets and tied database rows have no
  fixed order, so the same code gave different scores on different runs. Fixed by sorting
  explicitly.

More detail, with before/after numbers, in [`eval/README.md`](eval/README.md).

## Two ways to run it

Both use the exact same pipeline underneath:

- **API** — `POST /query` (FastAPI, `src/graphrag/api/main.py`). Run with
  `uvicorn graphrag.api.main:app`.
- **Web UI** — the Gradio app (`app.py`) that's running on the live demo above.

## Running locally

```bash
cp .env.example .env   # add your GROQ_API_KEY and DATABASE_URL
docker compose up -d   # or: brew install postgresql@18 pgvector
python scripts/ingest.py            # build the corpus
python -m graphrag.graph.pipeline   # build the knowledge graph
python app.py                       # start the web UI (or: uvicorn graphrag.api.main:app)
```

Run the evaluation: `python scripts/run_eval.py`. Run the tests: `pytest`.

## Stack

LangGraph · Postgres + pgvector · FastAPI · Gradio · sentence-transformers · Groq (Llama) · Docker
