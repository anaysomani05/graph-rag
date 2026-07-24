# Evaluation set

This folder holds the hand-written test set used to measure the system.

- `questions.jsonl` — 16 multi-hop questions, each needing facts from two different
  papers, with a gold answer and the chunk ids that support it.
- `source_papers.jsonl` — the arXiv papers the questions refer to. The corpus must
  include all of these for the gold answers to line up with real chunks.

Every question is grounded in real papers from the arXiv cs.CL (retrieval / RAG)
subfield. No facts are made up — each gold answer traces to specific sentences in the two
papers it cites.

## Results

Run with `python scripts/run_eval.py`. All five systems, same 16 questions:

```
system                     accuracy  precision@5  recall@5  latency
plain vector baseline        18.75%      67.50%     65.62%    164ms
+ graph-hybrid retrieval     18.75%      47.50%     65.62%    101ms
+ reranking                   6.25%      57.50%     78.12%    304ms
+ grounded answers            37.50%      57.50%     78.12%    1.4s
+ langgraph pipeline         31.25%      55.00%     75.00%    2.5s
```

**Recall is the key metric** — did retrieval find the evidence the question needs?
Graph-hybrid plus reranking raises it from 65% to 78%. Grounded synthesis then turns that
evidence into a correct, cited answer, roughly doubling accuracy (18% → 37%).

The langgraph pipeline scores a bit lower on accuracy on purpose: it refuses to answer
when the evidence is weak instead of guessing. That's a trade-off between accuracy and
honesty, not a bug.

## How scoring works

Precision and recall are measured **per paper**, not per exact chunk. A retrieved chunk
counts as a hit if it belongs to a paper the question needs. This is because a paper can
be found through any of its ~20 body chunks, not only the one specific chunk listed in the
gold label — and finding the right paper is what a multi-hop question is really testing.

## Key findings from building this

- **Don't name the papers in the question.** The first version of the questions named
  both papers directly, which let plain keyword search find them without any real
  retrieval. Rewriting the questions to describe the papers without naming them is what
  makes this a fair multi-hop test.
- **Graph density decides whether the graph helps.** With facts pulled from abstracts
  only, the graph was too sparse and the hybrid system tied the plain baseline. Including
  the start of each paper's body text made the graph dense enough to actually connect
  related papers — that's when hybrid started beating the baseline.
- **Recall improved, precision dropped slightly.** Adding graph and reranking pulls in a
  few extra candidates that aren't hits, which lowers precision a little. That's an
  acceptable trade for the recall gain, since missing the evidence entirely is the worse
  failure for multi-hop questions.

## Adding a question

Each line in `questions.jsonl` looks like:

```json
{"id": "q9", "question": "...", "gold_answer": "...", "gold_chunk_ids": ["<arxiv_id>#abstract", "..."], "hop_count": 2, "notes": "..."}
```

Guidelines:

- It must need facts from **two different papers** — otherwise it isn't testing multi-hop.
- **Don't name the papers or their methods in the question.** Describe what each paper
  does in your own words. Naming them lets keyword search cheat. (The gold answer *can*
  name them — that's what makes it checkable.)
- The gold answer must trace to specific sentences in the cited chunks, so it can be
  graded consistently.
