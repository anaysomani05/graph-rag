"""Hugging Face Space entry point (Gradio SDK).

Wraps the same GraphRAGPipeline that backs the FastAPI /query endpoint
(src/graphrag/api/main.py) — this Space is a demo UI over the identical
orchestration code, not a separate implementation.
"""

from __future__ import annotations

import gradio as gr

from graphrag.orchestration.graph import GraphRAGPipeline

pipeline: GraphRAGPipeline | None = None


def _get_pipeline() -> GraphRAGPipeline:
    global pipeline
    if pipeline is None:
        pipeline = GraphRAGPipeline()
        # Warm-up call — see api/main.py's lifespan for why this matters: the very
        # first inference through a freshly-loaded cross-encoder scores differently
        # than every subsequent call, which can flip the orchestration's relevance
        # gate on a genuinely answerable first question.
        pipeline.answer("warm-up query, response is discarded")
    return pipeline


def answer_question(question: str) -> tuple[str, str]:
    if not question.strip():
        return "", ""
    result = _get_pipeline().answer(question)
    if not result.citations:
        return result.answer, "_(no citations — see answer above)_"
    citations_md = "\n".join(
        f"{i}. {c.claim}\n   `{', '.join(c.chunk_ids)}`" for i, c in enumerate(result.citations, 1)
    )
    return result.answer, citations_md


EXAMPLE_QUESTIONS = [
    "One recent RAG efficiency paper moves similarity search directly into NAND "
    "flash memory to cut retrieval latency, claiming over 40x speedup versus a CPU "
    "baseline. A separate paper tackles a different cost bottleneck in the same "
    "kind of pipeline by compressing and reusing key-value caches during "
    "long-context generation, claiming up to 17x faster inference. What bottleneck "
    "does each target, and which stage of the pipeline (retrieval vs. generation) "
    "does each optimize?",
    "A benchmark paper found that research agents often retrieve the correct "
    "supporting evidence but still give the wrong final answer, because they stop "
    "before reconciling conflicting documents and defer to a plausible-looking but "
    "false one. A different paper describes a way to manipulate multi-hop agents "
    "purely by changing how true facts are positioned and emphasized, without "
    "adding any false information or instructions at all. What does each paper "
    "call its respective failure mechanism?",
]

with gr.Blocks(title="GraphRAG Research Assistant") as demo:
    gr.Markdown(
        "# GraphRAG Research Assistant\n"
        "Multi-hop QA over ~90 real arXiv papers (cs.CL retrieval/RAG subfield), "
        "using hybrid graph+vector retrieval, cross-encoder reranking, and grounded "
        "answer synthesis with per-claim citations. Ask a question that genuinely "
        "needs facts from two different papers to answer — see the "
        "[project README](https://github.com/anaysomani05/graph-rag) for the full "
        "engineering write-up and measured results against a flat-RAG baseline."
    )
    question_box = gr.Textbox(label="Question", lines=3, placeholder="Ask a multi-hop question...")
    submit_btn = gr.Button("Ask", variant="primary")
    answer_box = gr.Textbox(label="Answer", lines=6, interactive=False)
    citations_box = gr.Markdown(label="Citations")
    gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=question_box)

    submit_btn.click(fn=answer_question, inputs=question_box, outputs=[answer_box, citations_box])
    question_box.submit(fn=answer_question, inputs=question_box, outputs=[answer_box, citations_box])

if __name__ == "__main__":
    demo.launch()
