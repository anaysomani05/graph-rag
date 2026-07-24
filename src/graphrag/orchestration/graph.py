from __future__ import annotations

import json
import time

from groq import Groq
from langgraph.graph import END, START, StateGraph

from graphrag.config import settings
from graphrag.eval.schema import PredictionResult
from graphrag.ingestion.store import get_connection
from graphrag.orchestration.state import OrchestrationState
from graphrag.systems.hybrid_retrieval import HybridRetrieval
from graphrag.systems.reranker import CrossEncoderReranker
from graphrag.systems.synthesis import SourceCandidate, synthesize_answer

_PLANNER_PROMPT = """This question may require combining evidence from more than one \
document to answer fully. Produce 1 additional focused search query — a different \
phrasing or sub-aspect of the question — that together with the original question would \
help surface all the evidence needed. Do not just repeat the original question.

Question: {question}

Respond with ONLY a JSON array of strings, e.g. ["query one"]. If the question is \
already narrow enough that no additional query would help, respond with []."""

_NO_EVIDENCE_ANSWER = (
    "I don't have enough grounded evidence in the retrieved sources to answer this "
    "confidently."
)


def verify_answer(answer: str, citations: list) -> str:
    """The plan's "flags 'I don't know' instead of hallucinating" — if no claim
    survived synthesis's grounding check (empty citations) or the answer is blank,
    returns the honest fallback instead of an empty string a caller could mistake
    for "no error occurred." Pulled out of the node method (no self-state needed)
    so it's testable without constructing the full pipeline (DB, embedding model,
    cross-encoder, Groq client).
    """
    if not citations or not answer.strip():
        return _NO_EVIDENCE_ANSWER
    return answer


class GraphRAGPipeline:
    """Thin, linear LangGraph pipeline: planner -> retriever -> synthesizer -> verifier.

    Deliberately linear (no retry/loop edges) per the build plan — the point of using
    LangGraph even for a single-pass pipeline is that adding a real iterative
    verify/retry loop later is an incremental conditional edge, not a rewrite of a
    hand-rolled function-call chain (which is what systems/grounded_hybrid.py is).

    - planner: decomposes the question into 1-2 additional search queries. A genuine
      multi-hop question often mixes two different sub-topics into one query embedding
      that matches neither source paper well; searching sub-aspects separately gives
      hybrid retrieval more/better seeds to graph-expand from.
    - retriever: runs hybrid retrieval (vector + graph hop) for the original question
      and every sub-question, merges the candidate chunks, then reranks the merged set
      against the ORIGINAL question (sub-questions are search aids, not what the final
      answer needs to be relevant to).
    - synthesizer: unchanged grounded synthesis (see systems/synthesis.py) — two-layer
      citation grounding, paper-title-attributed sources.
    - verifier: the plan's "flags 'I don't know' instead of hallucinating" — if no
      claim survived synthesis's grounding check, the verifier replaces an empty
      answer with an explicit, honest "insufficient evidence" response rather than
      showing nothing or letting a later caller mistake empty-string for a real answer.
    """

    name = "langgraph_orchestrated"

    def __init__(
        self,
        hybrid: HybridRetrieval | None = None,
        reranker: CrossEncoderReranker | None = None,
        groq_client: Groq | None = None,
        rerank_top_n: int = 20,
        synthesis_top_n: int = 5,
        min_relevance_score: float = -2.0,
        max_sub_questions: int = 1,
    ):
        # Each sub-question means one more full HybridRetrieval.answer() call, and
        # each of those is several sequential DB round trips (vector search + a
        # graph-hop/subtopic-bridge lookup per seed). Fine on localhost; against a
        # network-latency-bound managed DB (Neon), cutting this from 2 to 1
        # sub-question took the pipeline from ~20s to ~8.5s just by itself, on top
        # of batching the per-seed graph queries (see graph/store.py). Increase
        # only once retrieval is made properly concurrent (separate connections
        # per query) — see eval/README.md's Day 5 latency notes.
        self.max_sub_questions = max_sub_questions
        self.hybrid = hybrid or HybridRetrieval()
        self.reranker = reranker or CrossEncoderReranker()
        self.min_relevance_score = min_relevance_score
        self.client = groq_client or Groq(api_key=settings.groq_api_key)
        self.conn = get_connection()
        self.rerank_top_n = rerank_top_n
        self.synthesis_top_n = synthesis_top_n
        self.app = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(OrchestrationState)
        graph.add_node("planner", self._planner_node)
        graph.add_node("retriever", self._retriever_node)
        graph.add_node("synthesizer", self._synthesizer_node)
        graph.add_node("verifier", self._verifier_node)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "retriever")
        graph.add_edge("retriever", "synthesizer")
        graph.add_edge("synthesizer", "verifier")
        graph.add_edge("verifier", END)
        return graph.compile()

    def _planner_node(self, state: OrchestrationState) -> dict:
        prompt = _PLANNER_PROMPT.format(question=state["question"])
        try:
            response = self.client.chat.completions.create(
                model=settings.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = "\n".join(
                    line for line in content.splitlines() if not line.strip().startswith("```")
                )
            sub_questions, _ = json.JSONDecoder().raw_decode(content.strip())
            sub_questions = [str(q) for q in sub_questions][: self.max_sub_questions]
        except Exception as e:
            print(f"WARN: planner call failed, proceeding with no sub-questions: {e}")
            sub_questions = []
        return {"sub_questions": sub_questions}

    def _retriever_node(self, state: OrchestrationState) -> dict:
        queries = [state["question"]] + state["sub_questions"]
        merged_ids: list[str] = []
        seen: set[str] = set()
        for q in queries:
            for chunk_id in self.hybrid.answer(q).retrieved_chunk_ids:
                if chunk_id not in seen:
                    merged_ids.append(chunk_id)
                    seen.add(chunk_id)

        candidate_ids = merged_ids[: self.rerank_top_n]
        rows = self.conn.execute(
            "SELECT chunk_id, text FROM chunks WHERE chunk_id = ANY(%s)", (candidate_ids,)
        ).fetchall()
        text_by_id = dict(rows)
        rerank_input = [(cid, text_by_id[cid]) for cid in candidate_ids if cid in text_by_id]
        reranked = self.reranker.rerank_with_scores(state["question"], rerank_input)

        # Relevance gate: an out-of-corpus question still returns *some* top-k vector
        # hits (cosine similarity always ranks something highest), but the
        # cross-encoder score reveals whether the best candidate is actually
        # relevant. Empirically, genuinely relevant pairs score roughly +2 to +4 on
        # this corpus; irrelevant ones score around -6 to -11 (see reranker.py). If
        # even the top reranked candidate scores below this, there's no real
        # evidence for this question — short-circuit to no candidates so the
        # verifier reports "insufficient evidence" instead of synthesis fabricating
        # a plausible-sounding answer from irrelevant chunks.
        if not reranked or reranked[0][1] < self.min_relevance_score:
            return {"candidates": []}
        reranked_ids = [chunk_id for chunk_id, _score in reranked[: self.synthesis_top_n]]

        title_rows = self.conn.execute(
            """
            SELECT c.chunk_id, p.title, c.text
            FROM chunks c JOIN papers p ON p.arxiv_id = c.arxiv_id
            WHERE c.chunk_id = ANY(%s)
            """,
            (reranked_ids,),
        ).fetchall()
        by_id = {chunk_id: SourceCandidate(chunk_id, title, text) for chunk_id, title, text in title_rows}
        candidates = [by_id[cid] for cid in reranked_ids if cid in by_id]
        return {"candidates": candidates}

    def _synthesizer_node(self, state: OrchestrationState) -> dict:
        answer, citations = synthesize_answer(state["question"], state["candidates"], client=self.client)
        return {"answer": answer, "citations": citations}

    def _verifier_node(self, state: OrchestrationState) -> dict:
        return {"answer": verify_answer(state["answer"], state["citations"])}

    def answer(self, question: str) -> PredictionResult:
        start = time.perf_counter()
        result = self.app.invoke(
            {"question": question, "sub_questions": [], "candidates": [], "answer": "", "citations": []}
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return PredictionResult(
            question_id="unused",
            predicted_answer=result["answer"],
            retrieved_chunk_ids=[c.chunk_id for c in result["candidates"]],
            latency_ms=latency_ms,
            citations=result["citations"],
        )
