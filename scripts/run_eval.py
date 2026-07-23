"""Run the eval harness against whichever systems currently exist.

Usage: python scripts/run_eval.py
"""

from groq import Groq

from graphrag.config import settings
from graphrag.eval.harness import compare_systems, load_questions, print_comparison
from graphrag.eval.judge import LLMJudge
from graphrag.orchestration.graph import GraphRAGPipeline
from graphrag.systems.flat_baseline import FlatVectorBaseline
from graphrag.systems.grounded_hybrid import GroundedHybridSystem
from graphrag.systems.hybrid_retrieval import HybridRetrieval
from graphrag.systems.reranked_hybrid import RerankedHybridRetrieval

if __name__ == "__main__":
    questions = load_questions("eval/questions.jsonl")
    hybrid = HybridRetrieval()
    reranked = RerankedHybridRetrieval(base=hybrid)
    systems = [
        FlatVectorBaseline(),
        hybrid,
        reranked,
        GroundedHybridSystem(reranked=reranked),
        GraphRAGPipeline(hybrid=hybrid),
    ]
    judge = LLMJudge(client=Groq(api_key=settings.groq_api_key), model=settings.groq_model)
    # GraphRAGPipeline alone makes 2 Groq calls/question (planner + synthesis), plus
    # the judge's own call — pacing needed to stay under the free-tier TPM cap (see
    # eval/README.md's Groq quota notes).
    reports = compare_systems(systems, questions, judge, k=5, delay_s=15.0)
    print_comparison(reports)
