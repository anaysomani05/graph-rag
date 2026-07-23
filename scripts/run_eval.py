"""Run the eval harness against whichever systems currently exist.

Usage: python scripts/run_eval.py
"""

from groq import Groq

from graphrag.config import settings
from graphrag.eval.harness import compare_systems, load_questions, print_comparison
from graphrag.eval.judge import LLMJudge
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
    ]
    judge = LLMJudge(client=Groq(api_key=settings.groq_api_key), model=settings.groq_model)
    reports = compare_systems(systems, questions, judge, k=5)
    print_comparison(reports)
