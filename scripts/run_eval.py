"""Run the eval harness against whichever systems currently exist.

Usage: python scripts/run_eval.py
"""

from graphrag.eval.harness import compare_systems, load_questions, print_comparison
from graphrag.eval.judge import LexicalOverlapJudge
from graphrag.systems.flat_baseline import FlatVectorBaseline

if __name__ == "__main__":
    questions = load_questions("eval/questions.jsonl")
    systems = [FlatVectorBaseline()]
    reports = compare_systems(systems, questions, LexicalOverlapJudge(), k=5)
    print_comparison(reports)
