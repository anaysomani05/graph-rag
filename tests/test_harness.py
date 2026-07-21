from graphrag.eval.harness import compare_systems, run_system
from graphrag.eval.judge import LexicalOverlapJudge
from graphrag.eval.schema import LabeledQuestion, PredictionResult


class FakeSystem:
    """A stand-in RAG system for exercising the harness before any real pipeline exists."""

    def __init__(self, name: str, answer_text: str, chunk_ids: list[str]):
        self.name = name
        self._answer_text = answer_text
        self._chunk_ids = chunk_ids

    def answer(self, question: str) -> PredictionResult:
        return PredictionResult(
            question_id="unused",
            predicted_answer=self._answer_text,
            retrieved_chunk_ids=self._chunk_ids,
            latency_ms=42.0,
        )


QUESTIONS = [
    LabeledQuestion(
        id="q1",
        question="Which paper's method would fail on the dataset used in the other paper?",
        gold_answer="Paper A's method fails because it assumes fixed-length inputs.",
        gold_chunk_ids=["paperA#3", "paperB#1"],
        hop_count=2,
    )
]


def test_run_system_scores_perfect_retrieval_and_answer():
    system = FakeSystem(
        "perfect",
        "Paper A's method fails because it assumes fixed-length inputs.",
        ["paperA#3", "paperB#1", "paperC#0"],
    )
    report = run_system(system, QUESTIONS, LexicalOverlapJudge(), k=5)
    assert report.n_questions == 1
    assert report.mean_accuracy == 1.0
    assert report.mean_recall_at_k == 1.0
    assert 0 < report.mean_precision_at_k <= 1.0


def test_run_system_scores_bad_retrieval_and_wrong_answer():
    system = FakeSystem("bad", "I don't know.", ["irrelevant#0"])
    report = run_system(system, QUESTIONS, LexicalOverlapJudge(), k=5)
    assert report.mean_accuracy == 0.0
    assert report.mean_precision_at_k == 0.0
    assert report.mean_recall_at_k == 0.0


def test_compare_systems_orders_reports_by_input_order():
    good = FakeSystem(
        "hybrid", "Paper A's method fails because it assumes fixed-length inputs.", ["paperA#3", "paperB#1"]
    )
    bad = FakeSystem("baseline", "unclear", ["paperA#3"])
    reports = compare_systems([bad, good], QUESTIONS, LexicalOverlapJudge(), k=5)
    assert [r.system_name for r in reports] == ["baseline", "hybrid"]
    assert reports[1].mean_accuracy > reports[0].mean_accuracy
