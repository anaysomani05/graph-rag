from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Protocol

from graphrag.eval.judge import AnswerJudge
from graphrag.eval.schema import LabeledQuestion, PredictionResult, QuestionScore, SystemReport
from graphrag.eval.scoring import precision_at_k, recall_at_k


class RAGSystem(Protocol):
    name: str

    def answer(self, question: str) -> PredictionResult: ...


def load_questions(path: str | Path) -> list[LabeledQuestion]:
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(LabeledQuestion.model_validate(json.loads(line)))
    return questions


def run_system(
    system: RAGSystem,
    questions: list[LabeledQuestion],
    judge: AnswerJudge,
    k: int = 5,
    delay_s: float = 0.0,
) -> SystemReport:
    """delay_s: pause between questions. Systems and judges that call an LLM (the
    orchestrated pipeline now calls Groq twice per question — planner + synthesis —
    on top of the judge's own grading call) can burst past Groq's free-tier
    tokens-per-minute limit with no pacing at all between questions; 0 is fine for
    systems/judges with no LLM calls."""
    scores: list[QuestionScore] = []
    for i, q in enumerate(questions):
        if i > 0 and delay_s > 0:
            time.sleep(delay_s)
        result: PredictionResult = system.answer(q.question)
        precision = precision_at_k(result.retrieved_chunk_ids, q.gold_chunk_ids, k)
        recall = recall_at_k(result.retrieved_chunk_ids, q.gold_chunk_ids, k)
        correct, reasoning = judge.grade(q.question, q.gold_answer, result.predicted_answer)
        scores.append(
            QuestionScore(
                question_id=q.id,
                precision_at_k=precision,
                recall_at_k=recall,
                correct=correct,
                judge_reasoning=reasoning,
                latency_ms=result.latency_ms,
            )
        )

    n = len(scores) or 1
    return SystemReport(
        system_name=getattr(system, "name", system.__class__.__name__),
        k=k,
        n_questions=len(scores),
        mean_accuracy=sum(s.correct for s in scores) / n,
        mean_precision_at_k=sum(s.precision_at_k for s in scores) / n,
        mean_recall_at_k=sum(s.recall_at_k for s in scores) / n,
        mean_latency_ms=sum(s.latency_ms for s in scores) / n,
        per_question=scores,
    )


def compare_systems(
    systems: list[RAGSystem],
    questions: list[LabeledQuestion],
    judge: AnswerJudge,
    k: int = 5,
    delay_s: float = 0.0,
) -> list[SystemReport]:
    return [run_system(system, questions, judge, k, delay_s=delay_s) for system in systems]


def print_comparison(reports: list[SystemReport]) -> None:
    header = f"{'system':<20}{'n':>5}{'accuracy':>12}{'precision@k':>14}{'recall@k':>12}{'latency(ms)':>14}"
    print(header)
    print("-" * len(header))
    for r in reports:
        print(
            f"{r.system_name:<20}{r.n_questions:>5}{r.mean_accuracy:>12.2%}"
            f"{r.mean_precision_at_k:>14.2%}{r.mean_recall_at_k:>12.2%}{r.mean_latency_ms:>14.1f}"
        )
