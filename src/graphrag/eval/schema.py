from __future__ import annotations

from pydantic import BaseModel


class LabeledQuestion(BaseModel):
    """One hand-labeled multi-hop question in the eval set."""

    id: str
    question: str
    gold_answer: str
    gold_chunk_ids: list[str]
    """Chunk ids that together contain enough information to answer the question.
    For a genuine multi-hop question these should span more than one paper."""
    hop_count: int = 2
    notes: str | None = None


class Citation(BaseModel):
    """One claim in a synthesized answer and the chunk ids that support it."""

    claim: str
    chunk_ids: list[str]


class PredictionResult(BaseModel):
    """What a system under test returns for one question."""

    question_id: str
    predicted_answer: str
    retrieved_chunk_ids: list[str]
    """Ranked, most relevant first. The harness slices to top-k itself."""
    latency_ms: float
    citations: list[Citation] = []
    """Populated only by systems that do grounded synthesis (see systems/synthesis.py).
    Empty for systems that just extract a placeholder answer from the top chunk."""


class QuestionScore(BaseModel):
    question_id: str
    precision_at_k: float
    recall_at_k: float
    correct: bool
    judge_reasoning: str | None = None
    latency_ms: float


class SystemReport(BaseModel):
    system_name: str
    k: int
    n_questions: int
    mean_accuracy: float
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_latency_ms: float
    per_question: list[QuestionScore]
