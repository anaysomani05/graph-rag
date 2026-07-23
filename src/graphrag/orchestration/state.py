from __future__ import annotations

from typing import TypedDict

from graphrag.eval.schema import Citation
from graphrag.systems.synthesis import SourceCandidate


class OrchestrationState(TypedDict):
    question: str
    sub_questions: list[str]
    candidates: list[SourceCandidate]
    answer: str
    citations: list[Citation]
