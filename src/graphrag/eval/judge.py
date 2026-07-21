from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())


class AnswerJudge(ABC):
    @abstractmethod
    def grade(self, question: str, gold_answer: str, predicted_answer: str) -> tuple[bool, str]:
        """Return (correct, reasoning)."""


class LexicalOverlapJudge(AnswerJudge):
    """Offline fallback with no API key required: token-overlap heuristic.

    Deliberately crude - only meant to unblock running the harness before LLM judge
    credentials are wired up. Swap for LLMJudge once Azure OpenAI/Groq creds exist,
    since token overlap will both over- and under-credit paraphrased answers.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def grade(self, question: str, gold_answer: str, predicted_answer: str) -> tuple[bool, str]:
        gold_tokens = set(_normalize(gold_answer).split())
        pred_tokens = set(_normalize(predicted_answer).split())
        if not gold_tokens:
            return False, "empty gold answer"
        overlap = len(gold_tokens & pred_tokens) / len(gold_tokens)
        correct = overlap >= self.threshold
        return correct, f"token overlap={overlap:.2f} (threshold={self.threshold})"


_GRADING_PROMPT = """You are grading whether a predicted answer correctly answers a question, \
given a gold reference answer. Focus on factual correctness, not phrasing or verbosity.

Question: {question}
Gold answer: {gold_answer}
Predicted answer: {predicted_answer}

Respond with ONLY a JSON object: {{"correct": true|false, "reasoning": "one sentence"}}"""


class LLMJudge(AnswerJudge):
    """Grades answers with an LLM. `client` must expose an OpenAI-compatible
    `chat.completions.create(model=..., messages=...)` method (works with both the
    `openai` SDK against Azure OpenAI and the `groq` SDK)."""

    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    def grade(self, question: str, gold_answer: str, predicted_answer: str) -> tuple[bool, str]:
        prompt = _GRADING_PROMPT.format(
            question=question, gold_answer=gold_answer, predicted_answer=predicted_answer
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(content)
            return bool(parsed["correct"]), str(parsed.get("reasoning", ""))
        except (json.JSONDecodeError, KeyError):
            return False, f"judge returned unparseable response: {content!r}"
