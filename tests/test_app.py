"""Regression tests for the Gradio entry point (app.py).

These exist because a real bug shipped to the live Space: app.py's answer_question
read `result.answer`, but PredictionResult exposes `predicted_answer` — no test
exercised app.py, so the AttributeError only surfaced as a runtime error in
production. A fake pipeline lets us test the field-mapping glue without a DB or
the sentence-transformer models.
"""

import app as app_module
from graphrag.eval.schema import Citation, PredictionResult


class _FakePipeline:
    def __init__(self, result: PredictionResult):
        self._result = result

    def answer(self, question: str) -> PredictionResult:
        return self._result


def _install_fake(monkeypatch, result: PredictionResult):
    monkeypatch.setattr(app_module, "_get_pipeline", lambda: _FakePipeline(result))


def test_answer_question_maps_answer_and_citations(monkeypatch):
    result = PredictionResult(
        question_id="q",
        predicted_answer="D-NOVA targets retrieval; C2KV targets generation.",
        retrieved_chunk_ids=["a#abstract", "b#abstract"],
        latency_ms=1.0,
        citations=[
            Citation(claim="D-NOVA targets retrieval.", chunk_ids=["a#abstract"]),
            Citation(claim="C2KV targets generation.", chunk_ids=["b#abstract"]),
        ],
    )
    _install_fake(monkeypatch, result)
    answer, citations_md = app_module.answer_question("some multi-hop question")
    assert answer == "D-NOVA targets retrieval; C2KV targets generation."
    assert "D-NOVA targets retrieval." in citations_md
    assert "`a#abstract`" in citations_md


def test_answer_question_handles_no_citations(monkeypatch):
    result = PredictionResult(
        question_id="q",
        predicted_answer="Insufficient evidence.",
        retrieved_chunk_ids=[],
        latency_ms=1.0,
        citations=[],
    )
    _install_fake(monkeypatch, result)
    answer, citations_md = app_module.answer_question("unanswerable question")
    assert answer == "Insufficient evidence."
    assert "no citations" in citations_md


def test_answer_question_blank_input_short_circuits(monkeypatch):
    # Must not touch the pipeline at all for empty input.
    monkeypatch.setattr(
        app_module, "_get_pipeline", lambda: (_ for _ in ()).throw(AssertionError("should not be called"))
    )
    assert app_module.answer_question("   ") == ("", "")
