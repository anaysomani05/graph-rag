from graphrag.eval.schema import Citation
from graphrag.orchestration.graph import _NO_EVIDENCE_ANSWER, verify_answer


def test_verify_answer_passes_through_when_grounded():
    citations = [Citation(claim="D-NOVA is 41.7x faster.", chunk_ids=["2607.17538v1#abstract"])]
    assert verify_answer("D-NOVA is 41.7x faster.", citations) == "D-NOVA is 41.7x faster."


def test_verify_answer_flags_no_citations():
    assert verify_answer("some answer text", []) == _NO_EVIDENCE_ANSWER


def test_verify_answer_flags_blank_answer():
    citations = [Citation(claim="x", chunk_ids=["a#b"])]
    assert verify_answer("   ", citations) == _NO_EVIDENCE_ANSWER
