from graphrag.systems.reranker import apply_diversity_cap


def test_diversity_cap_defers_excess_same_paper_chunks():
    ranked = [
        "paperA#chunk0",
        "paperA#chunk1",
        "paperA#chunk2",
        "paperB#chunk0",
    ]
    result = apply_diversity_cap(ranked, max_per_paper=2)
    assert result == ["paperA#chunk0", "paperA#chunk1", "paperB#chunk0", "paperA#chunk2"]


def test_diversity_cap_preserves_order_within_cap():
    ranked = ["paperA#chunk0", "paperB#chunk0", "paperA#chunk1", "paperB#chunk1"]
    result = apply_diversity_cap(ranked, max_per_paper=2)
    assert result == ranked  # nothing exceeds the cap, order unchanged


def test_diversity_cap_handles_empty_input():
    assert apply_diversity_cap([], max_per_paper=2) == []
