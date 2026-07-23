from graphrag.systems.synthesis import _lexical_overlap


def test_lexical_overlap_high_for_supported_claim():
    claim = "D-NOVA is up to 41.7x faster than a CPU baseline."
    source = "D-NOVA is up to 41.7x faster and 71x more energy-efficient than a CPU baseline."
    assert _lexical_overlap(claim, [source]) > 0.7


def test_lexical_overlap_low_for_unsupported_claim():
    claim = "C2KV achieves a 17x inference speedup for long-context generation."
    source = "D-NOVA moves similarity search directly into NAND flash memory."
    assert _lexical_overlap(claim, [source]) < 0.4


def test_lexical_overlap_checks_across_all_cited_sources():
    claim = "D-NOVA achieves a 41.7x speedup using Dual-Bound Tight Similarity Sensing."
    sources = [
        "D-NOVA is up to 41.7x faster than a CPU baseline.",
        "The Dual-Bound Tight Similarity Sensing metric is tailored for NAND strings.",
    ]
    assert _lexical_overlap(claim, sources) > 0.7
