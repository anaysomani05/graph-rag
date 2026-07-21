from __future__ import annotations

import re
from pathlib import Path

import fitz  # pymupdf


def extract_pdf_text(pdf_path: Path) -> str:
    """Extracts raw text from an arXiv PDF. Known limitation: naive PDF text
    extraction on multi-column arXiv layouts is lossy around equations, tables, and
    references (columns can interleave, formulas often extract as garbled symbols).
    Acceptable for v1 since chunking + retrieval is robust to some noisy chunks;
    revisit with arXiv LaTeX-source parsing if retrieval quality suffers because of it.
    """
    doc = fitz.open(pdf_path)
    try:
        pages = [page.get_text() for page in doc]
    finally:
        doc.close()
    return _clean("\n".join(pages))


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
