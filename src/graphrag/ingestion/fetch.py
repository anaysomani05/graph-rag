from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import aiohttp
import arxiv
from tenacity import retry, stop_after_attempt, wait_exponential

PDF_CACHE_DIR = Path("data/raw/pdfs")
SOURCE_PAPERS_PATH = Path("eval/source_papers.jsonl")

# Diverse queries across the retrieval/RAG corner of cs.CL, so the corpus isn't just
# near-duplicates of one search term.
QUERIES = [
    'cat:cs.CL AND (retrieval-augmented OR "retrieval augmented generation")',
    'cat:cs.CL AND "knowledge graph" AND (question answering OR reasoning)',
    'cat:cs.CL AND ("dense retrieval" OR reranking OR reranker)',
    'cat:cs.CL AND "multi-hop" AND (reasoning OR question answering)',
]


@dataclass
class PaperMeta:
    arxiv_id: str
    title: str
    abstract: str
    published: date
    pdf_url: str


def _to_meta(r: arxiv.Result) -> PaperMeta:
    return PaperMeta(
        arxiv_id=r.get_short_id(),
        title=r.title.strip(),
        abstract=r.summary.replace("\n", " ").strip(),
        published=r.published.date(),
        pdf_url=r.pdf_url,
    )


def _pinned_ids() -> list[str]:
    if not SOURCE_PAPERS_PATH.exists():
        return []
    with open(SOURCE_PAPERS_PATH) as f:
        return [json.loads(line)["arxiv_id"] for line in f if line.strip()]


def search_corpus(target_size: int = 90) -> list[PaperMeta]:
    """Search arXiv for the corpus. Guarantees the pinned source papers (the ones the
    hand-labeled eval questions reference) are included, then fills up to target_size
    with more papers from the same subfield."""
    client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)
    seen: dict[str, PaperMeta] = {}

    pinned = _pinned_ids()
    if pinned:
        for r in client.results(arxiv.Search(id_list=pinned)):
            seen[r.get_short_id()] = _to_meta(r)

    for query in QUERIES:
        if len(seen) >= target_size:
            break
        search = arxiv.Search(
            query=query,
            max_results=target_size,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        for r in client.results(search):
            if len(seen) >= target_size:
                break
            seen[r.get_short_id()] = _to_meta(r)

    return list(seen.values())


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _fetch_with_retry(session: aiohttp.ClientSession, url: str, dest: Path) -> None:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
        resp.raise_for_status()
        dest.write_bytes(await resp.read())


async def download_pdfs(
    papers: list[PaperMeta], concurrency: int = 4, politeness_delay_s: float = 1.0
) -> dict[str, Path]:
    """Downloads PDFs concurrently but capped and paced, to stay a polite consumer of
    arxiv.org rather than hammering it with ~90 simultaneous requests."""
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    paths: dict[str, Path] = {}

    async def _download(session: aiohttp.ClientSession, paper: PaperMeta) -> None:
        dest = PDF_CACHE_DIR / f"{paper.arxiv_id}.pdf"
        if dest.exists() and dest.stat().st_size > 0:
            paths[paper.arxiv_id] = dest
            return
        async with sem:
            await asyncio.sleep(politeness_delay_s)
            try:
                await _fetch_with_retry(session, paper.pdf_url, dest)
                paths[paper.arxiv_id] = dest
            except Exception as e:
                print(f"WARN: failed to download {paper.arxiv_id}: {e}")

    headers = {"User-Agent": "graphrag-research-assistant/0.1 (personal portfolio project)"}
    async with aiohttp.ClientSession(headers=headers) as session:
        await asyncio.gather(*[_download(session, p) for p in papers])

    return paths
