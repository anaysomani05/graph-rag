from __future__ import annotations

import asyncio

from sentence_transformers import SentenceTransformer

from graphrag.config import settings
from graphrag.ingestion.chunk import chunk_text
from graphrag.ingestion.fetch import download_pdfs, search_corpus
from graphrag.ingestion.parse import extract_pdf_text
from graphrag.ingestion.store import (
    ChunkRecord,
    create_schema,
    create_vector_index,
    get_connection,
    upsert_chunks,
    upsert_paper,
)


def run_ingestion(target_size: int = 90) -> None:
    print(f"Searching arXiv for ~{target_size} papers...")
    papers = search_corpus(target_size=target_size)
    print(f"Found {len(papers)} papers. Downloading PDFs...")

    pdf_paths = asyncio.run(download_pdfs(papers))
    print(f"Downloaded {len(pdf_paths)}/{len(papers)} PDFs.")

    model = SentenceTransformer(settings.embedding_model, device="cpu")
    conn = get_connection()
    create_schema(conn)

    total_chunks = 0
    for i, paper in enumerate(papers, 1):
        try:
            upsert_paper(conn, paper.arxiv_id, paper.title, paper.abstract, paper.published)

            texts = [paper.abstract]
            chunk_ids = [f"{paper.arxiv_id}#abstract"]

            pdf_path = pdf_paths.get(paper.arxiv_id)
            if pdf_path:
                try:
                    body_text = extract_pdf_text(pdf_path)
                    body_chunks = chunk_text(body_text)
                    texts.extend(body_chunks)
                    chunk_ids.extend(f"{paper.arxiv_id}#chunk{j}" for j in range(len(body_chunks)))
                except Exception as e:
                    print(f"WARN: failed to parse {paper.arxiv_id}, indexing abstract only: {e}")
            else:
                print(f"WARN: no PDF for {paper.arxiv_id}, indexing abstract only")

            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            records = [
                ChunkRecord(chunk_id=cid, arxiv_id=paper.arxiv_id, chunk_index=idx, text=t, embedding=emb)
                for idx, (cid, t, emb) in enumerate(zip(chunk_ids, texts, embeddings))
            ]
            upsert_chunks(conn, records)
            total_chunks += len(records)
            print(f"[{i}/{len(papers)}] {paper.arxiv_id}: {len(records)} chunks")
        except Exception as e:
            print(f"WARN: skipping {paper.arxiv_id} entirely, ingestion step failed: {e}")

    create_vector_index(conn)
    print(f"Done. {len(papers)} papers, {total_chunks} chunks ingested.")


if __name__ == "__main__":
    run_ingestion()
