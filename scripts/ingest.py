"""Run the full Day 1 ingestion pipeline: fetch corpus, download PDFs, parse, chunk,
embed, and store in Postgres+pgvector.

Usage: python scripts/ingest.py
"""

from graphrag.ingestion.pipeline import run_ingestion

if __name__ == "__main__":
    run_ingestion(target_size=90)
