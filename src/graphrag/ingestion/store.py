from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from graphrag.config import settings

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    published DATE
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    text TEXT NOT NULL,
    embedding vector(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_arxiv_id_idx ON chunks (arxiv_id);
"""


@dataclass
class ChunkRecord:
    chunk_id: str
    arxiv_id: str
    chunk_index: int
    text: str
    embedding: np.ndarray


def get_connection() -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url, autocommit=True)
    register_vector(conn)
    return conn


def create_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA_SQL)


def upsert_paper(
    conn: psycopg.Connection,
    arxiv_id: str,
    title: str,
    abstract: str,
    published: date | None,
) -> None:
    conn.execute(
        """
        INSERT INTO papers (arxiv_id, title, abstract, published)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (arxiv_id) DO UPDATE
        SET title = EXCLUDED.title, abstract = EXCLUDED.abstract, published = EXCLUDED.published
        """,
        (arxiv_id, title, abstract, published),
    )


def upsert_chunks(conn: psycopg.Connection, chunks: list[ChunkRecord]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (chunk_id, arxiv_id, chunk_index, text, embedding)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE
            SET text = EXCLUDED.text, embedding = EXCLUDED.embedding
            """,
            [(c.chunk_id, c.arxiv_id, c.chunk_index, c.text, c.embedding) for c in chunks],
        )


def create_vector_index(conn: psycopg.Connection) -> None:
    """Call after bulk load so ivfflat clusters against real data distribution."""
    n_chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    lists = max(1, min(100, n_chunks // 100))
    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})
        """
    )
    conn.execute("ANALYZE chunks")
