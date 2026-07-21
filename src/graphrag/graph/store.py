from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import psycopg

GRAPH_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    embedding vector(384)
);

CREATE TABLE IF NOT EXISTS edges (
    id SERIAL PRIMARY KEY,
    source_entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    target_entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    source_chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS edges_source_idx ON edges (source_entity_id);
CREATE INDEX IF NOT EXISTS edges_target_idx ON edges (target_entity_id);
CREATE INDEX IF NOT EXISTS edges_chunk_idx ON edges (source_chunk_id);
"""


@dataclass
class EdgeRecord:
    source_entity_id: str
    relation: str
    target_entity_id: str
    source_chunk_id: str


def create_graph_schema(conn: psycopg.Connection) -> None:
    conn.execute(GRAPH_SCHEMA_SQL)


def upsert_entities(
    conn: psycopg.Connection, entities: dict[str, str], embeddings: dict[str, np.ndarray]
) -> None:
    """entities: entity_id -> canonical_name. embeddings: entity_id -> vector, used for
    the subtopic-similarity bridge (see subtopic_bridge_chunk_ids) since exact entity
    dedup is deliberately strict and won't merge related-but-differently-worded
    sub-topics."""
    if not entities:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO entities (entity_id, canonical_name, embedding)
            VALUES (%s, %s, %s)
            ON CONFLICT (entity_id) DO NOTHING
            """,
            [(eid, name, embeddings.get(eid)) for eid, name in entities.items()],
        )


def insert_edges(conn: psycopg.Connection, edges: list[EdgeRecord]) -> None:
    if not edges:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO edges (source_entity_id, relation, target_entity_id, source_chunk_id)
            VALUES (%s, %s, %s, %s)
            """,
            [(e.source_entity_id, e.relation, e.target_entity_id, e.source_chunk_id) for e in edges],
        )


def _entity_degrees(conn: psycopg.Connection, entity_ids: set[str]) -> dict[str, int]:
    if not entity_ids:
        return {}
    ids = list(entity_ids)
    rows = conn.execute(
        """
        SELECT entity_id, count(*) FROM (
            SELECT source_entity_id AS entity_id FROM edges WHERE source_entity_id = ANY(%s)
            UNION ALL
            SELECT target_entity_id AS entity_id FROM edges WHERE target_entity_id = ANY(%s)
        ) t
        GROUP BY entity_id
        """,
        (ids, ids),
    ).fetchall()
    return dict(rows)


def neighbor_chunk_ids(
    conn: psycopg.Connection, chunk_id: str, hops: int = 1, max_entity_degree: int = 8
) -> set[str]:
    """Chunk ids reachable within `hops` graph hops from any entity mentioned in
    `chunk_id`, via the edges those entities participate in (as source or target).

    Entities connected to more than `max_entity_degree` papers are excluded from
    traversal (not from the seed set's own chunk lookup, only from hop expansion).
    Without this, generic entities like "RAG" or "LLM" — shared by dozens of
    unrelated papers — flood the neighbor set with noise instead of finding the
    specific paper a multi-hop question actually needs; see eval/README.md notes
    on hub-entity degree from the first extraction run (RAG: 80 edges, LLM: 30).
    """
    seed_entities: set[str] = set(
        row[0]
        for row in conn.execute(
            """
            SELECT DISTINCT source_entity_id FROM edges WHERE source_chunk_id = %s
            UNION
            SELECT DISTINCT target_entity_id FROM edges WHERE source_chunk_id = %s
            """,
            (chunk_id, chunk_id),
        ).fetchall()
    )

    seed_degrees = _entity_degrees(conn, seed_entities)
    frontier = {e for e in seed_entities if seed_degrees.get(e, 0) <= max_entity_degree}
    seen_entities = set(frontier)

    for _ in range(hops):
        if not frontier:
            break
        rows = conn.execute(
            """
            SELECT DISTINCT target_entity_id FROM edges WHERE source_entity_id = ANY(%s)
            UNION
            SELECT DISTINCT source_entity_id FROM edges WHERE target_entity_id = ANY(%s)
            """,
            (list(frontier), list(frontier)),
        ).fetchall()
        candidates = {r[0] for r in rows} - seen_entities
        candidate_degrees = _entity_degrees(conn, candidates)
        next_frontier = {e for e in candidates if candidate_degrees.get(e, 0) <= max_entity_degree}
        seen_entities |= next_frontier
        frontier = next_frontier

    if not seen_entities:
        return set()

    rows = conn.execute(
        """
        SELECT DISTINCT source_chunk_id FROM edges
        WHERE source_entity_id = ANY(%s) OR target_entity_id = ANY(%s)
        """,
        (list(seen_entities), list(seen_entities)),
    ).fetchall()
    return {r[0] for r in rows} - {chunk_id}


def subtopic_bridge_chunk_ids(
    conn: psycopg.Connection,
    chunk_id: str,
    relation: str = "addresses_subtopic",
    similarity_threshold: float = 0.55,
    top_n_per_subtopic: int = 3,
    max_results: int = 2,
) -> list[str]:
    """Bridges to other papers' chunks via semantic similarity between this chunk's
    "addresses_subtopic" entities and other papers' subtopic entities. Returns chunk
    ids ranked by best matching similarity, capped to max_results — callers merging
    this into a ranked list need the strongest bridges first and a small, predictable
    count, not an unordered, unbounded set (an earlier version returned a set with no
    ranking or overall cap, which let weak bridges flood a caller's top-k window).

    Exact entity-id dedup (see EntityDeduper, threshold 0.87) is deliberately strict,
    so two papers describing related-but-differently-worded sub-problems (e.g. "dense
    vector retrieval latency reduction" vs. "in-storage vector search acceleration")
    never collapse into the same discrete node and never connect via neighbor_chunk_ids.
    This is the looser, embedding-similarity counterpart that exists specifically to
    catch that case — the actual mechanism multi-hop questions about thematically
    related but textually dissimilar papers depend on.
    """
    own_subtopics = conn.execute(
        """
        SELECT en.embedding FROM edges e
        JOIN entities en ON en.entity_id = e.target_entity_id
        WHERE e.source_chunk_id = %s AND e.relation = %s AND en.embedding IS NOT NULL
        """,
        (chunk_id, relation),
    ).fetchall()

    best_similarity: dict[str, float] = {}
    for (embedding,) in own_subtopics:
        rows = conn.execute(
            """
            SELECT e.source_chunk_id, 1 - (en.embedding <=> %s) AS similarity
            FROM edges e
            JOIN entities en ON en.entity_id = e.target_entity_id
            WHERE e.relation = %s AND e.source_chunk_id != %s AND en.embedding IS NOT NULL
            ORDER BY en.embedding <=> %s
            LIMIT %s
            """,
            (embedding, relation, chunk_id, embedding, top_n_per_subtopic),
        ).fetchall()
        for bridged_chunk_id, similarity in rows:
            if similarity >= similarity_threshold:
                best_similarity[bridged_chunk_id] = max(
                    similarity, best_similarity.get(bridged_chunk_id, 0.0)
                )

    ranked = sorted(best_similarity.items(), key=lambda kv: -kv[1])
    return [chunk_id for chunk_id, _ in ranked[:max_results]]
