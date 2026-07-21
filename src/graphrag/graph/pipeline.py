from __future__ import annotations

import time

from groq import Groq
from sentence_transformers import SentenceTransformer

from graphrag.config import settings
from graphrag.graph.dedup import EntityDeduper
from graphrag.graph.extract import extract_triples
from graphrag.graph.store import EdgeRecord, create_graph_schema, insert_edges, upsert_entities
from graphrag.ingestion.store import get_connection


def run_extraction(politeness_delay_s: float = 10.0, arxiv_ids: list[str] | None = None) -> None:
    # Groq free tier caps llama-3.3-70b-versatile at 12,000 tokens/minute
    # (x-ratelimit-limit-tokens header). This prompt + response runs ~1500-2000
    # tokens/call, so 3s pacing still blew through the TPM budget after a few
    # calls. 10s keeps us at ~6 calls/min, comfortably under the cap. At that
    # pace, 90 papers takes ~15 min — longer than a single tool call budget, so
    # this function is resumable: pass arxiv_ids to process one batch at a time,
    # entities/edges are written after every paper (not only at the end), and
    # entity ids are content-addressed (see EntityDeduper) so batches never
    # collide with each other.
    conn = get_connection()
    create_graph_schema(conn)

    if arxiv_ids is not None:
        placeholders = ",".join(["%s"] * len(arxiv_ids))
        papers = conn.execute(
            f"SELECT arxiv_id, title, abstract FROM papers WHERE arxiv_id IN ({placeholders}) "
            "ORDER BY arxiv_id",
            arxiv_ids,
        ).fetchall()
    else:
        papers = conn.execute(
            "SELECT arxiv_id, title, abstract FROM papers ORDER BY arxiv_id"
        ).fetchall()
    print(f"Extracting entities/relations from {len(papers)} paper abstracts...")

    client = Groq(api_key=settings.groq_api_key)
    embedding_model = SentenceTransformer(settings.embedding_model)
    deduper = EntityDeduper(embedding_model)

    existing = conn.execute("SELECT entity_id, embedding FROM entities WHERE embedding IS NOT NULL").fetchall()
    if existing:
        deduper.seed_from_existing({eid: emb for eid, emb in existing})
        print(f"Seeded dedup index with {len(existing)} entities from previous batches.")

    n_raw_triples = 0
    n_failed = 0
    already_persisted: set[str] = set()

    for i, (arxiv_id, title, abstract) in enumerate(papers, 1):
        triples = extract_triples(arxiv_id, title, abstract, client=client)
        if not triples:
            n_failed += 1
        n_raw_triples += len(triples)

        paper_edges: list[EdgeRecord] = []
        for t in triples:
            source_id = deduper.entity_id_for(t.source)
            target_id = deduper.entity_id_for(t.target)
            paper_edges.append(
                EdgeRecord(
                    source_entity_id=source_id,
                    relation=t.relation,
                    target_entity_id=target_id,
                    source_chunk_id=t.source_chunk_id,
                )
            )

        new_entities = {
            eid: name for eid, name in deduper.all_entities().items() if eid not in already_persisted
        }
        if new_entities:
            new_ids = list(new_entities.keys())
            new_embeddings = embedding_model.encode(
                [new_entities[eid] for eid in new_ids], normalize_embeddings=True, show_progress_bar=False
            )
            upsert_entities(conn, new_entities, dict(zip(new_ids, new_embeddings)))
            already_persisted.update(new_ids)
        insert_edges(conn, paper_edges)

        print(f"[{i}/{len(papers)}] {arxiv_id}: {len(triples)} triples")
        if i < len(papers):
            time.sleep(politeness_delay_s)

    print(f"Done. {n_raw_triples} raw triples from {len(papers) - n_failed}/{len(papers)} papers processed.")


if __name__ == "__main__":
    run_extraction()
