from __future__ import annotations

import hashlib
import re

import numpy as np
from sentence_transformers import SentenceTransformer


def normalize_entity(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def _stable_entity_id(normalized: str) -> str:
    return "e_" + hashlib.sha1(normalized.encode()).hexdigest()[:12]


class EntityDeduper:
    """Crude greedy entity resolution: exact match on the normalized string first,
    then merge remaining distinct strings whose embeddings exceed a similarity
    threshold into the nearest existing cluster.

    This is not a rigorous clustering algorithm (no re-clustering, order-dependent,
    O(n^2) in the number of unique entities) — it exists to stop obvious duplicates
    like "Transformer" and "transformer model" from becoming disconnected graph
    nodes, since that silently kills the 1-2 hop neighbor lookups hybrid retrieval
    depends on. Good enough at ~500-1000 unique entities; revisit if it doesn't scale.

    entity_id is a stable hash of the normalized string, not a per-run counter —
    this matters because extraction runs in batches (Groq's free-tier TPM limit
    forces ~10s/call pacing, so a full 90-paper run has to be resumable across
    multiple process invocations). A counter-based id would silently collide
    across separate runs; a content-addressed id can't.
    """

    def __init__(self, model: SentenceTransformer, similarity_threshold: float = 0.87):
        self.model = model
        self.similarity_threshold = similarity_threshold
        self._cluster_normalized_names: list[str] = []
        self._cluster_embeddings: np.ndarray | None = None
        self._normalized_to_entity_id: dict[str, str] = {}
        self._canonical_names: dict[str, str] = {}

    def seed_from_existing(self, canonical_names: dict[str, np.ndarray]) -> None:
        """Preloads entities already persisted from a previous batch, so fuzzy
        matching in this batch can still merge against them. canonical_names:
        entity_id -> embedding."""
        for entity_id, embedding in canonical_names.items():
            self._cluster_normalized_names.append(f"__seed__{entity_id}")
            self._normalized_to_entity_id[f"__seed__{entity_id}"] = entity_id
            emb = embedding.to_numpy().reshape(1, -1)
            self._cluster_embeddings = (
                emb if self._cluster_embeddings is None else np.vstack([self._cluster_embeddings, emb])
            )

    def entity_id_for(self, raw_name: str) -> str:
        normalized = normalize_entity(raw_name) or "unknown"

        if normalized in self._normalized_to_entity_id:
            return self._normalized_to_entity_id[normalized]

        entity_id = _stable_entity_id(normalized)

        if self._cluster_embeddings is not None and len(self._cluster_normalized_names) > 0:
            query_emb = self.model.encode(normalized, normalize_embeddings=True)
            sims = self._cluster_embeddings @ query_emb
            best_idx = int(np.argmax(sims))
            if sims[best_idx] >= self.similarity_threshold:
                matched_name = self._cluster_normalized_names[best_idx]
                matched_entity_id = self._normalized_to_entity_id[matched_name]
                self._normalized_to_entity_id[normalized] = matched_entity_id
                if matched_entity_id not in self._canonical_names:
                    self._canonical_names[matched_entity_id] = raw_name.strip()
                return matched_entity_id

        self._normalized_to_entity_id[normalized] = entity_id
        self._canonical_names[entity_id] = raw_name.strip()
        self._cluster_normalized_names.append(normalized)
        emb = self.model.encode(normalized, normalize_embeddings=True).reshape(1, -1)
        self._cluster_embeddings = (
            emb if self._cluster_embeddings is None else np.vstack([self._cluster_embeddings, emb])
        )
        return entity_id

    def canonical_name(self, entity_id: str) -> str:
        return self._canonical_names[entity_id]

    def all_entities(self) -> dict[str, str]:
        """entity_id -> canonical_name, for every distinct NEW entity seen in this
        batch (excludes seeded entities from previous batches)."""
        return dict(self._canonical_names)
