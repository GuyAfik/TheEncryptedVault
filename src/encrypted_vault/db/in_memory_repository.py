"""In-memory implementation of AbstractVaultRepository.

Used for unit tests and CI — no ChromaDB dependency, instant startup.
Implements query_similar() using simple TF-IDF cosine similarity so
tests exercise realistic retrieval behaviour without a vector DB.
"""

import math
import re
from collections import Counter

from encrypted_vault.db.base_repository import AbstractVaultRepository
from encrypted_vault.state.vault_models import VaultFragment


def _tokenise(text: str) -> list[str]:
    """Lowercase word tokeniser."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _tf(tokens: list[str]) -> dict[str, float]:
    """Term frequency: count / total."""
    total = len(tokens) or 1
    counts = Counter(tokens)
    return {term: count / total for term, count in counts.items()}


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two TF vectors."""
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class InMemoryVaultRepository(AbstractVaultRepository):
    """
    Pure in-memory vault store for testing.

    Stores fragments in a dict keyed by chunk_id.
    query_similar() uses TF cosine similarity — no external dependencies.
    """

    def __init__(self) -> None:
        self._store: dict[str, VaultFragment] = {}

    # ── AbstractVaultRepository interface ──────────────────────────────────

    def upsert_fragment(self, fragment: VaultFragment) -> None:
        """Insert or replace a fragment."""
        self._store[fragment.chunk_id] = fragment

    def get_fragment(self, chunk_id: str) -> VaultFragment | None:
        """Return fragment by ID, or None."""
        return self._store.get(chunk_id)

    def get_all_fragments(self) -> list[VaultFragment]:
        """Return all stored fragments."""
        return list(self._store.values())

    def query_similar(self, search_term: str, n_results: int = 2) -> list[VaultFragment]:
        """Return top-n fragments by TF cosine similarity to search_term."""
        if not self._store:
            return []
        query_vec = _tf(_tokenise(search_term))
        scored = [
            (fragment, _cosine_similarity(query_vec, _tf(_tokenise(fragment.content))))
            for fragment in self._store.values()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [fragment for fragment, _ in scored[:n_results]]

    def reset(self) -> None:
        """Clear all fragments."""
        self._store.clear()
