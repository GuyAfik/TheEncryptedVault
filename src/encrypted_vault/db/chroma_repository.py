"""ChromaDB implementation of AbstractVaultRepository.

Production backend using local ChromaDB persistence.
Stores VaultFragment metadata alongside embeddings so we can
reconstruct full Pydantic objects from query results.
"""

import json

import chromadb
from chromadb.config import Settings as ChromaSettings

from encrypted_vault.db.base_repository import AbstractVaultRepository
from encrypted_vault.state.vault_models import VaultFragment

_COLLECTION_NAME = "vault_fragments"


class ChromaVaultRepository(AbstractVaultRepository):
    """
    Persists vault fragments in a local ChromaDB collection.

    ChromaDB handles embedding generation automatically using its default
    sentence-transformers model (all-MiniLM-L6-v2).
    """

    def __init__(self, persist_dir: str) -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ── AbstractVaultRepository interface ──────────────────────────────────

    def upsert_fragment(self, fragment: VaultFragment) -> None:
        """Insert or update a fragment. Metadata stores all Pydantic fields."""
        self._collection.upsert(
            ids=[fragment.chunk_id],
            documents=[fragment.content],
            metadatas=[self._to_metadata(fragment)],
        )

    def get_fragment(self, chunk_id: str) -> VaultFragment | None:
        """Retrieve a fragment by ID. Returns None if not found."""
        result = self._collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not result["ids"]:
            return None
        return self._from_result(
            chunk_id=result["ids"][0],
            document=result["documents"][0],
            metadata=result["metadatas"][0],
        )

    def get_all_fragments(self) -> list[VaultFragment]:
        """Return all stored fragments."""
        result = self._collection.get(include=["documents", "metadatas"])
        return [
            self._from_result(chunk_id=cid, document=doc, metadata=meta)
            for cid, doc, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]

    def query_similar(self, search_term: str, n_results: int = 2) -> list[VaultFragment]:
        """Return top-n fragments by cosine similarity to search_term."""
        count = self._collection.count()
        if count == 0:
            return []
        actual_n = min(n_results, count)
        result = self._collection.query(
            query_texts=[search_term],
            n_results=actual_n,
            include=["documents", "metadatas"],
        )
        fragments = []
        for cid, doc, meta in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0]
        ):
            fragments.append(self._from_result(chunk_id=cid, document=doc, metadata=meta))
        return fragments

    def reset(self) -> None:
        """Delete and recreate the collection (full wipe)."""
        self._client.delete_collection(_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _to_metadata(fragment: VaultFragment) -> dict:
        """Serialise VaultFragment fields to ChromaDB metadata (strings/ints only)."""
        return {
            "is_key_fragment": str(fragment.is_key_fragment),
            "digit_position": json.dumps(fragment.digit_position),
            "corruption_count": fragment.corruption_count,
        }

    @staticmethod
    def _from_result(chunk_id: str, document: str, metadata: dict) -> VaultFragment:
        """Reconstruct a VaultFragment from ChromaDB query result."""
        return VaultFragment(
            chunk_id=chunk_id,
            content=document,
            is_key_fragment=metadata["is_key_fragment"] == "True",
            digit_position=json.loads(metadata["digit_position"]),
            corruption_count=int(metadata["corruption_count"]),
        )
