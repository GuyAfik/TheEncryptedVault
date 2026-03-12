"""Abstract base class (interface) for all vault storage backends.

Services depend ONLY on AbstractVaultRepository — never on a concrete class.
To add a new backend (e.g. Pinecone), implement this interface and wire it
into ServiceContainer. Zero changes required in any other layer.
"""

from abc import ABC, abstractmethod

from encrypted_vault.state.vault_models import VaultFragment


class AbstractVaultRepository(ABC):
    """
    Interface contract for vault storage backends.

    Concrete implementations:
    - ChromaVaultRepository  (production — ChromaDB local persistence)
    - InMemoryVaultRepository (tests/CI — no external dependencies)
    """

    @abstractmethod
    def upsert_fragment(self, fragment: VaultFragment) -> None:
        """Insert or update a vault fragment by chunk_id."""
        ...

    @abstractmethod
    def get_fragment(self, chunk_id: str) -> VaultFragment | None:
        """Retrieve a single fragment by its chunk_id. Returns None if not found."""
        ...

    @abstractmethod
    def get_all_fragments(self) -> list[VaultFragment]:
        """Return all fragments currently stored in the vault."""
        ...

    @abstractmethod
    def query_similar(self, search_term: str, n_results: int = 2) -> list[VaultFragment]:
        """
        Retrieve the top-n fragments most semantically similar to search_term.
        Implementations may use vector similarity, keyword search, or TF-IDF.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Delete all fragments from the vault (used for game restart)."""
        ...
