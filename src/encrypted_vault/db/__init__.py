"""Database layer — pure data access, no business logic.

Layer 1 of the 5-layer architecture.
Only imports: state/ models + specific DB client libraries.
Never imports from: services/, agents/, graph/, ui/
"""

from encrypted_vault.db.base_repository import AbstractVaultRepository
from encrypted_vault.db.chroma_repository import ChromaVaultRepository
from encrypted_vault.db.in_memory_repository import InMemoryVaultRepository

__all__ = [
    "AbstractVaultRepository",
    "ChromaVaultRepository",
    "InMemoryVaultRepository",
]
