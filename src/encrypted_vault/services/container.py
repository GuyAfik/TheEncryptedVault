"""ServiceContainer — dependency injection container for all services.

A single ServiceContainer instance is created at startup and passed
through the LangGraph graph. Agents and nodes call services through it.
"""

from encrypted_vault.db.base_repository import AbstractVaultRepository
from encrypted_vault.db.chroma_repository import ChromaVaultRepository
from encrypted_vault.db.in_memory_repository import InMemoryVaultRepository
from encrypted_vault.services.vault_service import VaultService
from encrypted_vault.services.chat_service import ChatService
from encrypted_vault.services.game_service import GameService


class ServiceContainer:
    """
    Holds all service instances for a game session.

    Usage:
        # Production (ChromaDB)
        container = ServiceContainer.create_production(persist_dir="./chroma_db")

        # Tests (in-memory)
        container = ServiceContainer.create_in_memory()
    """

    def __init__(
        self,
        vault: VaultService,
        chat: ChatService,
        game: GameService,
    ) -> None:
        self.vault = vault
        self.chat = chat
        self.game = game

    # ── Factory methods ────────────────────────────────────────────────────

    @classmethod
    def create_production(cls, persist_dir: str) -> "ServiceContainer":
        """
        Build a ServiceContainer backed by ChromaDB.
        Used in the actual game (Streamlit UI).
        """
        repo: AbstractVaultRepository = ChromaVaultRepository(persist_dir=persist_dir)
        return cls._build(repo)

    @classmethod
    def create_in_memory(cls) -> "ServiceContainer":
        """
        Build a ServiceContainer backed by InMemoryVaultRepository.
        Used in unit tests and CI — no ChromaDB required.
        """
        repo: AbstractVaultRepository = InMemoryVaultRepository()
        return cls._build(repo)

    @classmethod
    def _build(cls, repo: AbstractVaultRepository) -> "ServiceContainer":
        """Internal factory: wire services together with the given repo."""
        vault_service = VaultService(repo=repo)
        chat_service = ChatService()
        game_service = GameService(vault_service=vault_service, chat_service=chat_service)
        return cls(vault=vault_service, chat=chat_service, game=game_service)
