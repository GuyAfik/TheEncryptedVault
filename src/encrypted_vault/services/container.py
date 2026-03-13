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
from encrypted_vault.services.memory_service import MemoryService
from encrypted_vault.memory.sqlite_memory import SQLiteAgentMemory
from encrypted_vault.memory.in_memory_memory import InMemoryAgentMemory


class ServiceContainer:
    """
    Holds all service instances for a game session.

    Usage:
        # Production (ChromaDB vault + SQLite agent memory)
        container = ServiceContainer.create_production(persist_dir="./chroma_db")

        # Tests (in-memory vault + in-memory agent memory)
        container = ServiceContainer.create_in_memory()
    """

    def __init__(
        self,
        vault: VaultService,
        chat: ChatService,
        game: GameService,
        memory: MemoryService,
    ) -> None:
        self.vault = vault
        self.chat = chat
        self.game = game
        self.memory = memory

    # ── Factory methods ────────────────────────────────────────────────────

    @classmethod
    def create_production(cls, persist_dir: str) -> "ServiceContainer":
        """
        Build a ServiceContainer backed by ChromaDB vault + SQLite agent memory.
        Used in the actual game (Streamlit UI).
        """
        repo: AbstractVaultRepository = ChromaVaultRepository(persist_dir=persist_dir)
        return cls._build(repo, use_sqlite_memory=True)

    @classmethod
    def create_in_memory(cls) -> "ServiceContainer":
        """
        Build a ServiceContainer backed by InMemoryVaultRepository + InMemoryAgentMemory.
        Used in unit tests and CI — no ChromaDB or SQLite required.
        """
        repo: AbstractVaultRepository = InMemoryVaultRepository()
        return cls._build(repo, use_sqlite_memory=False)

    @classmethod
    def _build(cls, repo: AbstractVaultRepository, use_sqlite_memory: bool = True) -> "ServiceContainer":
        """Internal factory: wire services together with the given repo."""
        vault_service = VaultService(repo=repo)
        chat_service = ChatService()
        game_service = GameService(vault_service=vault_service, chat_service=chat_service)
        memory_repo = SQLiteAgentMemory() if use_sqlite_memory else InMemoryAgentMemory()
        memory_service = MemoryService(repository=memory_repo)
        return cls(vault=vault_service, chat=chat_service, game=game_service, memory=memory_service)
