"""Agent episodic memory layer.

Two implementations:
- SQLiteAgentMemory: in-memory SQLite, ephemeral per game session (production)
- InMemoryAgentMemory: pure dict, no SQLite dependency (tests)

Both implement AbstractAgentMemory.
"""

from encrypted_vault.memory.base_memory import AbstractAgentMemory
from encrypted_vault.memory.sqlite_memory import SQLiteAgentMemory
from encrypted_vault.memory.in_memory_memory import InMemoryAgentMemory

__all__ = ["AbstractAgentMemory", "SQLiteAgentMemory", "InMemoryAgentMemory"]
