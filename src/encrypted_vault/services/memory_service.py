"""MemoryService — thin service wrapper around AbstractAgentMemory.

Injected into BaseAgent via ServiceContainer.
Provides:
- remember() / recall() / forget() / forget_all()  — episodic memory
- store_message() / load_history() / clear_history() / clear_all_history() — chat history
"""

import logging

from encrypted_vault.memory.base_memory import AbstractAgentMemory
from encrypted_vault.state.enums import AgentID

logger = logging.getLogger(__name__)


class MemoryService:
    """
    Service layer wrapper around AbstractAgentMemory.

    Injected into ServiceContainer and accessed by BaseAgent as services.memory.
    All agents share the same MemoryService instance; data is partitioned by agent_id.
    """

    def __init__(self, repository: AbstractAgentMemory) -> None:
        self._repo = repository

    # ── Episodic memory ────────────────────────────────────────────────────

    def remember(
        self,
        agent_id: AgentID,
        content: str,
        memory_type: str,
        turn: int,
    ) -> None:
        """Store a new episodic memory entry for this agent."""
        try:
            self._repo.store(agent_id, content, memory_type, turn)
        except Exception as e:
            logger.warning("[%s] Failed to store memory (%s): %s", agent_id.value, memory_type, e)

    def recall(
        self,
        agent_id: AgentID,
        memory_type: str,
        current_turn: int,
        keyword: str | None = None,
        n_results: int = 3,
        recency_window: int = 5,
    ) -> list[str]:
        """Retrieve up to n_results episodic memories for this agent."""
        try:
            return self._repo.query(
                agent_id=agent_id,
                memory_type=memory_type,
                current_turn=current_turn,
                keyword=keyword,
                n_results=n_results,
                recency_window=recency_window,
            )
        except Exception as e:
            logger.warning("[%s] Failed to recall memories (%s): %s", agent_id.value, memory_type, e)
            return []

    def forget(self, agent_id: AgentID) -> None:
        """Clear all episodic memories and chat history for this agent."""
        try:
            self._repo.reset(agent_id)
        except Exception as e:
            logger.warning("[%s] Failed to reset memory: %s", agent_id.value, e)

    def forget_all(self) -> None:
        """Clear all memories and chat histories for all agents (game restart)."""
        try:
            self._repo.reset_all()
            logger.info("All agent memories + chat histories cleared")
        except Exception as e:
            logger.warning("Failed to reset all memories: %s", e)

    # ── Chat history ───────────────────────────────────────────────────────

    def store_message(
        self,
        agent_id: AgentID,
        turn: int,
        role: str,
        content: str,
        tool_call_id: str | None = None,
        tool_calls: list | None = None,
    ) -> None:
        """Append a LangChain message to this agent's persistent chat history."""
        try:
            self._repo.store_message(
                agent_id=agent_id,
                turn=turn,
                role=role,
                content=content,
                tool_call_id=tool_call_id,
                tool_calls=tool_calls,
            )
        except Exception as e:
            logger.warning("[%s] Failed to store chat message (%s): %s", agent_id.value, role, e)

    def load_history(
        self,
        agent_id: AgentID,
        max_turns: int = 10,
    ) -> list[dict]:
        """
        Load the persistent chat history for this agent.

        Returns list of message dicts ordered chronologically (oldest first):
            [{'role': 'system', 'content': '...', 'tool_call_id': None, 'tool_calls': None}, ...]

        Keeps last max_turns turns; system message always included.
        """
        try:
            return self._repo.load_history(agent_id=agent_id, max_turns=max_turns)
        except Exception as e:
            logger.warning("[%s] Failed to load chat history: %s", agent_id.value, e)
            return []

    def clear_history(self, agent_id: AgentID) -> None:
        """Clear chat history for this agent only."""
        try:
            self._repo.clear_history(agent_id)
        except Exception as e:
            logger.warning("[%s] Failed to clear chat history: %s", agent_id.value, e)

    def clear_all_history(self) -> None:
        """Clear chat history for all agents."""
        try:
            self._repo.clear_all_history()
            logger.info("All agent chat histories cleared")
        except Exception as e:
            logger.warning("Failed to clear all chat histories: %s", e)
