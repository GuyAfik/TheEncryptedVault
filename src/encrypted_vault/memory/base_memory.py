"""AbstractAgentMemory — interface contract for all agent memory backends."""

from abc import ABC, abstractmethod

from encrypted_vault.state.enums import AgentID


class AbstractAgentMemory(ABC):
    """
    Interface for per-agent episodic memory storage.

    Implementations must be ephemeral per game session (reset on restart).
    Retrieval is by keyword match + memory_type filter + recency (last N turns).

    Memory types:
        'vault_clue'    — content retrieved from query_vault tool
        'social_claim'  — digit claims received via private messages
        'trust_event'   — trust updates after guess feedback cross-references claims
        'reasoning'     — 200-char summary of the agent's thought this turn
    """

    @abstractmethod
    def store(
        self,
        agent_id: AgentID,
        content: str,
        memory_type: str,
        turn: int,
    ) -> None:
        """Store a new memory entry for this agent."""

    @abstractmethod
    def query(
        self,
        agent_id: AgentID,
        memory_type: str,
        current_turn: int,
        keyword: str | None = None,
        n_results: int = 3,
        recency_window: int = 5,
    ) -> list[str]:
        """
        Retrieve up to n_results memories for this agent.

        Filters:
        - memory_type: exact match
        - turn >= current_turn - recency_window  (recent turns only)
        - keyword: optional substring filter on content

        Returns most recent entries first.
        """

    @abstractmethod
    def reset(self, agent_id: AgentID) -> None:
        """Clear all memories for this agent."""

    @abstractmethod
    def reset_all(self) -> None:
        """Clear all memories for all agents (called on game restart)."""
