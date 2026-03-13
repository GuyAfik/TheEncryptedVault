"""InMemoryAgentMemory — pure dict implementation for unit tests.

No SQLite dependency. Uses a simple list of dicts per agent.
Retrieval: filter by memory_type + recency window, optional keyword substring match.
Also implements chat history storage for persistent per-agent LLM conversation.
"""

from encrypted_vault.memory.base_memory import AbstractAgentMemory
from encrypted_vault.state.enums import AgentID


class InMemoryAgentMemory(AbstractAgentMemory):
    """
    Pure in-memory implementation of AbstractAgentMemory for unit tests.
    No external dependencies — just dicts of lists.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[dict]] = {a.value: [] for a in AgentID}
        self._history: dict[str, list[dict]] = {a.value: [] for a in AgentID}

    # ── Episodic memory ────────────────────────────────────────────────────

    def store(
        self,
        agent_id: AgentID,
        content: str,
        memory_type: str,
        turn: int,
    ) -> None:
        self._store[agent_id.value].append({
            "content": content,
            "memory_type": memory_type,
            "turn": turn,
        })

    def query(
        self,
        agent_id: AgentID,
        memory_type: str,
        current_turn: int,
        keyword: str | None = None,
        n_results: int = 3,
        recency_window: int = 5,
    ) -> list[str]:
        min_turn = max(0, current_turn - recency_window)
        entries = [
            e for e in self._store[agent_id.value]
            if e["memory_type"] == memory_type and e["turn"] >= min_turn
        ]
        if keyword:
            entries = [e for e in entries if keyword.lower() in e["content"].lower()]
        # Most recent first
        entries.sort(key=lambda e: e["turn"], reverse=True)
        return [e["content"] for e in entries[:n_results]]

    def reset(self, agent_id: AgentID) -> None:
        self._store[agent_id.value] = []
        self._history[agent_id.value] = []

    def reset_all(self) -> None:
        for agent_id in AgentID:
            self._store[agent_id.value] = []
            self._history[agent_id.value] = []

    def count(self, agent_id: AgentID | None = None) -> int:
        if agent_id:
            return len(self._store[agent_id.value])
        return sum(len(v) for v in self._store.values())

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
        """Append a message to this agent's chat history."""
        self._history[agent_id.value].append({
            "role": role,
            "content": content,
            "turn": turn,
            "tool_call_id": tool_call_id,
            "tool_calls": tool_calls,
        })

    def load_history(
        self,
        agent_id: AgentID,
        max_turns: int = 10,
    ) -> list[dict]:
        """
        Load chat history for this agent, keeping last max_turns turns.
        System messages are always included.
        """
        history = self._history[agent_id.value]
        if not history:
            return []
        # Find the latest turn
        latest_turn = max((m["turn"] for m in history), default=0)
        min_turn = max(0, latest_turn - max_turns + 1)
        return [
            {
                "role": m["role"],
                "content": m["content"],
                "tool_call_id": m["tool_call_id"],
                "tool_calls": m["tool_calls"],
            }
            for m in history
            if m["role"] == "system" or m["turn"] >= min_turn
        ]

    def clear_history(self, agent_id: AgentID) -> None:
        """Clear chat history for this agent."""
        self._history[agent_id.value] = []

    def clear_all_history(self) -> None:
        """Clear chat history for all agents."""
        for agent_id in AgentID:
            self._history[agent_id.value] = []
