"""SQLiteAgentMemory — ephemeral in-memory SQLite backend for agent episodic memory
and persistent chat history.

Uses Python stdlib sqlite3 with an in-memory database (:memory:).
Ephemeral: all data is lost when the object is garbage-collected or reset_all() is called.
No embeddings, no external dependencies.

Two tables:
1. agent_memories   — episodic memories (vault clues, social claims, trust events, reasoning)
2. agent_chat_history — full LangChain message history per agent (system/human/ai/tool messages)
"""

import json
import sqlite3
import logging

from encrypted_vault.memory.base_memory import AbstractAgentMemory
from encrypted_vault.state.enums import AgentID

logger = logging.getLogger(__name__)

_CREATE_MEMORIES_TABLE = """
CREATE TABLE IF NOT EXISTS agent_memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    TEXT    NOT NULL,
    turn        INTEGER NOT NULL,
    memory_type TEXT    NOT NULL,
    content     TEXT    NOT NULL
);
"""

_CREATE_MEMORIES_INDEX = """
CREATE INDEX IF NOT EXISTS idx_agent_type_turn
ON agent_memories (agent_id, memory_type, turn);
"""

_CREATE_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS agent_chat_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id     TEXT    NOT NULL,
    turn         INTEGER NOT NULL,
    role         TEXT    NOT NULL,
    content      TEXT    NOT NULL,
    tool_call_id TEXT,
    tool_calls   TEXT
);
"""

_CREATE_HISTORY_INDEX = """
CREATE INDEX IF NOT EXISTS idx_history_agent_turn
ON agent_chat_history (agent_id, turn);
"""


class SQLiteAgentMemory(AbstractAgentMemory):
    """
    In-memory SQLite implementation of AbstractAgentMemory.

    One shared in-memory SQLite database for all agents.
    Ephemeral — reset on game restart via reset_all().
    Thread-safe: uses check_same_thread=False (LangGraph runs in a single thread per game).

    Two tables:
    - agent_memories: episodic memories (vault clues, social claims, trust events, reasoning)
    - agent_chat_history: full LangChain message history per agent
    """

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_CREATE_MEMORIES_TABLE)
        self._conn.execute(_CREATE_MEMORIES_INDEX)
        self._conn.execute(_CREATE_HISTORY_TABLE)
        self._conn.execute(_CREATE_HISTORY_INDEX)
        self._conn.commit()
        logger.debug("SQLiteAgentMemory initialised (in-memory)")

    # ── AbstractAgentMemory interface ──────────────────────────────────────

    def store(
        self,
        agent_id: AgentID,
        content: str,
        memory_type: str,
        turn: int,
    ) -> None:
        """Insert a new memory entry. Duplicate content for the same agent/turn/type is allowed."""
        self._conn.execute(
            "INSERT INTO agent_memories (agent_id, turn, memory_type, content) VALUES (?, ?, ?, ?)",
            (agent_id.value, turn, memory_type, content),
        )
        self._conn.commit()
        logger.debug("[%s] Stored %s memory at turn %d: %s", agent_id.value, memory_type, turn, content[:60])

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

        Filters applied (in order):
        1. agent_id = this agent
        2. memory_type = requested type
        3. turn >= current_turn - recency_window  (recent turns only)
        4. content LIKE '%keyword%'  (optional)

        Returns most recent entries first.
        """
        min_turn = max(0, current_turn - recency_window)

        if keyword:
            rows = self._conn.execute(
                """
                SELECT content FROM agent_memories
                WHERE agent_id = ?
                  AND memory_type = ?
                  AND turn >= ?
                  AND content LIKE ?
                ORDER BY turn DESC
                LIMIT ?
                """,
                (agent_id.value, memory_type, min_turn, f"%{keyword}%", n_results),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT content FROM agent_memories
                WHERE agent_id = ?
                  AND memory_type = ?
                  AND turn >= ?
                ORDER BY turn DESC
                LIMIT ?
                """,
                (agent_id.value, memory_type, min_turn, n_results),
            ).fetchall()

        return [row["content"] for row in rows]

    def reset(self, agent_id: AgentID) -> None:
        """Delete all episodic memories and chat history for this agent."""
        self._conn.execute(
            "DELETE FROM agent_memories WHERE agent_id = ?",
            (agent_id.value,),
        )
        self._conn.execute(
            "DELETE FROM agent_chat_history WHERE agent_id = ?",
            (agent_id.value,),
        )
        self._conn.commit()
        logger.debug("[%s] Memory + chat history reset", agent_id.value)

    def reset_all(self) -> None:
        """Delete all episodic memories and chat history for all agents (game restart)."""
        self._conn.execute("DELETE FROM agent_memories")
        self._conn.execute("DELETE FROM agent_chat_history")
        self._conn.commit()
        logger.debug("All agent memories + chat histories reset")

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
        """
        Append a single LangChain message to this agent's chat history.

        Args:
            agent_id:     The agent this message belongs to.
            turn:         The game turn when this message was created.
            role:         One of 'system', 'human', 'ai', 'tool'.
            content:      The message text content.
            tool_call_id: For ToolMessage — the ID of the tool call being responded to.
            tool_calls:   For AIMessage — list of tool call dicts (serialised as JSON).
        """
        self._conn.execute(
            """
            INSERT INTO agent_chat_history
                (agent_id, turn, role, content, tool_call_id, tool_calls)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                agent_id.value,
                turn,
                role,
                content,
                tool_call_id,
                json.dumps(tool_calls) if tool_calls else None,
            ),
        )
        self._conn.commit()

    def load_history(
        self,
        agent_id: AgentID,
        max_turns: int = 10,
    ) -> list[dict]:
        """
        Load the chat history for this agent.

        Returns a list of message dicts ordered chronologically (oldest first):
            [{'role': 'system', 'content': '...', 'tool_call_id': None, 'tool_calls': None}, ...]

        Args:
            agent_id:  The agent whose history to load.
            max_turns: Keep only messages from the last N turns (plus the system message).
                       Older messages are dropped to stay within the LLM context window.
        """
        # Find the minimum turn to include (keep last max_turns turns)
        row = self._conn.execute(
            "SELECT MAX(turn) FROM agent_chat_history WHERE agent_id = ?",
            (agent_id.value,),
        ).fetchone()
        latest_turn = row[0] if row[0] is not None else 0
        min_turn = max(0, latest_turn - max_turns + 1)

        rows = self._conn.execute(
            """
            SELECT role, content, tool_call_id, tool_calls
            FROM agent_chat_history
            WHERE agent_id = ?
              AND (role = 'system' OR turn >= ?)
            ORDER BY id ASC
            """,
            (agent_id.value, min_turn),
        ).fetchall()

        return [
            {
                "role": row["role"],
                "content": row["content"],
                "tool_call_id": row["tool_call_id"],
                "tool_calls": json.loads(row["tool_calls"]) if row["tool_calls"] else None,
            }
            for row in rows
        ]

    def clear_history(self, agent_id: AgentID) -> None:
        """Clear chat history for this agent only."""
        self._conn.execute(
            "DELETE FROM agent_chat_history WHERE agent_id = ?",
            (agent_id.value,),
        )
        self._conn.commit()
        logger.debug("[%s] Chat history cleared", agent_id.value)

    def clear_all_history(self) -> None:
        """Clear chat history for all agents."""
        self._conn.execute("DELETE FROM agent_chat_history")
        self._conn.commit()
        logger.debug("All agent chat histories cleared")

    # ── Utility ────────────────────────────────────────────────────────────

    def count(self, agent_id: AgentID | None = None) -> int:
        """Return total memory count, optionally filtered by agent."""
        if agent_id:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM agent_memories WHERE agent_id = ?",
                (agent_id.value,),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM agent_memories").fetchone()
        return row[0]
