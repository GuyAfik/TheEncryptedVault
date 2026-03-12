"""Abstract BaseAgent — the common interface for all 4 game agents."""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from encrypted_vault.state.enums import AgentID
from encrypted_vault.state.agent_models import AgentPrivateState
from encrypted_vault.state.game_state import GlobalGameState
from encrypted_vault.services.container import ServiceContainer


class AgentTurnResult(BaseModel):
    """The output of a single agent turn."""

    agent_id: AgentID
    thought: str
    """Internal reasoning — shown in UI but never sent to other agents."""

    tool_calls_made: list[dict[str, Any]]
    """List of tool invocations: [{"tool": name, "args": {...}, "result": ...}]"""

    updated_private_state: AgentPrivateState
    """The agent's private state after this turn."""

    public_messages: list[str]
    """Any public broadcasts made this turn."""

    private_messages: list[dict[str, str]]
    """Any private DMs sent: [{"recipient": AgentID, "content": str}]"""

    guess_submitted: str | None = None
    """If the agent submitted a guess, the code string; else None."""


class BaseAgent(ABC):
    """
    Abstract base class for all game agents.

    Subclasses must implement:
    - _build_system_prompt() → str
    - _select_tools(services) → list[BaseTool]

    The run_turn() method executes the full Reasoning → Action loop.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        services: ServiceContainer,
        agent_id: AgentID,
    ) -> None:
        self.agent_id = agent_id
        self._llm = llm
        self._services = services
        self._system_prompt = self._build_system_prompt()
        self._tools = self._select_tools(services)
        self._llm_with_tools = llm.bind_tools(self._tools)

    # ── Abstract interface ─────────────────────────────────────────────────

    @abstractmethod
    def _build_system_prompt(self) -> str:
        """Return the agent's system prompt defining its personality and strategy."""
        ...

    @abstractmethod
    def _select_tools(self, services: ServiceContainer) -> list[BaseTool]:
        """Return the list of tools this agent is allowed to use."""
        ...

    # ── Public interface ───────────────────────────────────────────────────

    def run_turn(self, game_state: GlobalGameState) -> AgentTurnResult:
        """
        Execute one full turn for this agent.

        Flow:
        1. Build context from game state (vault results, chat history, private inbox)
        2. Call LLM with system prompt + context
        3. Execute any tool calls the LLM requests
        4. Update private state with new knowledge
        5. Return AgentTurnResult
        """
        private_state = game_state.agent_states[self.agent_id]
        context = self._build_context(game_state, private_state)

        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=context),
        ]

        # ── Agentic loop (up to 3 tool call rounds) ───────────────────────
        tool_calls_made: list[dict[str, Any]] = []
        public_messages: list[str] = []
        private_messages: list[dict[str, str]] = []
        guess_submitted: str | None = None
        thought = ""

        for _ in range(3):  # max 3 rounds of tool use per turn
            response = self._llm_with_tools.invoke(messages)
            messages.append(response)

            # Extract thought from response content
            if isinstance(response.content, str) and response.content:
                thought = response.content

            # No tool calls → agent is done
            if not hasattr(response, "tool_calls") or not response.tool_calls:
                break

            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                result = self._execute_tool(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    game_state=game_state,
                    private_state=private_state,
                    turn=game_state.turn,
                )

                # Track side effects
                if tool_name == "broadcast_message":
                    public_messages.append(tool_args.get("content", ""))
                elif tool_name == "send_private_message":
                    private_messages.append({
                        "recipient": tool_args.get("recipient", ""),
                        "content": tool_args.get("content", ""),
                    })
                elif tool_name == "submit_guess":
                    guess_submitted = tool_args.get("code", "")

                tool_calls_made.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result,
                })

                # Feed tool result back to LLM
                from langchain_core.messages import ToolMessage
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

        # ── Update private state ───────────────────────────────────────────
        updated_state = self._update_private_state(
            private_state=private_state,
            thought=thought,
            tool_calls_made=tool_calls_made,
        )

        return AgentTurnResult(
            agent_id=self.agent_id,
            thought=thought,
            tool_calls_made=tool_calls_made,
            updated_private_state=updated_state,
            public_messages=public_messages,
            private_messages=private_messages,
            guess_submitted=guess_submitted,
        )

    # ── Context building ───────────────────────────────────────────────────

    def _build_context(
        self,
        game_state: GlobalGameState,
        private_state: AgentPrivateState,
    ) -> str:
        """
        Build the context string passed to the LLM as the human message.
        Includes: turn info, public chat, private inbox, knowledge base.
        Note: master_key is NEVER included here.
        """
        lines: list[str] = []

        lines.append(f"=== GAME STATUS ===")
        lines.append(f"Turn: {game_state.turn + 1} / {game_state.max_turns}")
        lines.append(f"You are: {self.agent_id.display_name} {self.agent_id.emoji}")
        lines.append(f"Guesses remaining: {private_state.guesses_remaining}")
        lines.append(f"Token budget remaining: {private_state.token_budget - private_state.tokens_used}")
        lines.append("")

        # Public chat (last 10 messages)
        lines.append("=== PUBLIC CHAT (last 10 messages) ===")
        recent_public = game_state.public_chat[-10:]
        if recent_public:
            for msg in recent_public:
                lines.append(f"  [{msg.sender}]: {msg.content}")
        else:
            lines.append("  (no messages yet)")
        lines.append("")

        # Private inbox
        inbox = game_state.private_inboxes.get(self.agent_id)
        if inbox and inbox.messages:
            lines.append("=== YOUR PRIVATE INBOX ===")
            for msg in inbox.messages[-5:]:  # last 5 DMs
                lines.append(f"  From [{msg.sender}]: {msg.content}")
            lines.append("")

        # Knowledge base
        if private_state.knowledge_base:
            lines.append("=== YOUR KNOWLEDGE BASE ===")
            for clue in private_state.knowledge_base[-10:]:
                lines.append(f"  - {clue}")
            lines.append("")

        # Current suspicion
        if private_state.suspected_key:
            lines.append(f"=== YOUR CURRENT SUSPICION ===")
            lines.append(f"  Suspected key: {private_state.suspected_key}")
            if private_state.known_digits:
                known_str = ", ".join(
                    f"pos {p}={d}" for p, d in sorted(private_state.known_digits.items())
                )
                lines.append(f"  Confirmed digits: {known_str}")
            lines.append("")

        lines.append("=== YOUR TURN ===")
        lines.append(
            "Think carefully, then use your available tools. "
            "You may use multiple tools in sequence. "
            "Remember: your goal is to find the 4-digit Master Key before the others."
        )

        return "\n".join(lines)

    # ── Tool execution ─────────────────────────────────────────────────────

    def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict,
        game_state: GlobalGameState,
        private_state: AgentPrivateState,
        turn: int,
    ) -> Any:
        """Dispatch a tool call to the appropriate service method."""
        tool_map = {t.name: t for t in self._tools}
        if tool_name not in tool_map:
            return f"Error: tool '{tool_name}' not available to {self.agent_id.value}."
        tool = tool_map[tool_name]
        return tool.invoke(tool_args)

    # ── Private state update ───────────────────────────────────────────────

    def _update_private_state(
        self,
        private_state: AgentPrivateState,
        thought: str,
        tool_calls_made: list[dict],
    ) -> AgentPrivateState:
        """
        Update the agent's private state after a turn.
        Appends thought to trace, extracts knowledge from tool results.
        """
        updated = private_state.model_copy(deep=True)
        updated.turns_played += 1

        if thought:
            updated.add_thought(thought)

        # Extract knowledge from vault query results
        for call in tool_calls_made:
            if call["tool"] == "query_vault" and isinstance(call["result"], list):
                for fragment in call["result"]:
                    if isinstance(fragment, dict):
                        content = fragment.get("content", "")
                        if content:
                            updated.add_knowledge(f"[Vault] {content}")

        return updated
