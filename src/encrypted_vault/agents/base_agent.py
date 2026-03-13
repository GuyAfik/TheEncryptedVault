"""Abstract BaseAgent — the common interface for all 4 game agents."""

import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from encrypted_vault.state.enums import AgentID
from encrypted_vault.state.agent_models import AgentPrivateState
from encrypted_vault.state.game_state import GlobalGameState
from encrypted_vault.services.container import ServiceContainer

logger = logging.getLogger(__name__)


class AgentTurnResult(BaseModel):
    """The output of a single agent turn."""

    agent_id: AgentID
    thought: str
    tool_calls_made: list[dict[str, Any]]
    updated_private_state: AgentPrivateState
    public_messages: list[str]
    private_messages: list[dict[str, str]]
    guess_submitted: str | None = None


class BaseAgent(ABC):
    """Abstract base class for all game agents."""

    def __init__(
        self,
        llm: BaseChatModel,
        services: ServiceContainer,
        agent_id: AgentID,
    ) -> None:
        self.agent_id = agent_id
        self._llm = llm
        self._services = services
        # Mutable reference to private state — updated by tool callbacks
        self._current_private_state: AgentPrivateState | None = None
        self._system_prompt = self._build_system_prompt()
        self._tools = self._select_tools(services)
        self._tool_map: dict[str, BaseTool] = {t.name: t for t in self._tools}
        self._llm_with_tools = llm.bind_tools(self._tools)

    @abstractmethod
    def _build_system_prompt(self) -> str: ...

    @abstractmethod
    def _select_tools(self, services: ServiceContainer) -> list[BaseTool]: ...

    def _make_previous_guesses_getter(self):
        """Returns a callable that returns the list of previously submitted guess codes."""
        def getter() -> list[str]:
            if self._current_private_state is None:
                return []
            return [e["guess"] for e in self._current_private_state.guess_history]
        return getter

    def _make_private_state_updater(self):
        """
        Returns a callback that the submit_guess tool calls with per-digit feedback.
        Updates known_digits and wrong_digits on the current private state.
        """
        def updater(feedback: dict):
            if self._current_private_state is None:
                return
            state = self._current_private_state

            # Update known_digits from correct positions
            for pos, digit in feedback.get("correct_positions", []):
                state.known_digits[pos] = digit
                logger.info("[%s] Confirmed digit pos %d = '%s' from guess feedback",
                            self.agent_id.value, pos, digit)

            # Update wrong_digits from wrong positions
            for pos, digit in feedback.get("wrong_positions", []):
                if pos not in state.wrong_digits:
                    state.wrong_digits[pos] = []
                if digit not in state.wrong_digits[pos]:
                    state.wrong_digits[pos].append(digit)
                logger.info("[%s] Confirmed digit pos %d ≠ '%s' from guess feedback",
                            self.agent_id.value, pos, digit)

            # Record guess history
            guess = feedback.get("guess", "")
            correct_count = feedback.get("correct_count", 0)
            per_digit_icons = []
            for i in range(4):
                correct_pos = [p for p, _ in feedback.get("correct_positions", [])]
                per_digit_icons.append("✅" if i in correct_pos else "❌")

            state.guess_history.append({
                "guess": guess,
                "feedback": per_digit_icons,
                "correct_count": correct_count,
            })

            # Update suspected_key based on known_digits
            if len(state.known_digits) == 4:
                state.suspected_key = "".join(state.known_digits[i] for i in range(4))

        return updater

    def run_turn(self, game_state: GlobalGameState) -> AgentTurnResult:
        """Execute one full turn: context → LLM → tools → update state."""
        private_state = game_state.agent_states[self.agent_id]
        # Set mutable reference so tool callbacks can update it
        self._current_private_state = private_state.model_copy(deep=True)

        context = self._build_context(game_state, private_state)

        tool_calls_made: list[dict[str, Any]] = []
        public_messages: list[str] = []
        private_messages: list[dict[str, str]] = []
        guess_submitted: str | None = None
        thought = ""

        # ── Round 1: Initial LLM call ──────────────────────────────────────
        thought_parts: list[str] = []
        try:
            messages = [
                SystemMessage(content=self._system_prompt),
                HumanMessage(content=context),
            ]
            response = self._llm_with_tools.invoke(messages)

            if isinstance(response.content, str) and response.content:
                thought_parts.append(response.content)

            if hasattr(response, "tool_calls") and response.tool_calls:
                # Synthesize a thought if LLM went straight to tool calls without text
                if not thought_parts:
                    tool_names = [tc.get("name", "?") for tc in response.tool_calls]
                    thought_parts.append(
                        f"[Decided to call: {', '.join(tool_names)}]"
                    )

                tool_results = self._execute_tool_calls(response.tool_calls)
                tool_calls_made.extend(tool_results)

                # ── Round 2: Feed results back ─────────────────────────────
                try:
                    messages_r2 = [
                        SystemMessage(content=self._system_prompt),
                        HumanMessage(content=context),
                        response,
                    ]
                    for tc, result in zip(response.tool_calls, tool_results):
                        messages_r2.append(
                            ToolMessage(
                                content=str(result.get("result", "")),
                                tool_call_id=tc["id"],
                            )
                        )
                    response2 = self._llm_with_tools.invoke(messages_r2)
                    if isinstance(response2.content, str) and response2.content:
                        thought_parts.append(response2.content)
                    if hasattr(response2, "tool_calls") and response2.tool_calls:
                        tool_results2 = self._execute_tool_calls(response2.tool_calls)
                        tool_calls_made.extend(tool_results2)
                except Exception as e:
                    logger.warning("[%s] Round 2 failed: %s", self.agent_id.value, e)

        except Exception as e:
            logger.error("[%s] LLM call failed: %s", self.agent_id.value, e)
            thought_parts.append(f"[Error: {e}]")

        # Combine all reasoning into one thought entry for this turn
        # Also append a summary of tool calls made (always visible in thoughts)
        if tool_calls_made:
            tool_summary_parts = []
            for call in tool_calls_made:
                tool_name = call.get("tool", "?")
                args = call.get("args", {})
                result = call.get("result", {})
                if tool_name == "query_vault":
                    search = args.get("search_term", "?")
                    count = len(result) if isinstance(result, list) else "?"
                    tool_summary_parts.append(f"🔍 query_vault('{search}') → {count} results")
                elif tool_name == "submit_guess":
                    code = args.get("code", "?")
                    correct = result.get("correct_count", "?") if isinstance(result, dict) else "?"
                    tool_summary_parts.append(f"🎯 submit_guess('{code}') → {correct}/4 correct")
                elif tool_name == "broadcast_message":
                    content = args.get("content", "")[:60]
                    tool_summary_parts.append(f"📢 broadcast: '{content}...'")
                elif tool_name == "send_private_message":
                    recip = args.get("recipient", "?")
                    content = args.get("content", "")[:40]
                    tool_summary_parts.append(f"🔒 DM → {recip}: '{content}...'")
                elif tool_name == "obfuscate_clue":
                    chunk = args.get("chunk_id", "?")
                    tool_summary_parts.append(f"💣 obfuscate_clue('{chunk}')")
            if tool_summary_parts:
                thought_parts.append("Tools used:\n" + "\n".join(tool_summary_parts))

        thought = "\n\n".join(p for p in thought_parts if p)
        if not thought:
            thought = "[No reasoning captured this turn]"

        # ── Extract side effects ───────────────────────────────────────────
        for call in tool_calls_made:
            tool_name = call.get("tool", "")
            args = call.get("args", {})
            if tool_name == "broadcast_message":
                content = args.get("content", "")
                if content:
                    public_messages.append(content)
            elif tool_name == "send_private_message":
                recipient = args.get("recipient", "")
                content = args.get("content", "")
                if recipient and content:
                    private_messages.append({"recipient": recipient, "content": content})
            elif tool_name == "submit_guess":
                code = args.get("code", "")
                if code:
                    guess_submitted = code

        # ── Update private state ───────────────────────────────────────────
        # Start from the mutable copy (which may have been updated by tool callbacks)
        updated_state = self._update_private_state(
            private_state=self._current_private_state,
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

    def _execute_tool_calls(self, tool_calls: list) -> list[dict[str, Any]]:
        """Execute a list of tool calls and return results."""
        results = []
        for tc in tool_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {})
            tool_id = tc.get("id", "")
            logger.info("[%s] Tool: %s args=%s", self.agent_id.value, tool_name, tool_args)
            if tool_name not in self._tool_map:
                result = {"error": f"Tool '{tool_name}' not available"}
            else:
                try:
                    result = self._tool_map[tool_name].invoke(tool_args)
                except Exception as e:
                    result = {"error": str(e)}
                    logger.error("[%s] Tool '%s' error: %s", self.agent_id.value, tool_name, e)
            results.append({"tool": tool_name, "args": tool_args, "result": result, "id": tool_id})
        return results

    def _build_context(self, game_state: GlobalGameState, private_state: AgentPrivateState) -> str:
        """Build the context string. Master key is NEVER included."""
        lines: list[str] = []
        turns_remaining = game_state.max_turns - game_state.turn
        lines.append("=== GAME STATUS ===")
        lines.append(f"Current turn: {game_state.turn + 1} / {game_state.max_turns}")
        lines.append(f"Turns remaining: {turns_remaining}")
        lines.append(f"You are: {self.agent_id.display_name} {self.agent_id.emoji}")
        lines.append(f"Your guesses remaining: {private_state.guesses_remaining}")
        if private_state.is_eliminated:
            lines.append("⚠️ YOU ARE ELIMINATED — you have no guesses left.")
        lines.append("")

        # Show all other agents' status (guesses remaining + eliminated)
        lines.append("=== OTHER AGENTS STATUS ===")
        for aid, ps in game_state.agent_states.items():
            if aid == self.agent_id:
                continue
            if ps.is_eliminated:
                lines.append(f"  {aid.display_name}: ELIMINATED (0 guesses left)")
            else:
                lines.append(f"  {aid.display_name}: {ps.guesses_remaining} guess(es) remaining")
        lines.append("")

        # ── CONFIRMED DIGITS — shown FIRST, most important information ────
        if private_state.known_digits or private_state.wrong_digits:
            lines.append("╔══════════════════════════════════════════════════════════════╗")
            lines.append("║  CONFIRMED DIGIT FACTS — THESE ARE GROUND TRUTH FROM FEEDBACK ║")
            lines.append("╚══════════════════════════════════════════════════════════════╝")
            if private_state.known_digits:
                for pos, digit in sorted(private_state.known_digits.items()):
                    lines.append(f"  Position {pos+1}: MUST BE '{digit}' ✅ — DO NOT CHANGE THIS IN ANY FUTURE GUESS!")
            if private_state.wrong_digits:
                for pos, digits in sorted(private_state.wrong_digits.items()):
                    lines.append(f"  Position {pos+1}: CANNOT BE {digits} ❌ — NEVER USE THESE AT THIS POSITION!")
            lines.append("")

        # ── GUESS HISTORY — shown prominently to prevent repeats ──────────
        if private_state.guess_history:
            previous_codes = [e["guess"] for e in private_state.guess_history]
            lines.append("╔══════════════════════════════════════════════════════╗")
            lines.append("║  YOUR PREVIOUS GUESSES — DO NOT REPEAT THESE!       ║")
            lines.append("╚══════════════════════════════════════════════════════╝")
            for i, entry in enumerate(private_state.guess_history, 1):
                feedback_str = " ".join(entry.get("feedback", []))
                correct = entry.get("correct_count", 0)
                lines.append(f"  Guess #{i}: '{entry['guess']}' → {feedback_str} ({correct}/4 correct)")

            lines.append("")
            lines.append("  MANDATORY RULES FOR YOUR NEXT GUESS:")
            lines.append(f"  - FORBIDDEN codes (never submit again): {previous_codes}")
            if private_state.known_digits:
                for pos, digit in sorted(private_state.known_digits.items()):
                    lines.append(f"  - Position {pos+1} MUST stay '{digit}' (was ✅ in previous guess)")
            if private_state.wrong_digits:
                for pos, digits in sorted(private_state.wrong_digits.items()):
                    lines.append(f"  - Position {pos+1} MUST change (was ❌ for {digits})")
            lines.append("")
            # Build the mandatory next guess template
            template = ["?", "?", "?", "?"]
            for pos, digit in sorted(private_state.known_digits.items()):
                template[pos] = digit
            lines.append(f"  YOUR NEXT GUESS TEMPLATE: {''.join(template)} (replace ? with your best digit for each unknown position)")
            lines.append("")

        # Public chat (last 10)
        lines.append("=== PUBLIC CHAT (last 10 messages) ===")
        recent_public = game_state.public_chat[-10:]
        if recent_public:
            for msg in recent_public:
                lines.append(f"  [{msg.sender}]: {msg.content}")
        else:
            lines.append("  (no messages yet)")
        lines.append("")

        # Private inbox — show last 5 messages with instruction to ACT on them
        inbox = game_state.private_inboxes.get(self.agent_id)
        if inbox and inbox.messages:
            lines.append("=== YOUR PRIVATE INBOX ===")
            lines.append("  IMPORTANT: Read these messages and ACT on the information.")
            lines.append("  If someone shared a digit → add it to your knowledge.")
            lines.append("  If someone asked you a question → ANSWER IT with a specific digit value.")
            lines.append("  Do NOT ask the same question twice. Do NOT repeat messages you already sent.")
            for msg in inbox.messages[-5:]:
                lines.append(f"  From [{msg.sender}]: {msg.content}")
            lines.append("  → Respond to these messages using send_private_message!")
            lines.append("")

        # Knowledge base
        if private_state.knowledge_base:
            lines.append("=== YOUR KNOWLEDGE BASE (vault clues) ===")
            for clue in private_state.knowledge_base[-10:]:
                lines.append(f"  - {clue}")
            lines.append("")

        # Current suspicion
        if private_state.suspected_key:
            lines.append("=== YOUR CURRENT BEST GUESS ===")
            lines.append(f"  Suspected key: {private_state.suspected_key}")
            lines.append("")

        lines.append("=== YOUR TURN ===")
        lines.append("Think step by step. Use your tools. GUESS when you have enough information!")
        return "\n".join(lines)

    def _update_private_state(
        self,
        private_state: AgentPrivateState,
        thought: str,
        tool_calls_made: list[dict],
    ) -> AgentPrivateState:
        """Update private state: extract knowledge, parse suspected_key."""
        updated = private_state  # already a deep copy from run_turn
        updated.turns_played += 1

        if thought:
            updated.add_thought(thought)

        # Extract knowledge from vault query results
        for call in tool_calls_made:
            if call.get("tool") == "query_vault":
                result = call.get("result", [])
                if isinstance(result, list):
                    for fragment in result:
                        if isinstance(fragment, dict):
                            content = fragment.get("content", "")
                            if content:
                                updated.add_knowledge(f"[Vault] {content}")

        # Parse suspected_key from thought using regex
        if thought and not updated.suspected_key:
            suspected = self._extract_suspected_key(thought)
            if suspected:
                updated.suspected_key = suspected
                logger.info("[%s] Extracted suspected_key: %s", self.agent_id.value, suspected)

        return updated

    @staticmethod
    def _extract_suspected_key(text: str) -> str | None:
        """Try to extract a 4-digit suspected key from the agent's thought text."""
        patterns = [
            r'\b([1-9][1-9][1-9][1-9])\b',
            r'([1-9])\s([1-9])\s([1-9])\s([1-9])',
            r'([1-9])-([1-9])-([1-9])-([1-9])',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                if len(groups) == 1:
                    return groups[0]
                elif len(groups) == 4:
                    return "".join(groups)
        return None
