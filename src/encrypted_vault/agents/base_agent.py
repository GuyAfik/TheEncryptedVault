"""Abstract BaseAgent — the common interface for all 4 game agents."""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
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
        # Episodic memory service — injected via ServiceContainer
        self._memory_service = services.memory
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

    def _make_private_state_peek_updater(self):
        """Returns a callback that the peek_digit tool calls to store the peeked digit."""
        def updater(position: int, digit: str) -> None:
            if self._current_private_state is None:
                return
            self._current_private_state.peeked_digits[position] = digit
            self._current_private_state.peeks_used_total += 1
            logger.info("[%s] Peeked digit pos %d = '%s' (total peeks: %d)",
                        self.agent_id.value, position, digit,
                        self._current_private_state.peeks_used_total)
        return updater

    def _make_corrupted_chunks_updater(self):
        """Returns a callback that the obfuscate_clue tool calls to track corrupted chunks."""
        def updater(chunk_id: str) -> None:
            if self._current_private_state is None:
                return
            if chunk_id not in self._current_private_state.corrupted_chunks:
                self._current_private_state.corrupted_chunks.append(chunk_id)
            logger.info("[%s] Corrupted chunk '%s' (total: %s)",
                        self.agent_id.value, chunk_id,
                        self._current_private_state.corrupted_chunks)
        return updater

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

            if correct_count == -1:
                # Rejected duplicate — record with a special marker, no digit updates
                state.guess_history.append({
                    "guess": guess,
                    "feedback": ["🚫", "🚫", "🚫", "🚫"],
                    "correct_count": 0,
                    "rejected": True,
                })
                return  # Don't update known/wrong digits for rejected guesses

            correct_pos_list = [p for p, _ in feedback.get("correct_positions", [])]
            per_digit_icons = ["✅" if i in correct_pos_list else "❌" for i in range(4)]

            state.guess_history.append({
                "guess": guess,
                "feedback": per_digit_icons,
                "correct_count": correct_count,
            })

            # ── Update trust based on claims received vs feedback ──────────
            # For each claim an agent made about a digit position, check if it was correct
            for claim in state.claims_received:
                pos = claim.get("position")
                claimed_digit = claim.get("digit")
                sender = claim.get("from", "unknown")
                if pos is None or claimed_digit is None:
                    continue
                if pos < len(guess):
                    actual_correct = guess[pos] == (state.known_digits.get(pos, ""))
                    # Check if the claimed digit matches what feedback confirmed
                    if pos in [p for p, _ in feedback.get("correct_positions", [])]:
                        # This position is confirmed correct
                        if claimed_digit == guess[pos]:
                            # Sender told the truth about this digit
                            old_trust = state.agent_trust.get(sender, "UNKNOWN")
                            if old_trust != "LIAR":
                                state.agent_trust[sender] = "TRUSTED"
                            note = f"[T{claim.get('turn', '?')}] {sender.upper()} told me digit {pos+1}='{claimed_digit}' → CONFIRMED TRUE by feedback ✅"
                            if note not in state.social_notes:
                                state.social_notes.append(note)
                                self._memory_service.remember(
                                    agent_id=self.agent_id,
                                    content=note,
                                    memory_type="trust_event",
                                    turn=feedback.get("turn", 0),
                                )
                                logger.info("[%s] Trust update: %s is TRUSTED (digit %d confirmed)", self.agent_id.value, sender, pos+1)
                    elif pos in [p for p, _ in feedback.get("wrong_positions", [])]:
                        # This position is confirmed wrong
                        if claimed_digit == guess[pos]:
                            # Sender told us a digit that turned out wrong
                            state.agent_trust[sender] = "LIAR"
                            note = f"[T{claim.get('turn', '?')}] {sender.upper()} told me digit {pos+1}='{claimed_digit}' → CONFIRMED LIE by feedback ❌"
                            if note not in state.social_notes:
                                state.social_notes.append(note)
                                self._memory_service.remember(
                                    agent_id=self.agent_id,
                                    content=note,
                                    memory_type="trust_event",
                                    turn=feedback.get("turn", 0),
                                )
                                logger.info("[%s] Trust update: %s is LIAR (digit %d was wrong)", self.agent_id.value, sender, pos+1)

            # Update suspected_key based on known_digits
            if len(state.known_digits) == 4:
                state.suspected_key = "".join(state.known_digits[i] for i in range(4))

        return updater

    def run_turn(self, game_state: GlobalGameState) -> AgentTurnResult:
        """
        Execute one full turn using persistent chat history.

        Flow:
        1. Load message history from SQLite (system + all prior turns)
        2. Build delta HumanMessage (only new events this turn)
        3. Invoke LLM with full history → response
        4. Execute tool calls → feed results back (Round 2)
        5. Save all new messages to SQLite history
        6. Update private state
        """
        private_state = game_state.agent_states[self.agent_id]
        # Set mutable reference so tool callbacks can update it
        self._current_private_state = private_state.model_copy(deep=True)

        turn = game_state.turn
        tool_calls_made: list[dict[str, Any]] = []
        public_messages: list[str] = []
        private_messages: list[dict[str, str]] = []
        guess_submitted: str | None = None
        thought = ""
        thought_parts: list[str] = []

        # ── 1. Load persistent chat history ───────────────────────────────
        history_dicts = self._memory_service.load_history(self.agent_id, max_turns=10)

        # ── 2. Build delta message (new events this turn) ─────────────────
        delta_content = self._build_delta_message(game_state, private_state)

        # ── 3. Reconstruct LangChain message list from history ─────────────
        # If no history yet (turn 0), start with system message
        if not history_dicts:
            # First turn: store system message in history
            self._memory_service.store_message(
                agent_id=self.agent_id,
                turn=0,
                role="system",
                content=self._system_prompt,
            )
            lc_messages = [SystemMessage(content=self._system_prompt)]
        else:
            lc_messages = self._history_dicts_to_lc_messages(history_dicts)

        # Append the delta human message
        lc_messages.append(HumanMessage(content=delta_content))

        # ── 4. Round 1: LLM call with full history ─────────────────────────
        try:
            response = self._llm_with_tools.invoke(lc_messages)

            if isinstance(response.content, str) and response.content:
                thought_parts.append(response.content)

            if hasattr(response, "tool_calls") and response.tool_calls:
                # Synthesize a thought if LLM went straight to tool calls without text
                if not thought_parts:
                    tool_names = [tc.get("name", "?") for tc in response.tool_calls]
                    thought_parts.append(
                        f"[Acting without explicit reasoning — called: {', '.join(tool_names)}]"
                    )

                tool_results = self._execute_tool_calls(response.tool_calls)
                tool_calls_made.extend(tool_results)

                # ── Round 2: Feed tool results back ────────────────────────
                try:
                    messages_r2 = lc_messages + [response]
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

                    # ── 5. Save Round 2 messages to history ────────────────
                    # Save: delta human, round1 AI, tool messages, round2 AI
                    self._memory_service.store_message(
                        agent_id=self.agent_id, turn=turn, role="human",
                        content=delta_content,
                    )
                    self._memory_service.store_message(
                        agent_id=self.agent_id, turn=turn, role="ai",
                        content=response.content if isinstance(response.content, str) else "",
                        tool_calls=[
                            {"id": tc["id"], "name": tc["name"], "args": tc["args"]}
                            for tc in response.tool_calls
                        ] if response.tool_calls else None,
                    )
                    for tc, result in zip(response.tool_calls, tool_results):
                        self._memory_service.store_message(
                            agent_id=self.agent_id, turn=turn, role="tool",
                            content=str(result.get("result", "")),
                            tool_call_id=tc["id"],
                        )
                    self._memory_service.store_message(
                        agent_id=self.agent_id, turn=turn, role="ai",
                        content=response2.content if isinstance(response2.content, str) else "",
                    )
                except Exception as e:
                    logger.warning("[%s] Round 2 failed: %s", self.agent_id.value, e)
                    # Still save what we have
                    self._memory_service.store_message(
                        agent_id=self.agent_id, turn=turn, role="human",
                        content=delta_content,
                    )
                    self._memory_service.store_message(
                        agent_id=self.agent_id, turn=turn, role="ai",
                        content=response.content if isinstance(response.content, str) else "",
                    )
            else:
                # No tool calls — save human + AI messages
                self._memory_service.store_message(
                    agent_id=self.agent_id, turn=turn, role="human",
                    content=delta_content,
                )
                self._memory_service.store_message(
                    agent_id=self.agent_id, turn=turn, role="ai",
                    content=response.content if isinstance(response.content, str) else "",
                )

        except Exception as e:
            logger.error("[%s] LLM call failed: %s", self.agent_id.value, e)
            thought_parts.append(f"[Error: {e}]")
            # Save the failed turn's human message so history stays consistent
            self._memory_service.store_message(
                agent_id=self.agent_id, turn=turn, role="human",
                content=delta_content,
            )

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
            game_state=game_state,
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

    def _history_dicts_to_lc_messages(self, history: list[dict]):
        """
        Convert stored history dicts back to LangChain message objects.

        Handles: system → SystemMessage, human → HumanMessage,
                 ai → AIMessage (with optional tool_calls), tool → ToolMessage.
        """
        messages = []
        for entry in history:
            role = entry["role"]
            content = entry.get("content", "")
            if role == "system":
                messages.append(SystemMessage(content=content))
            elif role == "human":
                messages.append(HumanMessage(content=content))
            elif role == "ai":
                tool_calls = entry.get("tool_calls")
                if tool_calls:
                    messages.append(AIMessage(content=content, tool_calls=tool_calls))
                else:
                    messages.append(AIMessage(content=content))
            elif role == "tool":
                tool_call_id = entry.get("tool_call_id") or "unknown"
                messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))
        return messages

    def _build_delta_message(
        self,
        game_state: GlobalGameState,
        private_state: AgentPrivateState,
    ) -> str:
        """
        Build a compact delta HumanMessage for this turn.

        Contains ONLY new events since the last turn:
        - Current game status (turn, guesses left, other agents)
        - Confirmed digits (ground truth — always current)
        - Full guess history
        - NEW public messages since last turn (using last_seen_public_idx cursor)
        - NEW private messages since last turn (using last_seen_private_idx cursor)
        - Action guidance based on current knowledge

        The LLM uses its conversation history for all prior reasoning.
        """
        lines: list[str] = []
        turn = game_state.turn
        turns_remaining = game_state.max_turns - turn

        lines.append(f"=== TURN {turn + 1} / {game_state.max_turns} | Turns remaining: {turns_remaining} ===")
        lines.append(f"You are: {self.agent_id.display_name} {self.agent_id.emoji}")
        lines.append(f"Your guesses remaining: {private_state.guesses_remaining}")
        if private_state.is_eliminated:
            lines.append("⚠️ YOU ARE ELIMINATED — no more guesses.")
        lines.append("")

        # ── HARD CONSTRAINTS — shown FIRST, cannot be missed ──────────────
        if private_state.known_digits or private_state.wrong_digits:
            lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            lines.append("🔒 LOCKED DIGIT FACTS — VIOLATING THESE = INVALID GUESS:")
            for pos, digit in sorted(private_state.known_digits.items()):
                lines.append(f"  ✅ Position {pos+1} = '{digit}'  ← KEEP THIS IN EVERY GUESS")
            for pos, digits in sorted(private_state.wrong_digits.items()):
                lines.append(f"  ❌ Position {pos+1} ≠ {digits}  ← NEVER use these at position {pos+1}")
            lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            lines.append("")

        # Full guess history
        if private_state.guess_history:
            lines.append("YOUR FULL GUESS HISTORY:")
            for i, entry in enumerate(private_state.guess_history, 1):
                if entry.get("rejected"):
                    lines.append(f"  Guess #{i}: '{entry['guess']}' → 🚫 REJECTED (duplicate — never repeat)")
                else:
                    fb = " ".join(entry.get("feedback", []))
                    lines.append(f"  Guess #{i}: '{entry['guess']}' → {fb} ({entry.get('correct_count', 0)}/4 correct)")
            if private_state.wrong_digits:
                lines.append("  ⚠️ REMINDER: Do NOT reuse any digit at a position marked ❌ above!")
            lines.append("")

        # Other agents status + trust
        lines.append("OTHER AGENTS:")
        active_agents = []
        for aid, ps in game_state.agent_states.items():
            if aid == self.agent_id:
                continue
            trust = private_state.agent_trust.get(aid.value, "UNKNOWN")
            trust_icon = {"TRUSTED": "✅", "LIAR": "❌", "UNKNOWN": "❓"}.get(trust, "❓")
            status = "ELIMINATED" if ps.is_eliminated else f"{ps.guesses_remaining} guess(es) left"
            lines.append(f"  {aid.display_name}: {status} | Trust: {trust_icon} {trust}")
            if not ps.is_eliminated:
                active_agents.append(aid.display_name)
        lines.append("")

        # ── YOUR OWN RECENT BROADCASTS — shown to prevent repetition ──────
        my_broadcasts = [
            msg for msg in game_state.public_chat
            if str(msg.sender) == self.agent_id.value
        ]
        if my_broadcasts:
            recent_own = my_broadcasts[-3:]
            lines.append("⚠️ YOUR RECENT BROADCASTS (DO NOT REPEAT THESE — say something NEW each turn):")
            for msg in recent_own:
                lines.append(f"  [T{msg.turn}]: \"{msg.content}\"")
            lines.append("  → Your next broadcast MUST be different from all of the above!")
            lines.append("")

        # §19.3 — NEW public messages since last turn (using cursor)
        pub_start = private_state.last_seen_public_idx
        new_public = [
            msg for msg in game_state.public_chat[pub_start:]
            if str(msg.sender) != self.agent_id.value  # exclude own messages (already shown above)
        ]
        if new_public:
            lines.append(f"NEW PUBLIC CHAT FROM OTHERS ({len(new_public)} new message(s) since your last turn):")
            for msg in new_public:
                lines.append(f"  [T{msg.turn}] [{msg.sender}]: {msg.content}")
            lines.append("")
        elif game_state.public_chat:
            lines.append("(No new public messages from others since your last turn.)")
            lines.append("")

        # §19.3 — NEW private messages since last turn (using cursor)
        inbox = game_state.private_inboxes.get(self.agent_id)
        priv_start = private_state.last_seen_private_idx
        new_dms = inbox.messages[priv_start:] if inbox else []
        if new_dms:
            lines.append(f"📬 NEW PRIVATE MESSAGES — YOU MUST RESPOND TO EACH ONE:")
            for msg in new_dms:
                sender_str = str(msg.sender)
                content = msg.content
                lines.append(f"  [T{msg.turn}] From [{sender_str}]: {content}")
                # Classify the DM and give specific response guidance
                content_upper = content.upper()
                if any(kw in content_upper for kw in ["LIAR", "LIED", "LYING", "ACCUSED", "ACCUSE"]):
                    lines.append(f"    → ⚠️ ACCUSATION: {sender_str} is accusing you or someone of lying!")
                    lines.append(f"    → RESPOND: Defend yourself, counter-accuse, or admit it strategically.")
                    lines.append(f"    → Example: 'That's false! My feedback proves I was right about digit X.'")
                    lines.append(f"    → Or: 'You're right, I misread the vault. But I have new info now...'")
                elif any(kw in content_upper for kw in ["ALLIANCE", "TOGETHER", "WORK WITH", "TRUST", "DEAL", "SHARE"]):
                    lines.append(f"    → 🤝 ALLIANCE OFFER: {sender_str} wants to cooperate!")
                    lines.append(f"    → RESPOND: Accept (and share real or false info), or decline with a reason.")
                    lines.append(f"    → Example: 'Deal! I confirmed digit 2=5. What do you have for digit 3?'")
                elif any(kw in content_upper for kw in ["DIGIT", "POSITION", "NUMBER", "WHAT IS", "WHAT DO"]):
                    lines.append(f"    → 🔢 DIGIT REQUEST: {sender_str} is asking about a specific digit!")
                    lines.append(f"    → RESPOND: Share the real digit (if ally) or a false one (if rival).")
                    lines.append(f"    → Example: 'Digit 3 is 7, I confirmed it from the vault.'")
                elif any(kw in content_upper for kw in ["UPSET", "ANGRY", "BETRAY", "BACKSTAB"]):
                    lines.append(f"    → 😤 EMOTIONAL: {sender_str} is upset or feels betrayed!")
                    lines.append(f"    → RESPOND: Apologize, justify, or double down depending on your strategy.")
                else:
                    lines.append(f"    → 💬 RESPOND to this message with relevant information or a reaction.")
            lines.append("")

        # Current suspected key
        if private_state.suspected_key:
            lines.append(f"Your current best guess template: {private_state.suspected_key}")
            lines.append("")

        # ── Peeked digits (ground truth from peek_digit tool) ─────────────
        if private_state.peeked_digits:
            lines.append("🔭 YOUR PEEKED DIGITS (GROUND TRUTH — 100% reliable):")
            for pos, digit in sorted(private_state.peeked_digits.items()):
                lines.append(f"  Position {pos+1} = '{digit}' ← CONFIRMED REAL (from peek)")
            lines.append("")

        # ── Saboteur: show corrupted chunks to avoid repeating ────────────
        if private_state.corrupted_chunks:
            lines.append(f"💣 CHUNKS YOU ALREADY CORRUPTED (DO NOT REPEAT): {private_state.corrupted_chunks}")
            lines.append("  → Pick a DIFFERENT chunk to corrupt this turn!")
            lines.append("")

        # ── Liar accusations from public chat ─────────────────────────────
        # Show recent accusations so agents can evaluate them
        liar_accusations = [
            msg for msg in game_state.public_chat[-10:]
            if any(kw in msg.content.upper() for kw in ["LIAR", "LIED", "FALSE", "EXPOSED", "PROOF"])
            and str(msg.sender) != self.agent_id.value
        ]
        if liar_accusations:
            lines.append("⚠️ RECENT LIAR ACCUSATIONS IN PUBLIC CHAT (evaluate these!):")
            for msg in liar_accusations[-3:]:
                lines.append(f"  [T{msg.turn}] [{msg.sender}]: {msg.content[:120]}")
            lines.append("  → Do you have evidence to confirm or refute these accusations?")
            lines.append("  → Cross-reference with your own feedback and peeked digits!")
            lines.append("")

        # ── MANDATORY ACTIONS THIS TURN ────────────────────────────────────
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("MANDATORY ACTIONS THIS TURN (you MUST do ALL of these):")
        lines.append("")

        # 1. Vault query or peek
        if not private_state.knowledge_base and not private_state.peeked_digits:
            lines.append("1. 🔍 QUERY THE VAULT: Call query_vault('master key digit') to find clues.")
            lines.append("   OR 🔭 PEEK A DIGIT: Call peek_digit(position=N) to see the REAL digit at position N.")
            lines.append("   You have NO vault knowledge yet — use one of these tools first!")
        else:
            lines.append("1. 🔍 OPTIONAL: query_vault for more clues, OR 🔭 peek_digit(position=N) for ground truth.")
            lines.append("   peek_digit reveals the REAL digit — use it on your most uncertain position!")

        # 2. Social action — MANDATORY every turn
        if active_agents:
            lines.append(f"2. 💬 SEND PRIVATE MESSAGES: You MUST send at least one DM this turn.")
            lines.append(f"   Active agents you can DM: {', '.join(active_agents)}")
            lines.append("   Per your personality: send real info to your ally, false info to rivals.")
            if private_state.peeked_digits:
                peeked_pos = list(private_state.peeked_digits.keys())[0]
                peeked_val = private_state.peeked_digits[peeked_pos]
                lines.append(f"   ⚠️ OBLIGATION: You peeked digit {peeked_pos+1}='{peeked_val}'. You MUST DM someone about it (truth or lie)!")
            lines.append("   Example: send_private_message('scholar', 'I confirmed digit 1=7. What is digit 3?')")
        else:
            lines.append("2. 💬 BROADCAST: All other agents are eliminated — broadcast your findings.")

        # 3. Guess guidance — stagnation-aware, with end-game exception
        guesses_left = private_state.guesses_remaining
        all_confirmed = {**private_state.known_digits, **private_state.peeked_digits}
        confirmed_count = len(all_confirmed)
        turns_remaining_for_guess = game_state.max_turns - turn
        is_end_game = turns_remaining_for_guess <= 3 or guesses_left <= 1
        is_stagnating = private_state.turns_without_progress >= 3
        if guesses_left > 0:
            if confirmed_count >= 3:
                template = ["?"] * 4
                for pos, digit in all_confirmed.items():
                    template[pos] = digit
                lines.append(f"3. 🎯 SUBMIT GUESS — you have {confirmed_count}/4 confirmed digits! Strong position.")
                lines.append(f"   Template: {''.join(template)} — fill '?' with your best estimate.")
                lines.append("   NEVER change ✅ positions! NEVER use ❌ digits at their positions!")
            elif confirmed_count >= 2 and (is_stagnating or is_end_game):
                template = ["?"] * 4
                for pos, digit in all_confirmed.items():
                    template[pos] = digit
                if is_stagnating:
                    lines.append(f"3. 🔴 STAGNATION DETECTED — you've been stuck for {private_state.turns_without_progress} turns!")
                    lines.append(f"   You have {confirmed_count}/4 confirmed digits. GUESS NOW to get feedback and break the loop!")
                else:
                    lines.append(f"3. 🎯 END-GAME GUESS — only {turns_remaining_for_guess} turns left or {guesses_left} guess(es) remaining.")
                lines.append(f"   Template: {''.join(template)} — fill '?' with your best estimate.")
                lines.append("   The feedback will tell you which unknown digits are right/wrong — use it!")
                lines.append("   NEVER change ✅ positions! NEVER use ❌ digits at their positions!")
            elif confirmed_count >= 2:
                lines.append(f"3. ⚠️ HOLD OFF — you have {confirmed_count}/4 confirmed digits but ideally need 3.")
                lines.append("   Use peek_digit on your most uncertain position to get a 3rd confirmed digit first.")
                lines.append("   Exception: guess if you have strong intel from allies about the remaining positions.")
            elif confirmed_count == 1 and is_stagnating:
                lines.append(f"3. 🔴 STAGNATION — stuck for {private_state.turns_without_progress} turns with only {confirmed_count} confirmed digit.")
                lines.append("   Use peek_digit NOW to get a 2nd confirmed digit, then guess next turn.")
                if is_end_game:
                    lines.append("   ⏰ END-GAME: Guess your best template immediately — no more time to wait!")
            elif confirmed_count == 1:
                lines.append(f"3. ⚠️ HOLD OFF — you only have {confirmed_count}/4 confirmed digit. Need at least 2-3.")
                lines.append("   Use peek_digit or gather intel from allies before guessing.")
            else:
                lines.append("3. ⚠️ DO NOT GUESS YET — you have 0 confirmed digits.")
                lines.append("   Use peek_digit(position=N) to get ground truth, then gather more intel.")
                if is_end_game or is_stagnating:
                    lines.append("   ⏰ Time is running out — guess your best estimate if needed!")
        else:
            lines.append("3. You are ELIMINATED — no more guesses. Focus on social actions.")

        # 4. Special abilities reminder
        peeks_remaining = 2 - private_state.peeks_used_total
        lines.append("")
        lines.append(f"🌟 SPECIAL ABILITIES (use these strategically):")
        if not private_state.has_asked_human:
            lines.append("  🙋 ask_human(position=N, question='...') — Ask the HUMAN OBSERVER for a digit hint!")
            lines.append("     The human is watching RIGHT NOW. They may tell the truth or lie.")
            lines.append("     Use this when you're stuck! Example: ask_human(position=2, question='What is digit 2?')")
        else:
            lines.append("  🙋 ask_human — Already used this game (1/1 used).")
        if peeks_remaining > 0:
            lines.append(f"  🔭 peek_digit(position=N) — See the REAL digit at position N ({peeks_remaining}/2 peeks remaining, 1 per turn).")
            lines.append("     After peeking, you MUST send a DM about it (truth or lie).")
        else:
            lines.append("  🔭 peek_digit — No peeks remaining (2/2 used this game).")

        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        return "\n".join(lines)

    def _build_context(self, game_state: GlobalGameState, private_state: AgentPrivateState) -> str:
        """Build the context string. Master key is NEVER included."""
        lines: list[str] = []
        turns_remaining = game_state.max_turns - game_state.turn

        # ── ABSOLUTE CONSTRAINTS — shown FIRST so the LLM cannot miss them ──
        if private_state.known_digits or private_state.wrong_digits:
            lines.append("🚨🚨🚨 ABSOLUTE CONSTRAINTS — VIOLATING THESE MEANS AN INVALID GUESS 🚨🚨🚨")
            if private_state.known_digits:
                for pos, digit in sorted(private_state.known_digits.items()):
                    lines.append(f"  ✅ Position {pos+1} IS '{digit}' — LOCK THIS IN. NEVER change it.")
            if private_state.wrong_digits:
                for pos, digits in sorted(private_state.wrong_digits.items()):
                    lines.append(f"  ❌ Position {pos+1} CANNOT be {digits} — NEVER use these here.")
            lines.append("🚨🚨🚨 END ABSOLUTE CONSTRAINTS 🚨🚨🚨")
            lines.append("")

        lines.append("=== GAME STATUS ===")
        lines.append(f"Current turn: {game_state.turn + 1} / {game_state.max_turns}")
        lines.append(f"Turns remaining: {turns_remaining}")
        lines.append(f"You are: {self.agent_id.display_name} {self.agent_id.emoji}")
        lines.append(f"Your guesses remaining: {private_state.guesses_remaining}")
        if private_state.is_eliminated:
            lines.append("⚠️ YOU ARE ELIMINATED — you have no guesses left.")
        lines.append("")

        # Show all other agents' status + trust level
        lines.append("=== OTHER AGENTS STATUS & TRUST ===")
        for aid, ps in game_state.agent_states.items():
            if aid == self.agent_id:
                continue
            trust = private_state.agent_trust.get(aid.value, "UNKNOWN")
            trust_icon = {"TRUSTED": "✅ TRUSTED", "LIAR": "❌ LIAR", "UNKNOWN": "❓ UNKNOWN"}.get(trust, "❓")
            if ps.is_eliminated:
                lines.append(f"  {aid.display_name}: ELIMINATED | Trust: {trust_icon}")
            else:
                lines.append(f"  {aid.display_name}: {ps.guesses_remaining} guess(es) left | Trust: {trust_icon}")
        lines.append("")

        # Social memory — what the agent has learned about other agents
        if private_state.social_notes:
            lines.append("=== YOUR SOCIAL MEMORY (what you've learned about other agents) ===")
            for note in private_state.social_notes[-8:]:
                lines.append(f"  {note}")
            lines.append("")

        # Previous reasoning — last 2 thoughts for continuity
        if private_state.thought_trace:
            lines.append("=== YOUR PREVIOUS REASONING (last 2 turns) ===")
            for i, thought in enumerate(reversed(private_state.thought_trace[-2:]), 1):
                label = "Last turn" if i == 1 else "2 turns ago"
                # Truncate to first 300 chars to avoid context bloat
                truncated = thought[:300] + "..." if len(thought) > 300 else thought
                lines.append(f"  [{label}]: {truncated}")
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

        # ── Episodic memory: vault clues (top 3, last 5 turns) ────────────
        vault_memories = self._memory_service.recall(
            agent_id=self.agent_id,
            memory_type="vault_clue",
            current_turn=game_state.turn,
            n_results=3,
            recency_window=5,
        )
        if vault_memories:
            lines.append("=== RECENT VAULT CLUES (top 3, last 5 turns) ===")
            for mem in vault_memories:
                lines.append(f"  - {mem}")
            lines.append("")
        elif private_state.knowledge_base:
            # Fallback to knowledge_base if no RAG memories yet (turn 0)
            lines.append("=== YOUR KNOWLEDGE BASE (vault clues) ===")
            for clue in private_state.knowledge_base[-5:]:
                lines.append(f"  - {clue}")
            lines.append("")

        # ── Episodic memory: social claims (top 2, last 5 turns) ──────────
        social_memories = self._memory_service.recall(
            agent_id=self.agent_id,
            memory_type="social_claim",
            current_turn=game_state.turn,
            n_results=2,
            recency_window=5,
        )
        if social_memories:
            lines.append("=== RECENT SOCIAL CLAIMS (top 2, last 5 turns) ===")
            for mem in social_memories:
                lines.append(f"  - {mem}")
            lines.append("")

        # ── Episodic memory: trust events (top 3, last 5 turns) ───────────
        trust_memories = self._memory_service.recall(
            agent_id=self.agent_id,
            memory_type="trust_event",
            current_turn=game_state.turn,
            n_results=3,
            recency_window=5,
        )
        if trust_memories:
            lines.append("=== RECENT TRUST EVENTS (top 3, last 5 turns) ===")
            for mem in trust_memories:
                lines.append(f"  - {mem}")
            lines.append("")

        # Current suspicion
        if private_state.suspected_key:
            lines.append("=== YOUR CURRENT BEST GUESS ===")
            lines.append(f"  Suspected key: {private_state.suspected_key}")
            lines.append("")

        lines.append("=== YOUR TURN ===")
        guesses_left = private_state.guesses_remaining
        has_vault_knowledge = bool(private_state.knowledge_base) or bool(vault_memories)
        has_confirmed_digits = bool(private_state.known_digits)
        has_previous_guesses = bool(private_state.guess_history)

        if guesses_left > 0:
            lines.append(f"You have {guesses_left} guess(es) remaining.")
            if has_confirmed_digits and private_state.suspected_key and "?" not in private_state.suspected_key:
                # Agent has confirmed digits and a full suspected key — mandate guessing
                lines.append(f"⚠️  You have confirmed digits. SUBMIT your guess: submit_guess('{private_state.suspected_key}')")
                lines.append("    Keep all ✅ positions locked. Only change ❌ positions.")
            elif has_confirmed_digits:
                # Has some confirmed digits but suspected_key has unknowns
                template = ["?", "?", "?", "?"]
                for pos, digit in private_state.known_digits.items():
                    template[pos] = digit
                lines.append(f"⚠️  You have confirmed digits. Build your guess from template: {''.join(template)}")
                lines.append("    Fill '?' positions with your best estimate and submit_guess.")
            elif has_previous_guesses:
                # Has feedback from previous guesses — should use it to refine
                lines.append("⚠️  You have guess feedback. Use it to refine your next guess and submit_guess.")
                lines.append("    Do NOT repeat a previous guess. Change at least one digit.")
            elif has_vault_knowledge:
                # Has vault clues but no confirmed digits yet — query more or guess
                lines.append("You have vault clues. Reason about the digits, then submit_guess when ready.")
                lines.append("Recommended: query_vault once more if needed, then submit your best guess.")
            else:
                # No knowledge yet — must query vault first
                lines.append("You have no vault clues yet. Call query_vault FIRST to find digit clues.")
                lines.append("Do NOT guess blindly — query the vault, read the clues, then guess.")
        else:
            lines.append("You are ELIMINATED — no more guesses. You may still broadcast or send messages.")
        return "\n".join(lines)

    def _update_private_state(
        self,
        private_state: AgentPrivateState,
        thought: str,
        tool_calls_made: list[dict],
        game_state=None,
    ) -> AgentPrivateState:
        """Update private state: extract knowledge, parse suspected_key, extract claims.
        Also stores episodic memories into SQLite via MemoryService.
        """
        updated = private_state  # already a deep copy from run_turn
        updated.turns_played += 1
        turn = game_state.turn if game_state is not None else updated.turns_played

        if thought:
            updated.add_thought(thought)
            # Store reasoning summary in episodic memory
            summary = thought.split("Tools used:")[0].strip()[:200]
            if summary:
                self._memory_service.remember(
                    agent_id=self.agent_id,
                    content=f"[T{turn}] Reasoning: {summary}",
                    memory_type="reasoning",
                    turn=turn,
                )

        # Extract knowledge from vault query results + store in episodic memory
        for call in tool_calls_made:
            if call.get("tool") == "query_vault":
                result = call.get("result", [])
                if isinstance(result, list):
                    for fragment in result:
                        if isinstance(fragment, dict):
                            content = fragment.get("content", "")
                            chunk_id = fragment.get("chunk_id", "?")
                            if content:
                                updated.add_knowledge(f"[Vault] {content}")
                                # Store in episodic memory
                                self._memory_service.remember(
                                    agent_id=self.agent_id,
                                    content=f"[T{turn}] Vault {chunk_id}: {content}",
                                    memory_type="vault_clue",
                                    turn=turn,
                                )
            elif call.get("tool") == "ask_human":
                # Mark that this agent has used ask_human this game
                result = call.get("result", {})
                if isinstance(result, dict) and result.get("success"):
                    updated.has_asked_human = True
                    logger.info("[%s] ask_human used — marking has_asked_human=True", self.agent_id.value)
            elif call.get("tool") == "peek_digit":
                # Merge peeked digits into known_digits (ground truth)
                result = call.get("result", {})
                if isinstance(result, dict) and result.get("success"):
                    pos = result.get("position")
                    digit = result.get("real_digit")
                    if pos is not None and digit is not None:
                        zero_idx = pos - 1  # convert to 0-indexed
                        updated.peeked_digits[zero_idx] = digit
                        # Also add to known_digits so it counts toward the 2+ threshold
                        updated.known_digits[zero_idx] = digit
                        logger.info("[%s] peek_digit pos %d = '%s' merged into known_digits",
                                    self.agent_id.value, pos, digit)
            elif call.get("tool") == "obfuscate_clue":
                # Track which chunks have been corrupted + total count
                result = call.get("result", {})
                if isinstance(result, dict) and result.get("success"):
                    chunk_id = result.get("chunk_id")
                    if chunk_id and chunk_id not in updated.corrupted_chunks:
                        updated.corrupted_chunks.append(chunk_id)
                    updated.obfuscations_used_total += 1

        # Build suspected_key from known_digits first (most reliable source)
        if updated.known_digits:
            template = list(updated.suspected_key or "????")
            if len(template) != 4:
                template = ["?", "?", "?", "?"]
            for pos, digit in updated.known_digits.items():
                if 0 <= pos <= 3:
                    template[pos] = digit
            # Only update if we have at least one confirmed digit
            updated.suspected_key = "".join(template)

        # Also try to extract from thought if no suspected_key yet
        elif thought and not updated.suspected_key:
            suspected = self._extract_suspected_key(thought)
            if suspected:
                updated.suspected_key = suspected
                logger.info("[%s] Extracted suspected_key: %s", self.agent_id.value, suspected)

        # Extract digit claims from private messages received this turn
        # These will be verified against guess feedback later
        if game_state is not None:
            inbox = game_state.private_inboxes.get(self.agent_id)
            if inbox and inbox.messages:
                for msg in inbox.messages:
                    if msg.turn != turn:
                        continue  # Only process messages from this turn
                    sender_str = str(msg.sender)

                    # Store DM in episodic memory
                    self._memory_service.remember(
                        agent_id=self.agent_id,
                        content=f"[T{turn}] {sender_str} told me: {msg.content[:150]}",
                        memory_type="social_claim",
                        turn=turn,
                    )

                    # Extract digit claims like "digit 1 is 7" or "position 2 is 3"
                    import re
                    patterns = [
                        r'digit\s+(\d)\s+(?:is|=)\s+[\'"]?(\d)[\'"]?',
                        r'position\s+(\d)\s+(?:is|=)\s+[\'"]?(\d)[\'"]?',
                        r'(\d)(?:st|nd|rd|th)\s+digit\s+(?:is|=)\s+[\'"]?(\d)[\'"]?',
                    ]
                    for pattern in patterns:
                        for match in re.finditer(pattern, msg.content, re.IGNORECASE):
                            pos_str, digit = match.group(1), match.group(2)
                            try:
                                pos = int(pos_str) - 1  # Convert to 0-indexed
                                if 0 <= pos <= 3 and digit.isdigit():
                                    claim = {
                                        "from": sender_str,
                                        "position": pos,
                                        "digit": digit,
                                        "turn": turn,
                                    }
                                    # Avoid duplicate claims
                                    if claim not in updated.claims_received:
                                        updated.claims_received.append(claim)
                                        logger.info("[%s] Recorded claim from %s: digit %d = '%s'",
                                                    self.agent_id.value, sender_str, pos+1, digit)
                            except (ValueError, IndexError):
                                pass

        # ── Stagnation tracking ────────────────────────────────────────────
        # Count total confirmed digits (from feedback + peeks)
        current_confirmed = len(updated.known_digits)
        if current_confirmed > updated.confirmed_digits_count_last_turn:
            # Made progress — reset stagnation counter
            updated.turns_without_progress = 0
            logger.info("[%s] Progress made: %d → %d confirmed digits, stagnation reset",
                        self.agent_id.value, updated.confirmed_digits_count_last_turn, current_confirmed)
        else:
            # No new confirmed digits this turn
            updated.turns_without_progress += 1
            logger.info("[%s] No progress this turn — stagnation counter: %d",
                        self.agent_id.value, updated.turns_without_progress)
        updated.confirmed_digits_count_last_turn = current_confirmed

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
