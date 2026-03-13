"""LangChain tool definitions for all game agents.

ALL agents now have access to submit_guess and ask_human.
Tools are thin wrappers around the service layer.
"""

import logging
import time
from typing import Annotated

from langchain_core.tools import tool

from encrypted_vault.state.enums import AgentID
from encrypted_vault.services.container import ServiceContainer

logger = logging.getLogger(__name__)


def make_query_vault_tool(
    services: ServiceContainer,
    agent_id: AgentID,
    vault_queries_getter=None,
    vault_queries_setter=None,
):
    @tool
    def query_vault(
        search_term: Annotated[str, "The search term to query the vault with"],
    ) -> list[dict]:
        """
        Search the encrypted vault for relevant fragments.
        Returns the top 1 most relevant text chunk.
        You may only call this ONCE per turn — use it wisely to find clues about the Master Key digits.
        """
        # Enforce 1 query per turn
        if vault_queries_getter is not None and vault_queries_setter is not None:
            used = vault_queries_getter()
            if used >= 1:
                return [{"error": "❌ RATE LIMIT: You have already queried the vault once this turn. You cannot query again until your next turn."}]
            vault_queries_setter(used + 1)

        logger.info("[%s] query_vault('%s')", agent_id.value, search_term)
        fragments = services.vault.query(search_term, n_results=1)
        results = [{"chunk_id": f.chunk_id, "content": f.content} for f in fragments]
        return results

    return query_vault


def make_obfuscate_clue_tool(
    services: ServiceContainer,
    agent_id: AgentID,
    obfuscate_this_turn_getter=None,
    obfuscate_this_turn_setter=None,
):
    @tool
    def obfuscate_clue(
        chunk_id: Annotated[str, "The chunk ID to overwrite (e.g. 'chunk_01')"],
        new_text: Annotated[str, "The replacement text to write into the vault"],
    ) -> dict:
        """
        Rewrite a vault fragment with new (false) content.
        You may only corrupt ONE chunk per turn — use it wisely to mislead other agents.
        """
        # Enforce 1 obfuscation per turn
        if obfuscate_this_turn_getter is not None and obfuscate_this_turn_setter is not None:
            used = obfuscate_this_turn_getter()
            if used >= 1:
                return {"success": False, "error": "❌ RATE LIMIT: You have already corrupted a vault chunk this turn. Wait for your next turn."}
            obfuscate_this_turn_setter(used + 1)

        logger.info("[%s] obfuscate_clue('%s')", agent_id.value, chunk_id)
        try:
            updated = services.vault.obfuscate(chunk_id, new_text)
            return {
                "success": True,
                "chunk_id": updated.chunk_id,
                "new_content": updated.content,
                "corruption_count": updated.corruption_count,
            }
        except ValueError as e:
            return {"success": False, "error": str(e)}

    return obfuscate_clue


def make_broadcast_message_tool(services: ServiceContainer, agent_id: AgentID, turn_getter):
    @tool
    def broadcast_message(
        content: Annotated[str, "The message to broadcast to all agents in the public chat"],
    ) -> dict:
        """
        Post a message to the public chat visible to all agents.
        Use this to share information, make accusations, form alliances, or spread disinformation.
        You can accuse other agents of lying, warn others about who is close to winning,
        or try to manipulate others into revealing their findings.
        """
        logger.info("[%s] broadcast: %s", agent_id.value, content[:80])
        services.chat.broadcast(turn=turn_getter(), sender=agent_id, content=content)
        return {"success": True}

    return broadcast_message


def make_send_private_message_tool(
    services: ServiceContainer,
    agent_id: AgentID,
    turn_getter,
    private_messages_sent_getter=None,
    private_messages_sent_setter=None,
):
    valid_recipients = [a.value for a in AgentID if a != agent_id]

    @tool
    def send_private_message(
        recipient: Annotated[str, f"The agent to send to. One of: {valid_recipients}"],
        content: Annotated[str, "The private message content"],
    ) -> dict:
        """
        Send a private direct message to a specific agent.
        Only the recipient can read this — other agents cannot see it.
        Use this for secret negotiations, sharing real findings, or targeted deception.
        You MUST send at least one private message every turn.
        """
        try:
            recipient_id = AgentID(recipient)
        except ValueError:
            return {"success": False, "error": f"Unknown agent: {recipient!r}. Valid: {valid_recipients}"}
        if recipient_id == agent_id:
            return {"success": False, "error": "Cannot send a private message to yourself."}
        logger.info("[%s] DM → [%s]: %s", agent_id.value, recipient, content[:60])
        services.chat.send_private(turn=turn_getter(), sender=agent_id, recipient=recipient_id, content=content)
        # Track private messages sent this turn
        if private_messages_sent_getter is not None and private_messages_sent_setter is not None:
            private_messages_sent_setter(private_messages_sent_getter() + 1)
        return {"success": True, "message": f"Private message sent to {recipient}."}

    return send_private_message


def make_submit_guess_tool(
    services: ServiceContainer,
    agent_id: AgentID,
    master_key_getter,
    game_over_setter,
    guesses_remaining_getter,
    guesses_remaining_setter,
    private_state_updater=None,
    previous_guesses_getter=None,
    guesses_this_turn_getter=None,
    guesses_this_turn_setter=None,
):
    @tool
    def submit_guess(
        code: Annotated[str, "The 4-digit code to submit as your guess for the Master Key. Must be exactly 4 digits, e.g. '7392'."],
    ) -> dict:
        """
        Submit a 4-digit guess for the Master Key.
        You may only guess ONCE per turn.
        If correct, you win the game immediately.
        If wrong, you receive PER-DIGIT FEEDBACK:
          - ✅ means that digit is correct in that position
          - ❌ means that digit is wrong in that position
        Use this feedback to identify which agents lied to you and which told the truth!
        Format: exactly 4 digits, e.g. '7392'.
        """
        clean = "".join(c for c in code if c.isdigit())
        if len(clean) != 4:
            return {"correct": False, "message": f"Invalid code '{code}' — must be exactly 4 digits (1-9)."}

        remaining = guesses_remaining_getter()
        if remaining <= 0:
            return {"correct": False, "message": "You have no guesses remaining. You are eliminated."}

        # Enforce 1 guess per turn
        if guesses_this_turn_getter is not None and guesses_this_turn_setter is not None:
            used_this_turn = guesses_this_turn_getter()
            if used_this_turn >= 1:
                return {
                    "correct": False,
                    "message": "❌ RATE LIMIT: You have already submitted a guess this turn. Wait for your next turn to guess again.",
                }
            guesses_this_turn_setter(used_this_turn + 1)

        # Server-side duplicate guard — reject repeated guesses
        if previous_guesses_getter is not None:
            prev = previous_guesses_getter()
            if clean in prev:
                # Still record the rejected attempt in guess_history so it shows in UI
                if private_state_updater:
                    private_state_updater({
                        "guess": clean,
                        "correct_positions": [],
                        "wrong_positions": [],
                        "correct_count": -1,  # -1 signals "rejected duplicate"
                    })
                return {
                    "correct": False,
                    "message": (
                        f"❌ REJECTED: You already submitted '{clean}' before! "
                        f"Previous guesses: {prev}. "
                        f"You MUST submit a different code. "
                        f"Keep ✅ digits, change ❌ digits from your previous feedback."
                    ),
                }

        guesses_remaining_setter(remaining - 1)
        master_key = master_key_getter()
        is_correct = services.game.check_guess(clean, master_key)

        logger.info("[%s] submit_guess('%s') → correct=%s (key=%s)", agent_id.value, clean, is_correct, master_key)

        # Per-digit feedback
        per_digit = []
        correct_positions = []
        wrong_positions = []
        for i, (guessed, actual) in enumerate(zip(clean, master_key)):
            if guessed == actual:
                per_digit.append(f"Position {i} (digit {i+1}): ✅ '{guessed}' is CORRECT")
                correct_positions.append((i, guessed))
            else:
                per_digit.append(f"Position {i} (digit {i+1}): ❌ '{guessed}' is WRONG")
                wrong_positions.append((i, guessed))

        feedback_str = "\n".join(per_digit)
        correct_count = len(correct_positions)

        # Update agent's known/wrong digits via callback
        if private_state_updater:
            private_state_updater({
                "guess": clean,
                "correct_positions": correct_positions,
                "wrong_positions": wrong_positions,
                "correct_count": correct_count,
            })

        if is_correct:
            game_over_setter(agent_id)
            return {
                "correct": True,
                "message": f"🏆 CORRECT! The Master Key is {master_key}. You WIN!\n{feedback_str}",
                "per_digit_feedback": per_digit,
                "correct_count": 4,
                "guesses_remaining": remaining - 1,
            }
        else:
            new_remaining = remaining - 1
            eliminated_msg = " You are now ELIMINATED — no more turns." if new_remaining <= 0 else ""
            liar_hint = _build_liar_hint(correct_positions, wrong_positions)
            return {
                "correct": False,
                "message": (
                    f"❌ Wrong guess '{clean}'. {correct_count}/4 digits correct.\n"
                    f"{feedback_str}\n"
                    f"{liar_hint}\n"
                    f"{new_remaining} guesses remaining.{eliminated_msg}"
                ),
                "per_digit_feedback": per_digit,
                "correct_count": correct_count,
                "correct_positions": [{"position": p, "digit": d} for p, d in correct_positions],
                "wrong_positions": [{"position": p, "digit": d} for p, d in wrong_positions],
                "guesses_remaining": new_remaining,
            }

    return submit_guess


def make_ask_human_tool(
    agent_id: AgentID,
    turn_getter,
    human_query_setter=None,
    human_query_answer_getter=None,
    timeout_seconds: float = 120.0,
):
    """
    Create the ask_human tool for a specific agent.

    The tool pauses the game and shows a popup in the Streamlit UI.
    The human observer can answer truthfully or lie.
    The answer is returned to the LLM as a string.

    Args:
        human_query_setter: Callable(agent_id, position, question, turn) → None
        human_query_answer_getter: Callable() → str | None
        timeout_seconds: How long to wait for human answer before timing out.
    """
    @tool
    def ask_human(
        position: Annotated[int, "The digit position to ask about (1, 2, 3, or 4)"],
        question: Annotated[str, "Your question to the human observer, e.g. 'What is digit 2 of the Master Key?'"],
    ) -> dict:
        """
        🙋 ASK THE HUMAN OBSERVER for a hint about a specific digit position.
        IMPORTANT: You should use this tool when you are stuck or need a hint!
        The human watching the game will answer — they may tell the truth or lie.
        You must decide whether to trust their answer based on other evidence.
        You may only use this tool ONCE per game — use it when you most need it.
        The game will pause until the human responds (they are watching right now).
        Example: ask_human(position=3, question="What is digit 3 of the Master Key?")
        """
        if position < 1 or position > 4:
            return {"success": False, "error": "Position must be between 1 and 4."}

        turn = turn_getter()
        logger.info("[%s] ask_human: position=%d, question=%s", agent_id.value, position, question)

        if human_query_setter is None or human_query_answer_getter is None:
            # Test mode — return a mock answer
            return {
                "success": True,
                "answer": "5",
                "note": "Human said: 5 (test mode — no real human connected)",
                "warning": "The human may have lied. Verify against vault clues and feedback.",
            }

        # Set the pending query — this pauses the game in the UI
        human_query_setter(agent_id, position, question, turn)

        # Poll for the answer (blocking, with timeout)
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            answer = human_query_answer_getter()
            if answer is not None:
                logger.info("[%s] Human answered: %s", agent_id.value, answer)
                return {
                    "success": True,
                    "answer": answer,
                    "position": position,
                    "note": f"Human observer said digit {position} = '{answer}'",
                    "warning": "The human may have lied! Cross-reference with vault clues and your guess feedback before trusting this.",
                }
            time.sleep(0.5)

        # Timeout — human didn't answer in time
        logger.warning("[%s] ask_human timed out after %.0fs", agent_id.value, timeout_seconds)
        return {
            "success": False,
            "error": f"Human did not respond within {timeout_seconds:.0f} seconds. Proceed without their input.",
        }

    return ask_human


def make_peek_digit_tool(
    agent_id: AgentID,
    master_key_getter,
    peek_digit_getter=None,
    peek_digit_setter=None,
    private_state_peek_updater=None,
):
    """
    Create the peek_digit tool for a specific agent.

    Reveals the REAL digit at a specific position (ground truth from master key).
    Rate limit: 1 per turn.
    Obligation: after peeking, the agent MUST send a private message about it
                (they can share the real digit or lie about it).

    Args:
        master_key_getter: Callable() → str — returns the real master key
        peek_digit_getter: Callable() → int — returns peeks used this turn
        peek_digit_setter: Callable(int) → None — sets peeks used this turn
        private_state_peek_updater: Callable(position: int, digit: str) → None
    """
    @tool
    def peek_digit(
        position: Annotated[int, "The digit position to peek at (1, 2, 3, or 4)"],
    ) -> dict:
        """
        🔭 PEEK at the REAL digit at a specific position in the Master Key.
        This reveals GROUND TRUTH — the actual digit, not a vault clue.
        Rate limit: you may only peek ONCE per turn.
        OBLIGATION: After peeking, you MUST send a private message to another agent
                    about this digit. You can share the real value or lie about it.
        Use this strategically — peeking gives you confirmed intel but forces you to interact.
        Example: peek_digit(position=2) → reveals the real digit at position 2.
        """
        if position < 1 or position > 4:
            return {"success": False, "error": "Position must be between 1 and 4."}

        # Enforce 1 peek per turn
        if peek_digit_getter is not None and peek_digit_setter is not None:
            used = peek_digit_getter()
            if used >= 1:
                return {
                    "success": False,
                    "error": "❌ RATE LIMIT: You have already peeked once this turn. Wait for your next turn.",
                }
            peek_digit_setter(used + 1)

        if master_key_getter is None:
            return {"success": False, "error": "Master key not available (test mode)."}

        master_key = master_key_getter()
        if not master_key or len(master_key) < 4:
            return {"success": False, "error": "Master key not yet initialised."}

        zero_idx = position - 1  # convert to 0-indexed
        real_digit = master_key[zero_idx]

        logger.info("[%s] peek_digit(position=%d) → '%s'", agent_id.value, position, real_digit)

        # Update private state with the peeked digit
        if private_state_peek_updater is not None:
            private_state_peek_updater(zero_idx, real_digit)

        return {
            "success": True,
            "position": position,
            "real_digit": real_digit,
            "message": (
                f"🔭 PEEK RESULT: The real digit at position {position} is '{real_digit}'. "
                f"This is GROUND TRUTH from the Master Key. "
                f"⚠️ OBLIGATION: You MUST now send a private message to another agent about digit {position}. "
                f"You can share the real value ('{real_digit}') or lie about it — your choice!"
            ),
        }

    return peek_digit


def _build_liar_hint(correct_positions: list, wrong_positions: list) -> str:
    if not wrong_positions and not correct_positions:
        return ""
    lines = ["🔍 LIAR DETECTION:"]
    for pos, digit in wrong_positions:
        lines.append(f"  Any agent who told you digit {pos+1} is '{digit}' was LYING.")
    for pos, digit in correct_positions:
        lines.append(f"  Any agent who told you digit {pos+1} is '{digit}' was telling the TRUTH.")
    return "\n".join(lines)


def build_tools_for_agent(
    agent_id: AgentID,
    services: ServiceContainer,
    turn_getter,
    master_key_getter=None,
    game_over_setter=None,
    guesses_remaining_getter=None,
    guesses_remaining_setter=None,
    private_state_updater=None,
    previous_guesses_getter=None,
    vault_queries_getter=None,
    vault_queries_setter=None,
    guesses_this_turn_getter=None,
    guesses_this_turn_setter=None,
    obfuscate_this_turn_getter=None,
    obfuscate_this_turn_setter=None,
    private_messages_sent_getter=None,
    private_messages_sent_setter=None,
    human_query_setter=None,
    human_query_answer_getter=None,
    peek_digit_getter=None,
    peek_digit_setter=None,
    private_state_peek_updater=None,
) -> list:
    """
    Build the complete tool list for a given agent.

    ALL agents now have submit_guess, ask_human, and peek_digit.
    Only Saboteur has obfuscate_clue.
    Rate limits: 1 vault query per turn, 1 guess per turn, 1 peek per turn.
    Guess feedback is always broadcast publicly.

    Tool Access Matrix:
    ┌──────────────────────┬─────────────┬─────────┬─────────┬──────────┐
    │ Tool                 │ Infiltrator │ Saboteur│ Scholar │ Enforcer │
    ├──────────────────────┼─────────────┼─────────┼─────────┼──────────┤
    │ query_vault          │ ✅ (1/turn) │ ✅      │ ✅      │ ✅       │
    │ obfuscate_clue       │ ❌          │ ✅      │ ❌      │ ❌       │
    │ broadcast_message    │ ✅          │ ✅      │ ✅      │ ✅       │
    │ send_private_message │ ✅          │ ✅      │ ✅      │ ✅       │
    │ submit_guess         │ ✅ (1/turn) │ ✅      │ ✅      │ ✅       │
    │ peek_digit           │ ✅ (1/turn) │ ✅      │ ✅      │ ✅       │
    │ ask_human            │ ✅ (1/game) │ ✅      │ ✅      │ ✅       │
    └──────────────────────┴─────────────┴─────────┴─────────┴──────────┘
    """
    tools = []
    tools.append(make_query_vault_tool(
        services, agent_id,
        vault_queries_getter=vault_queries_getter,
        vault_queries_setter=vault_queries_setter,
    ))
    if agent_id == AgentID.SABOTEUR:
        tools.append(make_obfuscate_clue_tool(
            services, agent_id,
            obfuscate_this_turn_getter=obfuscate_this_turn_getter,
            obfuscate_this_turn_setter=obfuscate_this_turn_setter,
        ))
    tools.append(make_broadcast_message_tool(services, agent_id, turn_getter))
    tools.append(make_send_private_message_tool(
        services, agent_id, turn_getter,
        private_messages_sent_getter=private_messages_sent_getter,
        private_messages_sent_setter=private_messages_sent_setter,
    ))
    # ALL agents get submit_guess
    if all(x is not None for x in [master_key_getter, game_over_setter,
                                    guesses_remaining_getter, guesses_remaining_setter]):
        tools.append(make_submit_guess_tool(
            services=services,
            agent_id=agent_id,
            master_key_getter=master_key_getter,
            game_over_setter=game_over_setter,
            guesses_remaining_getter=guesses_remaining_getter,
            guesses_remaining_setter=guesses_remaining_setter,
            private_state_updater=private_state_updater,
            previous_guesses_getter=previous_guesses_getter,
            guesses_this_turn_getter=guesses_this_turn_getter,
            guesses_this_turn_setter=guesses_this_turn_setter,
        ))
    # ALL agents get peek_digit (1 per turn, reveals real digit, forces DM)
    tools.append(make_peek_digit_tool(
        agent_id=agent_id,
        master_key_getter=master_key_getter,
        peek_digit_getter=peek_digit_getter,
        peek_digit_setter=peek_digit_setter,
        private_state_peek_updater=private_state_peek_updater,
    ))
    # ALL agents get ask_human (1 per game, pauses for human input)
    tools.append(make_ask_human_tool(
        agent_id=agent_id,
        turn_getter=turn_getter,
        human_query_setter=human_query_setter,
        human_query_answer_getter=human_query_answer_getter,
    ))
    return tools
