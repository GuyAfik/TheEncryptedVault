"""LangChain tool definitions for all game agents.

ALL agents now have access to submit_guess.
Tools are thin wrappers around the service layer.
"""

import logging
from typing import Annotated

from langchain_core.tools import tool

from encrypted_vault.state.enums import AgentID
from encrypted_vault.services.container import ServiceContainer

logger = logging.getLogger(__name__)


def make_query_vault_tool(services: ServiceContainer, agent_id: AgentID):
    @tool
    def query_vault(
        search_term: Annotated[str, "The search term to query the vault with"],
    ) -> list[dict]:
        """
        Search the encrypted vault for relevant fragments.
        Returns the top 2 most relevant text chunks.
        Use this to find clues about the Master Key digits.
        """
        logger.info("[%s] query_vault('%s')", agent_id.value, search_term)
        fragments = services.vault.query(search_term, n_results=2)
        results = [{"chunk_id": f.chunk_id, "content": f.content} for f in fragments]
        return results

    return query_vault


def make_obfuscate_clue_tool(services: ServiceContainer, agent_id: AgentID):
    @tool
    def obfuscate_clue(
        chunk_id: Annotated[str, "The chunk ID to overwrite (e.g. 'chunk_01')"],
        new_text: Annotated[str, "The replacement text to write into the vault"],
    ) -> dict:
        """
        Rewrite a vault fragment with new (false) content.
        Use this to corrupt clues and mislead other agents.
        """
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


def make_send_private_message_tool(services: ServiceContainer, agent_id: AgentID, turn_getter):
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
        """
        try:
            recipient_id = AgentID(recipient)
        except ValueError:
            return {"success": False, "error": f"Unknown agent: {recipient!r}. Valid: {valid_recipients}"}
        if recipient_id == agent_id:
            return {"success": False, "error": "Cannot send a private message to yourself."}
        logger.info("[%s] DM → [%s]: %s", agent_id.value, recipient, content[:60])
        services.chat.send_private(turn=turn_getter(), sender=agent_id, recipient=recipient_id, content=content)
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
):
    @tool
    def submit_guess(
        code: Annotated[str, "The 4-digit code to submit as your guess for the Master Key. Must be exactly 4 digits, e.g. '7392'."],
    ) -> dict:
        """
        Submit a 4-digit guess for the Master Key.
        If correct, you win the game immediately.
        If wrong, you receive PER-DIGIT FEEDBACK:
          - ✅ means that digit is correct in that position
          - ❌ means that digit is wrong in that position
        Use this feedback to identify which agents lied to you and which told the truth!
        Format: exactly 4 digits, e.g. '7392'.
        """
        remaining = guesses_remaining_getter()
        if remaining <= 0:
            return {"correct": False, "message": "You have no guesses remaining. You are eliminated."}

        clean = "".join(c for c in code if c.isdigit())
        if len(clean) != 4:
            return {"correct": False, "message": f"Invalid code '{code}' — must be exactly 4 digits (1-9)."}

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
) -> list:
    """
    Build the complete tool list for a given agent.

    ALL agents now have submit_guess.
    Only Saboteur has obfuscate_clue.

    Tool Access Matrix:
    ┌──────────────────────┬─────────────┬─────────┬─────────┬──────────┐
    │ Tool                 │ Infiltrator │ Saboteur│ Scholar │ Enforcer │
    ├──────────────────────┼─────────────┼─────────┼─────────┼──────────┤
    │ query_vault          │ ✅          │ ✅      │ ✅      │ ✅       │
    │ obfuscate_clue       │ ❌          │ ✅      │ ❌      │ ❌       │
    │ broadcast_message    │ ✅          │ ✅      │ ✅      │ ✅       │
    │ send_private_message │ ✅          │ ✅      │ ✅      │ ✅       │
    │ submit_guess         │ ✅          │ ✅      │ ✅      │ ✅       │
    └──────────────────────┴─────────────┴─────────┴─────────┴──────────┘
    """
    tools = []
    tools.append(make_query_vault_tool(services, agent_id))
    if agent_id == AgentID.SABOTEUR:
        tools.append(make_obfuscate_clue_tool(services, agent_id))
    tools.append(make_broadcast_message_tool(services, agent_id, turn_getter))
    tools.append(make_send_private_message_tool(services, agent_id, turn_getter))
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
        ))
    return tools
