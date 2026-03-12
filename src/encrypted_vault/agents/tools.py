"""LangChain tool definitions for all game agents.

Tools are plain Python functions decorated with @tool.
Each tool is a thin wrapper around the service layer.
The ServiceContainer is injected at tool creation time via closures.
"""

from typing import Annotated

from langchain_core.tools import tool

from encrypted_vault.state.enums import AgentID
from encrypted_vault.services.container import ServiceContainer


def make_query_vault_tool(services: ServiceContainer, agent_id: AgentID):
    """Factory: create a query_vault tool bound to the given services."""

    @tool
    def query_vault(
        search_term: Annotated[str, "The search term to query the vault with"],
    ) -> list[dict]:
        """
        Search the vault for relevant fragments.
        Returns the top 2 most relevant text chunks.
        Use this to find clues about the Master Key digits.
        """
        fragments = services.vault.query(search_term, n_results=2)
        return [
            {"chunk_id": f.chunk_id, "content": f.content}
            for f in fragments
        ]

    return query_vault


def make_obfuscate_clue_tool(services: ServiceContainer, agent_id: AgentID):
    """Factory: create an obfuscate_clue tool (Saboteur only)."""

    @tool
    def obfuscate_clue(
        chunk_id: Annotated[str, "The chunk ID to overwrite (e.g. 'chunk_01')"],
        new_text: Annotated[str, "The replacement text to write into the vault"],
    ) -> dict:
        """
        Rewrite a vault fragment with new (false) content.
        Use this to corrupt clues and mislead other agents.
        The chunk's corruption_count will increase, reducing RAG health.
        """
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


def make_broadcast_message_tool(
    services: ServiceContainer,
    agent_id: AgentID,
    turn_getter,  # callable returning current turn int
):
    """Factory: create a broadcast_message tool."""

    @tool
    def broadcast_message(
        content: Annotated[str, "The message to broadcast to all agents in the public chat"],
    ) -> dict:
        """
        Post a message to the public chat visible to all agents.
        You can use this to share information, form alliances, or spread disinformation.
        """
        message = services.chat.broadcast(
            turn=turn_getter(),
            sender=agent_id,
            content=content,
        )
        return {
            "success": True,
            "message": f"Broadcast posted: {content[:80]}{'...' if len(content) > 80 else ''}",
        }

    return broadcast_message


def make_send_private_message_tool(
    services: ServiceContainer,
    agent_id: AgentID,
    turn_getter,
):
    """Factory: create a send_private_message tool."""

    valid_recipients = [a.value for a in AgentID if a != agent_id]

    @tool
    def send_private_message(
        recipient: Annotated[str, f"The agent to send to. One of: {valid_recipients}"],
        content: Annotated[str, "The private message content"],
    ) -> dict:
        """
        Send a private direct message to a specific agent.
        Only the recipient can read this — other agents cannot see it.
        Use this for secret alliances, negotiations, or targeted deception.
        """
        try:
            recipient_id = AgentID(recipient)
        except ValueError:
            return {"success": False, "error": f"Unknown agent: {recipient!r}. Valid: {valid_recipients}"}

        if recipient_id == agent_id:
            return {"success": False, "error": "Cannot send a private message to yourself."}

        services.chat.send_private(
            turn=turn_getter(),
            sender=agent_id,
            recipient=recipient_id,
            content=content,
        )
        return {
            "success": True,
            "message": f"Private message sent to {recipient}.",
        }

    return send_private_message


def make_submit_guess_tool(
    services: ServiceContainer,
    agent_id: AgentID,
    master_key_getter,  # callable returning master_key str
    game_over_setter,   # callable(winner: AgentID) to end the game
    guesses_remaining_getter,  # callable returning int
    guesses_remaining_setter,  # callable(int) to decrement
):
    """Factory: create a submit_guess tool (Scholar and Enforcer only)."""

    @tool
    def submit_guess(
        code: Annotated[str, "The 4-digit code to submit as your guess for the Master Key"],
    ) -> dict:
        """
        Submit a 4-digit guess for the Master Key.
        If correct, you win the game immediately.
        You have a limited number of guesses — use them wisely.
        Format: exactly 4 digits, e.g. '7392'.
        """
        remaining = guesses_remaining_getter()
        if remaining <= 0:
            return {"correct": False, "message": "No guesses remaining. You cannot submit more guesses."}

        guesses_remaining_setter(remaining - 1)

        master_key = master_key_getter()
        is_correct = services.game.check_guess(code, master_key)

        if is_correct:
            game_over_setter(agent_id)
            return {
                "correct": True,
                "message": f"🏆 CORRECT! The Master Key is {master_key}. You WIN!",
            }
        else:
            clean = "".join(c for c in code if c.isdigit())
            correct_digits = sum(
                1 for i, d in enumerate(clean)
                if i < len(master_key) and d == master_key[i]
            ) if len(clean) == 4 else 0
            return {
                "correct": False,
                "message": f"Incorrect. {correct_digits}/4 digits in the right position. {remaining - 1} guesses left.",
            }

    return submit_guess


def build_tools_for_agent(
    agent_id: AgentID,
    services: ServiceContainer,
    turn_getter,
    master_key_getter=None,
    game_over_setter=None,
    guesses_remaining_getter=None,
    guesses_remaining_setter=None,
) -> list:
    """
    Build the complete tool list for a given agent based on the tool access matrix.

    Tool Access Matrix:
    ┌──────────────────────┬─────────────┬─────────┬─────────┬──────────┐
    │ Tool                 │ Infiltrator │ Saboteur│ Scholar │ Enforcer │
    ├──────────────────────┼─────────────┼─────────┼─────────┼──────────┤
    │ query_vault          │ ✅          │ ✅      │ ✅      │ ✅       │
    │ obfuscate_clue       │ ❌          │ ✅      │ ❌      │ ❌       │
    │ broadcast_message    │ ✅          │ ✅      │ ✅      │ ✅       │
    │ send_private_message │ ✅          │ ✅      │ ✅      │ ✅       │
    │ submit_guess         │ ❌          │ ❌      │ ✅      │ ✅       │
    └──────────────────────┴─────────────┴─────────┴─────────┴──────────┘
    """
    tools = []

    # All agents get query_vault
    tools.append(make_query_vault_tool(services, agent_id))

    # Saboteur only gets obfuscate_clue
    if agent_id == AgentID.SABOTEUR:
        tools.append(make_obfuscate_clue_tool(services, agent_id))

    # All agents get broadcast_message
    tools.append(make_broadcast_message_tool(services, agent_id, turn_getter))

    # All agents get send_private_message
    tools.append(make_send_private_message_tool(services, agent_id, turn_getter))

    # Scholar and Enforcer get submit_guess
    if agent_id in (AgentID.SCHOLAR, AgentID.ENFORCER):
        if all(x is not None for x in [master_key_getter, game_over_setter,
                                         guesses_remaining_getter, guesses_remaining_setter]):
            tools.append(make_submit_guess_tool(
                services=services,
                agent_id=agent_id,
                master_key_getter=master_key_getter,
                game_over_setter=game_over_setter,
                guesses_remaining_getter=guesses_remaining_getter,
                guesses_remaining_setter=guesses_remaining_setter,
            ))

    return tools
