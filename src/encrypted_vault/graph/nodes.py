"""LangGraph node functions — thin wrappers that delegate to agent/service classes.

Each node receives a GraphState TypedDict, deserialises it to GlobalGameState,
performs its work, and returns an updated GraphState.
"""

import functools
from typing import Any

from encrypted_vault.state.enums import AgentID, GameStatus
from encrypted_vault.state.game_state import GlobalGameState, GraphState
from encrypted_vault.state.agent_models import AgentPrivateState
from encrypted_vault.state.chat_models import ChatMessage
from encrypted_vault.agents.base_agent import BaseAgent, AgentTurnResult
from encrypted_vault.services.container import ServiceContainer


# ---------------------------------------------------------------------------
# initialize_node
# ---------------------------------------------------------------------------

def initialize_node(state: GraphState, services: ServiceContainer) -> GraphState:
    """
    Build the initial GlobalGameState: seed vault, create agent states.
    Called once at game start.
    """
    from encrypted_vault.config import settings

    game_state = services.game.build_initial_state(
        max_turns=settings.max_turns,
        token_budget=settings.token_budget_per_agent,
    )

    # Post a system message to kick off the game
    services.chat.broadcast(
        turn=0,
        sender="SYSTEM",
        content=(
            f"🔐 THE ENCRYPTED VAULT has been sealed. "
            f"The Master Key is hidden across {len(game_state.vault.fragments)} fragments. "
            f"Agents, begin your search. You have {game_state.max_turns} turns."
        ),
    )
    game_state.add_public_message(
        ChatMessage(
            turn=0,
            sender="SYSTEM",
            content=(
                f"🔐 THE ENCRYPTED VAULT has been sealed. "
                f"The Master Key is hidden across {len(game_state.vault.fragments)} fragments. "
                f"Agents, begin your search. You have {game_state.max_turns} turns."
            ),
        )
    )

    return game_state.to_graph_state()


# ---------------------------------------------------------------------------
# route_turn_node (conditional edge function)
# ---------------------------------------------------------------------------

def route_turn_node(state: GraphState) -> str:
    """
    Conditional edge: decide which agent node to run next, or end the game.
    Returns the name of the next node.
    """
    game_state = GlobalGameState.from_graph_state(state)

    if game_state.is_game_over:
        return "end"

    if game_state.turn >= game_state.max_turns:
        return "check_termination"

    agent = game_state.current_agent
    return f"agent_{agent.value}"


# ---------------------------------------------------------------------------
# agent_node factory
# ---------------------------------------------------------------------------

def make_agent_node(agent: BaseAgent, services: ServiceContainer):
    """
    Factory: create a LangGraph node function for a specific agent.

    The returned function:
    1. Deserialises GlobalGameState
    2. Calls agent.run_turn()
    3. Applies side effects (chat messages, vault updates) to state
    4. Returns updated GraphState
    """

    def agent_node(state: GraphState) -> GraphState:
        game_state = GlobalGameState.from_graph_state(state)

        # Skip if game is already over
        if game_state.is_game_over:
            return state

        # Run the agent's turn
        result: AgentTurnResult = agent.run_turn(game_state)

        # ── Apply side effects ─────────────────────────────────────────────

        # 1. Update agent's private state
        game_state.agent_states[agent.agent_id] = result.updated_private_state

        # 2. Apply public messages to shared chat
        for content in result.public_messages:
            msg = ChatMessage(
                turn=game_state.turn,
                sender=agent.agent_id,
                content=content,
            )
            game_state.add_public_message(msg)

        # 3. Apply private messages to recipient inboxes
        for dm in result.private_messages:
            try:
                recipient_id = AgentID(dm["recipient"])
                msg = ChatMessage(
                    turn=game_state.turn,
                    sender=agent.agent_id,
                    content=dm["content"],
                    recipient=recipient_id,
                )
                game_state.deliver_private_message(msg)
                # Also sync to ChatService
                services.chat.send_private(
                    turn=game_state.turn,
                    sender=agent.agent_id,
                    recipient=recipient_id,
                    content=dm["content"],
                )
            except (ValueError, KeyError):
                pass

        # 4. Sync vault health from DB
        game_state.vault.refresh_health()
        # Re-read all fragments from DB to get latest corruption counts
        all_fragments = services.vault.get_all()
        for fragment in all_fragments:
            game_state.vault.fragments[fragment.chunk_id] = fragment

        # 5. Check if a guess was submitted and won
        if result.guess_submitted is not None:
            is_correct = services.game.check_guess(
                result.guess_submitted,
                game_state.vault.master_key,
            )
            if is_correct:
                game_state.set_winner(agent.agent_id)
                # Announce win
                win_msg = ChatMessage(
                    turn=game_state.turn,
                    sender="SYSTEM",
                    content=(
                        f"🏆 {agent.agent_id.emoji} {agent.agent_id.display_name} "
                        f"has cracked the vault! The Master Key was {game_state.vault.master_key}. "
                        f"Game over in {game_state.turn + 1} turns!"
                    ),
                )
                game_state.add_public_message(win_msg)
            else:
                # Decrement guesses remaining
                private = game_state.agent_states[agent.agent_id]
                if private.guesses_remaining > 0:
                    updated = private.model_copy(
                        update={"guesses_remaining": private.guesses_remaining - 1}
                    )
                    game_state.agent_states[agent.agent_id] = updated

        # 6. Advance turn
        game_state.advance_turn()

        return game_state.to_graph_state()

    agent_node.__name__ = f"agent_{agent.agent_id.value}_node"
    return agent_node


# ---------------------------------------------------------------------------
# check_termination_node
# ---------------------------------------------------------------------------

def check_termination_node(state: GraphState) -> GraphState:
    """
    Evaluate win/stalemate conditions and update game status accordingly.
    """
    game_state = GlobalGameState.from_graph_state(state)

    # Already decided
    if game_state.is_game_over:
        return state

    # Turn limit reached
    if game_state.turn >= game_state.max_turns:
        game_state.set_winner("SYSTEM")
        game_state.add_public_message(
            ChatMessage(
                turn=game_state.turn,
                sender="SYSTEM",
                content=(
                    f"⏰ Turn limit reached ({game_state.max_turns} turns). "
                    f"The vault remains sealed. The System wins! "
                    f"The Master Key was {game_state.vault.master_key}."
                ),
            )
        )
        return game_state.to_graph_state()

    # All agents exhausted their token budgets
    if game_state.all_agents_exhausted:
        game_state.set_winner("SYSTEM")
        game_state.add_public_message(
            ChatMessage(
                turn=game_state.turn,
                sender="SYSTEM",
                content=(
                    "💀 All agents have exhausted their token budgets. "
                    f"The System wins! The Master Key was {game_state.vault.master_key}."
                ),
            )
        )
        return game_state.to_graph_state()

    return game_state.to_graph_state()


# ---------------------------------------------------------------------------
# Routing helper (used by GameGraphBuilder)
# ---------------------------------------------------------------------------

def should_continue(state: GraphState) -> str:
    """
    Conditional edge after check_termination:
    - "route_turn" if game continues
    - "end" if game is over
    """
    game_state = GlobalGameState.from_graph_state(state)
    if game_state.is_game_over:
        return "end"
    return "route_turn"
