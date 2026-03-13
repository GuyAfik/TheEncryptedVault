"""GameGraphBuilder — constructs the LangGraph StateGraph for the game loop."""

import functools
import random

from langgraph.graph import StateGraph, END

from encrypted_vault.state.game_state import GraphState, GlobalGameState
from encrypted_vault.state.enums import AgentID
from encrypted_vault.agents.infiltrator import Infiltrator
from encrypted_vault.agents.saboteur import Saboteur
from encrypted_vault.agents.scholar import Scholar
from encrypted_vault.agents.enforcer import Enforcer
from encrypted_vault.services.container import ServiceContainer
from encrypted_vault.llm_factory import LLMFactory
from encrypted_vault.graph.nodes import (
    initialize_node,
    make_agent_node,
    check_termination_node,
)


class GameGraphBuilder:
    """
    Constructs and compiles the LangGraph StateGraph.

    All 4 agents have submit_guess.
    Agents are eliminated after 3 wrong guesses.
    Guess feedback is always broadcast publicly.
    Turn 20 with no correct guess → nobody wins.
    """

    def __init__(self, services: ServiceContainer) -> None:
        self._services = services

    def build(self):
        """Build and compile the LangGraph StateGraph."""
        llm = LLMFactory.create_default()

        # ── Shared mutable refs for tool callbacks ─────────────────────────
        _state_ref: dict = {"graph_state": None}
        _guesses: dict[AgentID, int] = {a: 3 for a in AgentID}
        # Per-turn rate-limit counters (reset at the start of each agent's turn)
        _vault_queries_this_turn: dict[AgentID, int] = {a: 0 for a in AgentID}
        _guesses_this_turn: dict[AgentID, int] = {a: 0 for a in AgentID}
        _private_messages_sent_this_turn: dict[AgentID, int] = {a: 0 for a in AgentID}
        _obfuscate_this_turn: int = 0  # Saboteur only

        def turn_getter() -> int:
            gs = _state_ref.get("graph_state")
            if gs:
                return GlobalGameState.from_graph_state(gs).turn
            return 0

        def master_key_getter() -> str:
            gs = _state_ref.get("graph_state")
            if gs:
                return GlobalGameState.from_graph_state(gs).vault.master_key
            return ""

        def game_over_setter(winner: AgentID) -> None:
            gs = _state_ref.get("graph_state")
            if gs:
                game_state = GlobalGameState.from_graph_state(gs)
                game_state.set_winner(winner)
                _state_ref["graph_state"] = game_state.to_graph_state()

        def make_guesses_getter(agent_id: AgentID):
            return lambda: _guesses.get(agent_id, 3)

        def make_guesses_setter(agent_id: AgentID):
            def setter(n: int):
                _guesses[agent_id] = n
            return setter

        def make_vault_queries_getter(agent_id: AgentID):
            return lambda: _vault_queries_this_turn.get(agent_id, 0)

        def make_vault_queries_setter(agent_id: AgentID):
            def setter(n: int):
                _vault_queries_this_turn[agent_id] = n
            return setter

        def make_guesses_this_turn_getter(agent_id: AgentID):
            return lambda: _guesses_this_turn.get(agent_id, 0)

        def make_guesses_this_turn_setter(agent_id: AgentID):
            def setter(n: int):
                _guesses_this_turn[agent_id] = n
            return setter

        def make_private_messages_sent_getter(agent_id: AgentID):
            return lambda: _private_messages_sent_this_turn.get(agent_id, 0)

        def make_private_messages_sent_setter(agent_id: AgentID):
            def setter(n: int):
                _private_messages_sent_this_turn[agent_id] = n
            return setter

        def obfuscate_this_turn_getter() -> int:
            return _obfuscate_this_turn

        def obfuscate_this_turn_setter(n: int) -> None:
            nonlocal _obfuscate_this_turn
            _obfuscate_this_turn = n

        # Expose reset functions so agent_node can call them at turn start
        def reset_turn_counters(agent_id: AgentID) -> None:
            """Reset per-turn rate-limit counters for the given agent."""
            nonlocal _obfuscate_this_turn
            _vault_queries_this_turn[agent_id] = 0
            _guesses_this_turn[agent_id] = 0
            _private_messages_sent_this_turn[agent_id] = 0
            if agent_id == AgentID.SABOTEUR:
                _obfuscate_this_turn = 0

        # Shared getter for private_messages_sent (used by make_agent_node)
        def private_messages_sent_getter(agent_id: AgentID) -> int:
            return _private_messages_sent_this_turn.get(agent_id, 0)

        # ── Instantiate all 4 agents (all get submit_guess) ────────────────
        def make_agent(cls, agent_id: AgentID, **extra_kwargs):
            kwargs = dict(
                llm=llm,
                services=self._services,
                turn_getter=turn_getter,
                master_key_getter=master_key_getter,
                game_over_setter=game_over_setter,
                guesses_remaining_getter=make_guesses_getter(agent_id),
                guesses_remaining_setter=make_guesses_setter(agent_id),
                private_state_updater_factory=lambda agent: agent._make_private_state_updater(),
                vault_queries_getter=make_vault_queries_getter(agent_id),
                vault_queries_setter=make_vault_queries_setter(agent_id),
                guesses_this_turn_getter=make_guesses_this_turn_getter(agent_id),
                guesses_this_turn_setter=make_guesses_this_turn_setter(agent_id),
                private_messages_sent_getter=make_private_messages_sent_getter(agent_id),
                private_messages_sent_setter=make_private_messages_sent_setter(agent_id),
            )
            kwargs.update(extra_kwargs)
            return cls(**kwargs)

        infiltrator = make_agent(Infiltrator, AgentID.INFILTRATOR)
        saboteur    = make_agent(Saboteur,    AgentID.SABOTEUR,
                                 obfuscate_this_turn_getter=obfuscate_this_turn_getter,
                                 obfuscate_this_turn_setter=obfuscate_this_turn_setter)
        scholar     = make_agent(Scholar,     AgentID.SCHOLAR)
        enforcer    = make_agent(Enforcer,    AgentID.ENFORCER)

        # ── Wrap agent nodes to keep _state_ref in sync ────────────────────
        def wrap_with_state_sync(node_fn, agent_id: AgentID):
            def wrapped(state: GraphState) -> GraphState:
                _state_ref["graph_state"] = state
                result = node_fn(state)
                _state_ref["graph_state"] = result
                return result
            wrapped.__name__ = f"agent_{agent_id.value}_node"
            return wrapped

        # ── Build StateGraph ───────────────────────────────────────────────
        graph = StateGraph(GraphState)

        graph.add_node(
            "initialize",
            functools.partial(
                initialize_node,
                services=self._services,
            ),
        )

        for agent_id, agent in [
            (AgentID.INFILTRATOR, infiltrator),
            (AgentID.SABOTEUR,    saboteur),
            (AgentID.SCHOLAR,     scholar),
            (AgentID.ENFORCER,    enforcer),
        ]:
            raw_node = make_agent_node(
                agent,
                self._services,
                reset_turn_counters,
                private_messages_sent_getter,
            )
            graph.add_node(
                f"agent_{agent_id.value}",
                wrap_with_state_sync(raw_node, agent_id),
            )

        graph.add_node("check_termination", check_termination_node)

        # ── Edges ──────────────────────────────────────────────────────────
        graph.set_entry_point("initialize")
        graph.add_edge("initialize", "check_termination")

        graph.add_conditional_edges(
            "check_termination",
            self._route_to_agent,
            {
                f"agent_{AgentID.INFILTRATOR.value}": f"agent_{AgentID.INFILTRATOR.value}",
                f"agent_{AgentID.SABOTEUR.value}":    f"agent_{AgentID.SABOTEUR.value}",
                f"agent_{AgentID.SCHOLAR.value}":     f"agent_{AgentID.SCHOLAR.value}",
                f"agent_{AgentID.ENFORCER.value}":    f"agent_{AgentID.ENFORCER.value}",
                "end": END,
            },
        )

        for agent_id in AgentID:
            graph.add_edge(f"agent_{agent_id.value}", "check_termination")

        return graph.compile()

    @staticmethod
    def _route_to_agent(state: GraphState) -> str:
        """Select next non-eliminated agent or end the game."""
        game_state = GlobalGameState.from_graph_state(state)

        if game_state.is_game_over or game_state.turn >= game_state.max_turns:
            return "end"

        # Find next non-eliminated agent in turn order
        total_agents = len(game_state.turn_order)
        for offset in range(total_agents):
            idx = (game_state.current_agent_index + offset) % total_agents
            agent_id = game_state.turn_order[idx]
            private = game_state.agent_states.get(agent_id)
            if private and not private.is_eliminated:
                if offset > 0:
                    game_state.current_agent_index = idx
                return f"agent_{agent_id.value}"

        # All agents eliminated
        return "end"
