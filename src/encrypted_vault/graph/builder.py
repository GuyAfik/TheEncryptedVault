"""GameGraphBuilder — constructs the LangGraph StateGraph for the game loop."""

import functools

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

    Graph topology:
        START
          │
        [initialize]
          │
        [check_termination] ◄──────────────────────────────┐
          │                                                 │
          ├─ game_over=True ──► END                         │
          │                                                 │
          ├─ agent=INFILTRATOR ──► [agent_infiltrator] ─────┤
          ├─ agent=SABOTEUR    ──► [agent_saboteur]    ─────┤
          ├─ agent=SCHOLAR     ──► [agent_scholar]     ─────┤
          └─ agent=ENFORCER    ──► [agent_enforcer]    ─────┘

    Usage:
        builder = GameGraphBuilder(services=container)
        graph = builder.build()
        for event in graph.stream(initial_state, stream_mode="values"):
            ...
    """

    def __init__(self, services: ServiceContainer) -> None:
        self._services = services

    def build(self):
        """Build and compile the LangGraph StateGraph. Returns a CompiledGraph."""
        llm = LLMFactory.create_default()

        # ── Shared mutable refs for tool callbacks ─────────────────────────
        # These allow tools to read/write game state during agent execution.
        _state_ref: dict = {"graph_state": None}
        _guesses: dict[AgentID, int] = {a: 3 for a in AgentID}

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

        # ── Instantiate agents ─────────────────────────────────────────────
        infiltrator = Infiltrator(
            llm=llm,
            services=self._services,
            turn_getter=turn_getter,
        )
        saboteur = Saboteur(
            llm=llm,
            services=self._services,
            turn_getter=turn_getter,
        )
        scholar = Scholar(
            llm=llm,
            services=self._services,
            turn_getter=turn_getter,
            master_key_getter=master_key_getter,
            game_over_setter=game_over_setter,
            guesses_remaining_getter=make_guesses_getter(AgentID.SCHOLAR),
            guesses_remaining_setter=make_guesses_setter(AgentID.SCHOLAR),
        )
        enforcer = Enforcer(
            llm=llm,
            services=self._services,
            turn_getter=turn_getter,
            master_key_getter=master_key_getter,
            game_over_setter=game_over_setter,
            guesses_remaining_getter=make_guesses_getter(AgentID.ENFORCER),
            guesses_remaining_setter=make_guesses_setter(AgentID.ENFORCER),
        )

        # ── Wrap agent nodes to update _state_ref ─────────────────────────
        # This keeps the shared state ref in sync so tool callbacks work.
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

        # Add initialize node
        graph.add_node(
            "initialize",
            functools.partial(initialize_node, services=self._services),
        )

        # Add agent nodes (wrapped for state sync)
        for agent_id, agent in [
            (AgentID.INFILTRATOR, infiltrator),
            (AgentID.SABOTEUR, saboteur),
            (AgentID.SCHOLAR, scholar),
            (AgentID.ENFORCER, enforcer),
        ]:
            raw_node = make_agent_node(agent, self._services)
            graph.add_node(
                f"agent_{agent_id.value}",
                wrap_with_state_sync(raw_node, agent_id),
            )

        # Add check_termination node
        graph.add_node("check_termination", check_termination_node)

        # ── Edges ──────────────────────────────────────────────────────────

        # Entry point
        graph.set_entry_point("initialize")

        # initialize → check_termination
        graph.add_edge("initialize", "check_termination")

        # check_termination → (agent node OR END) via single conditional edge
        graph.add_conditional_edges(
            "check_termination",
            self._route_to_agent,
            {
                f"agent_{AgentID.INFILTRATOR.value}": f"agent_{AgentID.INFILTRATOR.value}",
                f"agent_{AgentID.SABOTEUR.value}": f"agent_{AgentID.SABOTEUR.value}",
                f"agent_{AgentID.SCHOLAR.value}": f"agent_{AgentID.SCHOLAR.value}",
                f"agent_{AgentID.ENFORCER.value}": f"agent_{AgentID.ENFORCER.value}",
                "end": END,
            },
        )

        # Each agent → check_termination (loop)
        for agent_id in AgentID:
            graph.add_edge(f"agent_{agent_id.value}", "check_termination")

        return graph.compile()

    @staticmethod
    def _route_to_agent(state: GraphState) -> str:
        """
        Conditional edge function: select the next agent node or end the game.
        Called after check_termination on every iteration.
        """
        game_state = GlobalGameState.from_graph_state(state)

        if game_state.is_game_over or game_state.turn >= game_state.max_turns:
            return "end"

        agent = game_state.current_agent
        return f"agent_{agent.value}"
