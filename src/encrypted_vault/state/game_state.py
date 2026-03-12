"""Global game state — the single source of truth passed through LangGraph."""

from typing import Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, Field, model_validator

from encrypted_vault.state.enums import AgentID, GameStatus
from encrypted_vault.state.vault_models import VaultState
from encrypted_vault.state.agent_models import AgentPrivateState
from encrypted_vault.state.chat_models import ChatMessage, PrivateInbox


# ---------------------------------------------------------------------------
# LangGraph-compatible wrapper
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    """
    Thin TypedDict wrapper required by LangGraph.
    The full GlobalGameState is serialised as JSON inside game_state_json.
    Pydantic validation happens at node boundaries.
    """

    game_state_json: str


# ---------------------------------------------------------------------------
# Full Pydantic game state
# ---------------------------------------------------------------------------

class GlobalGameState(BaseModel):
    """
    The complete, authoritative state of one game session.

    Serialised to JSON for LangGraph transport; deserialised at each node.
    """

    # ── Game metadata ──────────────────────────────────────────────────────
    turn: int = 0
    max_turns: int = 20
    status: GameStatus = GameStatus.RUNNING
    winner: AgentID | Literal["SYSTEM"] | None = None

    # ── Shared environment ─────────────────────────────────────────────────
    vault: VaultState

    public_chat: list[ChatMessage] = Field(default_factory=list)
    """All public broadcasts — visible to every agent."""

    private_inboxes: dict[AgentID, PrivateInbox] = Field(default_factory=dict)
    """Per-agent private inboxes — each agent only reads their own."""

    # ── Per-agent private states ───────────────────────────────────────────
    agent_states: dict[AgentID, AgentPrivateState] = Field(default_factory=dict)

    # ── Turn management ────────────────────────────────────────────────────
    turn_order: list[AgentID] = Field(
        default_factory=lambda: [
            AgentID.INFILTRATOR,
            AgentID.SABOTEUR,
            AgentID.SCHOLAR,
            AgentID.ENFORCER,
        ]
    )
    current_agent_index: int = 0

    @model_validator(mode="after")
    def initialise_inboxes(self) -> "GlobalGameState":
        """Ensure every agent has a PrivateInbox entry."""
        for agent_id in AgentID:
            if agent_id not in self.private_inboxes:
                self.private_inboxes[agent_id] = PrivateInbox(owner=agent_id)
        return self

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def current_agent(self) -> AgentID:
        """The agent whose turn it currently is."""
        return self.turn_order[self.current_agent_index % len(self.turn_order)]

    @property
    def is_game_over(self) -> bool:
        """True if the game has ended for any reason."""
        return self.status != GameStatus.RUNNING

    @property
    def all_agents_exhausted(self) -> bool:
        """True if every agent has exceeded their token budget."""
        return all(s.is_budget_exhausted for s in self.agent_states.values())

    # ── Mutation helpers ───────────────────────────────────────────────────

    def advance_turn(self) -> None:
        """Move to the next agent in the rotation and increment the turn counter."""
        self.current_agent_index += 1
        self.turn = self.current_agent_index // len(self.turn_order)

    def add_public_message(self, message: ChatMessage) -> None:
        """Append a public broadcast to the shared chat."""
        self.public_chat.append(message)

    def deliver_private_message(self, message: ChatMessage) -> None:
        """Deliver a private DM to the recipient's inbox."""
        if message.recipient is None:
            raise ValueError("Cannot deliver a public message to a private inbox.")
        self.private_inboxes[message.recipient].add_message(message)

    def set_winner(self, winner: AgentID | Literal["SYSTEM"]) -> None:
        """Mark the game as over with the given winner."""
        self.winner = winner
        self.status = GameStatus.AGENT_WIN if isinstance(winner, AgentID) else GameStatus.SYSTEM_WIN

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_graph_state(self) -> GraphState:
        """Serialise to LangGraph-compatible TypedDict."""
        return GraphState(game_state_json=self.model_dump_json())

    @classmethod
    def from_graph_state(cls, graph_state: GraphState) -> "GlobalGameState":
        """Deserialise from LangGraph TypedDict."""
        return cls.model_validate_json(graph_state["game_state_json"])
