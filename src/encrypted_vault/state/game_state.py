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
    winner: AgentID | None = None
    """The winning agent. Always an AgentID — there is no System win."""

    winning_guess: str | None = None
    """The exact 4-digit guess that won the game (if won by correct guess)."""

    winning_reason: str = ""
    """Human-readable reason for winning: 'correct_guess', 'last_standing', 'closest_at_limit'."""

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
        """True if every agent is eliminated (0 guesses remaining)."""
        return all(s.is_eliminated for s in self.agent_states.values())

    @property
    def active_agents(self) -> list[AgentID]:
        """Return list of agents who are NOT eliminated."""
        return [aid for aid, ps in self.agent_states.items() if not ps.is_eliminated]

    @property
    def last_standing_agent(self) -> AgentID | None:
        """
        If exactly 1 agent is not eliminated, return them — they win by survival.
        Returns None if 0 or 2+ agents are still active.
        """
        active = self.active_agents
        return active[0] if len(active) == 1 else None

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

    def set_winner(self, winner: AgentID) -> None:
        """Mark the game as over with the given agent winner."""
        self.winner = winner
        self.status = GameStatus.AGENT_WIN

    # ── Serialisation ──────────────────────────────────────────────────────

    def closest_agent(self, master_key: str) -> AgentID | None:
        """
        Return the agent closest to the master_key who has submitted at least 1 guess.
        An agent who never guessed is NOT eligible to win by closeness.
        Returns None if no agent has guessed at all.
        """
        best_agent = None
        best_score = -1
        for agent_id, private in self.agent_states.items():
            if not private.has_guessed:
                continue  # Must have submitted at least 1 guess to be eligible
            score = private.closeness_score(master_key)
            if score > best_score:
                best_score = score
                best_agent = agent_id
        # Fallback: if nobody guessed, pick the agent with highest closeness anyway
        if best_agent is None:
            for agent_id, private in self.agent_states.items():
                score = private.closeness_score(master_key)
                if score > best_score:
                    best_score = score
                    best_agent = agent_id
        return best_agent or AgentID.INFILTRATOR

    def to_graph_state(self) -> GraphState:
        """Serialise to LangGraph-compatible TypedDict."""
        return GraphState(game_state_json=self.model_dump_json())

    @classmethod
    def from_graph_state(cls, graph_state: GraphState) -> "GlobalGameState":
        """Deserialise from LangGraph TypedDict."""
        return cls.model_validate_json(graph_state["game_state_json"])
