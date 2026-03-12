"""Agent private state Pydantic model."""

from pydantic import BaseModel, Field

from encrypted_vault.state.enums import AgentID


class AgentPrivateState(BaseModel):
    """
    Private state for a single agent.

    This is only passed to the agent's own node — other agents cannot see it.
    The UI (spectator) can see all private states.
    """

    agent_id: AgentID
    knowledge_base: list[str] = Field(default_factory=list)
    """Accumulated clues and deductions the agent has gathered."""

    suspected_key: str | None = None
    """The agent's current best 4-digit guess (may be partial, e.g. '7_9_')."""

    known_digits: dict[int, str] = Field(default_factory=dict)
    """Confirmed digit positions: {0: '7', 1: '3'} means pos 0=7, pos 1=3."""

    tokens_used: int = 0
    token_budget: int = 8000
    thought_trace: list[str] = Field(default_factory=list)
    """Internal reasoning log — shown in UI but never sent to other agents."""

    guesses_remaining: int = 3
    turns_played: int = 0

    def closeness_score(self, master_key: str) -> int:
        """
        Return 0-4: how many digits the agent has correct in the right position.
        Compares suspected_key against the real master_key.
        """
        if not self.suspected_key:
            return 0
        # Normalise: strip non-digits, pad/truncate to 4
        clean = "".join(c for c in self.suspected_key if c.isdigit())
        if len(clean) != 4:
            return 0
        return sum(1 for i, d in enumerate(clean) if i < len(master_key) and d == master_key[i])

    @property
    def is_budget_exhausted(self) -> bool:
        """True if the agent has used up their token budget."""
        return self.tokens_used >= self.token_budget

    def add_thought(self, thought: str) -> None:
        """Append a reasoning step to the thought trace."""
        self.thought_trace.append(thought)

    def add_knowledge(self, clue: str) -> None:
        """Add a new clue to the knowledge base (deduplicates)."""
        if clue not in self.knowledge_base:
            self.knowledge_base.append(clue)
