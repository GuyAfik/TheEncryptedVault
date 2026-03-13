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
    """The agent's current best 4-digit guess."""

    known_digits: dict[int, str] = Field(default_factory=dict)
    """Confirmed correct digit positions from guess feedback: {0: '7', 1: '3'}."""

    wrong_digits: dict[int, list[str]] = Field(default_factory=dict)
    """Digits confirmed WRONG at each position from guess feedback."""

    guess_history: list[dict] = Field(default_factory=list)
    """History of submitted guesses with per-digit feedback.
    Each entry: {'guess': '7392', 'feedback': ['✅', '❌', '✅', '❌'], 'correct_count': 2}"""

    thought_trace: list[str] = Field(default_factory=list)
    """Internal reasoning log — shown in UI but never sent to other agents."""

    guesses_remaining: int = 3
    turns_played: int = 0
    is_eliminated: bool = False
    """True when agent has used all 3 guesses without winning — they are out."""

    has_guessed: bool = False
    """True if the agent has submitted at least 1 guess.
    Required to be eligible for the 'closest agent wins' tiebreaker."""

    # ── Per-turn rate limits (reset at the start of each agent's turn) ────────
    vault_queries_this_turn: int = 0
    """Number of vault queries used this turn. Max 1 per turn."""

    guesses_this_turn: int = 0
    """Number of guesses submitted this turn. Max 1 per turn."""

    private_messages_sent_this_turn: int = 0
    """Number of private messages sent this turn. Used to detect if auto-DM fallback is needed."""

    peek_digit_this_turn: int = 0
    """Number of peek_digit calls this turn. Max 1 per turn."""

    # ── Per-game one-time abilities ────────────────────────────────────────
    has_asked_human: bool = False
    """True if this agent has already used ask_human this game (1 use per game)."""

    peeked_digits: dict[int, str] = Field(default_factory=dict)
    """Digits revealed by peek_digit: {0: '7'} means position 1 = '7' (confirmed real, 0-indexed)."""

    # ── Message history cursors (track what has already been shown to the LLM) ──
    last_seen_public_idx: int = 0
    """Index into game_state.public_chat — messages at index >= this are NEW this turn."""

    last_seen_private_idx: int = 0
    """Index into inbox.messages — messages at index >= this are NEW this turn."""

    # ── Social intelligence ────────────────────────────────────────────────

    agent_trust: dict[str, str] = Field(default_factory=dict)
    """Trust level per agent: {'saboteur': 'LIAR', 'infiltrator': 'TRUSTED', 'enforcer': 'UNKNOWN'}
    Updated automatically when guess feedback confirms or refutes what an agent told you."""

    social_notes: list[str] = Field(default_factory=list)
    """Persistent social observations: 'Infiltrator told me digit 1=7, confirmed TRUE by feedback'
    These are the agent's memory of social interactions and their outcomes."""

    claims_received: list[dict] = Field(default_factory=list)
    """Claims received from other agents: [{'from': 'infiltrator', 'position': 1, 'digit': '7', 'turn': 3}]
    Used to verify claims against guess feedback and update trust."""

    def closeness_score(self, master_key: str) -> int:
        """
        Return 0-4: how many digits the agent has correct in the right position.
        Uses known_digits (from guess feedback) first, then suspected_key.
        """
        # If we have confirmed digits from feedback, use those
        if self.known_digits:
            # Build best guess from known_digits + suspected_key
            best = list("0000")
            if self.suspected_key and len(self.suspected_key) == 4:
                for i, d in enumerate(self.suspected_key):
                    best[i] = d
            for pos, digit in self.known_digits.items():
                best[pos] = digit
            return sum(1 for i, d in enumerate(best) if i < len(master_key) and d == master_key[i])

        if not self.suspected_key:
            return 0
        clean = "".join(c for c in self.suspected_key if c.isdigit())
        if len(clean) != 4:
            return 0
        return sum(1 for i, d in enumerate(clean) if i < len(master_key) and d == master_key[i])

    def add_thought(self, thought: str) -> None:
        """Append a reasoning step to the thought trace."""
        self.thought_trace.append(thought)

    def add_knowledge(self, clue: str) -> None:
        """Add a new clue to the knowledge base (deduplicates)."""
        if clue not in self.knowledge_base:
            self.knowledge_base.append(clue)
