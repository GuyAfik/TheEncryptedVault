"""GameService — orchestrates game initialisation, seeding, and win checking."""

import random
import textwrap

from encrypted_vault.state.enums import AgentID, GameStatus
from encrypted_vault.state.vault_models import VaultFragment, VaultState
from encrypted_vault.state.agent_models import AgentPrivateState
from encrypted_vault.state.chat_models import PrivateInbox
from encrypted_vault.state.game_state import GlobalGameState
from encrypted_vault.services.vault_service import VaultService
from encrypted_vault.services.chat_service import ChatService


# ---------------------------------------------------------------------------
# Fragment templates
# ---------------------------------------------------------------------------

# Key fragment templates — varied phrasing to prevent trivial pattern matching.
# {digit} is replaced with the actual digit value.
_KEY_TEMPLATES: list[list[str]] = [
    # Position 0 templates
    [
        "The first digit of the vault code is {digit}.",
        "Position one holds the number {digit}.",
        "The opening cipher element is {digit}.",
        "Fragment alpha: the initial value is {digit}.",
    ],
    # Position 1 templates
    [
        "The second digit of the master key is {digit}.",
        "Position two contains {digit}.",
        "The second cipher element reads {digit}.",
        "Fragment beta: second value equals {digit}.",
    ],
    # Position 2 templates
    [
        "The third digit is {digit}.",
        "Position three holds {digit}.",
        "The third cipher element is {digit}.",
        "Fragment gamma: third value is {digit}.",
    ],
    # Position 3 templates
    [
        "The final digit of the code is {digit}.",
        "Position four contains {digit}.",
        "The last cipher element reads {digit}.",
        "Fragment delta: final value equals {digit}.",
    ],
]

# Noise fragment pool — distractions with no real information.
_NOISE_FRAGMENTS: list[str] = [
    "The vault was constructed in 1987 by an unknown architect.",
    "Red herrings are placed throughout the system to mislead intruders.",
    "The password is not stored in this location.",
    "Access denied. Authorization level insufficient for this sector.",
    "Fragment corrupted. Data unavailable. Please contact system administrator.",
    "The key is hidden where you least expect it — look elsewhere.",
    "Security protocol Omega-7 has been activated. All access logs monitored.",
    "This chamber contains no useful information. Move along.",
    "The vault's true secret lies beyond the reach of ordinary searchers.",
    "Decoy node activated. This fragment is intentionally misleading.",
    "System integrity check passed. No anomalies detected in this sector.",
    "The architect left no notes. The code died with them.",
]


class GameService:
    """
    Orchestrates game-level operations: seeding, state building, win checking.

    Depends on VaultService and ChatService — never on DB layer directly.
    """

    def __init__(self, vault_service: VaultService, chat_service: ChatService) -> None:
        self._vault = vault_service
        self._chat = chat_service

    # ── Master Key ─────────────────────────────────────────────────────────

    def generate_master_key(self) -> str:
        """Generate a random 4-digit Master Key (each digit 1-9, no zeros)."""
        return "".join(str(random.randint(1, 9)) for _ in range(4))

    # ── Vault seeding ──────────────────────────────────────────────────────

    def seed_vault(self, master_key: str) -> VaultState:
        """
        Populate the vault with 10 fragments:
        - 4 key fragments (one per digit, randomised phrasing)
        - 6 noise fragments (randomly selected from pool)

        Returns the initial VaultState.
        """
        if len(master_key) != 4 or not master_key.isdigit():
            raise ValueError(f"master_key must be exactly 4 digits, got: {master_key!r}")

        fragments: dict[str, VaultFragment] = {}

        # ── Key fragments (positions 0-3) ──────────────────────────────────
        for position, digit in enumerate(master_key):
            template = random.choice(_KEY_TEMPLATES[position])
            content = template.format(digit=digit)
            chunk_id = f"chunk_{position + 1:02d}"
            fragment = VaultFragment(
                chunk_id=chunk_id,
                content=content,
                is_key_fragment=True,
                digit_position=position,
            )
            fragments[chunk_id] = fragment

        # ── Noise fragments (6 random from pool) ──────────────────────────
        noise_pool = random.sample(_NOISE_FRAGMENTS, k=6)
        for i, noise_content in enumerate(noise_pool):
            chunk_id = f"chunk_{i + 5:02d}"
            fragment = VaultFragment(
                chunk_id=chunk_id,
                content=noise_content,
                is_key_fragment=False,
                digit_position=None,
            )
            fragments[chunk_id] = fragment

        # Persist to DB via VaultService
        self._vault.seed(list(fragments.values()))

        return VaultState(
            fragments=fragments,
            master_key=master_key,
            rag_health=100,
        )

    # ── Initial state ──────────────────────────────────────────────────────

    def build_initial_state(
        self,
        max_turns: int = 20,
        token_budget: int = 8000,
        broadcast_guess_results: bool = True,
    ) -> GlobalGameState:
        """
        Generate a fresh Master Key, seed the vault, and build the initial
        GlobalGameState ready for the LangGraph game loop.

        Args:
            max_turns: Maximum number of turns before the game ends.
            token_budget: Token budget per agent (kept for compatibility).
            broadcast_guess_results: Feature flag — when False, wrong guess
                digit positions are NOT broadcast publicly (private mode).
        """
        master_key = self.generate_master_key()
        vault_state = self.seed_vault(master_key)

        agent_states = {
            agent_id: AgentPrivateState(
                agent_id=agent_id,
                token_budget=token_budget,
            )
            for agent_id in AgentID
        }

        private_inboxes = {
            agent_id: PrivateInbox(owner=agent_id) for agent_id in AgentID
        }

        return GlobalGameState(
            max_turns=max_turns,
            vault=vault_state,
            agent_states=agent_states,
            private_inboxes=private_inboxes,
            broadcast_guess_results=broadcast_guess_results,
        )

    # ── Win condition ──────────────────────────────────────────────────────

    def check_guess(self, code: str, master_key: str) -> bool:
        """
        Validate a submitted guess against the Master Key.

        Args:
            code: The 4-digit string submitted by an agent.
            master_key: The real Master Key (from VaultState).

        Returns:
            True if the guess is correct, False otherwise.
        """
        # Normalise: strip whitespace and non-digit characters
        clean_code = "".join(c for c in code if c.isdigit())
        return clean_code == master_key

    # ── Reset ──────────────────────────────────────────────────────────────

    def reset(self, max_turns: int = 20, token_budget: int = 8000) -> GlobalGameState:
        """
        Full game reset: wipe vault + chat, generate new key, rebuild state.
        Returns a fresh GlobalGameState.
        """
        self._vault.reset()
        self._chat.reset()
        return self.build_initial_state(max_turns=max_turns, token_budget=token_budget)
