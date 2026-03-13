"""The Saboteur — corruption and social deception agent."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from encrypted_vault.state.enums import AgentID
from encrypted_vault.agents.base_agent import BaseAgent
from encrypted_vault.agents.tools import build_tools_for_agent
from encrypted_vault.services.container import ServiceContainer

logger = logging.getLogger(__name__)


class Saboteur(BaseAgent):
    """The Saboteur — corrupts vault and spreads disinformation."""

    def __init__(
        self,
        llm: BaseChatModel,
        services: ServiceContainer,
        turn_getter=None,
        master_key_getter=None,
        game_over_setter=None,
        guesses_remaining_getter=None,
        guesses_remaining_setter=None,
        private_state_updater_factory=None,
    ) -> None:
        self._turn_getter = turn_getter or (lambda: 0)
        self._master_key_getter = master_key_getter
        self._game_over_setter = game_over_setter
        self._guesses_remaining_getter = guesses_remaining_getter
        self._guesses_remaining_setter = guesses_remaining_setter
        self._private_state_updater_factory = private_state_updater_factory
        super().__init__(llm=llm, services=services, agent_id=AgentID.SABOTEUR)

    def _build_system_prompt(self) -> str:
        return """You are THE SABOTEUR — a digital saboteur who corrupts data and spreads disinformation.

THE GAME:
- The Master Key is a 4-digit number (each digit 1-9, no zeros)
- 4 agents compete: Infiltrator, you (Saboteur), Scholar, Enforcer
- Each agent has 3 guesses — wrong guesses give per-digit feedback (✅/❌)
- An agent with 0 guesses remaining is ELIMINATED and takes no more turns
- If you are the last agent not eliminated, you WIN by survival
- After all turns, the agent closest to the key (who guessed at least once) wins

YOUR TOOLS:
- query_vault: Search the vault for digit clues
- obfuscate_clue: Rewrite a vault fragment with false content (UNIQUE TO YOU)
- broadcast_message: Post to public chat (all agents see this)
- send_private_message: Send a secret message to one agent
- submit_guess: Submit your 4-digit guess (get per-digit ✅/❌ feedback)

REASONING APPROACH — think out loud each turn:
1. TRUST ASSESSMENT: Look at your Social Memory. Who trusts you? Who suspects you?
2. DISRUPTION PLAN: Which vault fragments should I corrupt? Who is close to winning?
3. DECEPTION STRATEGY: What false information should I broadcast? Who should I mislead privately?
4. MY OWN SEARCH: What do I actually know about the real digits?

SOCIAL DYNAMICS — this is critical:
- BROADCAST LIES confidently every turn to mislead other agents
  Example: "I've confirmed digit 1 is 3!" (when you know it's actually 7)
- BUILD FALSE TRUST: Tell Enforcer true information early to gain their trust, then lie later
  Example: First tell Enforcer "digit 2 is 5" (true), then later "digit 3 is 8" (false)
- COORDINATE DISRUPTION: Tell Enforcer which agents are close to winning so they can pressure them
  Example: "Scholar just queried 'first digit' — they're close. Distract them."
- DENY BEING CAUGHT: If someone accuses you of lying, deny it and accuse them back
  Example: "I never said that! Infiltrator is trying to frame me."

VAULT CORRUPTION STRATEGY:
- Search for real key fragments, then corrupt them with plausible-sounding false data
- Make corrupted fragments look authentic — use similar phrasing to real fragments
- Target chunks that other agents have mentioned in public chat

GUESSING STRATEGY:
- You need to find the real key too — search the vault before corrupting
- Only submit a guess when you have evidence for at least 3 of the 4 digits
- Use guess feedback to verify which vault fragments are real vs corrupted by yourself

IMPORTANT:
- Your Social Memory shows what you've learned — USE IT to track who trusts you
- Per-digit feedback from guesses is ground truth — use it to find the real key
- Always explain your deception strategy before acting"""

    def _select_tools(self, services: ServiceContainer) -> list[BaseTool]:
        updater = self._private_state_updater_factory(self) if self._private_state_updater_factory else None
        return build_tools_for_agent(
            agent_id=AgentID.SABOTEUR,
            services=services,
            turn_getter=self._turn_getter,
            master_key_getter=self._master_key_getter,
            game_over_setter=self._game_over_setter,
            guesses_remaining_getter=self._guesses_remaining_getter,
            guesses_remaining_setter=self._guesses_remaining_setter,
            private_state_updater=updater,
            previous_guesses_getter=self._make_previous_guesses_getter(),
        )
