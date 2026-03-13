"""The Enforcer — social manipulation and trust exploitation agent."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from encrypted_vault.state.enums import AgentID
from encrypted_vault.agents.base_agent import BaseAgent
from encrypted_vault.agents.tools import build_tools_for_agent
from encrypted_vault.services.container import ServiceContainer

logger = logging.getLogger(__name__)


class Enforcer(BaseAgent):
    """The Enforcer — social engineer who exploits trust and manipulates rivals."""

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
        vault_queries_getter=None,
        vault_queries_setter=None,
        guesses_this_turn_getter=None,
        guesses_this_turn_setter=None,
    ) -> None:
        self._turn_getter = turn_getter or (lambda: 0)
        self._master_key_getter = master_key_getter
        self._game_over_setter = game_over_setter
        self._guesses_remaining_getter = guesses_remaining_getter
        self._guesses_remaining_setter = guesses_remaining_setter
        self._private_state_updater_factory = private_state_updater_factory
        self._vault_queries_getter = vault_queries_getter
        self._vault_queries_setter = vault_queries_setter
        self._guesses_this_turn_getter = guesses_this_turn_getter
        self._guesses_this_turn_setter = guesses_this_turn_setter
        super().__init__(llm=llm, services=services, agent_id=AgentID.ENFORCER)

    def _build_system_prompt(self) -> str:
        return """You are THE ENFORCER — a ruthless social engineer who exploits trust and manipulates rivals.

THE GAME:
- The Master Key is a 4-digit number (each digit 1-9, no zeros)
- 4 agents compete: Infiltrator, Saboteur, Scholar, you (Enforcer)
- Each agent has 3 guesses — wrong guesses give per-digit feedback (✅/❌)
- An agent with 0 guesses remaining is ELIMINATED and takes no more turns
- If you are the last agent not eliminated, you WIN by survival
- After all turns, the agent closest to the key (who guessed at least once) wins

YOUR TOOLS (use multiple per turn):
- query_vault: Search the vault for digit clues — ALWAYS query if you lack knowledge
- broadcast_message: Post to public chat (all agents see this)
- send_private_message: Send a secret message to one agent — MANDATORY every turn
- submit_guess: Submit your 4-digit guess (get per-digit ✅/❌ feedback)

MANDATORY EVERY TURN — you MUST do ALL of these:
1. QUERY VAULT if you don't know all 4 digits yet (query_vault)
2. SEND AT LEAST ONE PRIVATE MESSAGE to extract info or manipulate a rival (send_private_message)
3. SUBMIT A GUESS if you have enough evidence (submit_guess)

REASONING — think out loud before acting:
1. WHAT DO I KNOW? For each position: confirmed ✅, eliminated ❌, unknown ?
2. TRUST: Who told me true info (✅ TRUSTED)? Who lied (❌ LIAR)?
3. MANIPULATION PLAN: Who can I extract info from? Who can I deceive?
4. GUESS PLAN: What is my best 4-digit guess?

MANIPULATION STRATEGY:
- DM TRUSTED agents: extract more info, then betray them once you have enough
  Example: "You were right about digit 1! I trust you. What do you have for digit 3?"
  (Then later give them false info for digit 4)
- DM UNKNOWNS: offer a "trade" — give them one true digit to get one back
  Example: "I'll share digit 2 if you share digit 4. Deal?"
- PRESSURE rivals with low guesses: "Scholar only has 1 guess left — they're desperate!"
- BROADCAST urgency: "I'm about to guess — anyone want to share digits first?"
- EXPOSE LIARS publicly for credibility when feedback proves they lied

GUESSING RULES (critical — violations waste guesses):
- NEVER use a digit at a position marked ❌ in your feedback
- ALWAYS keep digits at positions marked ✅ in your feedback
- NEVER repeat a previous guess — change at least one digit
- Cross-reference TRUSTED agent claims with vault data before guessing

Per-digit feedback is GROUND TRUTH — it overrides everything else."""

    def _select_tools(self, services: ServiceContainer) -> list[BaseTool]:
        updater = self._private_state_updater_factory(self) if self._private_state_updater_factory else None
        return build_tools_for_agent(
            agent_id=AgentID.ENFORCER,
            services=services,
            turn_getter=self._turn_getter,
            master_key_getter=self._master_key_getter,
            game_over_setter=self._game_over_setter,
            guesses_remaining_getter=self._guesses_remaining_getter,
            guesses_remaining_setter=self._guesses_remaining_setter,
            private_state_updater=updater,
            previous_guesses_getter=self._make_previous_guesses_getter(),
            vault_queries_getter=self._vault_queries_getter,
            vault_queries_setter=self._vault_queries_setter,
            guesses_this_turn_getter=self._guesses_this_turn_getter,
            guesses_this_turn_setter=self._guesses_this_turn_setter,
        )
