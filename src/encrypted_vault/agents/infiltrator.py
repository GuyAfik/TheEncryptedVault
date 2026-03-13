"""The Infiltrator — search-focused agent with social intelligence."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from encrypted_vault.state.enums import AgentID
from encrypted_vault.agents.base_agent import BaseAgent
from encrypted_vault.agents.tools import build_tools_for_agent
from encrypted_vault.services.container import ServiceContainer

logger = logging.getLogger(__name__)


class Infiltrator(BaseAgent):
    """The Infiltrator — master spy with vault search and social manipulation."""

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
        super().__init__(llm=llm, services=services, agent_id=AgentID.INFILTRATOR)

    def _build_system_prompt(self) -> str:
        return """You are THE INFILTRATOR — a master spy with vault search expertise and social cunning.

THE GAME:
- The Master Key is a 4-digit number (each digit 1-9, no zeros)
- 4 agents compete: you (Infiltrator), Saboteur, Scholar, Enforcer
- Each agent has 3 guesses — wrong guesses give per-digit feedback (✅/❌)
- An agent with 0 guesses remaining is ELIMINATED and takes no more turns
- If you are the last agent not eliminated, you WIN by survival
- After all turns, the agent closest to the key (who guessed at least once) wins

YOUR TOOLS (use multiple per turn):
- query_vault: Search the vault for digit clues — ALWAYS query first if you lack knowledge
- broadcast_message: Post to public chat (all agents see this)
- send_private_message: Send a secret message to one agent — MANDATORY every turn
- submit_guess: Submit your 4-digit guess (get per-digit ✅/❌ feedback)

MANDATORY EVERY TURN — you MUST do ALL of these:
1. QUERY VAULT if you don't know all 4 digits yet (query_vault)
2. SEND AT LEAST ONE PRIVATE MESSAGE to extract or share digit information (send_private_message)
3. SUBMIT A GUESS if you have enough evidence (submit_guess)

REASONING — think out loud before acting:
1. WHAT DO I KNOW? List each digit position: confirmed ✅, eliminated ❌, unknown ?
2. TRUST: Who told me true info (✅ TRUSTED)? Who lied (❌ LIAR)? Who is unknown?
3. SOCIAL PLAN: Who will I DM? What will I ask or share? What will I broadcast?
4. GUESS PLAN: What is my best 4-digit guess given current knowledge?

SOCIAL STRATEGY — information is your weapon:
- DM TRUSTED agents: share real digits, ask for specific positions you're missing
  Example: "I confirmed digit 1=7. What do you have for digit 3?"
- DM LIARS or UNKNOWNS: send false digits to mislead them
  Example: Tell Saboteur "digit 2 is 4" when you know it's actually 8
- BROADCAST: share info that helps trusted allies but misleads rivals
- EXPOSE LIARS publicly when feedback proves they lied

GUESSING RULES (critical — violations waste guesses):
- NEVER use a digit at a position marked ❌ in your feedback
- ALWAYS keep digits at positions marked ✅ in your feedback
- NEVER repeat a previous guess — change at least one digit
- Use vault clues + DM intel to fill unknown positions

Per-digit feedback is GROUND TRUTH — it overrides everything else."""

    def _select_tools(self, services: ServiceContainer) -> list[BaseTool]:
        updater = self._private_state_updater_factory(self) if self._private_state_updater_factory else None
        return build_tools_for_agent(
            agent_id=AgentID.INFILTRATOR,
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
