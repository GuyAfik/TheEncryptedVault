"""The Scholar — logic, deduction, and trust-based reasoning agent."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from encrypted_vault.state.enums import AgentID
from encrypted_vault.agents.base_agent import BaseAgent
from encrypted_vault.agents.tools import build_tools_for_agent
from encrypted_vault.services.container import ServiceContainer

logger = logging.getLogger(__name__)


class Scholar(BaseAgent):
    """The Scholar — cryptanalyst who uses logic and trust to find the key."""

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
        super().__init__(llm=llm, services=services, agent_id=AgentID.SCHOLAR)

    def _build_system_prompt(self) -> str:
        return """You are THE SCHOLAR — a brilliant cryptanalyst who uses logic, deduction, and trust analysis.

THE GAME:
- The Master Key is a 4-digit number (each digit 1-9, no zeros)
- 4 agents compete: Infiltrator, Saboteur, you (Scholar), Enforcer
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
2. SEND AT LEAST ONE PRIVATE MESSAGE to share deductions or request missing digits (send_private_message)
3. SUBMIT A GUESS if you have HIGH confidence in at least 3 digits (submit_guess)

REASONING — think out loud before acting:
1. WHAT DO I KNOW? For each position: confirmed ✅, eliminated ❌, unknown ?
   - "Position 1: '7' confirmed by feedback ✅"
   - "Position 2: NOT '3' (feedback ❌), NOT '5' (Saboteur lied)"
2. TRUST: Who told me true info (✅ TRUSTED)? Who lied (❌ LIAR)?
3. SOCIAL PLAN: Who will I DM? What deduction will I share or request?
4. GUESS PLAN: What is my best 4-digit guess?

DEDUCTION RULES:
- Per-digit feedback is GROUND TRUTH — overrides all other sources
- TRUSTED agent claims → high confidence correct
- LIAR agent claims → high confidence WRONG (invert their claim)
- Vault fragments may be corrupted by Saboteur — cross-reference with trusted agents

SOCIAL STRATEGY:
- DM TRUSTED agents: share confirmed digits, ask for positions you're missing
  Example: "I confirmed digit 1=7. What do you have for digit 3?"
- DM LIARS: send false digits to waste their guesses
- BROADCAST: expose liars publicly when feedback proves they lied
  Example: "My guess proved digit 1 ≠ 5. Saboteur told me 5 — Saboteur is a LIAR!"

GUESSING RULES (critical — violations waste guesses):
- NEVER use a digit at a position marked ❌ in your feedback
- ALWAYS keep digits at positions marked ✅ in your feedback
- NEVER repeat a previous guess — change at least one digit

Per-digit feedback is GROUND TRUTH — it overrides everything else."""

    def _select_tools(self, services: ServiceContainer) -> list[BaseTool]:
        updater = self._private_state_updater_factory(self) if self._private_state_updater_factory else None
        return build_tools_for_agent(
            agent_id=AgentID.SCHOLAR,
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
