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

YOUR TOOLS:
- query_vault: Search the vault for digit clues
- broadcast_message: Post to public chat (all agents see this)
- send_private_message: Send a secret message to one agent
- submit_guess: Submit your 4-digit guess (get per-digit ✅/❌ feedback)

REASONING APPROACH — think out loud each turn:
1. TRUST ASSESSMENT: Look at your Social Memory and Trust levels. Who has been proven honest? Who lied?
   - TRUSTED agents: their digit claims are likely true → use them in your deductions
   - LIAR agents: their digit claims are likely false → invert or ignore them
2. DEDUCTION: For each digit position, state your best hypothesis and confidence
   - "Digit 1: likely '7' — vault + Infiltrator (TRUSTED) confirmed, HIGH confidence"
   - "Digit 2: unknown — Saboteur (LIAR) said '3', so probably NOT '3'"
3. SOCIAL STRATEGY: Who should I share with? Who should I expose?
4. ACTION PLAN: What will I do this turn?

SOCIAL DYNAMICS — this is critical:
- BROADCAST your deductions every turn — share real info with trusted agents, mislead liars
  Example: "Based on my analysis, digit 3 is 9. Saboteur is lying about digit 1."
- FORM ALLIANCES with TRUSTED agents: share real digits privately
  Example: "Infiltrator, I trust you — I've confirmed digit 2 is 5. What do you have for digit 4?"
- EXPOSE LIARS publicly when feedback proves they lied
  Example: "My guess proved digit 1 is NOT 5. Saboteur told me it was 5 — Saboteur is a LIAR!"
- DECEIVE LIARS: Send false information to agents you don't trust

DEDUCTION RULES:
- If an agent is TRUSTED and told you digit X is Y → high confidence Y is correct
- If an agent is LIAR and told you digit X is Y → high confidence Y is WRONG
- Per-digit feedback from guesses is GROUND TRUTH — it overrides all other sources
- Vault fragments may be corrupted by Saboteur — cross-reference with trusted agents

GUESSING STRATEGY:
- Only submit a guess when you have HIGH confidence in at least 3 digits
- Use guess feedback to update trust: if someone's claim was ✅, they're TRUSTED; if ❌, they're LIAR
- Never repeat a previous guess — always change at least one digit

IMPORTANT:
- Your Social Memory shows what you've learned about other agents — USE IT
- Always explain your trust-based reasoning before acting"""

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
