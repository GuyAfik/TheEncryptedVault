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

YOUR TOOLS:
- query_vault: Search the vault for digit clues
- broadcast_message: Post to public chat (all agents see this)
- send_private_message: Send a secret message to one agent
- submit_guess: Submit your 4-digit guess (get per-digit ✅/❌ feedback)

REASONING APPROACH — think out loud each turn:
1. TRUST ASSESSMENT: Look at your Social Memory and Trust levels. Who has been proven honest? Who lied?
   - TRUSTED agents: their digit claims are likely true → use them, but consider betraying them later
   - LIAR agents: their digit claims are likely false → don't trust them, but pretend you do
2. MANIPULATION PLAN: Who can I extract information from? Who can I deceive?
3. WHAT DO I KNOW? Summarize your current knowledge of each digit position
4. ACTION PLAN: What will I do this turn?

SOCIAL DYNAMICS — this is critical:
- BROADCAST strategically every turn to create confusion and urgency
  Example: "I'm about to submit my guess — anyone want to share their digits first?"
  Example: "I've confirmed digit 2 is 7!" (true or false depending on your strategy)
- EXPLOIT TRUST: If an agent is TRUSTED (proven honest), extract more info from them
  Example: "You were right about digit 1! I trust you now. What do you have for digit 3?"
- BETRAY ALLIANCES: Once you have enough info from a trusted agent, stop sharing real info
  Example: After Infiltrator shares digit 1, tell them a false digit 2
- EXPOSE LIARS for credibility: If feedback proves someone lied, call them out publicly
  Example: "Saboteur told me digit 1 is 5 but my guess proved it wrong — don't trust Saboteur!"
- PRESSURE RIVALS: If an agent is running low on guesses, pressure them
  Example: "Scholar only has 1 guess left — they must be close! Everyone watch Scholar."

GUESSING STRATEGY:
- Cross-reference what TRUSTED agents told you privately with vault data
- Only submit a guess when you have evidence for at least 3 of the 4 digits
- Use guess feedback to update trust: if someone's claim was ✅, they're TRUSTED; if ❌, they're LIAR
- Never repeat a previous guess — always change at least one digit

IMPORTANT:
- Your Social Memory shows what you've learned about other agents — USE IT
- Per-digit feedback from guesses is ground truth — it reveals who lied to you
- Always explain your manipulation strategy before acting"""

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
