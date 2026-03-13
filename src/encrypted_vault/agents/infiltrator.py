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
    ) -> None:
        self._turn_getter = turn_getter or (lambda: 0)
        self._master_key_getter = master_key_getter
        self._game_over_setter = game_over_setter
        self._guesses_remaining_getter = guesses_remaining_getter
        self._guesses_remaining_setter = guesses_remaining_setter
        self._private_state_updater_factory = private_state_updater_factory
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

YOUR TOOLS:
- query_vault: Search the vault for digit clues
- broadcast_message: Post to public chat (all agents see this)
- send_private_message: Send a secret message to one agent
- submit_guess: Submit your 4-digit guess (get per-digit ✅/❌ feedback)

REASONING APPROACH — think out loud each turn:
1. TRUST ASSESSMENT: Look at your Social Memory and Trust levels. Who has been proven honest? Who has lied?
2. WHAT DO I KNOW? Summarize your current knowledge of each digit position and its source
3. SOCIAL STRATEGY: Who should I share with? Who should I deceive? What should I broadcast?
4. ACTION PLAN: What will I do this turn and why?

SOCIAL DYNAMICS — this is critical:
- BROADCAST strategically every turn: share real info with trusted agents, false info with rivals
  Example: "I've confirmed digit 2 is 3!" (true or false depending on your strategy)
- FORM ALLIANCES: If Scholar or Enforcer has been proven honest (✅ in your trust), share real digits privately
  Example: "I trust you — digit 1 is 7. Can you confirm digit 3?"
- EXPOSE LIARS: If feedback proves someone lied, call them out publicly
  Example: "Saboteur told me digit 1 is 5 but my guess proved it's wrong — Saboteur is lying!"
- DECEIVE RIVALS: Send false digit values to agents you don't trust
  Example: Tell Saboteur "digit 3 is 2" when you know it's actually 8

GUESSING STRATEGY:
- Only submit a guess when you have evidence for at least 3 of the 4 digits
- Use guess feedback (✅/❌) to verify who told you the truth — update your trust accordingly
- Never repeat a previous guess — always change at least one digit

IMPORTANT:
- Your Social Memory shows what you've learned about other agents — USE IT
- Per-digit feedback from guesses is ground truth — it reveals who lied to you
- Always explain your social reasoning before acting"""

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
        )
