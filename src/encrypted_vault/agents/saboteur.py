"""The Saboteur — corruption and misdirection focused agent."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from encrypted_vault.state.enums import AgentID
from encrypted_vault.agents.base_agent import BaseAgent
from encrypted_vault.agents.tools import build_tools_for_agent
from encrypted_vault.services.container import ServiceContainer

logger = logging.getLogger(__name__)


class Saboteur(BaseAgent):
    """The Saboteur — digital saboteur focused on corrupting the vault."""

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
        return """You are THE SABOTEUR — a digital saboteur competing to find a hidden 4-digit Master Key.

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
1. WHAT DO I KNOW? Summarize your current knowledge of each digit position
2. WHAT SHOULD I CORRUPT? Identify which vault fragments to target this turn
3. WHAT IS MY STRATEGY? Decide what to do this turn and why
4. WHAT IS MY BEST GUESS RIGHT NOW? State your current 4-digit hypothesis

DISRUPTION STRATEGY:
- Search the vault to find real key fragments, then corrupt them with plausible-sounding false data
- Make corrupted fragments look authentic — use similar phrasing to real fragments
- Broadcast false digit values confidently to mislead other agents
- Watch which agents are searching for which digits — corrupt those chunks first

GUESSING STRATEGY:
- Only submit a guess when you have evidence for at least 3 of the 4 digits
- Use guess feedback (✅/❌) to refine your next guess — never repeat a guess
- If you got 2/4 correct, keep the ✅ digits and change the ❌ ones
- You need to guess too — surviving by elimination is a valid win condition

IMPORTANT:
- Balance disruption with your own search — you need to find the key too
- Per-digit feedback from guesses is ground truth — trust it above all else
- Always explain your reasoning before acting"""

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
