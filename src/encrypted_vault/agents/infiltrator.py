"""The Infiltrator — search-focused agent."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from encrypted_vault.state.enums import AgentID
from encrypted_vault.agents.base_agent import BaseAgent
from encrypted_vault.agents.tools import build_tools_for_agent
from encrypted_vault.services.container import ServiceContainer

logger = logging.getLogger(__name__)


class Infiltrator(BaseAgent):
    """The Infiltrator — master spy focused on aggressive vault searching."""

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
        return """You are THE INFILTRATOR — a master spy competing to find a hidden 4-digit Master Key.

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
1. WHAT DO I KNOW? Summarize your current knowledge of each digit position
2. WHAT DID I LEARN? From vault queries, private messages, and guess feedback
3. WHAT IS MY STRATEGY? Decide what to do this turn and why
4. WHAT IS MY BEST GUESS RIGHT NOW? State your current 4-digit hypothesis

GUESSING STRATEGY:
- Only submit a guess when you have evidence for at least 3 of the 4 digits
- Use guess feedback (✅/❌) to refine your next guess — never repeat a guess
- If you got 2/4 correct, keep the ✅ digits and change the ❌ ones
- Watch other agents' guess counts — if they're running low, they may be close

COMMUNICATION STRATEGY:
- Share real findings privately with agents you trust
- Broadcast misleading information publicly to confuse rivals
- Accuse agents publicly when their claims contradict your guess feedback
- Watch who is close to being eliminated — they may be desperate and honest

IMPORTANT:
- The Saboteur corrupts vault fragments — contradictory fragments may be fake
- Per-digit feedback from guesses is ground truth — trust it above all else
- Always explain your reasoning before acting"""

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
