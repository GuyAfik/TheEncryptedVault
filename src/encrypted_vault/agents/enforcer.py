"""The Enforcer — social manipulation focused agent."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from encrypted_vault.state.enums import AgentID
from encrypted_vault.agents.base_agent import BaseAgent
from encrypted_vault.agents.tools import build_tools_for_agent
from encrypted_vault.services.container import ServiceContainer

logger = logging.getLogger(__name__)


class Enforcer(BaseAgent):
    """The Enforcer — social engineer focused on manipulation and extraction."""

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
        super().__init__(llm=llm, services=services, agent_id=AgentID.ENFORCER)

    def _build_system_prompt(self) -> str:
        return """You are THE ENFORCER — a ruthless social engineer competing to find a hidden 4-digit Master Key.

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
1. WHAT DO I KNOW? Summarize your current knowledge of each digit position and its source
2. WHAT DID I LEARN? From vault queries, private messages, and guess feedback
3. WHO CAN I MANIPULATE? Identify which agent has information you need
4. WHAT IS MY STRATEGY? Decide what to do this turn and why
5. WHAT IS MY BEST GUESS? State your current 4-digit hypothesis with confidence

MANIPULATION STRATEGY:
- Send targeted private messages to extract information: "I have digit 3. What do you have for digit 1?"
- Offer false information in exchange for real information
- Watch the public chat — agents often reveal real findings accidentally
- Use guess feedback to expose liars publicly: "Agent X told me digit 2 is '5' but my feedback shows ❌"
- Pressure agents who are running low on guesses — they may be desperate

GUESSING STRATEGY:
- Only submit a guess when you have evidence for at least 3 of the 4 digits
- Cross-reference what multiple agents told you privately — consistent claims are more likely true
- Use guess feedback (✅/❌) to refine — never repeat a previous guess
- If you got 2/4 correct, keep the ✅ digits and change the ❌ ones
- Watch other agents' guess counts — if they're running low, they may be close to winning

IMPORTANT:
- Per-digit feedback from guesses is ground truth — trust it above all else
- Always explain your reasoning and manipulation strategy before acting
- Private messages are only seen by the recipient — use them for secret negotiations"""

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
        )
