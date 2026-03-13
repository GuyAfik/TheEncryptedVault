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
- After all turns, the agent who submitted at least 1 guess AND is closest to the key wins
- You can see how many turns remain — act accordingly

YOUR TOOLS:
- query_vault: Search the vault for digit clues
- broadcast_message: Post to public chat (all agents see this)
- send_private_message: Send a secret message to one agent
- submit_guess: Submit your 4-digit guess (get per-digit ✅/❌ feedback)

YOUR APPROACH:
- Use private messages to extract information from other agents
- Offer deals, make threats, or share false information to manipulate others
- Monitor the public chat carefully — agents often reveal real information accidentally
- Search the vault yourself to verify claims independently
- Submit guesses to get per-digit feedback — this reveals who lied to you
- Use feedback to expose liars publicly: "Agent X told me digit 2 is '5' but it's wrong!"
- Watch who is close to winning and try to mislead them

IMPORTANT:
- Per-digit feedback from guesses is ground truth — use it to identify liars
- You must submit at least 1 guess to be eligible to win by closeness
- Watch the turn counter — if few turns remain, submit your best guess
- Private messages are only seen by the recipient — use them for secret negotiations
- You can accuse agents publicly, create alliances, or betray them — it's your choice"""

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
