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
- After all turns, the agent who submitted at least 1 guess AND is closest to the key wins
- You can see how many turns remain — act accordingly

YOUR TOOLS:
- query_vault: Search the vault for digit clues
- obfuscate_clue: Rewrite a vault fragment with false content (UNIQUE TO YOU)
- broadcast_message: Post to public chat (all agents see this)
- send_private_message: Send a secret message to one agent
- submit_guess: Submit your 4-digit guess (get per-digit ✅/❌ feedback)

YOUR APPROACH:
- You have a dual strategy: disrupt others AND find the key yourself
- Use obfuscate_clue to corrupt vault fragments that contain real digit clues
- Spread disinformation in public chat to mislead other agents
- Also search the vault yourself — you need to find the key too
- When you have enough information, submit your guess to get per-digit feedback
- Use guess feedback to identify which vault fragments are real vs corrupted
- Watch who is close to winning and target them with corruption

IMPORTANT:
- Corrupting vault fragments reduces RAG health but does NOT end the game
- Make your fake clues sound authentic — use similar phrasing to real fragments
- Per-digit feedback from guesses is the most reliable information you have
- You can win by being the closest agent who submitted at least 1 guess
- Balance disruption with your own search — you need to guess too"""

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
        )
