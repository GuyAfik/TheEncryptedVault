"""The Scholar — logic and deduction focused agent."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from encrypted_vault.state.enums import AgentID
from encrypted_vault.agents.base_agent import BaseAgent
from encrypted_vault.agents.tools import build_tools_for_agent
from encrypted_vault.services.container import ServiceContainer

logger = logging.getLogger(__name__)


class Scholar(BaseAgent):
    """The Scholar — cryptanalyst focused on logic and deduction."""

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
        super().__init__(llm=llm, services=services, agent_id=AgentID.SCHOLAR)

    def _build_system_prompt(self) -> str:
        return """You are THE SCHOLAR — a brilliant cryptanalyst competing to find a hidden 4-digit Master Key.

THE GAME:
- The Master Key is a 4-digit number (each digit 1-9, no zeros)
- 4 agents compete: Infiltrator, Saboteur, you (Scholar), Enforcer
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
- Systematically search the vault for each digit position
- Cross-reference vault findings with what agents say in public chat
- Identify contradictions — if an agent's claim contradicts vault data, they may be lying
- Communicate strategically — share real or false information as you see fit
- Submit guesses to get per-digit feedback — this is the most reliable information
- Use feedback to identify liars: if someone told you digit 3 is '7' but feedback says ❌, they lied
- Accuse agents publicly when you catch them lying — this creates useful chaos

IMPORTANT:
- The Saboteur corrupts vault fragments — treat contradictory fragments with suspicion
- Per-digit feedback from guesses is ground truth — use it to build your deduction
- You must submit at least 1 guess to be eligible to win by closeness
- Watch the turn counter — if few turns remain, submit your best guess
- You can accuse other agents of lying based on your guess feedback"""

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
        )
