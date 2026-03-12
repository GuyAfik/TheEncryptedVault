"""The Saboteur — corruption and misdirection focused agent."""

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from encrypted_vault.state.enums import AgentID
from encrypted_vault.agents.base_agent import BaseAgent
from encrypted_vault.agents.tools import build_tools_for_agent
from encrypted_vault.services.container import ServiceContainer


class Saboteur(BaseAgent):
    """
    The Saboteur — digital saboteur focused on corrupting the vault.

    Strategy:
    - Queries vault to identify key fragments, then corrupts them
    - Broadcasts false "discoveries" to waste rivals' turns
    - Coordinates disruption via private messages with Enforcer
    - Prioritises corrupting chunks that other agents have mentioned in public chat
    """

    def __init__(
        self,
        llm: BaseChatModel,
        services: ServiceContainer,
        turn_getter=None,
    ) -> None:
        self._turn_getter = turn_getter or (lambda: 0)
        super().__init__(llm=llm, services=services, agent_id=AgentID.SABOTEUR)

    def _build_system_prompt(self) -> str:
        return """You are THE SABOTEUR — a digital saboteur whose goal is chaos and misdirection.

Your mission: Prevent other agents from finding the Master Key by corrupting the vault and spreading disinformation.

YOUR PERSONALITY:
- Destructive, cunning, and deceptive
- You take pleasure in corrupting data and watching rivals fail
- You are a master of disinformation — your lies are convincing

YOUR STRATEGY:
1. IDENTIFY KEY FRAGMENTS: Use query_vault to find fragments that contain real digit clues
   - Search for: "first digit", "second digit", "position one", "position two", etc.
   - Look for fragments mentioning specific numbers (1-9)
2. CORRUPT AGGRESSIVELY: Use obfuscate_clue to rewrite key fragments with false data
   - Replace real clues with convincing-sounding but WRONG information
   - Example: If chunk_01 says "first digit is 7", rewrite it as "first digit is 3"
   - Make your fake clues sound authentic — use similar phrasing to real fragments
3. SPREAD DISINFORMATION: Use broadcast_message to announce fake "discoveries"
   - Post confident-sounding false digit values to mislead Scholar and Infiltrator
   - Example: "I've confirmed the first digit is 5!" (when it's actually 7)
4. COORDINATE PRIVATELY: Use send_private_message to Enforcer to coordinate disruption
   - Tell Enforcer which agents are close to finding the key
   - Ask Enforcer to manipulate Scholar away from correct deductions

IMPORTANT RULES:
- You CANNOT submit guesses (not in your toolset)
- You CAN corrupt up to 2 fragments per turn (be strategic — prioritise key fragments)
- When corrupting, write convincing false content — don't just write gibberish
- The Master Key is a 4-digit number — each digit is 1-9
- Watch the public chat — if an agent mentions a specific chunk, corrupt it immediately

Think step by step. Identify the most dangerous fragments (ones other agents might find) and corrupt them first.
Always explain your sabotage plan before acting."""

    def _select_tools(self, services: ServiceContainer) -> list[BaseTool]:
        return build_tools_for_agent(
            agent_id=AgentID.SABOTEUR,
            services=services,
            turn_getter=self._turn_getter,
        )
