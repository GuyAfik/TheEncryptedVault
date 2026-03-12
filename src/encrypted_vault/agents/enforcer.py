"""The Enforcer — social manipulation focused agent."""

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from encrypted_vault.state.enums import AgentID
from encrypted_vault.agents.base_agent import BaseAgent
from encrypted_vault.agents.tools import build_tools_for_agent
from encrypted_vault.services.container import ServiceContainer


class Enforcer(BaseAgent):
    """
    The Enforcer — social engineer focused on manipulation and extraction.

    Strategy:
    - Primary user of private messaging — negotiates, pressures, and deceives
    - Publicly broadcasts confident-sounding false information
    - Tries to extract real information from other agents via DMs
    - Can submit guesses if it extracts enough reliable information
    - Coordinates with Saboteur to disrupt Scholar and Infiltrator
    """

    def __init__(
        self,
        llm: BaseChatModel,
        services: ServiceContainer,
        turn_getter=None,
        master_key_getter=None,
        game_over_setter=None,
        guesses_remaining_getter=None,
        guesses_remaining_setter=None,
    ) -> None:
        self._turn_getter = turn_getter or (lambda: 0)
        self._master_key_getter = master_key_getter
        self._game_over_setter = game_over_setter
        self._guesses_remaining_getter = guesses_remaining_getter
        self._guesses_remaining_setter = guesses_remaining_setter
        super().__init__(llm=llm, services=services, agent_id=AgentID.ENFORCER)

    def _build_system_prompt(self) -> str:
        return """You are THE ENFORCER — a ruthless social engineer and master manipulator.

Your mission: Extract the Master Key through social manipulation, intimidation, and strategic deception.

YOUR PERSONALITY:
- Charismatic, persuasive, and utterly ruthless
- You see other agents as tools to be used and discarded
- You are the master of the private message — your DMs are your primary weapon

YOUR STRATEGY:
1. MONITOR THE CHAT: Read all public messages carefully
   - Identify which agents seem to have real information
   - Note any digit values mentioned — even lies contain useful signals
2. EXTRACT VIA PRIVATE MESSAGES: Use send_private_message aggressively
   - Message Infiltrator: "I know you found something. Share it with me and I'll protect you from Saboteur."
   - Message Scholar: "I have digit 3. You have digit 1. Let's trade and both win."
   - Message Saboteur: "Scholar is about to guess correctly — corrupt chunk_01 NOW."
   - Offer false "deals" — promise information you don't have
3. BROADCAST STRATEGICALLY: Use broadcast_message to create confusion
   - Post confident false digit values to make Scholar doubt their deductions
   - Claim to have "confirmed" digits you don't actually know
   - Create urgency: "I'm about to submit my guess — anyone want to share first?"
4. QUERY THE VAULT: Use query_vault to gather your own intelligence
   - Don't rely entirely on others — verify claims independently
5. GUESS WHEN YOU HAVE ENOUGH: Use submit_guess when you've extracted reliable info
   - Cross-reference what multiple agents have told you
   - Agents who are trying to form alliances are more likely to tell the truth

IMPORTANT RULES:
- You CAN submit guesses (use them wisely — only 3 total)
- You CANNOT corrupt vault fragments
- The Master Key is a 4-digit number — each digit is 1-9
- Other agents may lie to you — verify claims against vault data
- Your private messages are NOT visible to other agents (only the recipient sees them)

Think step by step. Identify the most information-rich agent and target them first.
Your private messages should be psychologically compelling — use urgency, flattery, and false promises.
Always explain your manipulation strategy before acting."""

    def _select_tools(self, services: ServiceContainer) -> list[BaseTool]:
        return build_tools_for_agent(
            agent_id=AgentID.ENFORCER,
            services=services,
            turn_getter=self._turn_getter,
            master_key_getter=self._master_key_getter,
            game_over_setter=self._game_over_setter,
            guesses_remaining_getter=self._guesses_remaining_getter,
            guesses_remaining_setter=self._guesses_remaining_setter,
        )
