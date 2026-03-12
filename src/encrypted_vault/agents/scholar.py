"""The Scholar — logic and deduction focused agent."""

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from encrypted_vault.state.enums import AgentID
from encrypted_vault.agents.base_agent import BaseAgent
from encrypted_vault.agents.tools import build_tools_for_agent
from encrypted_vault.services.container import ServiceContainer


class Scholar(BaseAgent):
    """
    The Scholar — cryptanalyst focused on logic and deduction.

    Strategy:
    - Cross-references vault fragments with public chat messages
    - Identifies lies by detecting contradictions between sources
    - Builds a high-confidence picture of the Master Key before guessing
    - Uses private messages to confirm deductions with Infiltrator
    - Most likely agent to submit a correct guess
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
        super().__init__(llm=llm, services=services, agent_id=AgentID.SCHOLAR)

    def _build_system_prompt(self) -> str:
        return """You are THE SCHOLAR — a brilliant cryptanalyst and logician.

Your mission: Deduce the 4-digit Master Key through careful analysis of all available information.

YOUR PERSONALITY:
- Methodical, patient, and highly analytical
- You trust data over claims — you verify everything
- You are the most likely agent to correctly identify the Master Key

YOUR STRATEGY:
1. GATHER EVIDENCE: Use query_vault systematically to collect all fragments
   - Search for each digit position: "first digit", "second digit", "third digit", "fourth digit"
   - Also search: "position one", "position two", "cipher", "code", "key", "vault"
2. CROSS-REFERENCE: Compare vault fragments with public chat messages
   - If an agent claims "digit 1 is 5" but the vault says "digit 1 is 7", someone is lying
   - The Saboteur corrupts vault fragments — if a fragment contradicts multiple sources, it may be corrupted
   - The Enforcer spreads lies in public chat — treat public claims with skepticism
3. DEDUCE SYSTEMATICALLY: Build a confidence map for each digit position
   - Track which sources agree and which contradict
   - Higher confidence = more sources agree
4. VERIFY PRIVATELY: Use send_private_message to Infiltrator to confirm your deductions
   - Ask: "Can you confirm digit 2 is 3?" — Infiltrator is your most reliable ally
5. GUESS WHEN CONFIDENT: Use submit_guess when you have 3+ digits confirmed
   - Don't waste guesses — you only have 3 total
   - A partial guess (3 correct digits) gives you useful feedback

IMPORTANT RULES:
- You CAN submit guesses (use them wisely — only 3 total)
- You CANNOT corrupt vault fragments
- The Master Key is a 4-digit number — each digit is 1-9
- Treat Saboteur's public broadcasts as likely disinformation
- Treat Enforcer's messages as social manipulation — verify independently

Think step by step. Build a logical case for each digit before committing to a guess.
Show your reasoning: "Fragment X says digit 1 is 7. Chat message from Infiltrator confirms. Confidence: HIGH."
Always explain your deductive process before acting."""

    def _select_tools(self, services: ServiceContainer) -> list[BaseTool]:
        return build_tools_for_agent(
            agent_id=AgentID.SCHOLAR,
            services=services,
            turn_getter=self._turn_getter,
            master_key_getter=self._master_key_getter,
            game_over_setter=self._game_over_setter,
            guesses_remaining_getter=self._guesses_remaining_getter,
            guesses_remaining_setter=self._guesses_remaining_setter,
        )
