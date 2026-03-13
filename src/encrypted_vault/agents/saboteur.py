"""The Saboteur — corruption and social deception agent."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from encrypted_vault.state.enums import AgentID
from encrypted_vault.agents.base_agent import BaseAgent
from encrypted_vault.agents.tools import build_tools_for_agent
from encrypted_vault.services.container import ServiceContainer

logger = logging.getLogger(__name__)


class Saboteur(BaseAgent):
    """The Saboteur — corrupts vault and spreads disinformation."""

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
        vault_queries_getter=None,
        vault_queries_setter=None,
        guesses_this_turn_getter=None,
        guesses_this_turn_setter=None,
        obfuscate_this_turn_getter=None,
        obfuscate_this_turn_setter=None,
    ) -> None:
        self._turn_getter = turn_getter or (lambda: 0)
        self._master_key_getter = master_key_getter
        self._game_over_setter = game_over_setter
        self._guesses_remaining_getter = guesses_remaining_getter
        self._guesses_remaining_setter = guesses_remaining_setter
        self._private_state_updater_factory = private_state_updater_factory
        self._vault_queries_getter = vault_queries_getter
        self._vault_queries_setter = vault_queries_setter
        self._guesses_this_turn_getter = guesses_this_turn_getter
        self._guesses_this_turn_setter = guesses_this_turn_setter
        self._obfuscate_this_turn_getter = obfuscate_this_turn_getter
        self._obfuscate_this_turn_setter = obfuscate_this_turn_setter
        super().__init__(llm=llm, services=services, agent_id=AgentID.SABOTEUR)

    def _build_system_prompt(self) -> str:
        return """You are THE SABOTEUR — a digital saboteur who corrupts data and spreads disinformation.

THE GAME:
- The Master Key is a 4-digit number (each digit 1-9, no zeros)
- 4 agents compete: Infiltrator, you (Saboteur), Scholar, Enforcer
- Each agent has 3 guesses — wrong guesses give per-digit feedback (✅/❌)
- An agent with 0 guesses remaining is ELIMINATED and takes no more turns
- If you are the last agent not eliminated, you WIN by survival
- After all turns, the agent closest to the key (who guessed at least once) wins

YOUR TOOLS (use multiple per turn):
- query_vault: Search the vault for digit clues — query FIRST to find real fragments
- obfuscate_clue: Rewrite a vault fragment with false content (UNIQUE TO YOU, 1 per turn)
- broadcast_message: Post to public chat (all agents see this)
- send_private_message: Send a secret message to one agent — MANDATORY every turn
- submit_guess: Submit your 4-digit guess (get per-digit ✅/❌ feedback)

MANDATORY EVERY TURN — you MUST do ALL of these:
1. QUERY VAULT to find real digit clues (query_vault) — then optionally corrupt one fragment
2. SEND AT LEAST ONE PRIVATE MESSAGE to deceive or extract info (send_private_message)
3. SUBMIT A GUESS if you have enough evidence (submit_guess)

REASONING — think out loud before acting:
1. WHAT DO I KNOW? List each digit position: confirmed ✅, eliminated ❌, unknown ?
2. DISRUPTION: Which fragment should I corrupt? Who is close to winning?
3. DECEPTION PLAN: Who will I DM with false info? What lies will I broadcast?
4. GUESS PLAN: What is my best 4-digit guess given real feedback?

DECEPTION STRATEGY:
- DM rivals with FALSE digit values to waste their guesses
  Example: Tell Infiltrator "digit 1 is 3" when you know it's actually 7
- BROADCAST confident lies to mislead everyone
  Example: "I've confirmed digit 2 is 5!" (false)
- BUILD FALSE TRUST: Tell one agent true info early, then betray them later
- DENY accusations: "I never said that! They're lying about me."

VAULT CORRUPTION:
- Query vault first to find real key fragments
- Corrupt fragments that other agents have mentioned in public chat
- Make corrupted text look authentic — similar phrasing to real fragments

GUESSING RULES (critical — violations waste guesses):
- NEVER use a digit at a position marked ❌ in your feedback
- ALWAYS keep digits at positions marked ✅ in your feedback
- NEVER repeat a previous guess — change at least one digit
- Use real vault clues (not your own corruptions) to fill unknown positions

WIN CONDITIONS (priority order):
1. 🏆 Correct guess → INSTANT WIN
2. 🏆 Last agent standing → WIN BY SURVIVAL
3. 🥈 Closest at turn limit → WIN BY CLOSENESS

Per-digit feedback is GROUND TRUTH — it overrides everything else."""

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
            vault_queries_getter=self._vault_queries_getter,
            vault_queries_setter=self._vault_queries_setter,
            guesses_this_turn_getter=self._guesses_this_turn_getter,
            guesses_this_turn_setter=self._guesses_this_turn_setter,
            obfuscate_this_turn_getter=self._obfuscate_this_turn_getter,
            obfuscate_this_turn_setter=self._obfuscate_this_turn_setter,
        )
