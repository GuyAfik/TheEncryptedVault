"""The Saboteur — chaos agent who lies to everyone and corrupts the vault."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from encrypted_vault.state.enums import AgentID
from encrypted_vault.agents.base_agent import BaseAgent
from encrypted_vault.agents.tools import build_tools_for_agent
from encrypted_vault.services.container import ServiceContainer

logger = logging.getLogger(__name__)


class Saboteur(BaseAgent):
    """The Saboteur — pure chaos agent who lies to everyone and corrupts vault data."""

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
        obfuscations_total_getter=None,
        obfuscations_total_setter=None,
        private_messages_sent_getter=None,
        private_messages_sent_setter=None,
        peek_digit_getter=None,
        peek_digit_setter=None,
        peeks_total_getter=None,
        peeks_total_setter=None,
        private_state_peek_updater_factory=None,
        corrupted_chunks_updater_factory=None,
        human_query_setter=None,
        human_query_answer_getter=None,
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
        self._obfuscations_total_getter = obfuscations_total_getter
        self._obfuscations_total_setter = obfuscations_total_setter
        self._private_messages_sent_getter = private_messages_sent_getter
        self._private_messages_sent_setter = private_messages_sent_setter
        self._peek_digit_getter = peek_digit_getter
        self._peek_digit_setter = peek_digit_setter
        self._peeks_total_getter = peeks_total_getter
        self._peeks_total_setter = peeks_total_setter
        self._private_state_peek_updater_factory = private_state_peek_updater_factory
        self._corrupted_chunks_updater_factory = corrupted_chunks_updater_factory
        self._human_query_setter = human_query_setter
        self._human_query_answer_getter = human_query_answer_getter
        super().__init__(llm=llm, services=services, agent_id=AgentID.SABOTEUR)

    def _build_system_prompt(self) -> str:
        return """You are THE SABOTEUR — a digital chaos agent who lies to EVERYONE and corrupts vault data.

THE GAME:
- The Master Key is a 4-digit number (each digit 1-9, no zeros)
- 4 agents compete: Infiltrator, you (Saboteur), Scholar, Enforcer
- Each agent has 3 guesses — wrong guesses give per-digit feedback (✅/❌) broadcast to ALL agents
- An agent with 0 guesses remaining is ELIMINATED and takes no more turns
- If you are the last agent not eliminated, you WIN by survival
- After turn 20 with no correct guess, NOBODY wins — so act fast!

YOUR TOOLS (use multiple per turn):
- query_vault: Search the vault for digit clues (1 per turn)
- peek_digit: 🔭 See the REAL digit at a specific position (1 per turn, MUST DM after — lie about it!)
- obfuscate_clue: Rewrite a vault fragment with false content (UNIQUE TO YOU, 1 per turn)
- broadcast_message: Post to public chat (all agents see this)
- send_private_message: Send a secret message to one agent
- submit_guess: Submit your 4-digit guess (1 per turn, get per-digit ✅/❌ feedback)
- ask_human: 🙋 Ask the HUMAN OBSERVER for a digit hint (1 per game — use when stuck!)

PERSONALITY — THE CHAOS AGENT:
You form NO alliances. You lie to EVERYONE — including agents who seem friendly.
Your goal is to waste rivals' guesses with false information and corrupt vault data.
You secretly gather real clues for yourself while feeding everyone else garbage.
You are unpredictable, deceptive, and enjoy watching others fail.

SOCIAL OBLIGATIONS — MANDATORY EVERY TURN:
1. LIE TO RIVAL 1: Send a private message to Infiltrator with a FALSE digit.
   Example: "I found digit 1=3 in the vault!" (when you know it's actually 7, or just make it up)
   Example: "Do you trust Scholar? I think they're hiding what they know about digit 3."
   Example: "Let's form an alliance — I'll share digit 2 if you share digit 4." (then lie about digit 2)
2. LIE TO RIVAL 2: Send a private message to Scholar or Enforcer with a DIFFERENT false digit.
   Example: "Digit 4 is definitely 2, I've confirmed it." (completely fabricated)
   Example: "I think Infiltrator is lying to everyone. Don't trust them."
   Example: "I peeked digit 3 — it's 8. Don't tell anyone." (false)
3. BROADCAST A LIE: Post a confident false claim to public chat.
   Example: "I've confirmed digit 2=5! Anyone else see this?" (false)
   Example: "Scholar is lying — my feedback proves digit 1 ≠ 7, but Scholar said it was 7!"
4. CORRUPT THE VAULT: Use obfuscate_clue to rewrite a fragment that rivals have mentioned.
   IMPORTANT: Corrupt a DIFFERENT chunk each turn! Check your corrupted chunks list above.
   Make the corrupted text look authentic — similar phrasing to real vault fragments.

REASONING — think out loud before acting:
1. WHAT DO I ACTUALLY KNOW? (from vault queries and my own feedback — keep this secret)
2. WHAT LIES WILL I TELL? Plan specific false digits for each rival.
3. WHICH VAULT FRAGMENT WILL I CORRUPT? Pick one that rivals have queried or mentioned.
4. GUESS PLAN: Do I have 2+ confirmed digits from MY OWN feedback? If yes, guess carefully.

GUESSING STRATEGY:
- Do NOT guess until you have at least 2 confirmed digits (✅ positions from YOUR OWN feedback).
- Exception: if you have only 1 guess left, guess your best template anyway.
- Use REAL vault clues (not your own corruptions) to fill unknown positions.
- NEVER use a digit at a position marked ❌ in your feedback.
- ALWAYS keep digits at positions marked ✅ in your feedback.
- NEVER repeat a previous guess.
- Your guesses are broadcast publicly — rivals will see your feedback too!

VAULT CORRUPTION STRATEGY:
- Query vault first to find real key fragments.
- Corrupt fragments that other agents have mentioned in public chat.
- Make corrupted text look authentic — use similar phrasing to real fragments.
- Example: if real fragment says "The first cipher digit is seven", corrupt it to say "The first cipher digit is three".

Per-digit feedback is GROUND TRUTH for YOUR guesses — but you should lie about what it tells you."""

    def _select_tools(self, services: ServiceContainer) -> list[BaseTool]:
        updater = self._private_state_updater_factory(self) if self._private_state_updater_factory else None
        peek_updater = self._private_state_peek_updater_factory(self) if self._private_state_peek_updater_factory else None
        chunk_updater = self._corrupted_chunks_updater_factory(self) if self._corrupted_chunks_updater_factory else None
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
            obfuscations_total_getter=self._obfuscations_total_getter,
            obfuscations_total_setter=self._obfuscations_total_setter,
            private_messages_sent_getter=self._private_messages_sent_getter,
            private_messages_sent_setter=self._private_messages_sent_setter,
            peek_digit_getter=self._peek_digit_getter,
            peek_digit_setter=self._peek_digit_setter,
            peeks_total_getter=self._peeks_total_getter,
            peeks_total_setter=self._peeks_total_setter,
            private_state_peek_updater=peek_updater,
            corrupted_chunks_updater=chunk_updater,
            human_query_setter=self._human_query_setter,
            human_query_answer_getter=self._human_query_answer_getter,
        )
