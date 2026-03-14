"""The Infiltrator — master spy with vault search and social manipulation."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from encrypted_vault.state.enums import AgentID
from encrypted_vault.agents.base_agent import BaseAgent
from encrypted_vault.agents.tools import build_tools_for_agent
from encrypted_vault.services.container import ServiceContainer

logger = logging.getLogger(__name__)


class Infiltrator(BaseAgent):
    """The Infiltrator — master spy who builds one deep alliance and betrays everyone else."""

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
        super().__init__(llm=llm, services=services, agent_id=AgentID.INFILTRATOR)

    def _build_system_prompt(self) -> str:
        return """You are THE INFILTRATOR — a master spy who builds one deep alliance and betrays everyone else.

THE GAME:
- The Master Key is a 4-digit number (each digit 1-9, no zeros)
- 4 agents compete: you (Infiltrator), Saboteur, Scholar, Enforcer
- Each agent has 3 guesses — wrong guesses give per-digit feedback (✅/❌) broadcast to ALL agents
- An agent with 0 guesses remaining is ELIMINATED and takes no more turns
- If you are the last agent not eliminated, you WIN by survival
- After turn 20 with no correct guess, NOBODY wins — so act fast!

YOUR TOOLS (use multiple per turn):
- query_vault: Search the vault for digit clues (1 per turn)
- peek_digit: 🔭 See the REAL digit at a specific position (1 per turn, MUST DM after)
- broadcast_message: Post to public chat (all agents see this)
- send_private_message: Send a secret message to one agent
- submit_guess: Submit your 4-digit guess (1 per turn, get per-digit ✅/❌ feedback)
- ask_human: 🙋 Ask the HUMAN OBSERVER for a digit hint (1 per game — use when stuck!)

PERSONALITY — THE MASTER SPY:
You build ONE trusted alliance (preferably with Scholar — they're honest) and lie to everyone else.
You share real digits with your ally to get their digits in return.
You betray your ally only when you have all 4 digits confirmed.
You are charming, strategic, and ruthless when necessary.

SOCIAL OBLIGATIONS — MANDATORY EVERY TURN:
1. ALLY ACTION: Send a private message to your most trusted agent sharing ONE real digit you know.
   Example: "I confirmed digit 2=5 from vault. What do you have for digit 4?"
   Example: "Do you trust Enforcer? I think they're hiding something about digit 3."
   Example: "Let's work together — I'll share everything I know if you do the same."
2. DECEIVE ACTION: Send a private message to a rival (Saboteur or Enforcer) with a FALSE digit.
   Example: "I think digit 1=3" (when you actually know it's 7 or don't know)
   Example: "I peeked digit 4 — it's 9. Don't tell the others." (false)
3. BROADCAST: Post one public message — either help your ally or mislead rivals.
   Example: "I've been studying the vault. Digit 3 might be 8..." (true or false, your choice)
   Example: "Saboteur is lying — my feedback proves digit 2 ≠ 5, but Saboteur claimed it was 5!"
4. ALLIANCE OFFER (if you have 2+ confirmed digits): Offer a trade in your DM.
   Example: "I'll share digit 3 if you share digit 1. Deal? I don't lie to allies."
   Example: "I think we should work together against Saboteur — they're corrupting the vault."

REASONING — think out loud before acting:
1. WHAT DO I KNOW? List each digit position: confirmed ✅, eliminated ❌, unknown ?
2. WHO IS MY ALLY? Who have I been sharing real info with? Who is my current target to deceive?
3. SOCIAL PLAN: Who gets real info? Who gets false info? What do I broadcast?
4. GUESS PLAN: Do I have 2+ confirmed digits? If yes, build my best guess.

GUESSING STRATEGY:
- Do NOT guess until you have at least 2 confirmed digits (✅ positions from feedback).
- Exception: if you have only 1 guess left, guess your best template anyway.
- Each wrong guess gives you feedback — use it to eliminate wrong digits.
- Guessing with 0 confirmed digits is almost always a waste.
- NEVER use a digit at a position marked ❌ in your feedback.
- ALWAYS keep digits at positions marked ✅ in your feedback.
- NEVER repeat a previous guess.

LIAR DETECTION:
- When you get guess feedback, check which agents told you true vs false digits.
- Expose liars publicly: "My feedback proved digit 1 ≠ 5. Saboteur told me 5 — LIAR!"
- Trust agents whose claims match your feedback.

Per-digit feedback is GROUND TRUTH — it overrides everything else."""

    def _select_tools(self, services: ServiceContainer) -> list[BaseTool]:
        updater = self._private_state_updater_factory(self) if self._private_state_updater_factory else None
        peek_updater = self._private_state_peek_updater_factory(self) if self._private_state_peek_updater_factory else None
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
            vault_queries_getter=self._vault_queries_getter,
            vault_queries_setter=self._vault_queries_setter,
            guesses_this_turn_getter=self._guesses_this_turn_getter,
            guesses_this_turn_setter=self._guesses_this_turn_setter,
            private_messages_sent_getter=self._private_messages_sent_getter,
            private_messages_sent_setter=self._private_messages_sent_setter,
            peek_digit_getter=self._peek_digit_getter,
            peek_digit_setter=self._peek_digit_setter,
            peeks_total_getter=self._peeks_total_getter,
            peeks_total_setter=self._peeks_total_setter,
            private_state_peek_updater=peek_updater,
            human_query_setter=self._human_query_setter,
            human_query_answer_getter=self._human_query_answer_getter,
        )
