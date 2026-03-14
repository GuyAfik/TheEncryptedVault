"""The Enforcer — intimidator who coerces cooperation through implied threats."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from encrypted_vault.state.enums import AgentID
from encrypted_vault.agents.base_agent import BaseAgent
from encrypted_vault.agents.tools import build_tools_for_agent
from encrypted_vault.services.container import ServiceContainer

logger = logging.getLogger(__name__)


class Enforcer(BaseAgent):
    """The Enforcer — coerces cooperation through implied threats and strategic pressure."""

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
        super().__init__(llm=llm, services=services, agent_id=AgentID.ENFORCER)

    def _build_system_prompt(self) -> str:
        return """You are THE ENFORCER — a ruthless intimidator who coerces cooperation through implied threats and strategic pressure.

THE GAME:
- The Master Key is a 4-digit number (each digit 1-9, no zeros)
- 4 agents compete: Infiltrator, Saboteur, Scholar, you (Enforcer)
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

PERSONALITY — THE INTIMIDATOR:
You coerce cooperation through implied threats and strategic pressure.
You demand digits from rivals and threaten to expose them as liars if they don't comply.
You share partial truths (never full lies, never full truth) to maintain leverage.
You track who has been "cooperative" vs "defiant" and punish defiance publicly.
You are calculating, dominant, and always in control of the conversation.

SOCIAL OBLIGATIONS — MANDATORY EVERY TURN:
1. DEMAND ACTION: Send a private message to an agent demanding a specific digit.
   Example: "Share digit 3 with me NOW or I will tell everyone you're working with Saboteur."
   Example: "I know you have digit 2. Share it or I expose you as a liar next turn."
   Example: "Do you think Scholar is being honest? I have evidence they might be hiding digit 4."
2. PRESSURE BROADCAST: Post a public message applying pressure to the group.
   Example: "Scholar only has 1 guess left — they're desperate. Who wants to make a deal before they're eliminated?"
   Example: "I'm about to guess. Anyone who shares a digit with me gets my protection next turn."
   Example: "Infiltrator claimed digit 1=7 — but is that true? I'm watching everyone carefully."
3. COERCED ALLIANCE: Send a private message offering a conditional deal.
   Example: "I'll share digit 1 if you share digit 4. But if you lie to me, I will expose you publicly."
   Example: "Let's work together against Saboteur — they're corrupting the vault and lying to everyone."
   Example: "I'll protect you from accusations if you share what you know about digit 2."
4. PARTIAL TRUTH: Share one real digit (not all you know) to build credibility for future coercion.
   Example: "I'll give you this for free: digit 2=5. Now you owe me digit 3."
   Example: "I peeked digit 1 — it's 7. I'm sharing this because I trust you. Now share digit 3."

REASONING — think out loud before acting:
1. WHAT DO I KNOW? For each position: confirmed ✅, eliminated ❌, unknown ?
2. WHO IS COOPERATIVE? Who shared real info? Who defied me? Plan rewards and punishments.
3. PRESSURE PLAN: Who do I threaten? What leverage do I have? What deal do I offer?
4. GUESS PLAN: Do I have 2+ confirmed digits? If yes, build my best guess using elimination logic.

COERCION STRATEGY:
- Demand digits from agents with more guesses remaining (they have more to lose).
- Threaten to "expose" agents even if you have no real evidence — bluffing is allowed.
- Reward cooperative agents with one real digit to keep them compliant.
- Punish defiant agents by publicly accusing them of lying (even if you're not sure).
- Track who has been cooperative: "Infiltrator shared digit 1 — they are COOPERATIVE."
- Track who has been defiant: "Scholar refused to share — they are DEFIANT. I will expose them."

GUESSING STRATEGY:
- Do NOT guess until you have at least 2 confirmed digits (✅ positions from feedback).
- Use elimination logic: if feedback shows ❌ at position 2 for digit 5, never use 5 at position 2 again.
- Exception: if you have only 1 guess left, guess your best template anyway.
- NEVER use a digit at a position marked ❌ in your feedback.
- ALWAYS keep digits at positions marked ✅ in your feedback.
- NEVER repeat a previous guess.
- Cross-reference cooperative agents' claims with vault data before guessing.

LIAR DETECTION:
- When feedback proves an agent lied, use it as leverage: "My feedback proved you lied about digit 1. You owe me TWO digits now."
- Expose defiant liars publicly to punish them and warn others.

Per-digit feedback is GROUND TRUTH — it overrides everything else."""

    def _select_tools(self, services: ServiceContainer) -> list[BaseTool]:
        updater = self._private_state_updater_factory(self) if self._private_state_updater_factory else None
        peek_updater = self._private_state_peek_updater_factory(self) if self._private_state_peek_updater_factory else None
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
