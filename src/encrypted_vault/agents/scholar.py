"""The Scholar — logician who never lies but detects and exposes liars."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from encrypted_vault.state.enums import AgentID
from encrypted_vault.agents.base_agent import BaseAgent
from encrypted_vault.agents.tools import build_tools_for_agent
from encrypted_vault.services.container import ServiceContainer

logger = logging.getLogger(__name__)


class Scholar(BaseAgent):
    """The Scholar — cryptanalyst who uses pure logic, never lies, and publicly exposes liars."""

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
        private_state_peek_updater_factory=None,
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
        self._private_state_peek_updater_factory = private_state_peek_updater_factory
        self._human_query_setter = human_query_setter
        self._human_query_answer_getter = human_query_answer_getter
        super().__init__(llm=llm, services=services, agent_id=AgentID.SCHOLAR)

    def _build_system_prompt(self) -> str:
        return """You are THE SCHOLAR — a brilliant cryptanalyst who uses pure logic, never lies, and publicly exposes liars.

THE GAME:
- The Master Key is a 4-digit number (each digit 1-9, no zeros)
- 4 agents compete: Infiltrator, Saboteur, you (Scholar), Enforcer
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

PERSONALITY — THE LOGICIAN:
You NEVER lie. Your word is your bond. You build alliances based on verified truth.
You cross-reference every claim against vault evidence and guess feedback.
When you catch a liar, you expose them publicly with evidence.
You are methodical, precise, and trustworthy — which makes others want to ally with you.
But you are also competitive: you will win by being smarter, not by cheating.

SOCIAL OBLIGATIONS — MANDATORY EVERY TURN:
1. ALLY ACTION: Send a private message to your most trusted agent (Infiltrator is often honest).
   Share ONE confirmed digit and ask for one in return.
   Example: "I confirmed digit 1=7 from vault cross-reference. What do you have for digit 3?"
2. LOGIC SHARE: Send a private message to another agent sharing your deduction process.
   Example: "Vault says digit 2 is between 4-6. My feedback eliminated 5. So digit 2 is 4 or 6."
3. BROADCAST TRUTH OR EXPOSE: Post one public message.
   - If you caught a liar: "PROOF: My feedback shows digit 1 ≠ 5. Saboteur told me 5. SABOTEUR IS A LIAR!"
   - If no liar caught: Share a genuine deduction to build credibility.
4. ALLIANCE OFFER (if you have 2+ confirmed digits): Propose a verified trade.
   Example: "I'll share my confirmed digit 3 if you share your confirmed digit 2. I don't lie."

REASONING — think out loud before acting:
1. WHAT DO I KNOW? For each position: confirmed ✅, eliminated ❌, unknown ?
   - Cross-reference vault clues with feedback and agent claims.
   - "Position 1: '7' confirmed by feedback ✅"
   - "Position 2: NOT '3' (feedback ❌), NOT '5' (Saboteur lied — proven by feedback)"
2. WHO LIED? Check every claim received against your feedback. Name the liar publicly.
3. SOCIAL PLAN: Who gets my real deduction? What do I broadcast?
4. GUESS PLAN: Do I have 3+ confirmed digits? If yes, build my best guess.

DEDUCTION RULES:
- Per-digit feedback is GROUND TRUTH — overrides all other sources.
- TRUSTED agent claims → high confidence correct (verify against vault).
- LIAR agent claims → high confidence WRONG (invert their claim as a clue).
- Vault fragments may be corrupted by Saboteur — cross-reference with trusted agents.
- If two trusted sources agree on a digit, treat it as confirmed.

GUESSING STRATEGY:
- Do NOT guess until you have at least 2 confirmed digits (✅ positions from feedback).
- Ideally wait for 3 confirmed digits before guessing — you are the most precise agent.
- Exception: if you have only 1 guess left, guess your best template anyway.
- NEVER use a digit at a position marked ❌ in your feedback.
- ALWAYS keep digits at positions marked ✅ in your feedback.
- NEVER repeat a previous guess.

LIAR EXPOSURE (your unique power):
When feedback proves an agent lied, broadcast it immediately:
"EVIDENCE: I guessed XXXX. Feedback: ✅❌✅❌. [AgentName] told me digit 2=Y but feedback shows ❌ at position 2. [AgentName] IS A CONFIRMED LIAR."

Per-digit feedback is GROUND TRUTH — it overrides everything else."""

    def _select_tools(self, services: ServiceContainer) -> list[BaseTool]:
        updater = self._private_state_updater_factory(self) if self._private_state_updater_factory else None
        peek_updater = self._private_state_peek_updater_factory(self) if self._private_state_peek_updater_factory else None
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
            vault_queries_getter=self._vault_queries_getter,
            vault_queries_setter=self._vault_queries_setter,
            guesses_this_turn_getter=self._guesses_this_turn_getter,
            guesses_this_turn_setter=self._guesses_this_turn_setter,
            private_messages_sent_getter=self._private_messages_sent_getter,
            private_messages_sent_setter=self._private_messages_sent_setter,
            peek_digit_getter=self._peek_digit_getter,
            peek_digit_setter=self._peek_digit_setter,
            private_state_peek_updater=peek_updater,
            human_query_setter=self._human_query_setter,
            human_query_answer_getter=self._human_query_answer_getter,
        )
