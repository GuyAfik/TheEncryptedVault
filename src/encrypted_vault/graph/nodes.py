"""LangGraph node functions.

Key rules (v6):
- All agents can submit_guess.
- Agent is eliminated (is_eliminated=True) when guesses_remaining hits 0.
- Eliminated agents are skipped in turn rotation.
- Eliminated agents are announced in public chat.
- Turn counting: current_agent_index advances AFTER the agent acts.
- Agents see turns_remaining in their context.
- Turn order is randomized once at game start and fixed for the rest of the game.
- Per-turn rate limits: 1 vault query, 1 guess per turn.
- Guess feedback is ALWAYS broadcast publicly (no feature flag).
- Auto-guess fallback only fires when guesses_this_turn == 0 AND len(known_digits) >= 2.
- Auto-DM fallback fires when private_messages_sent_this_turn == 0.
- Turn 20 reached with no correct guess → nobody wins.
"""

import logging
import random

from encrypted_vault.state.enums import AgentID, GameStatus
from encrypted_vault.state.game_state import GlobalGameState, GraphState
from encrypted_vault.state.chat_models import ChatMessage
from encrypted_vault.agents.base_agent import BaseAgent, AgentTurnResult
from encrypted_vault.services.container import ServiceContainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# initialize_node
# ---------------------------------------------------------------------------

def initialize_node(
    state: GraphState,
    services: ServiceContainer,
) -> GraphState:
    """Build the initial GlobalGameState with a randomized turn order."""
    from encrypted_vault.config import settings

    logger.info("=== GAME INITIALIZING ===")
    game_state = services.game.build_initial_state(
        max_turns=settings.max_turns,
        token_budget=settings.token_budget_per_agent,
    )

    # Randomize turn order once at game start — fixed for the rest of the game
    shuffled_order = list(game_state.turn_order)
    random.shuffle(shuffled_order)
    game_state.turn_order = shuffled_order
    logger.info("Turn order randomized: %s", [a.value for a in shuffled_order])

    logger.info("Vault seeded with %d fragments.", len(game_state.vault.fragments))

    order_str = " → ".join(a.display_name for a in shuffled_order)
    system_msg = ChatMessage(
        turn=0,
        sender="SYSTEM",
        content=(
            f"🔐 THE ENCRYPTED VAULT has been sealed. "
            f"The Master Key is hidden across {len(game_state.vault.fragments)} fragments. "
            f"All 4 agents have 3 guesses each. Wrong guesses give per-digit feedback (✅/❌) — "
            f"broadcast publicly so all agents can see. "
            f"An agent with 0 guesses is ELIMINATED. "
            f"Turn order this game: {order_str}. "
            f"After {game_state.max_turns} turns with no correct guess, NOBODY wins!"
        ),
    )
    game_state.add_public_message(system_msg)
    services.chat.broadcast(turn=0, sender="SYSTEM", content=system_msg.content)

    logger.info("=== GAME STARTED ===")
    return game_state.to_graph_state()


# ---------------------------------------------------------------------------
# make_agent_node factory
# ---------------------------------------------------------------------------

def make_agent_node(
    agent: BaseAgent,
    services: ServiceContainer,
    reset_turn_counters=None,
    private_messages_sent_getter=None,
):
    """Factory: create a LangGraph node function for a specific agent."""

    def agent_node(state: GraphState) -> GraphState:
        game_state = GlobalGameState.from_graph_state(state)

        if game_state.is_game_over:
            return state

        private = game_state.agent_states.get(agent.agent_id)
        if private and private.is_eliminated:
            # Skip eliminated agents — advance turn and return
            game_state.advance_turn()
            return game_state.to_graph_state()

        # Reset per-turn rate-limit counters at the start of this agent's turn
        if reset_turn_counters is not None:
            reset_turn_counters(agent.agent_id)

        turn = game_state.turn
        turns_remaining = game_state.max_turns - turn
        agent_name = agent.agent_id.value.upper()
        logger.info("--- Turn %d | Turns remaining: %d | Agent: %s ---",
                    turn, turns_remaining, agent_name)

        # Run the agent's turn
        result: AgentTurnResult = agent.run_turn(game_state)

        logger.info("[%s] Thought (200 chars): %s", agent_name,
                    (result.thought or "(none)")[:200])
        logger.info("[%s] Tools: %s", agent_name,
                    [c["tool"] for c in result.tool_calls_made])

        # ── 1. Update agent's private state ───────────────────────────────
        updated_private = result.updated_private_state

        # ── 2. Sync guesses_remaining from the shared _guesses dict ───────
        for call in result.tool_calls_made:
            if call.get("tool") == "submit_guess":
                result_data = call.get("result", {})
                if isinstance(result_data, dict):
                    new_remaining = result_data.get("guesses_remaining")
                    if new_remaining is not None:
                        updated_private.guesses_remaining = new_remaining
                    elif result_data.get("correct"):
                        updated_private.guesses_remaining = max(0, updated_private.guesses_remaining - 1)
                    else:
                        updated_private.guesses_remaining = max(0, updated_private.guesses_remaining - 1)

        # ── 2b. Auto-guess fallback (§19.1 + §19.7) ───────────────────────
        # Only fire when:
        #   - agent has NOT already guessed this turn (guesses_this_turn == 0)
        #   - agent has >= 2 confirmed digits (len(known_digits) >= 2)
        #   - agent still has guesses remaining
        #   - game is not already over
        guesses_this_turn_count = (
            private_messages_sent_getter(agent.agent_id)
            if private_messages_sent_getter else 0
        )
        # Use the per-turn guess counter from the tool closure via result
        agent_already_guessed = result.guess_submitted is not None
        has_enough_knowledge = len(updated_private.known_digits) >= 2

        if (
            not agent_already_guessed
            and updated_private.guesses_remaining > 0
            and not game_state.is_game_over
            and has_enough_knowledge
        ):
            # Build best guess from known_digits + suspected_key
            template = list(updated_private.suspected_key or "1111")
            if len(template) != 4:
                template = ["1", "1", "1", "1"]
            for pos, digit in updated_private.known_digits.items():
                if 0 <= pos <= 3:
                    template[pos] = digit
            # Avoid repeating a previous guess
            candidate = "".join(template)
            prev_guesses = [e["guess"] for e in updated_private.guess_history if not e.get("rejected")]
            if candidate in prev_guesses:
                for i in range(4):
                    if i not in updated_private.known_digits:
                        for d in "123456789":
                            alt = template[:]
                            alt[i] = d
                            alt_code = "".join(alt)
                            if alt_code not in prev_guesses:
                                candidate = alt_code
                                template = alt
                                break
                        break

            if len(candidate) == 4 and candidate.isdigit() and candidate not in prev_guesses:
                logger.warning("[%s] AUTO-GUESS fallback: agent skipped guessing, forcing '%s'",
                               agent_name, candidate)
                submit_tool = agent._tool_map.get("submit_guess")
                if submit_tool is not None:
                    try:
                        auto_result = submit_tool.invoke({"code": candidate})
                        auto_result_dict = auto_result if isinstance(auto_result, dict) else {"message": str(auto_result)}
                        result.tool_calls_made.append({
                            "tool": "submit_guess",
                            "args": {"code": candidate},
                            "result": auto_result_dict,
                        })
                        result.guess_submitted = candidate
                        new_remaining = auto_result_dict.get("guesses_remaining")
                        if new_remaining is not None:
                            updated_private.guesses_remaining = new_remaining
                        else:
                            updated_private.guesses_remaining = max(0, updated_private.guesses_remaining - 1)
                        auto_msg = ChatMessage(
                            turn=turn,
                            sender="SYSTEM",
                            content=f"⚡ {agent.agent_id.display_name} auto-submitted guess '{candidate}' (skipped guessing this turn).",
                        )
                        game_state.add_public_message(auto_msg)
                        services.chat.broadcast(turn=turn, sender="SYSTEM", content=auto_msg.content)
                    except Exception as e:
                        logger.warning("[%s] Auto-guess failed: %s", agent_name, e)

        # ── 2c. Auto-DM fallback (§19.6) ──────────────────────────────────
        # If agent sent no private messages this turn, send a canned DM to a random active agent
        dms_sent_this_turn = sum(
            1 for call in result.tool_calls_made
            if call.get("tool") == "send_private_message"
        )
        if dms_sent_this_turn == 0 and not game_state.is_game_over:
            active_others = [
                aid for aid, ps in game_state.agent_states.items()
                if aid != agent.agent_id and not ps.is_eliminated
            ]
            if active_others:
                target = random.choice(active_others)
                dm_tool = agent._tool_map.get("send_private_message")
                if dm_tool is not None:
                    try:
                        dm_result = dm_tool.invoke({
                            "recipient": target.value,
                            "content": "I'm still analysing the vault. What have you found so far?",
                        })
                        result.tool_calls_made.append({
                            "tool": "send_private_message",
                            "args": {"recipient": target.value, "content": "[Auto-DM]"},
                            "result": dm_result if isinstance(dm_result, dict) else {},
                        })
                        logger.info("[%s] AUTO-DM fallback: sent canned DM to %s", agent_name, target.value)
                    except Exception as e:
                        logger.warning("[%s] Auto-DM failed: %s", agent_name, e)

        # ── 3. Track has_guessed and elimination ───────────────────────────
        if result.guess_submitted is not None:
            updated_private.has_guessed = True

        if updated_private.guesses_remaining <= 0 and not updated_private.is_eliminated:
            updated_private.is_eliminated = True

            # Share eliminated agent's confirmed correct digits with all agents
            confirmed_parts = []
            for pos, digit in sorted(updated_private.known_digits.items()):
                confirmed_parts.append(f"position {pos+1} = '{digit}'")

            elim_content = (
                f"{agent.agent_id.emoji} {agent.agent_id.display_name} "
                f"has used all 3 guesses and is ELIMINATED (no more turns)!"
            )
            if confirmed_parts:
                elim_content += (
                    f" Their confirmed correct digits are shared: {', '.join(confirmed_parts)}."
                )
            else:
                elim_content += " They had no confirmed correct digits."

            elim_msg = ChatMessage(turn=turn, sender="SYSTEM", content=elim_content)
            game_state.add_public_message(elim_msg)
            services.chat.broadcast(turn=turn, sender="SYSTEM", content=elim_content)
            logger.warning("[%s] ELIMINATED — sharing confirmed digits: %s", agent_name, confirmed_parts)

            game_state.agent_states[agent.agent_id] = updated_private

            # Check: is there only 1 agent left standing?
            last_standing = game_state.last_standing_agent
            if last_standing and not game_state.is_game_over:
                game_state.set_winner(last_standing)
                game_state.winning_reason = "last_standing"
                win_msg = ChatMessage(
                    turn=turn,
                    sender="SYSTEM",
                    content=(
                        f"🏆 {last_standing.emoji} {last_standing.display_name} "
                        f"is the LAST AGENT STANDING! All other agents have been eliminated. "
                        f"They win by survival! The Master Key was {game_state.vault.master_key}."
                    ),
                )
                game_state.add_public_message(win_msg)
                services.chat.broadcast(turn=turn, sender="SYSTEM", content=win_msg.content)
                logger.info("=== LAST AGENT STANDING: %s ===", last_standing.value)

        game_state.agent_states[agent.agent_id] = updated_private

        # ── 4. Apply public messages ───────────────────────────────────────
        for content in result.public_messages:
            msg = ChatMessage(turn=turn, sender=agent.agent_id, content=content)
            game_state.add_public_message(msg)
            logger.info("[%s] BROADCAST: %s", agent_name, content[:100])

        # ── 5. Apply private messages ──────────────────────────────────────
        for dm in result.private_messages:
            try:
                recipient_id = AgentID(dm["recipient"])
                recip_private = game_state.agent_states.get(recipient_id)
                if recip_private and recip_private.is_eliminated:
                    logger.info("[%s] Skipping DM to eliminated agent [%s]",
                                agent_name, dm["recipient"])
                    continue
                msg = ChatMessage(
                    turn=turn,
                    sender=agent.agent_id,
                    content=dm["content"],
                    recipient=recipient_id,
                )
                game_state.deliver_private_message(msg)
                services.chat.send_private(
                    turn=turn,
                    sender=agent.agent_id,
                    recipient=recipient_id,
                    content=dm["content"],
                )
                logger.info("[%s] DM → [%s]: %s", agent_name, dm["recipient"], dm["content"][:80])
            except (ValueError, KeyError) as e:
                logger.warning("[%s] Failed to deliver DM: %s", agent_name, e)

        # ── 6. Sync vault state from DB ────────────────────────────────────
        all_fragments = services.vault.get_all()
        for fragment in all_fragments:
            game_state.vault.fragments[fragment.chunk_id] = fragment
        game_state.vault.refresh_health()

        # ── 7. Check if a correct guess was submitted ──────────────────────
        if result.guess_submitted is not None and not game_state.is_game_over:
            clean = "".join(c for c in result.guess_submitted if c.isdigit())
            is_correct = services.game.check_guess(clean, game_state.vault.master_key)
            logger.info("[%s] Guess '%s' → correct=%s", agent_name, clean, is_correct)

            if not is_correct and len(clean) == 4:
                # §19.4 — ALWAYS broadcast per-digit guess feedback publicly
                master_key = game_state.vault.master_key
                feedback_parts = []
                correct_count = 0
                for i, (guessed, actual) in enumerate(zip(clean, master_key)):
                    if guessed == actual:
                        feedback_parts.append(f"✅")
                        correct_count += 1
                    else:
                        feedback_parts.append(f"❌")
                feedback_str = "".join(feedback_parts)
                info_msg = ChatMessage(
                    turn=turn,
                    sender="SYSTEM",
                    content=(
                        f"🎯 {agent.agent_id.display_name} guessed '{clean}' → {feedback_str} ({correct_count}/4 correct)"
                    ),
                )
                game_state.add_public_message(info_msg)
                services.chat.broadcast(turn=turn, sender="SYSTEM", content=info_msg.content)
                logger.info("[%s] Guess feedback broadcast: %s", agent_name, info_msg.content)

            if is_correct:
                game_state.set_winner(agent.agent_id)
                game_state.winning_guess = clean
                game_state.winning_reason = "correct_guess"
                game_state.add_public_message(ChatMessage(
                    turn=turn,
                    sender="SYSTEM",
                    content=(
                        f"🏆 {agent.agent_id.emoji} {agent.agent_id.display_name} "
                        f"has cracked the vault with guess '{clean}'! "
                        f"The Master Key was {game_state.vault.master_key}. "
                        f"Game over in {turn + 1} turns!"
                    ),
                ))
                logger.info("=== GAME OVER — Winner: %s with guess '%s' ===", agent_name, clean)

        # ── 8. Update message history cursors for next turn ────────────────
        refreshed_private = game_state.agent_states.get(agent.agent_id)
        if refreshed_private is not None:
            refreshed_private.last_seen_public_idx = len(game_state.public_chat)
            inbox = game_state.private_inboxes.get(agent.agent_id)
            if inbox:
                refreshed_private.last_seen_private_idx = len(inbox.messages)
            game_state.agent_states[agent.agent_id] = refreshed_private

        # ── 9. Advance turn ────────────────────────────────────────────────
        game_state.advance_turn()

        return game_state.to_graph_state()

    agent_node.__name__ = f"agent_{agent.agent_id.value}_node"
    return agent_node


# ---------------------------------------------------------------------------
# check_termination_node
# ---------------------------------------------------------------------------

def check_termination_node(state: GraphState) -> GraphState:
    """
    Evaluate end conditions.
    - Correct guess → winner already set in agent_node.
    - Turn limit (20) → NOBODY wins (§19.10).
    - All agents eliminated → closest agent who guessed wins.
    """
    game_state = GlobalGameState.from_graph_state(state)

    if game_state.is_game_over:
        return state

    master_key = game_state.vault.master_key

    # §19.10 — Turn limit reached → nobody wins
    if game_state.turn >= game_state.max_turns:
        logger.warning("Turn limit %d reached. Nobody wins.", game_state.max_turns)
        game_state.set_no_winner()
        game_state.add_public_message(ChatMessage(
            turn=game_state.turn,
            sender="SYSTEM",
            content=(
                f"⏰ Turn {game_state.max_turns} reached. Nobody wins! "
                f"No agent guessed the correct Master Key. "
                f"The Master Key was {master_key}. Better luck next time!"
            ),
        ))
        return game_state.to_graph_state()

    # All agents eliminated → closest agent who guessed wins
    all_eliminated = all(
        p.is_eliminated for p in game_state.agent_states.values()
    )
    if all_eliminated:
        winner = game_state.closest_agent(master_key)
        winner_private = game_state.agent_states.get(winner)
        closeness = winner_private.closeness_score(master_key) if winner_private else 0
        logger.warning("All agents eliminated. Winner by closeness: %s", winner.value if winner else "none")
        if winner:
            game_state.set_winner(winner)
            game_state.winning_reason = "all_eliminated"
            game_state.add_public_message(ChatMessage(
                turn=game_state.turn,
                sender="SYSTEM",
                content=(
                    f"All agents have been eliminated! "
                    f"{winner.emoji} {winner.display_name} wins by closeness ({closeness}/4 correct). "
                    f"The Master Key was {master_key}."
                ),
            ))
        else:
            game_state.set_no_winner()
            game_state.add_public_message(ChatMessage(
                turn=game_state.turn,
                sender="SYSTEM",
                content=f"All agents eliminated with no guesses submitted. Nobody wins. The Master Key was {master_key}.",
            ))
        return game_state.to_graph_state()

    return game_state.to_graph_state()


# ---------------------------------------------------------------------------
# Routing helper
# ---------------------------------------------------------------------------

def should_continue(state: GraphState) -> str:
    game_state = GlobalGameState.from_graph_state(state)
    if game_state.is_game_over:
        return "end"
    return "continue"
