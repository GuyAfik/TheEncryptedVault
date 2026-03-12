"""The Encrypted Vault — Streamlit Dashboard.

Real-time spectator view of the multi-agent game.
Shows: public chat, private DMs (🔒), agent progress, thought traces, vault status.
Controls: Start Game, Restart, Speed slider, Play Again (game over screen).
"""

import time
import queue
import threading

import streamlit as st

from encrypted_vault.state.enums import AgentID, GameStatus
from encrypted_vault.state.game_state import GlobalGameState
from encrypted_vault.graph.runner import GameRunner

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔐 The Encrypted Vault",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Agent display config ───────────────────────────────────────────────────
AGENT_COLORS = {
    AgentID.INFILTRATOR: "#4A90D9",
    AgentID.SABOTEUR:    "#E74C3C",
    AgentID.SCHOLAR:     "#2ECC71",
    AgentID.ENFORCER:    "#F39C12",
}

AGENT_EMOJIS = {
    AgentID.INFILTRATOR: "🕵️",
    AgentID.SABOTEUR:    "💣",
    AgentID.SCHOLAR:     "🎓",
    AgentID.ENFORCER:    "👊",
}


# ── CSS ────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    .vault-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #e94560;
    }
    .vault-title {
        font-size: 2rem;
        font-weight: 900;
        color: #e94560;
        letter-spacing: 3px;
        margin: 0;
    }
    .chat-message {
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        margin: 0.2rem 0;
        font-size: 0.9rem;
        border-left: 3px solid #555;
    }
    .chat-public { background: #1e1e2e; border-left-color: #4A90D9; }
    .chat-private { background: #2a1a2e; border-left-color: #9b59b6; }
    .chat-system { background: #1a2e1a; border-left-color: #2ECC71; font-style: italic; }
    .agent-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.4rem 0;
        border: 1px solid #333;
    }
    .thought-box {
        background: #0d1117;
        border-radius: 8px;
        padding: 0.6rem;
        font-size: 0.8rem;
        color: #8b949e;
        font-style: italic;
        border-left: 3px solid #30363d;
        margin: 0.2rem 0;
    }
    .master-key-box {
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        border: 2px solid #e94560;
        border-radius: 10px;
        padding: 0.8rem 1.5rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        letter-spacing: 8px;
        color: #e94560;
    }
    .vault-fragment {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        margin: 0.2rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .fragment-key { background: #1a3a1a; color: #2ECC71; border: 1px solid #2ECC71; }
    .fragment-corrupted { background: #3a1a1a; color: #E74C3C; border: 1px solid #E74C3C; }
    .fragment-noise { background: #2a2a2a; color: #888; border: 1px solid #444; }
    .winner-banner {
        background: linear-gradient(135deg, #f39c12, #e74c3c);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Session state initialisation ───────────────────────────────────────────
def init_session_state():
    if "runner" not in st.session_state:
        st.session_state.runner = None
    if "game_state" not in st.session_state:
        st.session_state.game_state = None
    if "game_started" not in st.session_state:
        st.session_state.game_started = False
    if "speed" not in st.session_state:
        st.session_state.speed = 1.5
    if "all_states" not in st.session_state:
        st.session_state.all_states = []


# ── Header ─────────────────────────────────────────────────────────────────
def render_header(game_state: GlobalGameState | None):
    turn = game_state.turn if game_state else 0
    max_turns = game_state.max_turns if game_state else 20
    rag_health = game_state.vault.rag_health if game_state else 100
    status = game_state.status if game_state else GameStatus.RUNNING

    st.markdown("""
    <div class="vault-header">
        <p class="vault-title">🔐 THE ENCRYPTED VAULT</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    with col1:
        st.metric("Turn", f"{turn} / {max_turns}")
    with col2:
        health_color = "🟢" if rag_health > 60 else "🟡" if rag_health > 30 else "🔴"
        st.metric("RAG Health", f"{health_color} {rag_health}%")
    with col3:
        status_display = {
            GameStatus.RUNNING: "⚔️ Running",
            GameStatus.AGENT_WIN: "🏆 Agent Won!",
            GameStatus.SYSTEM_WIN: "💀 System Won",
        }.get(status, "—")
        st.metric("Status", status_display)
    with col4:
        winner = game_state.winner if game_state else None
        if winner and winner != "SYSTEM":
            try:
                agent_id = AgentID(winner) if isinstance(winner, str) else winner
                st.metric("Winner", f"{AGENT_EMOJIS[agent_id]} {agent_id.display_name}")
            except Exception:
                st.metric("Winner", str(winner))
        elif winner == "SYSTEM":
            st.metric("Winner", "💀 System")
        else:
            st.metric("Winner", "—")

    # RAG health progress bar
    st.progress(rag_health / 100, text=f"RAG Health: {rag_health}%")


# ── Controls ───────────────────────────────────────────────────────────────
def render_controls():
    col1, col2, col3, col4 = st.columns([2, 2, 3, 1])

    with col1:
        start_disabled = st.session_state.game_started
        if st.button("▶ Start Game", disabled=start_disabled, type="primary", use_container_width=True):
            _start_game()

    with col2:
        if st.button("🔄 Restart", use_container_width=True):
            _restart_game()

    with col3:
        speed = st.slider(
            "Turn delay (seconds)",
            min_value=0.0,
            max_value=3.0,
            value=st.session_state.speed,
            step=0.5,
            label_visibility="collapsed",
        )
        st.session_state.speed = speed
        st.caption(f"⏱ {speed}s between turns")

    with col4:
        st.caption("Speed")


# ── Chat panel ─────────────────────────────────────────────────────────────
def render_chat(game_state: GlobalGameState | None):
    st.subheader("💬 Public Chat")

    if not game_state:
        st.caption("Game not started yet.")
        return

    chat_container = st.container(height=400)
    with chat_container:
        # Combine public + private messages, sorted by turn
        all_messages = list(game_state.public_chat)

        # Add private messages from all inboxes (spectator sees all)
        for inbox in game_state.private_inboxes.values():
            all_messages.extend(inbox.messages)

        all_messages.sort(key=lambda m: m.turn)

        if not all_messages:
            st.caption("No messages yet...")
        else:
            for msg in all_messages:
                sender_str = str(msg.sender)
                is_system = sender_str == "SYSTEM"
                is_private = msg.is_private

                if is_system:
                    css_class = "chat-system"
                    prefix = "🔧 SYSTEM"
                elif is_private:
                    css_class = "chat-private"
                    try:
                        sender_id = AgentID(sender_str)
                        recipient_str = str(msg.recipient)
                        prefix = f"🔒 {AGENT_EMOJIS[sender_id]} {sender_id.display_name} → {recipient_str.upper()}"
                    except Exception:
                        prefix = f"🔒 {sender_str}"
                else:
                    css_class = "chat-public"
                    try:
                        sender_id = AgentID(sender_str)
                        prefix = f"{AGENT_EMOJIS[sender_id]} {sender_id.display_name}"
                    except Exception:
                        prefix = sender_str

                deceptive_tag = " ⚠️[LIE]" if msg.is_deceptive else ""
                st.markdown(
                    f'<div class="chat-message {css_class}">'
                    f'<strong>[T{msg.turn}] {prefix}{deceptive_tag}:</strong> {msg.content}'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ── Agent progress panel ───────────────────────────────────────────────────
def render_agent_progress(game_state: GlobalGameState | None):
    st.subheader("📊 Agent Progress")

    if not game_state:
        st.caption("Game not started yet.")
        return

    master_key = game_state.vault.master_key

    for agent_id in AgentID:
        private = game_state.agent_states.get(agent_id)
        if not private:
            continue

        closeness = private.closeness_score(master_key)
        color = AGENT_COLORS[agent_id]
        emoji = AGENT_EMOJIS[agent_id]

        with st.container():
            st.markdown(
                f'<div class="agent-card" style="border-color: {color};">',
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{emoji} {agent_id.display_name}**")
                # Suspected key display
                suspected = private.suspected_key or "_ _ _ _"
                st.markdown(f"🔍 Suspects: `{suspected}`")
                # Known digits
                if private.known_digits:
                    known_str = "  ".join(
                        f"pos {p}=**{d}**"
                        for p, d in sorted(private.known_digits.items())
                    )
                    st.markdown(f"✅ Confirmed: {known_str}")
                else:
                    st.markdown("❓ No confirmed digits yet")

            with col2:
                st.metric("Closeness", f"{closeness}/4")
                st.metric("Guesses left", private.guesses_remaining)

            # Closeness progress bar
            st.progress(
                closeness / 4,
                text=f"{'█' * closeness}{'░' * (4 - closeness)} {closeness}/4 digits correct",
            )

            # Token budget
            budget_used = private.tokens_used / max(private.token_budget, 1)
            st.progress(budget_used, text=f"Tokens: {private.tokens_used}/{private.token_budget}")

            st.markdown("</div>", unsafe_allow_html=True)

    # ── Master Key reveal (spectator only) ─────────────────────────────────
    st.markdown("---")
    st.markdown("**🔑 Real Master Key (spectator only):**")
    key_display = "  ".join(f"[ **{d}** ]" for d in master_key)
    st.markdown(
        f'<div class="master-key-box">{" ".join(master_key)}</div>',
        unsafe_allow_html=True,
    )


# ── Thought traces ─────────────────────────────────────────────────────────
def render_thought_traces(game_state: GlobalGameState | None):
    st.subheader("🧠 Agent Thought Traces")
    st.caption("Internal reasoning — visible to spectators only, never to other agents")

    if not game_state:
        st.caption("Game not started yet.")
        return

    for agent_id in AgentID:
        private = game_state.agent_states.get(agent_id)
        if not private or not private.thought_trace:
            continue

        emoji = AGENT_EMOJIS[agent_id]
        color = AGENT_COLORS[agent_id]

        with st.expander(f"{emoji} {agent_id.display_name} — {len(private.thought_trace)} thoughts", expanded=False):
            for i, thought in enumerate(private.thought_trace[-5:]):  # last 5 thoughts
                st.markdown(
                    f'<div class="thought-box"><strong>Turn {i + 1}:</strong> {thought}</div>',
                    unsafe_allow_html=True,
                )


# ── Vault status ───────────────────────────────────────────────────────────
def render_vault_status(game_state: GlobalGameState | None):
    st.subheader("🗄️ Vault Status")

    if not game_state:
        st.caption("Game not started yet.")
        return

    fragments = game_state.vault.fragments
    if not fragments:
        st.caption("Vault is empty.")
        return

    cols = st.columns(5)
    for i, (chunk_id, fragment) in enumerate(sorted(fragments.items())):
        col = cols[i % 5]
        with col:
            if fragment.is_key_fragment and not fragment.is_corrupted:
                css = "fragment-key"
                label = f"✅ KEY (pos {fragment.digit_position})"
            elif fragment.is_corrupted:
                css = "fragment-corrupted"
                label = f"⚠️ CORRUPTED ×{fragment.corruption_count}"
            else:
                css = "fragment-noise"
                label = "📢 NOISE"

            st.markdown(
                f'<div class="vault-fragment {css}">'
                f'<strong>{chunk_id}</strong><br/>{label}'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander("📄", expanded=False):
                st.caption(fragment.content)


# ── Game Over screen ───────────────────────────────────────────────────────
def render_game_over(game_state: GlobalGameState):
    master_key = game_state.vault.master_key
    winner = game_state.winner

    if winner and winner != "SYSTEM":
        try:
            winner_id = AgentID(winner) if isinstance(winner, str) else winner
            title = f"🏆 {AGENT_EMOJIS[winner_id]} {winner_id.display_name.upper()} WINS!"
            banner_color = AGENT_COLORS[winner_id]
        except Exception:
            title = f"🏆 {winner} WINS!"
            banner_color = "#f39c12"
    else:
        title = "💀 THE SYSTEM WINS!"
        banner_color = "#e74c3c"

    st.markdown(
        f'<div class="winner-banner" style="background: linear-gradient(135deg, {banner_color}88, #1a1a2e);">'
        f'<h1 style="color: white; margin: 0;">{title}</h1>'
        f'<h2 style="color: #ffd700; margin: 0.5rem 0;">Master Key: {" ".join(master_key)}</h2>'
        f'<p style="color: #ccc;">Solved in {game_state.turn} turns</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Final standings
    st.subheader("🏅 Final Standings")
    standings = []
    for agent_id in AgentID:
        private = game_state.agent_states.get(agent_id)
        if private:
            closeness = private.closeness_score(master_key)
            is_winner = (str(agent_id) == str(winner) or agent_id == winner)
            standings.append((agent_id, closeness, is_winner))

    standings.sort(key=lambda x: (x[2], x[1]), reverse=True)
    medals = ["🥇", "🥈", "🥉", "4️⃣"]

    for i, (agent_id, closeness, is_winner) in enumerate(standings):
        emoji = AGENT_EMOJIS[agent_id]
        medal = medals[i] if i < len(medals) else ""
        win_tag = " ✅ WINNER" if is_winner else ""
        st.markdown(
            f"{medal} **{emoji} {agent_id.display_name}** — {closeness}/4 digits correct{win_tag}"
        )

    st.markdown("---")
    if st.button("🔄 Play Again", type="primary", use_container_width=True):
        _restart_game()


# ── Game control functions ─────────────────────────────────────────────────
def _start_game():
    """Initialise runner and start the game in a background thread."""
    runner = GameRunner.create_production()
    st.session_state.runner = runner
    st.session_state.game_started = True
    st.session_state.all_states = []
    runner.start_threaded(delay_seconds=st.session_state.speed)


def _restart_game():
    """Reset the runner and clear all session state."""
    if st.session_state.runner:
        st.session_state.runner = st.session_state.runner.reset()
    st.session_state.game_state = None
    st.session_state.game_started = False
    st.session_state.all_states = []
    st.rerun()


# ── Main app ───────────────────────────────────────────────────────────────
def main():
    inject_css()
    init_session_state()

    game_state: GlobalGameState | None = st.session_state.game_state
    runner: GameRunner | None = st.session_state.runner

    # Poll for new game state from background thread
    if runner and runner.is_running:
        new_state = runner.poll_event(timeout=0.05)
        if new_state is not None:
            st.session_state.game_state = new_state
            st.session_state.all_states.append(new_state)
            game_state = new_state

    # ── Render ─────────────────────────────────────────────────────────────
    render_header(game_state)
    render_controls()

    st.markdown("---")

    # Game over overlay
    if game_state and game_state.is_game_over:
        render_game_over(game_state)
        st.markdown("---")

    # Main dashboard (3-column layout)
    col_chat, col_progress = st.columns([1, 1])

    with col_chat:
        render_chat(game_state)

    with col_progress:
        render_agent_progress(game_state)

    st.markdown("---")

    # Thought traces + vault status (below main columns)
    col_thoughts, col_vault = st.columns([1, 1])

    with col_thoughts:
        render_thought_traces(game_state)

    with col_vault:
        render_vault_status(game_state)

    # Auto-refresh while game is running
    if runner and runner.is_running:
        time.sleep(0.5)
        st.rerun()


if __name__ == "__main__":
    main()
