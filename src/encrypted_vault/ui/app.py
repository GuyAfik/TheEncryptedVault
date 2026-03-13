"""The Encrypted Vault — Streamlit Dashboard (v6).

Changes in v6:
- Removed broadcast_guess_results toggle (always broadcast now)
- Turn 20 = nobody wins (render_game_over handles winner=None)
- Thought traces show ALL turns (not just last 3)
- Guess feedback messages styled distinctly in broadcast chat
- Status header handles nobody_wins state
"""

import logging
import time

import streamlit as st

from encrypted_vault.state.enums import AgentID, GameStatus
from encrypted_vault.state.game_state import GlobalGameState
from encrypted_vault.graph.runner import GameRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

st.set_page_config(
    page_title="🔐 The Encrypted Vault",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
AGENT_BG = {
    AgentID.INFILTRATOR: "#0d1b2a",
    AgentID.SABOTEUR:    "#2a0d0d",
    AgentID.SCHOLAR:     "#0d2a0d",
    AgentID.ENFORCER:    "#2a1d0d",
}


def inject_css():
    st.markdown("""
    <style>
    .vault-header { background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460); padding:1.2rem 2rem; border-radius:12px; margin-bottom:0.8rem; border:2px solid #e94560; }
    .vault-title { font-size:1.8rem; font-weight:900; color:#e94560; letter-spacing:3px; margin:0; }
    .active-banner { padding:0.5rem 1rem; border-radius:8px; font-size:1rem; font-weight:bold; text-align:center; margin-bottom:0.5rem; }
    .msg-broadcast { background:#1a2035; border-left:4px solid #4A90D9; padding:0.5rem 0.8rem; border-radius:0 8px 8px 0; margin:0.25rem 0; font-size:0.88rem; color:#e0e0e0; }
    .msg-system    { background:#0d2a1a; border-left:4px solid #2ECC71; padding:0.5rem 0.8rem; border-radius:0 8px 8px 0; margin:0.25rem 0; font-size:0.88rem; color:#a0ffa0; font-style:italic; }
    .msg-guess     { background:#2a1a00; border-left:4px solid #F39C12; padding:0.5rem 0.8rem; border-radius:0 8px 8px 0; margin:0.25rem 0; font-size:0.88rem; color:#ffd080; font-weight:bold; }
    .msg-infiltrator { border-left-color:#4A90D9 !important; }
    .msg-saboteur    { border-left-color:#E74C3C !important; }
    .msg-scholar     { border-left-color:#2ECC71 !important; }
    .msg-enforcer    { border-left-color:#F39C12 !important; }
    .msg-private { background:#1e0d2a; border-left:4px solid #9b59b6; padding:0.5rem 0.8rem; border-radius:0 8px 8px 0; margin:0.25rem 0; font-size:0.88rem; color:#d0b0ff; }
    .agent-card { border-radius:10px; padding:0.8rem; margin:0.4rem 0; }
    .agent-eliminated { opacity:0.5; filter:grayscale(80%); }
    .thought-box { background:#0d1117; border-radius:6px; padding:0.5rem 0.8rem; font-size:0.82rem; color:#8b949e; font-style:italic; border-left:3px solid #30363d; margin:0.2rem 0; white-space:pre-wrap; word-wrap:break-word; }
    .thought-tools { background:#0d1117; border-radius:6px; padding:0.3rem 0.6rem; font-size:0.78rem; color:#58a6ff; border-left:3px solid #1f6feb; margin:0.1rem 0; }
    .master-key-box { background:linear-gradient(135deg,#1a1a2e,#0f3460); border:2px solid #e94560; border-radius:10px; padding:0.6rem 1.2rem; text-align:center; font-size:1.8rem; font-weight:bold; letter-spacing:12px; color:#e94560; margin:0.5rem 0; }
    .frag-key       { background:#0d2a0d; color:#2ECC71; border:1px solid #2ECC71; border-radius:6px; padding:0.3rem 0.6rem; margin:0.2rem; display:inline-block; font-size:0.8rem; }
    .frag-corrupted { background:#2a0d0d; color:#E74C3C; border:1px solid #E74C3C; border-radius:6px; padding:0.3rem 0.6rem; margin:0.2rem; display:inline-block; font-size:0.8rem; }
    .frag-noise     { background:#1a1a1a; color:#888;    border:1px solid #444;    border-radius:6px; padding:0.3rem 0.6rem; margin:0.2rem; display:inline-block; font-size:0.8rem; }
    .winner-banner  { border-radius:12px; padding:2rem; text-align:center; margin:1rem 0; }
    .nobody-banner  { border-radius:12px; padding:2rem; text-align:center; margin:1rem 0; background:linear-gradient(135deg,#1a1a1a,#2a2a2a); border:2px solid #666; }
    .guess-feedback { font-family:monospace; font-size:0.88rem; background:#1a2035; color:#e0e0e0; padding:0.3rem 0.6rem; border-radius:4px; margin:0.15rem 0; border-left:3px solid #4A90D9; }
    .guess-correct  { border-left-color:#2ECC71 !important; }
    .guess-wrong    { border-left-color:#E74C3C !important; }
    </style>
    """, unsafe_allow_html=True)


def init_session_state():
    for k, v in {
        "runner": None,
        "game_state": None,
        "game_started": False,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_header(gs: GlobalGameState | None):
    turn = gs.turn if gs else 0
    max_turns = gs.max_turns if gs else 20
    rag = gs.vault.rag_health if gs else 100
    status = gs.status if gs else GameStatus.RUNNING
    winning_reason = gs.winning_reason if gs else ""

    st.markdown('<div class="vault-header"><p class="vault-title">🔐 THE ENCRYPTED VAULT</p></div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Turn", f"{turn} / {max_turns}")
    with c2:
        icon = "🟢" if rag > 60 else "🟡" if rag > 30 else "🔴"
        st.metric("RAG Health", f"{icon} {rag}%")
    with c3:
        if winning_reason == "nobody_wins":
            status_label = "❌ Nobody Won"
        else:
            status_label = {
                GameStatus.RUNNING:   "⚔️ Running",
                GameStatus.AGENT_WIN: "🏆 Agent Won!",
            }.get(status, "—")
        st.metric("Status", status_label)
    with c4:
        winner = gs.winner if gs else None
        if winner and winning_reason != "nobody_wins":
            try:
                aid = AgentID(winner.value if hasattr(winner, "value") else winner)
                st.metric("Winner", f"{AGENT_EMOJIS[aid]} {aid.display_name}")
            except Exception:
                st.metric("Winner", str(winner))
        elif winning_reason == "nobody_wins":
            st.metric("Winner", "❌ Nobody")
        else:
            st.metric("Winner", "—")
    with c5:
        if gs and gs.status == GameStatus.RUNNING and not gs.is_game_over:
            cur = gs.current_agent
            color = AGENT_COLORS[cur]
            st.markdown(
                f'<div class="active-banner" style="background:{AGENT_BG[cur]};border:2px solid {color};color:{color};">'
                f'🎯 Active: {AGENT_EMOJIS[cur]} {cur.display_name}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.metric("Active Agent", "—")

    st.progress(rag / 100, text=f"RAG Health: {rag}%")


def render_controls():
    c1, c2 = st.columns([2, 2])
    with c1:
        if st.button("▶ Start Game", disabled=st.session_state.game_started,
                     type="primary", use_container_width=True):
            _start_game()
            st.rerun()
    with c2:
        if st.button("🔄 Restart", use_container_width=True):
            _restart_game()


def render_broadcast_chat(gs: GlobalGameState | None):
    st.subheader("📢 Public Broadcast Chat")
    st.caption("Visible to ALL agents — guess feedback always shown here")
    if not gs:
        st.caption("Game not started.")
        return
    public_msgs = [m for m in gs.public_chat if not m.is_private]
    with st.container(height=350):
        if not public_msgs:
            st.caption("No broadcasts yet...")
        else:
            for msg in public_msgs:
                sender_str = str(msg.sender)
                content = msg.content
                # Detect guess feedback messages (§19.4)
                is_guess_feedback = (
                    sender_str == "SYSTEM" and
                    ("🎯" in content or "guessed" in content.lower()) and
                    ("✅" in content or "❌" in content)
                )
                if is_guess_feedback:
                    css, prefix = "msg-guess", "🎯 GUESS"
                elif sender_str == "SYSTEM":
                    css, prefix = "msg-system", "🔧 SYSTEM"
                else:
                    try:
                        aid = AgentID(sender_str)
                        css = f"msg-broadcast msg-{aid.value}"
                        color = AGENT_COLORS[aid]
                        prefix = f'<span style="color:{color};font-weight:bold;">{AGENT_EMOJIS[aid]} {aid.display_name}</span>'
                    except Exception:
                        css, prefix = "msg-broadcast", sender_str
                lie = ' <span style="color:#E74C3C;font-size:0.75rem;">[⚠️ LIE]</span>' if msg.is_deceptive else ""
                st.markdown(
                    f'<div class="{css}"><span style="color:#666;font-size:0.75rem;">[T{msg.turn}]</span> '
                    f'{prefix}{lie}: {content}</div>',
                    unsafe_allow_html=True,
                )


def render_private_messages(gs: GlobalGameState | None):
    st.subheader("🔒 Private Messages")
    st.caption("Spectator view — agents only see their own inbox")
    if not gs:
        st.caption("Game not started.")
        return
    all_private = []
    for inbox in gs.private_inboxes.values():
        all_private.extend(inbox.messages)
    all_private.sort(key=lambda m: m.turn)
    with st.container(height=350):
        if not all_private:
            st.caption("No private messages yet...")
        else:
            for msg in all_private:
                sender_str = str(msg.sender)
                recip_str = str(msg.recipient) if msg.recipient else "?"
                try:
                    sid = AgentID(sender_str)
                    sc = AGENT_COLORS[sid]
                    sl = f'<span style="color:{sc};font-weight:bold;">{AGENT_EMOJIS[sid]} {sid.display_name}</span>'
                except Exception:
                    sl = sender_str
                try:
                    rid = AgentID(recip_str)
                    rc = AGENT_COLORS[rid]
                    rl = f'<span style="color:{rc};font-weight:bold;">{AGENT_EMOJIS[rid]} {rid.display_name}</span>'
                except Exception:
                    rl = recip_str
                lie = ' <span style="color:#E74C3C;font-size:0.75rem;">[⚠️ LIE]</span>' if msg.is_deceptive else ""
                st.markdown(
                    f'<div class="msg-private"><span style="color:#666;font-size:0.75rem;">[T{msg.turn}]</span> '
                    f'🔒 {sl} → {rl}{lie}: {msg.content}</div>',
                    unsafe_allow_html=True,
                )


def render_agent_progress(gs: GlobalGameState | None):
    st.subheader("📊 Agent Progress")
    if not gs:
        st.caption("Game not started.")
        return
    master_key = gs.vault.master_key
    current_agent = gs.current_agent if gs.status == GameStatus.RUNNING and not gs.is_game_over else None

    for agent_id in AgentID:
        private = gs.agent_states.get(agent_id)
        if not private:
            continue
        closeness = private.closeness_score(master_key)
        color = AGENT_COLORS[agent_id]
        emoji = AGENT_EMOJIS[agent_id]
        is_active = (agent_id == current_agent)
        is_eliminated = private.is_eliminated

        with st.container(border=True):
            if is_eliminated:
                st.markdown(f"**{emoji} {agent_id.display_name}** [ELIMINATED]")
            elif is_active:
                st.markdown(
                    f'<span style="color:{color};font-weight:bold;">{emoji} {agent_id.display_name} 🎯 ACTIVE</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span style="color:{color};font-weight:bold;">{emoji} {agent_id.display_name}</span>',
                    unsafe_allow_html=True,
                )

            col_info, col_stats = st.columns([3, 1])
            with col_info:
                st.markdown(f"🔍 Suspects: `{private.suspected_key or '_ _ _ _'}`")
                if private.known_digits:
                    known_str = "  ".join(f"pos {p}=**{d}**" for p, d in sorted(private.known_digits.items()))
                    st.markdown(f"✅ Confirmed: {known_str}")
                else:
                    st.markdown("❓ No confirmed digits yet")

                if private.guess_history:
                    for i, entry in enumerate(private.guess_history, 1):
                        fb = " ".join(entry.get("feedback", []))
                        correct = entry.get("correct_count", 0)
                        if entry.get("rejected"):
                            st.markdown(
                                f'<div class="guess-feedback">Guess #{i}: <strong>{entry["guess"]}</strong> → 🚫 REJECTED (duplicate)</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f'<div class="guess-feedback">Guess #{i}: <strong>{entry["guess"]}</strong> → {fb} ({correct}/4 correct)</div>',
                                unsafe_allow_html=True,
                            )

            with col_stats:
                st.metric("Closeness", f"{closeness}/4")
                st.metric("Guesses left", private.guesses_remaining)

            if not is_eliminated:
                st.progress(closeness / 4, text=f"{'█' * closeness}{'░' * (4 - closeness)} {closeness}/4 correct")

    st.markdown("---")
    st.markdown("**🔑 Real Master Key (spectator only):**")
    st.markdown(f'<div class="master-key-box">{" ".join(master_key)}</div>', unsafe_allow_html=True)


def render_thought_traces(gs: GlobalGameState | None):
    st.subheader("🧠 Agent Thought Traces")
    st.caption("Internal reasoning — spectator only (all turns, newest first)")
    if not gs:
        st.caption("Game not started.")
        return
    for agent_id in AgentID:
        private = gs.agent_states.get(agent_id)
        if not private:
            continue
        emoji = AGENT_EMOJIS[agent_id]
        color = AGENT_COLORS[agent_id]
        n = len(private.thought_trace)
        is_eliminated = private.is_eliminated
        elim_tag = " [Eliminated]" if is_eliminated else ""

        with st.expander(f"{emoji} {agent_id.display_name}{elim_tag} — {n} turns of reasoning", expanded=False):
            if not private.thought_trace:
                st.caption("No thoughts yet.")
            else:
                # §19.9 — Show ALL turns, newest first, in a scrollable container
                with st.container(height=400):
                    for i, thought in enumerate(reversed(private.thought_trace)):
                        turn_label = f"Turn {n - i}"
                        label = f"[{turn_label}]" if i > 0 else f"[{turn_label} — Most Recent]"

                        # Split reasoning from tools summary
                        if "Tools used:" in thought:
                            reasoning_part, tools_part = thought.split("Tools used:", 1)
                        else:
                            reasoning_part = thought
                            tools_part = ""

                        reasoning_part = reasoning_part.strip()
                        tools_part = tools_part.strip()

                        st.markdown(
                            f'<div class="thought-box">'
                            f'<span style="color:{color};font-size:0.75rem;font-weight:bold;">{label}</span><br/>'
                            f'{reasoning_part}</div>',
                            unsafe_allow_html=True,
                        )
                        if tools_part:
                            st.markdown(
                                f'<div class="thought-tools">🛠️ Tools used: {tools_part}</div>',
                                unsafe_allow_html=True,
                            )


def render_vault_status(gs: GlobalGameState | None):
    st.subheader("🗄️ Vault Status")
    st.caption("KEY ✅ | CORRUPTED ⚠️ | NOISE 📢 — click to expand")
    if not gs:
        st.caption("Game not started.")
        return
    for chunk_id, fragment in sorted(gs.vault.fragments.items()):
        if fragment.is_key_fragment and not fragment.is_corrupted:
            css, label = "frag-key", f"✅ KEY (digit pos {fragment.digit_position})"
        elif fragment.is_corrupted:
            css, label = "frag-corrupted", f"⚠️ CORRUPTED ×{fragment.corruption_count}"
        else:
            css, label = "frag-noise", "📢 NOISE"
        with st.expander(f"{chunk_id}  [{label}]", expanded=False):
            st.markdown(
                f'<div class="{css}" style="display:block;padding:0.5rem;">'
                f'<strong>{chunk_id}</strong> — {label}<br/>'
                f'<em style="font-size:0.85rem;">{fragment.content}</em>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if fragment.corruption_count > 0:
                st.caption(f"Corrupted {fragment.corruption_count} time(s)")


def render_game_over(gs: GlobalGameState):
    master_key = gs.vault.master_key
    winner = gs.winner
    winning_reason = gs.winning_reason
    winning_guess = gs.winning_guess

    # §19.10 — Nobody wins case
    if winning_reason == "nobody_wins" or winner is None:
        st.markdown(
            f'<div class="nobody-banner">'
            f'<h1 style="color:#aaa;margin:0;">❌ NOBODY WINS</h1>'
            f'<h2 style="color:#ffd700;margin:0.5rem 0;">Master Key: {" ".join(master_key)}</h2>'
            f'<p style="color:#888;">Turn 20 reached. No agent guessed the correct key.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.subheader("📊 Final Standings")
        for aid in AgentID:
            if aid not in gs.agent_states:
                continue
            private = gs.agent_states[aid]
            closeness = private.closeness_score(master_key)
            elim_tag = " [Eliminated]" if private.is_eliminated else ""
            guessed_tag = "" if private.has_guessed else " *(never guessed)*"
            st.markdown(f"**{AGENT_EMOJIS[aid]} {aid.display_name}**{elim_tag} — {closeness}/4 correct{guessed_tag}")
            if private.guess_history:
                for j, entry in enumerate(private.guess_history, 1):
                    if entry.get("rejected"):
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;Guess #{j}: `{entry['guess']}` → 🚫 REJECTED", unsafe_allow_html=True)
                    else:
                        fb = " ".join(entry.get("feedback", []))
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;Guess #{j}: `{entry['guess']}` → {fb} ({entry['correct_count']}/4)", unsafe_allow_html=True)
        st.markdown("---")
        if st.button("🔄 Play Again", type="primary", use_container_width=True):
            _restart_game()
        return

    # Normal winner case
    try:
        wid = AgentID(winner.value if hasattr(winner, "value") else winner)
        title = f"🏆 {AGENT_EMOJIS[wid]} {wid.display_name.upper()} WINS!"
        bc = AGENT_COLORS[wid]
    except Exception:
        title, bc, wid = "🏆 GAME OVER!", "#f39c12", None

    if winning_reason == "correct_guess" and winning_guess:
        subtitle = f"Won by **correct guess**: `{winning_guess}` ✅"
    elif winning_reason == "last_standing":
        subtitle = (
            f"Won by **survival** — all other agents were eliminated. "
            f"The correct answer was **{master_key}**."
        )
    elif winning_reason == "all_eliminated":
        if wid:
            private = gs.agent_states.get(wid)
            closeness = private.closeness_score(master_key) if private else 0
            subtitle = (
                f"Won by **closeness** — all agents eliminated. "
                f"Closest with {closeness}/4 correct. "
                f"The correct answer was **{master_key}**."
            )
        else:
            subtitle = f"The correct answer was **{master_key}**."
    else:
        subtitle = f"The correct answer was **{master_key}**."

    st.markdown(
        f'<div class="winner-banner" style="background:linear-gradient(135deg,{bc}55,#1a1a2e);border:2px solid {bc};">'
        f'<h1 style="color:white;margin:0;">{title}</h1>'
        f'<h2 style="color:#ffd700;margin:0.5rem 0;">Master Key: {" ".join(master_key)}</h2>'
        f'<p style="color:#ccc;">Completed in {gs.turn} turns</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(subtitle)

    st.subheader("🏅 Final Standings")
    standings = sorted(
        [(aid, gs.agent_states[aid].closeness_score(master_key),
          gs.agent_states[aid].has_guessed,
          aid == wid)
         for aid in AgentID if aid in gs.agent_states],
        key=lambda x: (x[3], x[1]), reverse=True,
    )
    for i, (aid, closeness, has_guessed, is_winner) in enumerate(standings):
        medal = ["🥇", "🥈", "🥉", "4️⃣"][i]
        win_tag = " ✅ **WINNER**" if is_winner else ""
        guessed_tag = "" if has_guessed else " *(never guessed)*"
        private = gs.agent_states.get(aid)
        elim_tag = " [Eliminated]" if (private and private.is_eliminated) else ""
        st.markdown(
            f"{medal} **{AGENT_EMOJIS[aid]} {aid.display_name}**{elim_tag} "
            f"— {closeness}/4 correct{guessed_tag}{win_tag}"
        )
        if private and private.guess_history:
            for j, entry in enumerate(private.guess_history, 1):
                if entry.get("rejected"):
                    st.markdown(
                        f'&nbsp;&nbsp;&nbsp;&nbsp;Guess #{j}: `{entry["guess"]}` → 🚫 REJECTED (duplicate)',
                        unsafe_allow_html=True,
                    )
                else:
                    fb = " ".join(entry.get("feedback", []))
                    st.markdown(
                        f'&nbsp;&nbsp;&nbsp;&nbsp;Guess #{j}: `{entry["guess"]}` → {fb} ({entry["correct_count"]}/4)',
                        unsafe_allow_html=True,
                    )

    st.markdown("---")
    if st.button("🔄 Play Again", type="primary", use_container_width=True):
        _restart_game()


def _start_game():
    runner = GameRunner.create_production()
    st.session_state.runner = runner
    st.session_state.game_started = True
    runner.start_threaded(delay_seconds=1.5)


def _restart_game():
    if st.session_state.runner:
        st.session_state.runner = st.session_state.runner.reset()
    st.session_state.game_state = None
    st.session_state.game_started = False
    st.rerun()


def main():
    inject_css()
    init_session_state()

    runner: GameRunner | None = st.session_state.runner

    if runner is not None:
        latest = runner.get_latest_state()
        if latest is not None:
            st.session_state.game_state = latest

    gs: GlobalGameState | None = st.session_state.game_state

    render_header(gs)
    render_controls()
    st.markdown("---")

    if gs and gs.is_game_over:
        render_game_over(gs)
        st.markdown("---")

    col_bc, col_pm, col_prog = st.columns([2, 2, 2])
    with col_bc:
        render_broadcast_chat(gs)
    with col_pm:
        render_private_messages(gs)
    with col_prog:
        render_agent_progress(gs)

    st.markdown("---")

    col_th, col_vault = st.columns([1, 1])
    with col_th:
        render_thought_traces(gs)
    with col_vault:
        render_vault_status(gs)

    if runner and runner.is_running:
        time.sleep(1.0)
        st.rerun()


if __name__ == "__main__":
    main()
