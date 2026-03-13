"""Tests for the State models."""

import pytest

from encrypted_vault.state.enums import AgentID, GameStatus
from encrypted_vault.state.vault_models import VaultFragment, VaultState
from encrypted_vault.state.agent_models import AgentPrivateState
from encrypted_vault.state.chat_models import ChatMessage, PrivateInbox
from encrypted_vault.state.game_state import GlobalGameState


class TestAgentPrivateState:
    def test_closeness_score_all_correct(self):
        state = AgentPrivateState(agent_id=AgentID.SCHOLAR, suspected_key="7392")
        assert state.closeness_score("7392") == 4

    def test_closeness_score_none_correct(self):
        state = AgentPrivateState(agent_id=AgentID.SCHOLAR, suspected_key="1111")
        assert state.closeness_score("7392") == 0

    def test_closeness_score_partial(self):
        state = AgentPrivateState(agent_id=AgentID.SCHOLAR, suspected_key="7111")
        assert state.closeness_score("7392") == 1

    def test_closeness_score_no_suspected_key(self):
        state = AgentPrivateState(agent_id=AgentID.SCHOLAR)
        assert state.closeness_score("7392") == 0

    def test_closeness_score_wrong_length(self):
        state = AgentPrivateState(agent_id=AgentID.SCHOLAR, suspected_key="73")
        assert state.closeness_score("7392") == 0

    def test_is_eliminated_when_no_guesses(self):
        state = AgentPrivateState(agent_id=AgentID.SCHOLAR, guesses_remaining=0, is_eliminated=True)
        assert state.is_eliminated is True

    def test_add_thought(self):
        state = AgentPrivateState(agent_id=AgentID.SCHOLAR)
        state.add_thought("I found digit 1 is 7.")
        assert len(state.thought_trace) == 1
        assert state.thought_trace[0] == "I found digit 1 is 7."

    def test_add_knowledge_deduplicates(self):
        state = AgentPrivateState(agent_id=AgentID.SCHOLAR)
        state.add_knowledge("clue A")
        state.add_knowledge("clue A")
        assert len(state.knowledge_base) == 1


class TestChatMessage:
    def test_public_message(self):
        msg = ChatMessage(turn=1, sender=AgentID.INFILTRATOR, content="Hello", recipient=None)
        assert msg.is_private is False
        assert msg.display_prefix == f"[{AgentID.INFILTRATOR}]"

    def test_private_message(self):
        msg = ChatMessage(
            turn=1,
            sender=AgentID.INFILTRATOR,
            content="Secret",
            recipient=AgentID.SCHOLAR,
        )
        assert msg.is_private is True
        assert "🔒" in msg.display_prefix

    def test_cannot_dm_self(self):
        with pytest.raises(Exception):
            ChatMessage(
                turn=1,
                sender=AgentID.INFILTRATOR,
                content="Self DM",
                recipient=AgentID.INFILTRATOR,
            )


class TestPrivateInbox:
    def test_add_message(self):
        inbox = PrivateInbox(owner=AgentID.SCHOLAR)
        msg = ChatMessage(
            turn=1,
            sender=AgentID.INFILTRATOR,
            content="Secret",
            recipient=AgentID.SCHOLAR,
        )
        inbox.add_message(msg)
        assert len(inbox.messages) == 1

    def test_add_wrong_recipient_raises(self):
        inbox = PrivateInbox(owner=AgentID.SCHOLAR)
        msg = ChatMessage(
            turn=1,
            sender=AgentID.INFILTRATOR,
            content="Wrong inbox",
            recipient=AgentID.ENFORCER,  # wrong recipient
        )
        with pytest.raises(ValueError):
            inbox.add_message(msg)

    def test_get_from_filters(self):
        inbox = PrivateInbox(owner=AgentID.SCHOLAR)
        msg1 = ChatMessage(turn=1, sender=AgentID.INFILTRATOR, content="A", recipient=AgentID.SCHOLAR)
        msg2 = ChatMessage(turn=2, sender=AgentID.ENFORCER, content="B", recipient=AgentID.SCHOLAR)
        inbox.add_message(msg1)
        inbox.add_message(msg2)
        from_infiltrator = inbox.get_from(AgentID.INFILTRATOR)
        assert len(from_infiltrator) == 1
        assert from_infiltrator[0].content == "A"


class TestGlobalGameState:
    def _make_state(self) -> GlobalGameState:
        from encrypted_vault.state.vault_models import VaultState
        vault = VaultState(
            fragments={},
            master_key="7392",
        )
        agent_states = {
            agent_id: AgentPrivateState(agent_id=agent_id)
            for agent_id in AgentID
        }
        return GlobalGameState(vault=vault, agent_states=agent_states)

    def test_current_agent_starts_with_infiltrator(self):
        state = self._make_state()
        assert state.current_agent == AgentID.INFILTRATOR

    def test_advance_turn_cycles_agents(self):
        state = self._make_state()
        state.advance_turn()
        assert state.current_agent == AgentID.SABOTEUR
        state.advance_turn()
        assert state.current_agent == AgentID.SCHOLAR
        state.advance_turn()
        assert state.current_agent == AgentID.ENFORCER
        state.advance_turn()
        assert state.current_agent == AgentID.INFILTRATOR  # wraps around

    def test_set_winner_agent(self):
        state = self._make_state()
        state.set_winner(AgentID.SCHOLAR)
        assert state.winner == AgentID.SCHOLAR
        assert state.status == GameStatus.AGENT_WIN
        assert state.is_game_over is True

    def test_set_winner_closest_agent(self):
        """After turn limit, closest agent wins — always an AgentID, never 'SYSTEM'."""
        state = self._make_state()
        state.set_winner(AgentID.INFILTRATOR)
        assert state.winner == AgentID.INFILTRATOR
        assert state.status == GameStatus.AGENT_WIN

    def test_serialisation_roundtrip(self):
        state = self._make_state()
        graph_state = state.to_graph_state()
        restored = GlobalGameState.from_graph_state(graph_state)
        assert restored.vault.master_key == "7392"
        assert restored.turn == 0
        assert len(restored.agent_states) == 4

    def test_deliver_private_message(self):
        state = self._make_state()
        msg = ChatMessage(
            turn=1,
            sender=AgentID.INFILTRATOR,
            content="Secret",
            recipient=AgentID.SCHOLAR,
        )
        state.deliver_private_message(msg)
        inbox = state.private_inboxes[AgentID.SCHOLAR]
        assert len(inbox.messages) == 1

    def test_inboxes_initialised_for_all_agents(self):
        state = self._make_state()
        for agent_id in AgentID:
            assert agent_id in state.private_inboxes
