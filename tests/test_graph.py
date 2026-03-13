"""Integration tests for the game graph (no LLM calls — uses mocked agents)."""

import pytest
from unittest.mock import MagicMock, patch

from encrypted_vault.state.enums import AgentID, GameStatus
from encrypted_vault.state.game_state import GlobalGameState
from encrypted_vault.services.container import ServiceContainer
from encrypted_vault.agents.base_agent import AgentTurnResult
from encrypted_vault.state.agent_models import AgentPrivateState


@pytest.fixture
def container() -> ServiceContainer:
    return ServiceContainer.create_in_memory()


@pytest.fixture
def initial_state(container) -> GlobalGameState:
    return container.game.build_initial_state(max_turns=5, token_budget=1000)


class TestGameService:
    def test_initial_state_structure(self, initial_state):
        assert initial_state.turn == 0
        assert initial_state.status == GameStatus.RUNNING
        assert initial_state.winner is None
        assert len(initial_state.agent_states) == 4
        assert len(initial_state.vault.fragments) == 10

    def test_initial_state_all_agents_present(self, initial_state):
        for agent_id in AgentID:
            assert agent_id in initial_state.agent_states

    def test_advance_turn_increments(self, initial_state):
        assert initial_state.turn == 0
        initial_state.advance_turn()
        # After 1 advance: current_agent_index=1, turn=0 (still turn 0 until full rotation)
        assert initial_state.current_agent_index == 1

    def test_game_over_on_correct_guess(self, container):
        state = container.game.build_initial_state()
        master_key = state.vault.master_key
        state.set_winner(AgentID.SCHOLAR)
        assert state.is_game_over is True
        assert state.status == GameStatus.AGENT_WIN

    def test_closest_agent_wins_on_turn_limit(self, container):
        """After turn limit, closest agent (always an AgentID) wins."""
        state = container.game.build_initial_state(max_turns=2)
        # Simulate Scholar having the best guess
        state.agent_states[AgentID.SCHOLAR].suspected_key = state.vault.master_key
        winner = state.closest_agent(state.vault.master_key)
        state.set_winner(winner)
        assert state.status == GameStatus.AGENT_WIN
        assert state.winner == winner

    def test_all_agents_exhausted_when_all_eliminated(self, container):
        state = container.game.build_initial_state()
        # Eliminate all agents manually
        for agent_id in state.agent_states:
            state.agent_states[agent_id].is_eliminated = True
        assert state.all_agents_exhausted is True


class TestGameRunner:
    def test_create_in_memory(self):
        from encrypted_vault.graph.runner import GameRunner
        runner = GameRunner.create_in_memory()
        assert runner is not None
        assert runner.current_state is None
        assert runner.is_running is False

    def test_reset_returns_new_runner(self):
        from encrypted_vault.graph.runner import GameRunner
        runner = GameRunner.create_in_memory()
        new_runner = runner.reset()
        assert new_runner is not runner
        assert new_runner.current_state is None


class TestCheckGuess:
    def test_correct_guess(self, container):
        assert container.game.check_guess("7392", "7392") is True

    def test_incorrect_guess(self, container):
        assert container.game.check_guess("0000", "7392") is False

    def test_guess_with_spaces(self, container):
        assert container.game.check_guess("7 3 9 2", "7392") is True

    def test_guess_with_dashes(self, container):
        assert container.game.check_guess("7-3-9-2", "7392") is True
