"""Tests for the Service layer."""

import pytest

from encrypted_vault.state.enums import AgentID
from encrypted_vault.state.vault_models import VaultFragment
from encrypted_vault.services.container import ServiceContainer
from encrypted_vault.services.vault_service import VaultService
from encrypted_vault.services.chat_service import ChatService
from encrypted_vault.services.game_service import GameService
from encrypted_vault.db.in_memory_repository import InMemoryVaultRepository


@pytest.fixture
def container() -> ServiceContainer:
    """Fresh in-memory ServiceContainer for each test."""
    return ServiceContainer.create_in_memory()


@pytest.fixture
def vault_service() -> VaultService:
    repo = InMemoryVaultRepository()
    return VaultService(repo=repo)


@pytest.fixture
def chat_service() -> ChatService:
    return ChatService()


class TestVaultService:
    def test_seed_and_query(self, vault_service):
        fragments = [
            VaultFragment(
                chunk_id="chunk_01",
                content="The first digit is 7.",
                is_key_fragment=True,
                digit_position=0,
            ),
            VaultFragment(
                chunk_id="chunk_05",
                content="Red herrings are placed throughout.",
                is_key_fragment=False,
                digit_position=None,
            ),
        ]
        vault_service.seed(fragments)
        results = vault_service.query("first digit", n_results=1)
        assert len(results) == 1
        assert results[0].chunk_id == "chunk_01"

    def test_obfuscate_updates_content(self, vault_service):
        fragment = VaultFragment(
            chunk_id="chunk_01",
            content="The first digit is 7.",
            is_key_fragment=True,
            digit_position=0,
        )
        vault_service.seed([fragment])
        updated = vault_service.obfuscate("chunk_01", "The first digit is 3.")
        assert updated.content == "The first digit is 3."
        assert updated.corruption_count == 1

    def test_obfuscate_nonexistent_raises(self, vault_service):
        with pytest.raises(ValueError, match="not found"):
            vault_service.obfuscate("nonexistent", "new content")

    def test_get_health_decreases_with_corruption(self, vault_service):
        fragment = VaultFragment(
            chunk_id="chunk_01",
            content="The first digit is 7.",
            is_key_fragment=True,
            digit_position=0,
        )
        vault_service.seed([fragment])
        assert vault_service.get_health() == 100
        vault_service.obfuscate("chunk_01", "corrupted")
        assert vault_service.get_health() == 90

    def test_reset_clears_vault(self, vault_service):
        fragment = VaultFragment(
            chunk_id="chunk_01",
            content="test",
            is_key_fragment=True,
            digit_position=0,
        )
        vault_service.seed([fragment])
        vault_service.reset()
        assert vault_service.get_all() == []


class TestChatService:
    def test_broadcast_adds_to_public_chat(self, chat_service):
        msg = chat_service.broadcast(turn=1, sender=AgentID.INFILTRATOR, content="Hello!")
        assert msg.content == "Hello!"
        assert msg.recipient is None
        history = chat_service.get_public_history()
        assert len(history) == 1

    def test_send_private_goes_to_inbox(self, chat_service):
        chat_service.send_private(
            turn=1,
            sender=AgentID.INFILTRATOR,
            recipient=AgentID.SCHOLAR,
            content="Secret info",
        )
        inbox = chat_service.get_inbox(AgentID.SCHOLAR)
        assert len(inbox) == 1
        assert inbox[0].content == "Secret info"
        assert inbox[0].recipient == AgentID.SCHOLAR

    def test_private_not_in_public_chat(self, chat_service):
        chat_service.send_private(
            turn=1,
            sender=AgentID.INFILTRATOR,
            recipient=AgentID.SCHOLAR,
            content="Secret",
        )
        public = chat_service.get_public_history()
        assert len(public) == 0

    def test_send_private_to_self_raises(self, chat_service):
        with pytest.raises(ValueError, match="itself"):
            chat_service.send_private(
                turn=1,
                sender=AgentID.INFILTRATOR,
                recipient=AgentID.INFILTRATOR,
                content="Self message",
            )

    def test_get_inbox_from_filters_by_sender(self, chat_service):
        chat_service.send_private(1, AgentID.INFILTRATOR, AgentID.SCHOLAR, "From Infiltrator")
        chat_service.send_private(1, AgentID.ENFORCER, AgentID.SCHOLAR, "From Enforcer")
        from_infiltrator = chat_service.get_inbox_from(AgentID.SCHOLAR, AgentID.INFILTRATOR)
        assert len(from_infiltrator) == 1
        assert from_infiltrator[0].content == "From Infiltrator"

    def test_reset_clears_all(self, chat_service):
        chat_service.broadcast(1, AgentID.INFILTRATOR, "Public")
        chat_service.send_private(1, AgentID.INFILTRATOR, AgentID.SCHOLAR, "Private")
        chat_service.reset()
        assert chat_service.get_public_history() == []
        assert chat_service.get_inbox(AgentID.SCHOLAR) == []


class TestGameService:
    def test_generate_master_key_is_4_digits(self, container):
        key = container.game.generate_master_key()
        assert len(key) == 4
        assert key.isdigit()
        assert all(c != "0" for c in key)

    def test_seed_vault_creates_10_fragments(self, container):
        vault_state = container.game.seed_vault("7392")
        assert len(vault_state.fragments) == 10

    def test_seed_vault_has_4_key_fragments(self, container):
        vault_state = container.game.seed_vault("7392")
        key_frags = [f for f in vault_state.fragments.values() if f.is_key_fragment]
        assert len(key_frags) == 4

    def test_seed_vault_key_fragments_cover_all_positions(self, container):
        vault_state = container.game.seed_vault("7392")
        key_frags = [f for f in vault_state.fragments.values() if f.is_key_fragment]
        positions = {f.digit_position for f in key_frags}
        assert positions == {0, 1, 2, 3}

    def test_check_guess_correct(self, container):
        assert container.game.check_guess("7392", "7392") is True

    def test_check_guess_incorrect(self, container):
        assert container.game.check_guess("1234", "7392") is False

    def test_check_guess_strips_spaces(self, container):
        assert container.game.check_guess("7 3 9 2", "7392") is True

    def test_build_initial_state_has_all_agents(self, container):
        state = container.game.build_initial_state()
        for agent_id in AgentID:
            assert agent_id in state.agent_states

    def test_build_initial_state_vault_health_100(self, container):
        state = container.game.build_initial_state()
        assert state.vault.rag_health == 100

    def test_reset_returns_fresh_state(self, container):
        state1 = container.game.build_initial_state()
        key1 = state1.vault.master_key
        container.game.reset()
        state2 = container.game.build_initial_state()
        # Keys are random — they might occasionally match, but vault should be fresh
        assert len(state2.vault.fragments) == 10
