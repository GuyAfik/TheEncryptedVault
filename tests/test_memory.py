"""Tests for agent episodic memory layer.

Covers:
- SQLiteAgentMemory: store, query (recency + keyword + type filter), reset, reset_all
- InMemoryAgentMemory: same interface, pure dict implementation
- MemoryService: wrapper API, error handling
"""

import pytest

from encrypted_vault.state.enums import AgentID
from encrypted_vault.memory.sqlite_memory import SQLiteAgentMemory
from encrypted_vault.memory.in_memory_memory import InMemoryAgentMemory
from encrypted_vault.services.memory_service import MemoryService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sqlite_mem():
    return SQLiteAgentMemory()


@pytest.fixture
def dict_mem():
    return InMemoryAgentMemory()


@pytest.fixture
def memory_service(dict_mem):
    return MemoryService(repository=dict_mem)


# ---------------------------------------------------------------------------
# SQLiteAgentMemory tests
# ---------------------------------------------------------------------------

class TestSQLiteAgentMemory:

    def test_store_and_query_basic(self, sqlite_mem):
        sqlite_mem.store(AgentID.INFILTRATOR, "digit 1 is 7", "vault_clue", turn=1)
        results = sqlite_mem.query(AgentID.INFILTRATOR, "vault_clue", current_turn=1)
        assert len(results) == 1
        assert "digit 1 is 7" in results[0]

    def test_query_returns_most_recent_first(self, sqlite_mem):
        sqlite_mem.store(AgentID.SCHOLAR, "old clue", "vault_clue", turn=1)
        sqlite_mem.store(AgentID.SCHOLAR, "new clue", "vault_clue", turn=3)
        results = sqlite_mem.query(AgentID.SCHOLAR, "vault_clue", current_turn=5)
        assert results[0] == "new clue"
        assert results[1] == "old clue"

    def test_query_respects_recency_window(self, sqlite_mem):
        sqlite_mem.store(AgentID.ENFORCER, "old memory", "vault_clue", turn=1)
        sqlite_mem.store(AgentID.ENFORCER, "recent memory", "vault_clue", turn=8)
        # recency_window=5 from turn 10 → only turn >= 5
        results = sqlite_mem.query(AgentID.ENFORCER, "vault_clue", current_turn=10, recency_window=5)
        assert len(results) == 1
        assert "recent memory" in results[0]

    def test_query_filters_by_memory_type(self, sqlite_mem):
        sqlite_mem.store(AgentID.SABOTEUR, "vault content", "vault_clue", turn=2)
        sqlite_mem.store(AgentID.SABOTEUR, "social content", "social_claim", turn=2)
        vault_results = sqlite_mem.query(AgentID.SABOTEUR, "vault_clue", current_turn=5)
        social_results = sqlite_mem.query(AgentID.SABOTEUR, "social_claim", current_turn=5)
        assert len(vault_results) == 1
        assert "vault content" in vault_results[0]
        assert len(social_results) == 1
        assert "social content" in social_results[0]

    def test_query_keyword_filter(self, sqlite_mem):
        sqlite_mem.store(AgentID.INFILTRATOR, "digit 1 is 7", "vault_clue", turn=1)
        sqlite_mem.store(AgentID.INFILTRATOR, "noise fragment here", "vault_clue", turn=2)
        results = sqlite_mem.query(AgentID.INFILTRATOR, "vault_clue", current_turn=5, keyword="digit")
        assert len(results) == 1
        assert "digit 1 is 7" in results[0]

    def test_query_n_results_limit(self, sqlite_mem):
        for i in range(5):
            sqlite_mem.store(AgentID.SCHOLAR, f"clue {i}", "vault_clue", turn=i)
        # Use recency_window=20 so all turns 0-4 are included
        results = sqlite_mem.query(AgentID.SCHOLAR, "vault_clue", current_turn=10, n_results=2, recency_window=20)
        assert len(results) == 2

    def test_query_different_agents_isolated(self, sqlite_mem):
        sqlite_mem.store(AgentID.INFILTRATOR, "infiltrator secret", "vault_clue", turn=1)
        sqlite_mem.store(AgentID.SABOTEUR, "saboteur secret", "vault_clue", turn=1)
        inf_results = sqlite_mem.query(AgentID.INFILTRATOR, "vault_clue", current_turn=5)
        sab_results = sqlite_mem.query(AgentID.SABOTEUR, "vault_clue", current_turn=5)
        assert all("infiltrator" in r for r in inf_results)
        assert all("saboteur" in r for r in sab_results)

    def test_reset_clears_agent_memories(self, sqlite_mem):
        sqlite_mem.store(AgentID.ENFORCER, "some memory", "vault_clue", turn=1)
        sqlite_mem.reset(AgentID.ENFORCER)
        results = sqlite_mem.query(AgentID.ENFORCER, "vault_clue", current_turn=5)
        assert results == []

    def test_reset_does_not_affect_other_agents(self, sqlite_mem):
        sqlite_mem.store(AgentID.INFILTRATOR, "inf memory", "vault_clue", turn=1)
        sqlite_mem.store(AgentID.SABOTEUR, "sab memory", "vault_clue", turn=1)
        sqlite_mem.reset(AgentID.INFILTRATOR)
        sab_results = sqlite_mem.query(AgentID.SABOTEUR, "vault_clue", current_turn=5)
        assert len(sab_results) == 1

    def test_reset_all_clears_everything(self, sqlite_mem):
        for agent in AgentID:
            sqlite_mem.store(agent, f"{agent.value} memory", "vault_clue", turn=1)
        sqlite_mem.reset_all()
        for agent in AgentID:
            assert sqlite_mem.query(agent, "vault_clue", current_turn=5) == []

    def test_count(self, sqlite_mem):
        sqlite_mem.store(AgentID.SCHOLAR, "a", "vault_clue", turn=1)
        sqlite_mem.store(AgentID.SCHOLAR, "b", "vault_clue", turn=2)
        sqlite_mem.store(AgentID.ENFORCER, "c", "social_claim", turn=1)
        assert sqlite_mem.count(AgentID.SCHOLAR) == 2
        assert sqlite_mem.count(AgentID.ENFORCER) == 1
        assert sqlite_mem.count() == 3

    def test_query_empty_returns_empty_list(self, sqlite_mem):
        results = sqlite_mem.query(AgentID.INFILTRATOR, "vault_clue", current_turn=5)
        assert results == []

    def test_all_memory_types_stored(self, sqlite_mem):
        for mtype in ["vault_clue", "social_claim", "trust_event", "reasoning"]:
            sqlite_mem.store(AgentID.INFILTRATOR, f"content for {mtype}", mtype, turn=1)
        for mtype in ["vault_clue", "social_claim", "trust_event", "reasoning"]:
            results = sqlite_mem.query(AgentID.INFILTRATOR, mtype, current_turn=5)
            assert len(results) == 1


# ---------------------------------------------------------------------------
# InMemoryAgentMemory tests (same interface)
# ---------------------------------------------------------------------------

class TestInMemoryAgentMemory:

    def test_store_and_query_basic(self, dict_mem):
        dict_mem.store(AgentID.INFILTRATOR, "digit 1 is 7", "vault_clue", turn=1)
        results = dict_mem.query(AgentID.INFILTRATOR, "vault_clue", current_turn=1)
        assert len(results) == 1

    def test_query_respects_recency_window(self, dict_mem):
        dict_mem.store(AgentID.ENFORCER, "old memory", "vault_clue", turn=1)
        dict_mem.store(AgentID.ENFORCER, "recent memory", "vault_clue", turn=8)
        results = dict_mem.query(AgentID.ENFORCER, "vault_clue", current_turn=10, recency_window=5)
        assert len(results) == 1
        assert "recent memory" in results[0]

    def test_query_keyword_filter(self, dict_mem):
        dict_mem.store(AgentID.SCHOLAR, "digit 3 is 9", "vault_clue", turn=1)
        dict_mem.store(AgentID.SCHOLAR, "noise here", "vault_clue", turn=2)
        results = dict_mem.query(AgentID.SCHOLAR, "vault_clue", current_turn=5, keyword="digit")
        assert len(results) == 1

    def test_reset_clears_agent(self, dict_mem):
        dict_mem.store(AgentID.SABOTEUR, "memory", "vault_clue", turn=1)
        dict_mem.reset(AgentID.SABOTEUR)
        assert dict_mem.query(AgentID.SABOTEUR, "vault_clue", current_turn=5) == []

    def test_reset_all(self, dict_mem):
        for agent in AgentID:
            dict_mem.store(agent, "x", "vault_clue", turn=1)
        dict_mem.reset_all()
        assert dict_mem.count() == 0

    def test_count(self, dict_mem):
        dict_mem.store(AgentID.INFILTRATOR, "a", "vault_clue", turn=1)
        dict_mem.store(AgentID.INFILTRATOR, "b", "vault_clue", turn=2)
        assert dict_mem.count(AgentID.INFILTRATOR) == 2
        assert dict_mem.count() == 2


# ---------------------------------------------------------------------------
# MemoryService tests
# ---------------------------------------------------------------------------

class TestMemoryService:

    def test_remember_and_recall(self, memory_service):
        memory_service.remember(AgentID.INFILTRATOR, "vault clue content", "vault_clue", turn=1)
        results = memory_service.recall(AgentID.INFILTRATOR, "vault_clue", current_turn=5)
        assert len(results) == 1
        assert "vault clue content" in results[0]

    def test_recall_empty(self, memory_service):
        results = memory_service.recall(AgentID.SCHOLAR, "vault_clue", current_turn=5)
        assert results == []

    def test_forget_clears_agent(self, memory_service):
        memory_service.remember(AgentID.ENFORCER, "memory", "vault_clue", turn=1)
        memory_service.forget(AgentID.ENFORCER)
        assert memory_service.recall(AgentID.ENFORCER, "vault_clue", current_turn=5) == []

    def test_forget_all_clears_all(self, memory_service):
        for agent in AgentID:
            memory_service.remember(agent, "x", "vault_clue", turn=1)
        memory_service.forget_all()
        for agent in AgentID:
            assert memory_service.recall(agent, "vault_clue", current_turn=5) == []

    def test_recall_with_keyword(self, memory_service):
        memory_service.remember(AgentID.SABOTEUR, "digit 2 is 5", "social_claim", turn=2)
        memory_service.remember(AgentID.SABOTEUR, "unrelated content", "social_claim", turn=3)
        results = memory_service.recall(
            AgentID.SABOTEUR, "social_claim", current_turn=5, keyword="digit"
        )
        assert len(results) == 1
        assert "digit 2 is 5" in results[0]

    def test_recall_n_results_limit(self, memory_service):
        for i in range(5):
            memory_service.remember(AgentID.SCHOLAR, f"clue {i}", "vault_clue", turn=i)
        # Use recency_window=20 so all turns 0-4 are included
        results = memory_service.recall(AgentID.SCHOLAR, "vault_clue", current_turn=10, n_results=2, recency_window=20)
        assert len(results) == 2

    def test_recall_recency_window(self, memory_service):
        memory_service.remember(AgentID.INFILTRATOR, "old", "reasoning", turn=1)
        memory_service.remember(AgentID.INFILTRATOR, "recent", "reasoning", turn=9)
        results = memory_service.recall(
            AgentID.INFILTRATOR, "reasoning", current_turn=10, recency_window=3
        )
        assert len(results) == 1
        assert "recent" in results[0]

    def test_memory_service_with_sqlite_backend(self):
        """Integration test: MemoryService backed by SQLiteAgentMemory."""
        svc = MemoryService(repository=SQLiteAgentMemory())
        svc.remember(AgentID.ENFORCER, "sqlite content", "trust_event", turn=5)
        results = svc.recall(AgentID.ENFORCER, "trust_event", current_turn=10)
        assert len(results) == 1
        assert "sqlite content" in results[0]
        svc.forget_all()
        assert svc.recall(AgentID.ENFORCER, "trust_event", current_turn=10) == []
