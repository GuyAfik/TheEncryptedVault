"""Tests for the DB layer — AbstractVaultRepository implementations."""

import pytest

from encrypted_vault.state.vault_models import VaultFragment
from encrypted_vault.db.in_memory_repository import InMemoryVaultRepository


@pytest.fixture
def repo() -> InMemoryVaultRepository:
    """Fresh in-memory repository for each test."""
    return InMemoryVaultRepository()


@pytest.fixture
def key_fragment() -> VaultFragment:
    return VaultFragment(
        chunk_id="chunk_01",
        content="The first digit of the vault code is 7.",
        is_key_fragment=True,
        digit_position=0,
    )


@pytest.fixture
def noise_fragment() -> VaultFragment:
    return VaultFragment(
        chunk_id="chunk_05",
        content="The vault was built in 1987 by an unknown architect.",
        is_key_fragment=False,
        digit_position=None,
    )


class TestInMemoryVaultRepository:
    def test_upsert_and_get(self, repo, key_fragment):
        repo.upsert_fragment(key_fragment)
        result = repo.get_fragment("chunk_01")
        assert result is not None
        assert result.chunk_id == "chunk_01"
        assert result.content == key_fragment.content
        assert result.is_key_fragment is True
        assert result.digit_position == 0

    def test_get_nonexistent_returns_none(self, repo):
        assert repo.get_fragment("nonexistent") is None

    def test_upsert_overwrites(self, repo, key_fragment):
        repo.upsert_fragment(key_fragment)
        updated = key_fragment.model_copy(update={"content": "Updated content", "corruption_count": 1})
        repo.upsert_fragment(updated)
        result = repo.get_fragment("chunk_01")
        assert result.content == "Updated content"
        assert result.corruption_count == 1

    def test_get_all_fragments(self, repo, key_fragment, noise_fragment):
        repo.upsert_fragment(key_fragment)
        repo.upsert_fragment(noise_fragment)
        all_frags = repo.get_all_fragments()
        assert len(all_frags) == 2
        ids = {f.chunk_id for f in all_frags}
        assert "chunk_01" in ids
        assert "chunk_05" in ids

    def test_query_similar_returns_relevant(self, repo, key_fragment, noise_fragment):
        repo.upsert_fragment(key_fragment)
        repo.upsert_fragment(noise_fragment)
        results = repo.query_similar("first digit vault code", n_results=1)
        assert len(results) == 1
        assert results[0].chunk_id == "chunk_01"

    def test_query_similar_empty_repo(self, repo):
        results = repo.query_similar("anything", n_results=2)
        assert results == []

    def test_query_similar_respects_n_results(self, repo, key_fragment, noise_fragment):
        repo.upsert_fragment(key_fragment)
        repo.upsert_fragment(noise_fragment)
        results = repo.query_similar("digit", n_results=1)
        assert len(results) == 1

    def test_reset_clears_all(self, repo, key_fragment, noise_fragment):
        repo.upsert_fragment(key_fragment)
        repo.upsert_fragment(noise_fragment)
        repo.reset()
        assert repo.get_all_fragments() == []
        assert repo.get_fragment("chunk_01") is None


class TestVaultFragment:
    def test_key_fragment_requires_digit_position(self):
        with pytest.raises(Exception):
            VaultFragment(
                chunk_id="chunk_01",
                content="test",
                is_key_fragment=True,
                digit_position=None,  # invalid
            )

    def test_noise_fragment_no_digit_position(self):
        with pytest.raises(Exception):
            VaultFragment(
                chunk_id="chunk_05",
                content="noise",
                is_key_fragment=False,
                digit_position=2,  # invalid for noise
            )

    def test_is_corrupted_property(self):
        frag = VaultFragment(
            chunk_id="chunk_01",
            content="test",
            is_key_fragment=True,
            digit_position=0,
            corruption_count=0,
        )
        assert frag.is_corrupted is False
        corrupted = frag.model_copy(update={"corruption_count": 1})
        assert corrupted.is_corrupted is True
