"""VaultService — business logic for RAG vault operations."""

from encrypted_vault.db.base_repository import AbstractVaultRepository
from encrypted_vault.state.vault_models import VaultFragment


class VaultService:
    """
    Encapsulates all business logic for interacting with the vault.

    Depends on AbstractVaultRepository — never on a concrete DB class.
    """

    def __init__(self, repo: AbstractVaultRepository) -> None:
        self._repo = repo

    # ── Read operations ────────────────────────────────────────────────────

    def query(self, search_term: str, n_results: int = 2) -> list[VaultFragment]:
        """
        Retrieve the top-n vault fragments most relevant to search_term.
        This is the agent-facing search tool.
        """
        return self._repo.query_similar(search_term, n_results=n_results)

    def get_all(self) -> list[VaultFragment]:
        """Return all fragments (used by UI for vault status display)."""
        return self._repo.get_all_fragments()

    def get_fragment(self, chunk_id: str) -> VaultFragment | None:
        """Return a specific fragment by ID."""
        return self._repo.get_fragment(chunk_id)

    def get_health(self) -> int:
        """
        Compute RAG health as a 0-100 score.
        Each corruption event reduces health by 10 points.
        """
        fragments = self._repo.get_all_fragments()
        if not fragments:
            return 100
        total_corruption = sum(f.corruption_count for f in fragments)
        return max(0, 100 - total_corruption * 10)

    # ── Write operations ───────────────────────────────────────────────────

    def obfuscate(self, chunk_id: str, new_text: str) -> VaultFragment:
        """
        Rewrite a vault fragment's content (Saboteur's primary tool).

        Increments corruption_count and persists the updated fragment.
        Raises ValueError if chunk_id does not exist.
        """
        fragment = self._repo.get_fragment(chunk_id)
        if fragment is None:
            raise ValueError(f"Fragment '{chunk_id}' not found in vault.")

        updated = fragment.model_copy(
            update={
                "content": new_text,
                "corruption_count": fragment.corruption_count + 1,
            }
        )
        self._repo.upsert_fragment(updated)
        return updated

    def seed(self, fragments: list[VaultFragment]) -> None:
        """Bulk-insert fragments (used during game initialisation)."""
        for fragment in fragments:
            self._repo.upsert_fragment(fragment)

    def reset(self) -> None:
        """Wipe all fragments (used for game restart)."""
        self._repo.reset()
