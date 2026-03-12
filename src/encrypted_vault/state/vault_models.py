"""Vault-related Pydantic models."""

from pydantic import BaseModel, Field, model_validator


class VaultFragment(BaseModel):
    """A single text chunk stored in the RAG vault."""

    chunk_id: str
    content: str
    is_key_fragment: bool = False
    digit_position: int | None = None  # 0-3 for key fragments; None for noise
    corruption_count: int = 0

    @model_validator(mode="after")
    def validate_digit_position(self) -> "VaultFragment":
        if self.is_key_fragment and self.digit_position is None:
            raise ValueError("Key fragments must have a digit_position (0-3).")
        if not self.is_key_fragment and self.digit_position is not None:
            raise ValueError("Noise fragments must not have a digit_position.")
        if self.digit_position is not None and self.digit_position not in range(4):
            raise ValueError("digit_position must be 0, 1, 2, or 3.")
        return self

    @property
    def is_corrupted(self) -> bool:
        """True if this fragment has been obfuscated at least once."""
        return self.corruption_count > 0


class VaultState(BaseModel):
    """The complete state of the RAG vault."""

    fragments: dict[str, VaultFragment] = Field(default_factory=dict)
    master_key: str  # e.g. "7392" — never exposed to agents
    rag_health: int = Field(default=100, ge=0, le=100)

    def compute_health(self) -> int:
        """Recompute RAG health from current corruption counts."""
        total_corruption = sum(f.corruption_count for f in self.fragments.values())
        return max(0, 100 - total_corruption * 10)

    def refresh_health(self) -> None:
        """Update rag_health field from current fragment state."""
        self.rag_health = self.compute_health()

    @property
    def key_fragments(self) -> list[VaultFragment]:
        """Return only the fragments that hold Master Key digits."""
        return [f for f in self.fragments.values() if f.is_key_fragment]

    @property
    def noise_fragments(self) -> list[VaultFragment]:
        """Return only the noise (distraction) fragments."""
        return [f for f in self.fragments.values() if not f.is_key_fragment]
