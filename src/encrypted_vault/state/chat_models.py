"""Chat-related Pydantic models for public and private messaging."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from encrypted_vault.state.enums import AgentID


class ChatMessage(BaseModel):
    """
    A single message in the game's communication system.

    - recipient=None  → public broadcast (visible to all agents)
    - recipient=AgentID → private DM (visible only to recipient + UI spectator)
    """

    turn: int
    sender: AgentID | Literal["SYSTEM"]
    content: str
    is_deceptive: bool = False
    """Metadata flag — not visible to other agents; shown in UI for spectators."""

    recipient: AgentID | None = None
    """None = public; AgentID = private direct message."""

    @model_validator(mode="after")
    def sender_cannot_dm_self(self) -> "ChatMessage":
        if self.recipient is not None and self.sender == self.recipient:
            raise ValueError("An agent cannot send a private message to itself.")
        return self

    @property
    def is_private(self) -> bool:
        """True if this is a private direct message."""
        return self.recipient is not None

    @property
    def display_prefix(self) -> str:
        """Human-readable prefix for UI display."""
        if self.is_private:
            return f"🔒 [{self.sender} → {self.recipient}]"
        return f"[{self.sender}]"


class PrivateInbox(BaseModel):
    """
    Each agent's private inbox of direct messages received.
    Only the owning agent (and the UI spectator) can read this.
    """

    owner: AgentID
    messages: list[ChatMessage] = Field(default_factory=list)

    def add_message(self, message: ChatMessage) -> None:
        """Append a private message to this inbox."""
        if message.recipient != self.owner:
            raise ValueError(
                f"Message recipient {message.recipient} does not match inbox owner {self.owner}."
            )
        self.messages.append(message)

    def get_from(self, sender: AgentID) -> list[ChatMessage]:
        """Return all messages received from a specific sender."""
        return [m for m in self.messages if m.sender == sender]
