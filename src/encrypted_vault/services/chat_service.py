"""ChatService — business logic for public and private messaging."""

from encrypted_vault.state.enums import AgentID
from encrypted_vault.state.chat_models import ChatMessage, PrivateInbox


class ChatService:
    """
    Manages the public chat and private inbox system.

    Holds in-memory state for the current game session.
    State is also mirrored in GlobalGameState for LangGraph transport.
    """

    def __init__(self) -> None:
        self._public_chat: list[ChatMessage] = []
        self._inboxes: dict[AgentID, PrivateInbox] = {
            agent_id: PrivateInbox(owner=agent_id) for agent_id in AgentID
        }

    # ── Public channel ─────────────────────────────────────────────────────

    def broadcast(
        self,
        turn: int,
        sender: AgentID | str,
        content: str,
        is_deceptive: bool = False,
    ) -> ChatMessage:
        """
        Post a public message visible to all agents.

        Args:
            turn: Current game turn number.
            sender: The AgentID (or "SYSTEM") sending the message.
            content: The message text.
            is_deceptive: Metadata flag for UI display (not shown to agents).
        """
        message = ChatMessage(
            turn=turn,
            sender=sender,
            content=content,
            is_deceptive=is_deceptive,
            recipient=None,
        )
        self._public_chat.append(message)
        return message

    def get_public_history(self, last_n: int | None = None) -> list[ChatMessage]:
        """Return public chat history, optionally limited to last N messages."""
        if last_n is None:
            return list(self._public_chat)
        return self._public_chat[-last_n:]

    # ── Private channel ────────────────────────────────────────────────────

    def send_private(
        self,
        turn: int,
        sender: AgentID,
        recipient: AgentID,
        content: str,
        is_deceptive: bool = False,
    ) -> ChatMessage:
        """
        Send a private direct message to a specific agent.

        The message is stored in the recipient's inbox only.
        The UI spectator can see all private messages.

        Raises:
            ValueError: If sender == recipient.
        """
        if sender == recipient:
            raise ValueError("An agent cannot send a private message to itself.")

        message = ChatMessage(
            turn=turn,
            sender=sender,
            content=content,
            is_deceptive=is_deceptive,
            recipient=recipient,
        )
        self._inboxes[recipient].add_message(message)
        return message

    def get_inbox(self, agent_id: AgentID) -> list[ChatMessage]:
        """Return all private messages received by agent_id."""
        return list(self._inboxes[agent_id].messages)

    def get_inbox_from(self, agent_id: AgentID, sender: AgentID) -> list[ChatMessage]:
        """Return private messages received by agent_id from a specific sender."""
        return self._inboxes[agent_id].get_from(sender)

    def get_all_private_messages(self) -> list[ChatMessage]:
        """Return all private messages across all inboxes (for UI spectator)."""
        all_messages: list[ChatMessage] = []
        for inbox in self._inboxes.values():
            all_messages.extend(inbox.messages)
        # Sort by turn for chronological display
        all_messages.sort(key=lambda m: m.turn)
        return all_messages

    # ── State sync ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all messages (used for game restart)."""
        self._public_chat.clear()
        for agent_id in AgentID:
            self._inboxes[agent_id] = PrivateInbox(owner=agent_id)

    def get_inboxes_snapshot(self) -> dict[AgentID, PrivateInbox]:
        """Return a copy of all inboxes for state serialisation."""
        return dict(self._inboxes)
