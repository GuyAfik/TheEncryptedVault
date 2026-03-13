"""Enumerations shared across all layers."""

from enum import Enum


class AgentID(str, Enum):
    """Unique identifier for each agent in the game."""

    INFILTRATOR = "infiltrator"
    SABOTEUR = "saboteur"
    SCHOLAR = "scholar"
    ENFORCER = "enforcer"

    @property
    def emoji(self) -> str:
        """Return the display emoji for this agent."""
        return {
            AgentID.INFILTRATOR: "🕵️",
            AgentID.SABOTEUR: "💣",
            AgentID.SCHOLAR: "🎓",
            AgentID.ENFORCER: "👊",
        }[self]

    @property
    def display_name(self) -> str:
        """Return the capitalised display name."""
        return self.value.capitalize()


class GameStatus(str, Enum):
    """Current status of the game."""

    RUNNING = "running"
    AGENT_WIN = "agent_win"
    # Note: There is no SYSTEM_WIN — an agent always wins.
    # After 20 turns, the agent closest to the key is declared winner.
