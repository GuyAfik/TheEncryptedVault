"""Agent layer — 4 OOP agent classes with reasoning + tool selection.

Layer 3 of the 5-layer architecture.
Only imports from: services/, state/ models, llm_factory.
Never imports from: graph/, ui/
"""

from encrypted_vault.agents.base_agent import BaseAgent, AgentTurnResult
from encrypted_vault.agents.infiltrator import Infiltrator
from encrypted_vault.agents.saboteur import Saboteur
from encrypted_vault.agents.scholar import Scholar
from encrypted_vault.agents.enforcer import Enforcer

__all__ = [
    "BaseAgent",
    "AgentTurnResult",
    "Infiltrator",
    "Saboteur",
    "Scholar",
    "Enforcer",
]
