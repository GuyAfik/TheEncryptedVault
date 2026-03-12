"""GameRunner — high-level interface for running and resetting the game.

This is the entry point used by the Streamlit UI (Layer 5).
It yields GlobalGameState after each turn for real-time UI updates.
"""

import queue
import threading
from collections.abc import Generator

from encrypted_vault.state.game_state import GlobalGameState, GraphState
from encrypted_vault.services.container import ServiceContainer
from encrypted_vault.graph.builder import GameGraphBuilder
from encrypted_vault.config import settings


class GameRunner:
    """
    Manages the lifecycle of a single game session.

    Usage (Streamlit):
        runner = GameRunner.create_production()
        for state in runner.start():
            update_ui(state)

    Usage (tests):
        runner = GameRunner.create_in_memory()
        states = list(runner.start())
    """

    def __init__(self, services: ServiceContainer) -> None:
        self._services = services
        self._current_state: GlobalGameState | None = None
        self._event_queue: queue.Queue[GlobalGameState | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    # ── Factory methods ────────────────────────────────────────────────────

    @classmethod
    def create_production(cls) -> "GameRunner":
        """Create a GameRunner backed by ChromaDB (production)."""
        container = ServiceContainer.create_production(
            persist_dir=settings.chroma_persist_dir
        )
        return cls(services=container)

    @classmethod
    def create_in_memory(cls) -> "GameRunner":
        """Create a GameRunner backed by InMemoryVaultRepository (tests)."""
        container = ServiceContainer.create_in_memory()
        return cls(services=container)

    # ── Game lifecycle ─────────────────────────────────────────────────────

    def start(self) -> Generator[GlobalGameState, None, None]:
        """
        Run the game and yield GlobalGameState after each agent turn.

        This is a synchronous generator — suitable for direct iteration in tests.
        For Streamlit, use start_async() which runs in a background thread.
        """
        builder = GameGraphBuilder(services=self._services)
        graph = builder.build()

        # Build initial state
        initial_game_state = self._services.game.build_initial_state(
            max_turns=settings.max_turns,
            token_budget=settings.token_budget_per_agent,
        )
        initial_graph_state = initial_game_state.to_graph_state()

        # Stream events from LangGraph
        for event in graph.stream(initial_graph_state, stream_mode="values"):
            if isinstance(event, dict) and "game_state_json" in event:
                game_state = GlobalGameState.from_graph_state(event)
                self._current_state = game_state
                yield game_state

                if game_state.is_game_over:
                    break

    def start_threaded(self, delay_seconds: float = 1.0) -> None:
        """
        Run the game in a background thread, pushing states to an event queue.
        Used by the Streamlit UI for non-blocking real-time updates.

        Args:
            delay_seconds: Pause between turns (for UI readability).
        """
        import time

        def _run():
            try:
                for state in self.start():
                    self._event_queue.put(state)
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)
            except Exception as e:
                # Signal error to UI
                self._event_queue.put(None)
                raise
            finally:
                self._event_queue.put(None)  # Sentinel: game over

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def poll_event(self, timeout: float = 0.1) -> GlobalGameState | None:
        """
        Non-blocking poll for the next game state from the event queue.
        Returns None if no new state is available yet, or if game is over.

        Used by Streamlit's polling loop.
        """
        try:
            return self._event_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def reset(self) -> "GameRunner":
        """
        Full game reset: wipe vault + chat, return a fresh GameRunner.

        The Streamlit UI calls this when [🔄 Restart] is clicked.
        Returns a new GameRunner instance with a clean state.
        """
        # Stop any running thread
        if self._thread and self._thread.is_alive():
            # Drain the queue to unblock the thread
            while not self._event_queue.empty():
                try:
                    self._event_queue.get_nowait()
                except queue.Empty:
                    break

        # Reset services (wipes vault + chat)
        self._services.game.reset(
            max_turns=settings.max_turns,
            token_budget=settings.token_budget_per_agent,
        )

        # Return a fresh runner with the same services
        return GameRunner(services=self._services)

    # ── State access ───────────────────────────────────────────────────────

    @property
    def current_state(self) -> GlobalGameState | None:
        """The most recently yielded game state."""
        return self._current_state

    @property
    def is_running(self) -> bool:
        """True if the background thread is active."""
        return self._thread is not None and self._thread.is_alive()
