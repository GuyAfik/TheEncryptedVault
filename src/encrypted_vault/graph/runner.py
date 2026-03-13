"""GameRunner — high-level interface for running and resetting the game.

Yields GlobalGameState after each turn for real-time UI updates.
Uses a thread-safe queue for background execution.
"""

import logging
import queue
import threading
import time
from collections.abc import Generator

from encrypted_vault.state.game_state import GlobalGameState, GraphState
from encrypted_vault.services.container import ServiceContainer
from encrypted_vault.graph.builder import GameGraphBuilder
from encrypted_vault.config import settings

logger = logging.getLogger(__name__)

# Sentinel value to signal end of game
_GAME_OVER = object()


class GameRunner:
    """
    Manages the lifecycle of a single game session.

    The game runs in a background thread and pushes GlobalGameState
    objects into a thread-safe queue. The Streamlit UI polls this queue.

    Usage (Streamlit):
        runner = GameRunner.create_production()
        runner.start_threaded(delay_seconds=1.5)
        # In UI loop:
        state = runner.get_latest_state()  # non-blocking
        # Human-in-the-loop:
        query = runner.get_pending_human_query()
        if query:
            runner.answer_human_query("7")

    Usage (tests):
        runner = GameRunner.create_in_memory()
        states = list(runner.start())
    """

    def __init__(self, services: ServiceContainer) -> None:
        self._services = services
        self._state_queue: queue.Queue = queue.Queue(maxsize=100)
        self._latest_state: GlobalGameState | None = None
        self._latest_state_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Human-in-the-loop: reference to the builder's answer callback
        self._answer_human_query_fn = None
        self._builder: "GameGraphBuilder | None" = None

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
        Run the game synchronously and yield GlobalGameState after each agent turn.
        Suitable for tests and direct iteration.
        """
        builder = GameGraphBuilder(services=self._services)
        self._builder = builder
        graph = builder.build()
        # Wire up the human query answer function after build()
        self._answer_human_query_fn = getattr(builder, "_answer_human_query", None)

        initial_game_state = self._services.game.build_initial_state(
            max_turns=settings.max_turns,
            token_budget=settings.token_budget_per_agent,
        )
        initial_graph_state = initial_game_state.to_graph_state()

        logger.info("GameRunner.start() — streaming graph events")

        for event in graph.stream(initial_graph_state, stream_mode="values"):
            if isinstance(event, dict) and "game_state_json" in event:
                game_state = GlobalGameState.from_graph_state(event)
                with self._latest_state_lock:
                    self._latest_state = game_state
                yield game_state
                if game_state.is_game_over:
                    logger.info("Game over — stopping stream")
                    break

    def start_threaded(
        self,
        delay_seconds: float = 1.5,
    ) -> None:
        """
        Run the game in a background daemon thread.
        States are pushed to _state_queue and also stored in _latest_state.
        The Streamlit UI reads _latest_state via get_latest_state().

        Args:
            delay_seconds: Pause between turns for UI readability.
        """
        self._stop_event.clear()

        def _run():
            try:
                for state in self.start():
                    if self._stop_event.is_set():
                        break
                    # Store latest state (thread-safe)
                    with self._latest_state_lock:
                        self._latest_state = state
                    # Push to queue (non-blocking; drop if full)
                    try:
                        self._state_queue.put_nowait(state)
                    except queue.Full:
                        pass
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)
            except Exception as e:
                logger.error("GameRunner thread error: %s", e, exc_info=True)
            finally:
                # Push sentinel to signal completion
                try:
                    self._state_queue.put_nowait(_GAME_OVER)
                except queue.Full:
                    pass
                logger.info("GameRunner thread finished")

        self._thread = threading.Thread(target=_run, daemon=True, name="GameRunner")
        self._thread.start()
        logger.info("GameRunner background thread started (delay=%.1fs)", delay_seconds)

    def get_latest_state(self) -> GlobalGameState | None:
        """
        Return the most recently produced game state.
        Thread-safe. Non-blocking. Returns None if no state yet.
        Used by Streamlit UI for polling.
        """
        with self._latest_state_lock:
            return self._latest_state

    def drain_queue(self) -> list[GlobalGameState]:
        """
        Drain all pending states from the queue.
        Returns the list of new states (may be empty).
        """
        states = []
        while True:
            try:
                item = self._state_queue.get_nowait()
                if item is _GAME_OVER:
                    break
                states.append(item)
            except queue.Empty:
                break
        return states

    def get_pending_human_query(self):
        """
        Return the pending HumanQueryRequest if an agent is waiting for human input.
        Returns None if no query is pending.
        Thread-safe.
        """
        with self._latest_state_lock:
            state = self._latest_state
        if state and state.pending_human_query:
            return state.pending_human_query
        return None

    def answer_human_query(self, answer: str) -> None:
        """
        Provide the human's answer to the pending agent query.
        This unblocks the ask_human tool and resumes the game.
        Thread-safe.
        """
        logger.info("GameRunner.answer_human_query('%s')", answer)
        if self._answer_human_query_fn is not None:
            self._answer_human_query_fn(answer)
        # Also update the latest state so the UI sees the query is resolved
        with self._latest_state_lock:
            if self._latest_state and self._latest_state.pending_human_query:
                self._latest_state.resolve_human_query(answer)

    def reset(self) -> "GameRunner":
        """
        Full game reset: stop thread, wipe vault + chat + agent memories, return a fresh GameRunner.
        """
        logger.info("GameRunner.reset() called")
        self._stop_event.set()

        # Reset vault + chat
        self._services.game.reset(
            max_turns=settings.max_turns,
            token_budget=settings.token_budget_per_agent,
        )

        # Reset all agent episodic memories (ephemeral per game session)
        self._services.memory.forget_all()
        logger.info("Agent episodic memories cleared on reset")

        return GameRunner(services=self._services)

    # ── State access ───────────────────────────────────────────────────────

    @property
    def current_state(self) -> GlobalGameState | None:
        return self.get_latest_state()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
