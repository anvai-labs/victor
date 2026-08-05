# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Centralized RL coordinator with unified SQLite storage.

Provides both sync and async interfaces:
- Sync methods: `record_outcome()`, `get_recommendation()` (for sync callers)
- Async methods: `record_outcome_async()`, `get_recommendation_async()` (for async orchestrator)

The async methods use `asyncio.to_thread()` to offload SQLite operations to a thread pool,
preventing event loop blocking while maintaining the same synchronous SQLite internals.

Database:
    Uses the unified database at ~/.victor/victor.db via victor.core.database.
    All RL tables are consolidated in this single database for easier management.
"""

from victor.core.json_utils import json_dumps
import asyncio
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.framework.rl.experiment_coordinator import get_experiment_coordinator

if TYPE_CHECKING:
    from victor.framework.rl.option_framework import OptionRegistry
from victor.core.database import get_database
from victor.core.schema import Tables, Schema
from victor.core.constants import DEFAULT_VERTICAL

logger = logging.getLogger(__name__)


def _correlation_for_outcome(outcome: Any) -> tuple:
    """Resolve (session_id, turn_id) for an RL outcome (R1 correlation spine).

    Prefers the live request context (set per session/turn in
    ``victor.core.context``); falls back to values stashed on ``outcome.metadata``.
    Returns ``(None, None)`` if unavailable — the columns are nullable, and
    correlation must never break outcome recording.
    """
    try:
        from victor.core.context import get_session_id, get_turn_id

        meta = getattr(outcome, "metadata", None) or {}
        sid = get_session_id() or meta.get("session_id") or None
        tid = get_turn_id() or meta.get("turn_id") or None
        return (sid, tid)
    except Exception:  # correlation is non-critical
        return (None, None)


@dataclass
class PromptCandidateSuiteWorkflowResult:
    """Result of processing a prompt-candidate benchmark suite."""

    prompt_optimizer_sync: Optional[Any] = None
    prompt_rollout: Optional[Dict[str, Any]] = None
    prompt_rollout_analysis: Optional[Dict[str, Any]] = None
    prompt_rollout_decision: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize workflow state for CLI artifacts."""
        payload: Dict[str, Any] = {}
        if self.prompt_optimizer_sync is not None:
            payload["prompt_optimizer_sync"] = (
                self.prompt_optimizer_sync.to_dict()
                if hasattr(self.prompt_optimizer_sync, "to_dict")
                else self.prompt_optimizer_sync
            )
        if self.prompt_rollout is not None:
            payload["prompt_rollout"] = self.prompt_rollout
        if self.prompt_rollout_analysis is not None:
            payload["prompt_rollout_analysis"] = self.prompt_rollout_analysis
        if self.prompt_rollout_decision is not None:
            payload["prompt_rollout_decision"] = self.prompt_rollout_decision
        return payload


class AsyncWriterQueue:
    """Async writer queue for RL outcomes with background flushing.

    Provides non-blocking outcome recording for async code paths (like
    the agent loop) by queueing outcomes and flushing them in a background
    task.

    Key Features:
        - Non-blocking: `queue_async()` never blocks the event loop
        - Backpressure: Configurable max queue size with drop policy
        - Batched writes: Flushes in batches for efficiency
        - Auto-flush: Background task flushes at configurable intervals
        - Graceful shutdown: Ensures all outcomes are flushed on close

    Example:
        # Initialize with coordinator
        writer = AsyncWriterQueue(coordinator)
        await writer.start()

        # Queue outcomes (non-blocking)
        await writer.queue_async("model_selector", outcome, "coding")

        # Shutdown (flushes remaining)
        await writer.stop()

    Attributes:
        coordinator: RLCoordinator to write to
        batch_size: Number of outcomes per flush
        flush_interval: Seconds between auto-flushes
        max_queue_size: Maximum queue depth before dropping
    """

    def __init__(
        self,
        coordinator: "RLCoordinator",
        batch_size: int = 50,
        flush_interval: float = 5.0,
        max_queue_size: int = 1000,
    ) -> None:
        """Initialize async writer queue.

        Args:
            coordinator: RLCoordinator to write to
            batch_size: Outcomes per batch before flush
            flush_interval: Seconds between auto-flushes
            max_queue_size: Max queue depth (drops oldest if exceeded)
        """
        self.coordinator = coordinator
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_queue_size = max_queue_size

        self._queue: asyncio.Queue[tuple[str, RLOutcome, str]] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._flush_task: Optional[asyncio.Task[None]] = None
        self._running = False

        # Metrics
        self._outcomes_queued = 0
        self._outcomes_flushed = 0
        self._outcomes_dropped = 0
        self._flush_count = 0

    async def start(self) -> None:
        """Start the background flush task."""
        if self._running:
            return

        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.debug("RL: AsyncWriterQueue started")

    async def stop(self) -> None:
        """Stop the background flush task and flush remaining outcomes."""
        if not self._running:
            return

        self._running = False

        # Cancel the flush loop
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush of remaining items
        await self._do_flush()
        logger.debug(
            f"RL: AsyncWriterQueue stopped "
            f"(queued={self._outcomes_queued}, flushed={self._outcomes_flushed}, "
            f"dropped={self._outcomes_dropped})"
        )

    async def queue_async(
        self,
        learner_name: str,
        outcome: RLOutcome,
        vertical: str = DEFAULT_VERTICAL,
    ) -> bool:
        """Queue an outcome for async writing.

        Non-blocking: Returns immediately. If queue is full, drops oldest
        outcome and logs a warning.

        Args:
            learner_name: Name of learner to update
            outcome: Outcome data
            vertical: Which vertical this came from

        Returns:
            True if queued, False if dropped due to full queue
        """
        item = (learner_name, outcome, vertical)

        try:
            self._queue.put_nowait(item)
            self._outcomes_queued += 1
            return True
        except asyncio.QueueFull:
            # Drop oldest and try again
            try:
                dropped = self._queue.get_nowait()
                self._outcomes_dropped += 1
                logger.warning(
                    f"RL: Writer queue full, dropped oldest outcome "
                    f"({dropped[0]}, {dropped[1].task_type})"
                )
                self._queue.put_nowait(item)
                self._outcomes_queued += 1
                return True
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                self._outcomes_dropped += 1
                logger.warning(f"RL: Failed to queue outcome for {learner_name}")
                return False

    async def _flush_loop(self) -> None:
        """Background task that periodically flushes the queue."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._do_flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"RL: Flush loop error: {e}")

    async def _do_flush(self) -> int:
        """Flush pending outcomes to database.

        Returns:
            Number of outcomes flushed
        """
        if self._queue.empty():
            return 0

        # Collect batch
        batch: list[tuple[str, RLOutcome, str]] = []
        while len(batch) < self.batch_size and not self._queue.empty():
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        if not batch:
            return 0

        # Flush to database in thread pool
        count = await asyncio.to_thread(self._write_batch, batch)
        self._outcomes_flushed += count
        self._flush_count += 1
        logger.debug(f"RL: Flushed {count} outcomes (batch {self._flush_count})")
        return count

    def _write_batch(self, batch: list[tuple[str, RLOutcome, str]]) -> int:
        """Write a batch of outcomes to database (sync, runs in thread pool).

        Args:
            batch: List of (learner_name, outcome, vertical) tuples

        Returns:
            Number of outcomes written
        """
        from datetime import datetime as dt

        db = self.coordinator.db
        cursor = db.cursor()
        count = 0

        try:
            for learner_name, outcome, vertical in batch:
                outcome.vertical = vertical

                # Record in learner-specific tables
                learner = self.coordinator.get_learner(learner_name)
                if learner:
                    learner.record_outcome(outcome)

                # Record in shared outcomes table
                from victor.core.schema import Tables

                cursor.execute(
                    f"""
                    INSERT INTO {Tables.RL_OUTCOME} (
                        learner_id, provider, model, task_type, vertical,
                        repo_id, success, quality_score, metadata, session_id, turn_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        learner_name,  # Maps to learner_id column
                        outcome.provider,
                        outcome.model,
                        outcome.task_type,
                        outcome.vertical or "general",
                        self.coordinator._repo_id,
                        1 if outcome.success else 0,
                        outcome.quality_score,
                        outcome.to_dict()["metadata"],
                        *_correlation_for_outcome(outcome),
                    ),
                )
                count += 1

            # Single commit for all outcomes
            db.commit()
            return count

        except Exception as e:
            logger.error(f"RL: Failed to write batch of {len(batch)} outcomes: {e}")
            db.rollback()
            return 0

    def get_metrics(self) -> Dict[str, Any]:
        """Get writer queue metrics.

        Returns:
            Dictionary with queue metrics
        """
        return {
            "outcomes_queued": self._outcomes_queued,
            "outcomes_flushed": self._outcomes_flushed,
            "outcomes_dropped": self._outcomes_dropped,
            "flush_count": self._flush_count,
            "queue_depth": self._queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "batch_size": self.batch_size,
            "flush_interval": self.flush_interval,
        }


class BatchedOutcomeWriter:
    """Batched writer for RL outcomes to reduce database commits.

    Instead of committing each outcome immediately, this class queues
    outcomes and writes them in batches. This significantly reduces
    I/O overhead when recording multiple outcomes.

    Example:
        writer = BatchedOutcomeWriter(coordinator, batch_size=50)
        for outcome in outcomes:
            writer.queue_outcome("learner", outcome, "vertical")
        writer.flush()  # Commit all queued outcomes

        # Or use as context manager
        with BatchedOutcomeWriter(coordinator) as writer:
            writer.queue_outcome("learner", outcome, "vertical")
        # Auto-flush on exit

    Attributes:
        coordinator: RLCoordinator to write to
        batch_size: Max outcomes per batch before auto-flush
        _queue: List of queued (learner_name, outcome, vertical) tuples
    """

    def __init__(
        self,
        coordinator: "RLCoordinator",
        batch_size: int = 50,
        auto_flush_on_exit: bool = True,
    ) -> None:
        """Initialize batched writer.

        Args:
            coordinator: RLCoordinator to write to
            batch_size: Max outcomes per batch before auto-flush
            auto_flush_on_exit: Flush on context manager exit
        """
        self.coordinator = coordinator
        self.batch_size = batch_size
        self.auto_flush_on_exit = auto_flush_on_exit
        self._queue: list[tuple[str, RLOutcome, str]] = []
        self._flush_count = 0

    def __enter__(self) -> "BatchedOutcomeWriter":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager, optionally flushing."""
        if self.auto_flush_on_exit and self._queue:
            self.flush()

    def queue_outcome(
        self,
        learner_name: str,
        outcome: RLOutcome,
        vertical: str = DEFAULT_VERTICAL,
    ) -> None:
        """Queue an outcome for batched writing.

        Args:
            learner_name: Name of learner to update
            outcome: Outcome data
            vertical: Which vertical this came from
        """
        self._queue.append((learner_name, outcome, vertical))

        # Auto-flush if batch is full
        if len(self._queue) >= self.batch_size:
            self.flush()

    def flush(self) -> int:
        """Flush all queued outcomes to database.

        Writes all outcomes in a single transaction for efficiency.

        Returns:
            Number of outcomes written
        """
        if not self._queue:
            return 0

        count = len(self._queue)
        queue_copy = self._queue.copy()
        self._queue.clear()

        # Use coordinator's db connection for transaction
        db = self.coordinator.db
        cursor = db.cursor()

        try:
            for learner_name, outcome, vertical in queue_copy:
                outcome.vertical = vertical

                # Record in learner-specific tables
                learner = self.coordinator.get_learner(learner_name)
                if learner:
                    # Use learner's record_outcome but skip learner's commit
                    learner.record_outcome(outcome)

                # Record in shared outcomes table
                cursor.execute(
                    f"""
                    INSERT INTO {Tables.RL_OUTCOME} (
                        learner_id, provider, model, task_type, vertical,
                        repo_id, success, quality_score, metadata, session_id, turn_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        learner_name,  # Maps to learner_id column
                        outcome.provider,
                        outcome.model,
                        outcome.task_type,
                        outcome.vertical or "general",
                        self.coordinator._repo_id,
                        1 if outcome.success else 0,
                        outcome.quality_score,
                        outcome.to_dict()["metadata"],
                        *_correlation_for_outcome(outcome),
                    ),
                )

            # Single commit for all outcomes
            db.commit()
            self._flush_count += count
            logger.debug(f"RL: Flushed batch of {count} outcomes")
            return count

        except Exception as e:
            logger.error(f"RL: Failed to flush {count} outcomes: {e}")
            db.rollback()
            # Re-queue failed outcomes
            self._queue.extend(queue_copy)
            raise

    def get_queue_size(self) -> int:
        """Get number of queued outcomes."""
        return len(self._queue)

    def get_flush_count(self) -> int:
        """Get total number of outcomes flushed."""
        return self._flush_count

    async def flush_async(self) -> int:
        """Async version of flush."""
        return await asyncio.to_thread(self.flush)


class RLCoordinator:
    """Centralized coordinator for all RL learners.

    Responsibilities:
    - Manage unified SQLite database for all learners
    - Register and coordinate learners
    - Collect telemetry across verticals
    - Enable cross-vertical learning
    - Export metrics for monitoring

    Architecture:
    - Uses unified database: ~/.victor/victor.db (via victor.core.database)
    - Each learner gets own tables (prefixed with learner name)
    - Shared outcomes table for cross-learner analysis
    - Telemetry table for monitoring

    Usage:
        coordinator = get_rl_coordinator()
        coordinator.record_outcome("continuation_patience", outcome, "coding")
        rec = coordinator.get_recommendation("continuation_patience", ...)
    """

    def __init__(self, storage_path: Optional[Path] = None, db_path: Optional[Path] = None):
        """Initialize RL coordinator.

        Args:
            storage_path: Directory for RL data (e.g., ~/.victor/rl_data/) - legacy, now ignored
            db_path: Path to SQLite database - legacy, now uses unified database
        """
        # Legacy storage_path kept for backward compatibility
        if storage_path is None:
            storage_path = Path.home() / ".victor" / "rl_data"
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Use unified database from victor.core.database
        self._db_manager = get_database()
        self.db = self._db_manager.get_connection()
        self.db_path = self._db_manager.db_path

        # Registry of learners
        self._learners: Dict[str, BaseLearner] = {}

        # Active learners allowlist (None = no restriction)
        self._active_learners: Optional[Set[str]] = None

        # Session-scoped provider/model override (set by the orchestrator from
        # the session profile, so GEPA uses the ACTUAL provider — not the
        # global settings default).
        self._session_provider: Optional[str] = None
        self._session_model: Optional[str] = None

        # Async writer queue for non-blocking outcome recording
        self._writer_queue: Optional[AsyncWriterQueue] = None
        self._writer_queue_enabled = False

        # Lifecycle state tracking
        self._is_closed = False

        # Per-repo isolation: outcomes are tagged with repo_id
        self._repo_id: Optional[str] = None

        # Tool name prefixes to exclude from RL recording (test fixtures)
        self._excluded_tool_prefixes = (
            "dummy_",
            "test_",
            "flaky_",
            "error_",
            "always_",
            "mock_",
        )

        # Lazy-initialized OptionRegistry for hierarchical RL options
        self._option_registry: Optional["OptionRegistry"] = None

        # Ensure core tables exist
        self._ensure_core_tables()

        # Migrate schema: add repo_id column if missing
        self._migrate_add_repo_id()

        # Migrate schema: add Phase 1 columns (feedback_source, session_id, etc.)
        self._migrate_add_phase1_columns()

        # Auto-register default learners
        self._register_default_learners()

        # Connect to RL hooks and metrics for event-driven updates
        self._connect_hooks_and_metrics()

        logger.info(f"RL: Coordinator initialized with unified database at {self.db_path}")

    def _ensure_db_connection(self) -> None:
        """Reacquire a usable SQLite connection if the cached one was closed.

        Integration tests reset global database singletons between test cases.
        The RL coordinator is a separate singleton and can outlive that reset,
        leaving cached learner instances and the raw SQLite connection stale.
        """
        if self._is_closed:
            raise sqlite3.ProgrammingError("Cannot operate on a closed database.")

        try:
            self.db.execute("SELECT 1")
            return
        except Exception:
            logger.info("RL: Reopening coordinator database connection")

        self.db = self._db_manager.get_connection()
        self._learners.clear()
        self._ensure_core_tables()
        self._migrate_add_repo_id()
        self._migrate_add_phase1_columns()

    def _ensure_core_tables(self) -> None:
        """Create core tables for telemetry and cross-learner analysis."""
        cursor = self.db.cursor()

        # Shared outcomes table for all learners (uses schema constant)
        cursor.execute(Schema.RL_OUTCOME)

        # Index for fast lookups
        cursor.executescript(Schema.RL_OUTCOME_INDEXES)

        # Telemetry/metrics table for monitoring (uses schema constant)
        cursor.execute(Schema.RL_METRIC)

        # Unified RL state tables — all learners write here, partitioned by learner_id
        cursor.execute(Schema.RL_Q_VALUE)
        cursor.execute(Schema.RL_Q_VALUE_INDEXES)
        cursor.execute(Schema.RL_TRANSITION)
        cursor.execute(Schema.RL_TRANSITION_INDEXES)
        cursor.execute(Schema.RL_PARAM)
        cursor.execute(Schema.RL_TASK_STAT)

        self.db.commit()
        logger.debug(
            "RL: Core tables ensured (including unified rl_q_value/rl_transition/rl_param/rl_task_stat)"
        )

    def _migrate_add_repo_id(self) -> None:
        """Add repo_id column to rl_outcome if missing (backward-compatible)."""
        try:
            cursor = self.db.cursor()
            cursor.execute(f"PRAGMA table_info({Tables.RL_OUTCOME})")
            columns = {row[1] for row in cursor.fetchall()}
            if "repo_id" not in columns:
                cursor.execute(
                    f"ALTER TABLE {Tables.RL_OUTCOME} ADD COLUMN repo_id TEXT DEFAULT NULL"
                )
                cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_rl_outcome_repo "
                    f"ON {Tables.RL_OUTCOME}(repo_id)"
                )
                self.db.commit()
                logger.info("RL: Migrated rl_outcome — added repo_id column")
        except Exception as e:
            logger.debug(f"RL: repo_id migration skipped: {e}")

    def _migrate_add_phase1_columns(self) -> None:
        """Add Priority 4 Phase 1 columns to rl_outcome (backward-compatible)."""
        phase1_columns = {
            "feedback_source": "TEXT DEFAULT NULL",
            "user_feedback": "TEXT DEFAULT NULL",
            "helpful": "INTEGER DEFAULT NULL",
            "correction": "TEXT DEFAULT NULL",
            "session_summary": "TEXT DEFAULT NULL",
            "session_id": "TEXT DEFAULT NULL",
            # Correlation spine (R1): turn_id joins an outcome to the offered/invoked
            # capture records (tool.supply / tool.intent) for the same turn.
            "turn_id": "TEXT DEFAULT NULL",
        }
        try:
            cursor = self.db.cursor()
            cursor.execute(f"PRAGMA table_info({Tables.RL_OUTCOME})")
            existing = {row[1] for row in cursor.fetchall()}
            added = []
            for col, typedef in phase1_columns.items():
                if col not in existing:
                    cursor.execute(f"ALTER TABLE {Tables.RL_OUTCOME} ADD COLUMN {col} {typedef}")
                    added.append(col)
            if added:
                cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_rl_outcome_feedback_source "
                    f"ON {Tables.RL_OUTCOME}(feedback_source, created_at)"
                )
                cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_rl_outcome_session_id "
                    f"ON {Tables.RL_OUTCOME}(session_id)"
                )
                cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_rl_outcome_session_turn "
                    f"ON {Tables.RL_OUTCOME}(session_id, turn_id)"
                )
                self.db.commit()
                logger.info("RL: Migrated rl_outcome — added columns: %s", added)
        except Exception as e:
            logger.debug("RL: Phase 1 migration skipped: %s", e)

    def set_session_provider(self, provider: str, model: str) -> None:
        """Set the session-scoped provider/model for GEPA reflection.

        Called by the orchestrator on init so GEPA uses the ACTUAL session
        provider (e.g., zai/glm-5.2 from -p zai-coding), not the global
        settings default (which may be ollama on a dev machine).
        """
        self._session_provider = provider
        self._session_model = model

    def set_repo_context(self, repo_id: Optional[str]) -> None:
        """Set current repo context for outcome isolation.

        When set, all recorded outcomes are tagged with this repo_id,
        and queries prefer repo-specific data over global data.

        Args:
            repo_id: Repository identifier (e.g., directory name), or None for global
        """
        self._repo_id = repo_id
        logger.debug(f"RL: Repo context set to {repo_id!r}")

    def get_option_registry(self) -> "OptionRegistry":
        """Get the OptionRegistry, creating it lazily on first access.

        The OptionRegistry manages hierarchical RL options (Explore, Implement,
        Debug, Review) — temporal abstractions for multi-step agent skills.
        It is NOT a learner; use this method instead of ``get_learner()``.

        Returns:
            OptionRegistry instance with default options registered.
        """
        if self._option_registry is None:
            from victor.framework.rl.option_framework import OptionRegistry

            self._option_registry = OptionRegistry()
            logger.info(
                "RL: OptionRegistry initialized with %d options",
                len(self._option_registry._options),
            )
        return self._option_registry

    def _should_record_tool(self, tool_name: str) -> bool:
        """Check if a tool outcome should be recorded (filters test fixtures)."""
        if not tool_name:
            return True
        return not any(tool_name.startswith(p) for p in self._excluded_tool_prefixes)

    def _register_default_learners(self) -> None:
        """Register default learners.

        Note: Actual learner instances will be created lazily on first use
        to avoid circular imports and unnecessary initialization.
        """
        # Learners are registered when first accessed via get_learner()
        pass

    def _connect_hooks_and_metrics(self) -> None:
        """Connect to RL hooks registry and metrics exporter.

        This enables event-driven learner activation and metrics collection.
        """
        try:
            # Connect to hooks registry
            from victor.framework.rl.hooks import get_rl_hooks

            hooks = get_rl_hooks(coordinator=self)
            logger.debug("RL: Connected to hooks registry")

            # Connect to metrics exporter
            from victor.framework.rl.metrics import get_rl_metrics

            get_rl_metrics(coordinator=self, hooks=hooks)
            logger.debug("RL: Connected to metrics exporter")

        except ImportError as e:
            logger.debug("RL: Hooks/metrics not available: %s", e)

    def register_learner(self, name: str, learner: BaseLearner) -> None:
        """Register a learner with the coordinator.

        Args:
            name: Learner name (e.g., "continuation_patience")
            learner: Learner instance
        """
        if name in self._learners:
            logger.warning(f"RL: Learner '{name}' already registered, replacing")

        self._learners[name] = learner
        logger.debug(f"RL: Registered learner '{name}'")

    def set_active_learners(self, learners: List[str]) -> None:
        """Set the active learners allowlist.

        Only learners in this list will be lazily created. Already-created
        learners remain accessible regardless of this setting.

        Args:
            learners: List of learner names to allow
        """
        self._active_learners = set(learners)
        logger.info(f"RL: Active learners set to: {sorted(self._active_learners)}")

    def get_active_learner_names(self) -> Optional[Set[str]]:
        """Get the current active learners allowlist.

        Returns:
            Set of active learner names, or None if no restriction
        """
        return self._active_learners

    def get_learner(self, name: str) -> Optional[BaseLearner]:
        """Get a learner by name, creating it lazily if needed.

        Args:
            name: Learner name

        Returns:
            Learner instance or None if unknown or not in active_learners
        """
        self._ensure_db_connection()

        if name in self._learners:
            return self._learners[name]

        # Guard: skip creation if not in active_learners allowlist
        if self._active_learners is not None and name not in self._active_learners:
            logger.debug(f"RL: Learner '{name}' not in active_learners, skipping")
            return None

        # Lazy initialization of learners
        learner = self._create_learner(name)
        if learner:
            self.register_learner(name, learner)

        return learner

    def _create_learner(self, name: str) -> Optional[BaseLearner]:
        """Create a learner instance on demand.

        Args:
            name: Learner name

        Returns:
            Learner instance or None if unknown
        """
        # Import here to avoid circular dependencies
        try:
            if name == "continuation_patience":
                from victor.framework.rl.learners.continuation_patience import (
                    ContinuationPatienceLearner,
                )

                return ContinuationPatienceLearner(
                    name=name, db_connection=self.db, learning_rate=0.1
                )
            elif name == "continuation_prompts":
                from victor.framework.rl.learners.continuation_prompts import (
                    ContinuationPromptLearner,
                )

                return ContinuationPromptLearner(
                    name=name, db_connection=self.db, learning_rate=0.1
                )
            elif name == "semantic_threshold":
                from victor.framework.rl.learners.semantic_threshold import (
                    SemanticThresholdLearner,
                )

                return SemanticThresholdLearner(name=name, db_connection=self.db, learning_rate=0.1)
            elif name == "model_selector":
                from victor.framework.rl.learners.model_selector import (
                    ModelSelectorLearner,
                )

                return ModelSelectorLearner(name=name, db_connection=self.db, learning_rate=0.1)
            elif name == "cache_eviction":
                from victor.framework.rl.learners.cache_eviction import (
                    CacheEvictionLearner,
                )

                return CacheEvictionLearner(name=name, db_connection=self.db, learning_rate=0.1)
            elif name == "grounding_threshold":
                from victor.framework.rl.learners.grounding_threshold import (
                    GroundingThresholdLearner,
                )

                return GroundingThresholdLearner(
                    name=name, db_connection=self.db, learning_rate=0.1
                )
            elif name == "quality_weights":
                from victor.framework.rl.learners.quality_weights import (
                    QualityWeightLearner,
                )

                return QualityWeightLearner(name=name, db_connection=self.db, learning_rate=0.05)
            elif name == "tool_selector":
                from victor.framework.rl.learners.tool_selector import (
                    ToolSelectorLearner,
                )

                return ToolSelectorLearner(name=name, db_connection=self.db, learning_rate=0.05)
            elif name == "mode_transition":
                from victor.framework.rl.learners.mode_transition import (
                    ModeTransitionLearner,
                )

                return ModeTransitionLearner(name=name, db_connection=self.db, learning_rate=0.1)
            elif name == "prompt_template":
                from victor.framework.rl.learners.prompt_template import (
                    PromptTemplateLearner,
                )

                return PromptTemplateLearner(name=name, db_connection=self.db, learning_rate=0.1)
            elif name == "team_composition":
                from victor.agent.teams.learner import TeamCompositionLearner

                # TeamCompositionLearner has different signature - uses db_path instead of db_connection
                return TeamCompositionLearner(learning_rate=0.1)
            elif name == "cross_vertical":
                from victor.framework.rl.learners.cross_vertical import (
                    CrossVerticalLearner,
                )

                return CrossVerticalLearner(name=name, db_connection=self.db, learning_rate=0.1)
            elif name == "workflow_execution":
                from victor.framework.rl.learners.workflow_execution import (
                    WorkflowExecutionLearner,
                )

                return WorkflowExecutionLearner(name=name, db_connection=self.db, learning_rate=0.1)
            elif name == "context_pruning":
                from victor.framework.rl.learners.context_pruning import (
                    ContextPruningLearner,
                )

                return ContextPruningLearner(name=name, db_connection=self.db, learning_rate=0.15)
            elif name == "prompt_optimizer":
                # Check settings: skip when prompt optimization is off
                try:
                    from victor.config.settings import get_settings

                    settings = get_settings()
                    po_cfg = getattr(settings, "prompt_optimization", None)
                    if po_cfg is not None and not po_cfg.enabled:
                        logger.debug("Prompt optimization disabled — skipping prompt_optimizer")
                        return None
                except Exception:
                    pass  # Settings unavailable — allow creation

                from victor.framework.rl.learners.prompt_optimizer import (
                    PromptOptimizerLearner,
                )

                # GEPA v2: use tiered service + Pareto when gepa.enabled
                strategy = None
                use_pareto = False
                max_prompt_chars = 1500
                try:
                    from victor.config.settings import get_settings

                    settings = get_settings()
                    po_cfg = getattr(settings, "prompt_optimization", None)
                    gepa_cfg = po_cfg.gepa if po_cfg else getattr(settings, "gepa", None)
                    if gepa_cfg and gepa_cfg.enabled:
                        from victor.config.gepa_settings import GEPAModelSpec
                        from victor.framework.rl.gepa_strategy_adapter import (
                            GEPAServiceStrategy,
                        )
                        from victor.framework.rl.gepa_tier_manager import (
                            GEPATierManager,
                        )

                        main_model_spec = None
                        if getattr(gepa_cfg, "use_main_model", True):
                            # Prefer session-scoped provider/model (set by the
                            # orchestrator from the session profile) over global
                            # settings — fixes GEPA calling the wrong provider
                            # (e.g., ollama instead of zai when -p zai-coding).
                            main_provider = self._session_provider
                            main_model = self._session_model
                            if not main_provider or not main_model:
                                provider_cfg = getattr(settings, "provider", None)
                                main_provider = (
                                    getattr(provider_cfg, "default_provider", None)
                                    if provider_cfg
                                    else None
                                )
                                main_model = (
                                    getattr(provider_cfg, "default_model", None)
                                    if provider_cfg
                                    else None
                                )

                            # KV cache isolation: resolve a different model for
                            # GEPA reflection so it doesn't thrash the main
                            # agent's cache namespace. Same provider (keeps API
                            # keys/auth), different model (separate cache).
                            gepa_model_override = getattr(gepa_cfg, "model", None)
                            if gepa_model_override and gepa_model_override != main_model:
                                # Explicit override from settings/profile.
                                main_model = gepa_model_override
                                logger.info(
                                    "[GEPA] cache_mode=ISOLATED model=%s "
                                    "(explicit override, ≠ session %s)",
                                    main_model,
                                    self._session_model,
                                )
                            elif main_model:
                                # Auto-resolve: try the provider's "edge" tier
                                # model (cheaper + different cache namespace).
                                try:
                                    from victor.config.decision_settings import (
                                        DecisionServiceSettings,
                                    )

                                    decision_cfg = DecisionServiceSettings()
                                    provider_tiers = decision_cfg.provider_model_tiers.get(
                                        str(main_provider), {}
                                    )
                                    edge_model = provider_tiers.get("edge")
                                    if edge_model and edge_model != main_model:
                                        main_model = edge_model
                                        logger.info(
                                            "[GEPA] cache_mode=ISOLATED model=%s "
                                            "(auto-resolved edge tier for %s, "
                                            "≠ session %s)",
                                            main_model,
                                            main_provider,
                                            self._session_model,
                                        )
                                    else:
                                        logger.warning(
                                            "[GEPA] cache_mode=SHARED model=%s — "
                                            "no different tier model found for "
                                            "provider %s. Mid-session prompt "
                                            "optimization can cause KV cache miss "
                                            "and cost full token price. Set "
                                            "settings.prompt_optimization.gepa.model "
                                            "to isolate, or batch GEPA at "
                                            "session-end.",
                                            main_model,
                                            main_provider,
                                        )
                                except Exception:
                                    logger.warning(
                                        "[GEPA] cache_mode=SHARED model=%s — "
                                        "could not resolve tier models. Mid-session "
                                        "prompt optimization can cause KV cache "
                                        "miss and cost full token price.",
                                        main_model,
                                    )
                            if main_provider and main_model:
                                _local = str(main_provider) in (
                                    "ollama",
                                    "lmstudio",
                                    "vllm",
                                    "mlx",
                                )
                                main_model_spec = GEPAModelSpec(
                                    provider=str(main_provider),
                                    model=str(main_model),
                                    timeout_s=120.0 if _local else 60.0,
                                )

                        tier_mgr = GEPATierManager(gepa_cfg, main_model_spec=main_model_spec)
                        strategy = GEPAServiceStrategy(tier_mgr)
                        use_pareto = True
                        max_prompt_chars = getattr(gepa_cfg, "max_prompt_chars", 1500)
                        logger.info(
                            "GEPA v2 enabled: tier=%s, pareto=True, main_model=%s",
                            tier_mgr.get_current_tier(),
                            (
                                f"{main_model_spec.provider}/{main_model_spec.model}"
                                if main_model_spec
                                else "none"
                            ),
                        )
                except Exception as e:
                    logger.debug("GEPA v2 init skipped: %s", e)

                return PromptOptimizerLearner(
                    name=name,
                    db_connection=self.db,
                    learning_rate=0.1,
                    strategy=strategy,
                    use_pareto=use_pareto,
                    max_prompt_chars=max_prompt_chars,
                )
            elif name == "user_feedback":
                from victor.framework.rl.learners.user_feedback import (
                    UserFeedbackLearner,
                )

                return UserFeedbackLearner(name=name, db_connection=self.db, learning_rate=0.1)
            elif name == "option_framework":
                # OptionRegistry is not a learner — access via get_option_registry()
                logger.debug("RL: option_framework is not a learner — use get_option_registry()")
                return None
            else:
                logger.warning(f"RL: Unknown learner '{name}'")
                return None
        except (ImportError, TypeError) as e:
            logger.warning(f"RL: Failed to create learner '{name}': {e}")
            return None

    def record_outcome(
        self,
        learner_name: str,
        outcome: RLOutcome,
        vertical: str = DEFAULT_VERTICAL,
    ) -> None:
        """Record an outcome for a specific learner.

        This is the main entry point for providing feedback to learners.

        Args:
            learner_name: Name of learner to update
            outcome: Outcome data
            vertical: Which vertical this came from (coding, devops, data_science)
        """
        if self._is_closed:
            logger.debug(
                "RL: Coordinator is closed; dropping outcome for %s (normal during teardown)",
                learner_name,
            )
            return

        self._ensure_db_connection()
        outcome.vertical = vertical

        # Filter test fixture tools from RL recording
        tool_name = outcome.metadata.get("tool_name", "") if outcome.metadata else ""
        if not self._should_record_tool(tool_name):
            logger.debug(f"RL: Skipping test fixture tool '{tool_name}'")
            return

        learner = self.get_learner(learner_name)
        if not learner:
            logger.warning(f"RL: Unknown learner '{learner_name}', skipping outcome")
            return

        try:
            # Record in learner-specific tables
            learner.record_outcome(outcome)

            # Record in shared outcomes table with repo_id and Phase 1 columns
            meta = outcome.metadata or {}
            cursor = self.db.cursor()
            cursor.execute(
                f"""
                INSERT INTO {Tables.RL_OUTCOME} (
                    learner_id, provider, model, task_type, vertical,
                    repo_id, success, quality_score, metadata,
                    feedback_source, user_feedback, helpful, correction, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    learner_name,
                    outcome.provider,
                    outcome.model,
                    outcome.task_type,
                    outcome.vertical or "general",
                    self._repo_id,
                    1 if outcome.success else 0,
                    outcome.quality_score,
                    outcome.to_dict()["metadata"],
                    meta.get("feedback_source"),
                    meta.get("user_feedback"),
                    (
                        1
                        if meta.get("helpful") is True
                        else (0 if meta.get("helpful") is False else None)
                    ),
                    meta.get("correction"),
                    meta.get("session_id"),
                ),
            )
            self.db.commit()

            # Also record telemetry metric
            self._record_metric(
                learner_name,
                "outcome_recorded",
                outcome.quality_score,
                {"success": outcome.success, "task_type": outcome.task_type},
            )

            logger.debug(
                f"RL: Recorded outcome for {learner_name} "
                f"({outcome.provider}:{outcome.model}:{outcome.task_type})"
            )

        except Exception as e:
            logger.error(f"RL: Failed to record outcome for {learner_name}: {e}")
            self.db.rollback()

    def _record_metric(
        self,
        learner_id: str,
        metric_type: str,
        metric_value: float,
        metadata: Optional[dict] = None,
    ) -> None:
        """Record a telemetry metric to rl_metric table.

        Args:
            learner_id: Learner name/ID
            metric_type: Type of metric (e.g., 'outcome_recorded', 'q_value_update')
            metric_value: Numeric metric value
            metadata: Optional additional metadata dict
        """
        try:
            self._ensure_db_connection()
            import json

            cursor = self.db.cursor()
            cursor.execute(
                f"""
                INSERT INTO {Tables.RL_METRIC}
                (learner_id, metric_type, metric_value, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (
                    learner_id,
                    metric_type,
                    metric_value,
                    json_dumps(metadata) if metadata else None,
                ),
            )
            self.db.commit()
        except Exception as e:
            logger.debug(f"RL: Failed to record metric: {e}")

    def record_metric(
        self,
        learner_id: str,
        metric_type: str,
        metric_value: float,
        metadata: Optional[dict] = None,
    ) -> None:
        """Public interface to record telemetry metrics.

        Args:
            learner_id: Learner name/ID
            metric_type: Type of metric
            metric_value: Numeric metric value
            metadata: Optional additional metadata
        """
        self._record_metric(learner_id, metric_type, metric_value, metadata)

    def get_recommendation(
        self,
        learner_name: str,
        provider: str,
        model: str,
        task_type: str,
    ) -> Optional[RLRecommendation]:
        """Get recommendation from a learner.

        Args:
            learner_name: Name of learner to query
            provider: Provider name
            model: Model name
            task_type: Task type

        Returns:
            Recommendation with value and confidence, or None if no data
        """
        self._ensure_db_connection()
        learner = self.get_learner(learner_name)
        if not learner:
            logger.debug(f"RL: Unknown learner '{learner_name}', no recommendation")
            return None

        try:
            return learner.get_recommendation(provider, model, task_type)
        except Exception as e:
            logger.error(f"RL: Failed to get recommendation from {learner_name}: {e}")
            return None

    def create_prompt_rollout_experiment(
        self,
        *,
        section_name: str,
        provider: str,
        treatment_hash: str,
        control_hash: Optional[str] = None,
        traffic_split: float = 0.1,
        min_samples_per_variant: int = 100,
    ) -> Optional[str]:
        """Create and start a prompt rollout experiment through the prompt optimizer.

        Returns:
            Experiment ID if created and started successfully, else None.
        """
        learner = self.get_learner("prompt_optimizer")
        if learner is None or not hasattr(learner, "create_rollout_experiment"):
            return None

        try:
            experiment_coordinator = get_experiment_coordinator(self.db)
            return learner.create_rollout_experiment(
                experiment_coordinator,
                section_name=section_name,
                provider=provider,
                treatment_hash=treatment_hash,
                control_hash=control_hash,
                traffic_split=traffic_split,
                min_samples_per_variant=min_samples_per_variant,
            )
        except Exception as e:
            logger.warning("RL: Failed to create prompt rollout experiment: %s", e)
            return None

    @staticmethod
    def get_prompt_rollout_experiment_id(
        *,
        section_name: str,
        provider: str,
        treatment_hash: str,
    ) -> str:
        """Return the canonical experiment id for a prompt rollout candidate."""
        return f"prompt_optimizer_{section_name.lower()}_{provider or 'default'}_{treatment_hash}"

    @staticmethod
    def _get_prompt_rollout_auto_action(result: Any) -> Optional[str]:
        """Infer the safe action from rollout experiment analysis."""
        recommendation = getattr(result, "recommendation", "")
        is_significant = bool(getattr(result, "is_significant", False))
        treatment_better = bool(getattr(result, "treatment_better", False))

        if recommendation.startswith("Roll out treatment") and is_significant and treatment_better:
            return "rollout"
        if recommendation.startswith("Keep control") and is_significant and not treatment_better:
            return "rollback"
        return None

    @staticmethod
    def _resolve_prompt_rollout_context(sync_result: Any, suite: Any) -> Dict[str, str]:
        """Resolve rollout identity for the benchmark-approved suite winner."""
        approved_hash = getattr(sync_result, "approved_prompt_candidate_hash", None)
        if not approved_hash:
            raise ValueError("no benchmark-approved candidate available for rollout")

        section_name = None
        provider = None

        for decision in getattr(sync_result, "decisions", []) or []:
            if getattr(decision, "prompt_candidate_hash", None) != approved_hash:
                continue
            section_name = getattr(decision, "section_name", None)
            provider = getattr(decision, "provider", None)
            if section_name and provider:
                break

        if not section_name or not provider:
            for run in getattr(suite, "runs", []) or []:
                config = getattr(run, "config", None)
                if getattr(config, "prompt_candidate_hash", None) != approved_hash:
                    continue
                section_name = getattr(config, "prompt_section_name", None)
                provider = getattr(config, "provider", None)
                if section_name and provider:
                    break

        if not section_name or not provider:
            raise ValueError("approved candidate metadata unavailable for rollout")

        experiment_id = RLCoordinator.get_prompt_rollout_experiment_id(
            section_name=section_name,
            provider=provider,
            treatment_hash=approved_hash,
        )
        return {
            "experiment_id": experiment_id,
            "section_name": section_name,
            "provider": provider,
            "prompt_candidate_hash": approved_hash,
        }

    def analyze_prompt_rollout_experiment(
        self,
        *,
        section_name: str,
        provider: str,
        treatment_hash: str,
    ) -> Optional[Dict[str, Any]]:
        """Analyze the rollout experiment for a prompt candidate."""
        experiment_id = RLCoordinator.get_prompt_rollout_experiment_id(
            section_name=section_name,
            provider=provider,
            treatment_hash=treatment_hash,
        )
        experiment_coordinator = get_experiment_coordinator(self.db)
        status = experiment_coordinator.get_experiment_status(experiment_id)
        if status is None:
            return None

        result = experiment_coordinator.analyze_experiment(experiment_id)
        if result is None:
            return {
                "experiment_id": experiment_id,
                "section_name": section_name,
                "provider": provider,
                "prompt_candidate_hash": treatment_hash,
                "status": status.get("status"),
                "analysis_available": False,
                "recommendation": "Prompt rollout analysis unavailable",
                "auto_action": None,
                "details": {},
            }

        return {
            "experiment_id": experiment_id,
            "section_name": section_name,
            "provider": provider,
            "prompt_candidate_hash": treatment_hash,
            "status": status.get("status"),
            "analysis_available": True,
            "recommendation": result.recommendation,
            "auto_action": RLCoordinator._get_prompt_rollout_auto_action(result),
            "is_significant": result.is_significant,
            "treatment_better": result.treatment_better,
            "effect_size": result.effect_size,
            "p_value": result.p_value,
            "confidence_interval": result.confidence_interval,
            "details": result.details,
        }

    def apply_prompt_rollout_recommendation(
        self,
        *,
        section_name: str,
        provider: str,
        treatment_hash: str,
        dry_run: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Apply the rollout/rollback recommendation for a prompt experiment."""
        report = RLCoordinator.analyze_prompt_rollout_experiment(
            self,
            section_name=section_name,
            provider=provider,
            treatment_hash=treatment_hash,
        )
        if report is None:
            return None
        if not report.get("analysis_available", False):
            return {
                "experiment_id": report["experiment_id"],
                "action": None,
                "applied": False,
                "dry_run": dry_run,
                "reason": report.get("recommendation"),
            }

        action = report.get("auto_action")
        if not action:
            return {
                "experiment_id": report["experiment_id"],
                "action": None,
                "applied": False,
                "dry_run": dry_run,
                "reason": report.get("recommendation"),
            }

        if dry_run:
            return {
                "experiment_id": report["experiment_id"],
                "action": action,
                "applied": False,
                "dry_run": True,
            }

        experiment_coordinator = get_experiment_coordinator(self.db)
        if action == "rollout":
            updated = experiment_coordinator.rollout_treatment(report["experiment_id"])
        else:
            updated = experiment_coordinator.rollback_experiment(report["experiment_id"])
        if not updated:
            return {
                "experiment_id": report["experiment_id"],
                "action": action,
                "applied": False,
                "dry_run": False,
                "reason": f"unable to apply prompt rollout decision for {report['experiment_id']}",
            }
        return {
            "experiment_id": report["experiment_id"],
            "action": action,
            "applied": True,
            "dry_run": False,
        }

    def process_prompt_candidate_evaluation_suite(
        self,
        suite: Any,
        *,
        min_pass_rate: float = 0.5,
        promote_best: bool = False,
        create_rollout: bool = False,
        rollout_control_hash: Optional[str] = None,
        rollout_traffic_split: float = 0.1,
        rollout_min_samples_per_variant: int = 100,
        analyze_rollout: bool = False,
        apply_rollout_decision: bool = False,
        rollout_decision_dry_run: bool = False,
    ) -> PromptCandidateSuiteWorkflowResult:
        """Process a prompt-candidate suite through sync and rollout stages."""
        if create_rollout and promote_best:
            raise ValueError("--create-rollout cannot be combined with promote_best")
        if apply_rollout_decision and promote_best:
            raise ValueError("--apply-rollout-decision cannot be combined with promote_best")
        if create_rollout and not 0.0 < rollout_traffic_split < 1.0:
            raise ValueError("rollout_traffic_split must be between 0 and 1")
        if create_rollout and rollout_min_samples_per_variant <= 0:
            raise ValueError("rollout_min_samples_per_variant must be greater than 0")

        learner = self.get_learner("prompt_optimizer")
        if learner is None or not hasattr(learner, "sync_evaluation_suite"):
            raise ValueError("prompt optimizer not available")

        sync_result = learner.sync_evaluation_suite(
            suite,
            min_pass_rate=min_pass_rate,
            promote_best=promote_best,
        )
        workflow = PromptCandidateSuiteWorkflowResult(prompt_optimizer_sync=sync_result)

        if not (create_rollout or analyze_rollout or apply_rollout_decision):
            return workflow

        try:
            context = RLCoordinator._resolve_prompt_rollout_context(sync_result, suite)
        except ValueError as exc:
            error = str(exc)
            if create_rollout:
                workflow.prompt_rollout = {"created": False, "error": error}
            if analyze_rollout:
                workflow.prompt_rollout_analysis = {
                    "analysis_available": False,
                    "recommendation": error,
                }
            if apply_rollout_decision:
                workflow.prompt_rollout_decision = {
                    "applied": False,
                    "dry_run": rollout_decision_dry_run,
                    "error": error,
                }
            return workflow

        if create_rollout:
            experiment_id = RLCoordinator.create_prompt_rollout_experiment(
                self,
                section_name=context["section_name"],
                provider=context["provider"],
                treatment_hash=context["prompt_candidate_hash"],
                control_hash=rollout_control_hash,
                traffic_split=rollout_traffic_split,
                min_samples_per_variant=rollout_min_samples_per_variant,
            )
            workflow.prompt_rollout = (
                {"created": True, "experiment_id": experiment_id}
                if experiment_id
                else {
                    "created": False,
                    "error": "unable to start prompt rollout experiment",
                }
            )

        if analyze_rollout:
            report = RLCoordinator.analyze_prompt_rollout_experiment(
                self,
                section_name=context["section_name"],
                provider=context["provider"],
                treatment_hash=context["prompt_candidate_hash"],
            )
            workflow.prompt_rollout_analysis = (
                report
                if report is not None
                else {
                    "analysis_available": False,
                    "recommendation": (
                        f"prompt rollout experiment not found: {context['experiment_id']}"
                    ),
                }
            )

        if apply_rollout_decision:
            decision = RLCoordinator.apply_prompt_rollout_recommendation(
                self,
                section_name=context["section_name"],
                provider=context["provider"],
                treatment_hash=context["prompt_candidate_hash"],
                dry_run=rollout_decision_dry_run,
            )
            workflow.prompt_rollout_decision = (
                decision
                if decision is not None
                else {
                    "applied": False,
                    "dry_run": rollout_decision_dry_run,
                    "error": f"prompt rollout experiment not found: {context['experiment_id']}",
                }
            )

        return workflow

    @staticmethod
    def format_evolution_report(
        section: str,
        generation: int,
        old_text: str,
        new_text: str,
        alpha: float = 1.0,
        beta_val: float = 1.0,
        sample_count: int = 0,
    ) -> str:
        """Format an auditable evolution report for user visibility.

        Returns a structured text report showing before/after prompt text,
        generation number, and Thompson Sampling statistics.
        """
        max_preview = 300
        old_preview = (old_text[:max_preview] + "...") if len(old_text) > max_preview else old_text
        new_preview = (new_text[:max_preview] + "...") if len(new_text) > max_preview else new_text
        mean = alpha / (alpha + beta_val) if (alpha + beta_val) > 0 else 0.5

        lines = [
            f"[GEPA Evolution] section={section} gen-{generation}",
            f"  Stats: α={alpha:.1f} β={beta_val:.1f} mean={mean:.2f} samples={sample_count}",
            f"  Before ({len(old_text)} chars): {old_preview}",
            f"  After  ({len(new_text)} chars): {new_preview}",
        ]
        return "\n".join(lines)

    def try_evolve_on_session_end(self, provider: str, model: str):
        """EvoTest: try evolving one prompt section after conversation end.

        Lightweight evolution — picks one section (round-robin), collects
        recent traces, evolves if enough data. Non-blocking, best-effort.

        Returns:
            dict with report on success, True for legacy compat, False on skip.
        """
        learner = self._learners.get("prompt_optimizer")
        if learner is None:
            return False

        sections = []
        section_resolver = getattr(type(learner), "get_evolvable_sections", None)
        if callable(section_resolver):
            try:
                resolved_sections = section_resolver(learner)
                if isinstance(resolved_sections, (list, tuple)):
                    sections = list(resolved_sections)
            except Exception:
                logger.debug("RL: Failed to resolve evolvable sections", exc_info=True)
        if not sections:
            configured_sections = getattr(learner, "EVOLVABLE_SECTIONS", [])
            if isinstance(configured_sections, (list, tuple)):
                sections = list(configured_sections)
        if not sections:
            return False

        idx = getattr(self, "_evolution_section_idx", 0)
        section = sections[idx % len(sections)]
        self._evolution_section_idx = idx + 1

        rec = learner.get_recommendation(provider, model, "default", section_name=section)
        current_text = rec.value if rec else ""
        if not current_text:
            return False

        try:
            result = learner.evolve(section, current_text, provider)
            if result:
                report = self.format_evolution_report(
                    section=section,
                    generation=getattr(result, "generation", 0),
                    old_text=current_text,
                    new_text=getattr(result, "text", ""),
                    alpha=getattr(result, "alpha", 1.0),
                    beta_val=getattr(result, "beta_val", 1.0),
                    sample_count=getattr(result, "sample_count", 0),
                )
                logger.info(f"[evotest] {report}")
                return {"evolved": True, "section": section, "report": report}
        except Exception as e:
            logger.debug(f"[evotest] Evolution skipped: {e}")
        return False

    def maybe_background_evolve(self, provider: str = "", model: str = "") -> None:
        """Check staleness and spawn background evolution if needed.

        Called once at chat startup.  If the newest prompt candidate is older
        than ``_EVOLUTION_STALE_DAYS``, a daemon thread evolves all evolvable
        sections from existing trace data.  The thread is fire-and-forget —
        it never blocks the user.

        A filesystem lock prevents concurrent evolution runs across
        multiple ``victor chat`` sessions.
        """
        import threading
        from datetime import datetime, timedelta, timezone

        _EVOLUTION_STALE_DAYS = 7
        lock_path = self.storage_path / ".evolution_lock"

        # Quick check: skip if we already spawned a thread this session
        if getattr(self, "_bg_evolution_started", False):
            return
        self._bg_evolution_started = True

        # Check newest candidate timestamp
        try:
            cursor = self.db.cursor()
            cursor.execute("SELECT MAX(created_at) FROM agent_prompt_candidate")
            row = cursor.fetchone()
            if row and row[0]:
                # Parse ISO timestamp
                last_evolved = datetime.fromisoformat(row[0])
                if last_evolved.tzinfo is None:
                    last_evolved = last_evolved.replace(tzinfo=timezone.utc)
                age = datetime.now(timezone.utc) - last_evolved
                if age < timedelta(days=_EVOLUTION_STALE_DAYS):
                    logger.debug(
                        "RL: Last evolution was %s ago — still fresh, skipping background evolve",
                        age,
                    )
                    return
                logger.info(
                    "RL: Last evolution was %s ago (>%dd) — starting background evolve",
                    age,
                    _EVOLUTION_STALE_DAYS,
                )
            else:
                logger.info("RL: No prompt candidates found — starting background evolve")
        except Exception as e:
            logger.debug("RL: Could not check candidate staleness: %s", e)
            return

        def _evolve_in_background():
            """Run evolution for all sections in a daemon thread."""
            # Filesystem lock to prevent concurrent evolution
            try:
                lock_path.touch(exist_ok=True)
                if lock_path.stat().st_size > 0:
                    logger.debug("RL: Background evolution lock held, skipping")
                    return
                lock_path.write_text(str(datetime.now(timezone.utc).isoformat()))
            except Exception:
                return

            try:
                learner = self.get_learner("prompt_optimizer")
                if learner is None:
                    return

                sections = []
                section_resolver = getattr(type(learner), "get_evolvable_sections", None)
                if callable(section_resolver):
                    try:
                        resolved_sections = section_resolver(learner)
                        if isinstance(resolved_sections, (list, tuple)):
                            sections = list(resolved_sections)
                    except Exception:
                        logger.debug(
                            "RL: Failed to resolve evolvable sections in background evolve",
                            exc_info=True,
                        )
                if not sections:
                    configured_sections = getattr(learner, "EVOLVABLE_SECTIONS", [])
                    if isinstance(configured_sections, (list, tuple)):
                        sections = list(configured_sections)
                evolved_count = 0
                for section in sections:
                    try:
                        rec = learner.get_recommendation(
                            provider or "",
                            model or "",
                            "default",
                            section_name=section,
                        )
                        current_text = rec.value if rec else ""
                        if not current_text:
                            continue
                        result = learner.evolve(section, current_text, provider or "")
                        if result:
                            evolved_count += 1
                    except Exception as e:
                        logger.debug("RL: Background evolve '%s' failed: %s", section, e)

                if evolved_count:
                    logger.info(
                        "RL: Background evolution complete — %d/%d sections evolved",
                        evolved_count,
                        len(sections),
                    )
            except Exception as e:
                logger.debug("RL: Background evolution failed: %s", e)
            finally:
                try:
                    lock_path.unlink(missing_ok=True)
                except Exception:
                    pass

        thread = threading.Thread(
            target=_evolve_in_background,
            name="rl-background-evolve",
            daemon=True,
        )
        thread.start()

    # =========================================================================
    # Async Wrappers - Non-blocking versions for async orchestrator
    # =========================================================================

    async def record_outcome_async(
        self,
        learner_name: str,
        outcome: RLOutcome,
        vertical: str = DEFAULT_VERTICAL,
        use_queue: Optional[bool] = None,
    ) -> None:
        """Async version of record_outcome - offloads SQLite to thread pool.

        Use this from async code (like ChatService.stream_chat) to avoid
        blocking the event loop during SQLite operations.

        When the writer queue is enabled (via enable_writer_queue()), outcomes
        are queued for background batched writing instead of immediate commits.

        Args:
            learner_name: Name of learner to update
            outcome: Outcome data
            vertical: Which vertical this came from (coding, devops, data_science)
            use_queue: Override queue behavior (None=use default, True=force queue,
                       False=force immediate)
        """
        # Determine whether to use writer queue
        should_queue = use_queue if use_queue is not None else self._writer_queue_enabled

        if should_queue and self._writer_queue:
            # Queue for batched writing (non-blocking)
            await self._writer_queue.queue_async(learner_name, outcome, vertical)
        else:
            # Direct write (offloaded to thread pool)
            await asyncio.to_thread(self.record_outcome, learner_name, outcome, vertical)

    async def get_recommendation_async(
        self,
        learner_name: str,
        provider: str,
        model: str,
        task_type: str,
    ) -> Optional[RLRecommendation]:
        """Async version of get_recommendation - offloads SQLite to thread pool.

        Use this from async code to avoid blocking the event loop.

        Args:
            learner_name: Name of learner to query
            provider: Provider name
            model: Model name
            task_type: Task type

        Returns:
            Recommendation with value and confidence, or None if no data
        """
        return await asyncio.to_thread(
            self.get_recommendation, learner_name, provider, model, task_type
        )

    async def get_all_recommendations_async(
        self, provider: str, model: str, task_type: str
    ) -> Dict[str, RLRecommendation]:
        """Async version of get_all_recommendations.

        Args:
            provider: Provider name
            model: Model name
            task_type: Task type

        Returns:
            Dictionary mapping learner name to recommendation
        """
        return await asyncio.to_thread(self.get_all_recommendations, provider, model, task_type)

    async def create_prompt_rollout_experiment_async(
        self,
        *,
        section_name: str,
        provider: str,
        treatment_hash: str,
        control_hash: Optional[str] = None,
        traffic_split: float = 0.1,
        min_samples_per_variant: int = 100,
    ) -> Optional[str]:
        """Async version of create_prompt_rollout_experiment."""
        return await asyncio.to_thread(
            self.create_prompt_rollout_experiment,
            section_name=section_name,
            provider=provider,
            treatment_hash=treatment_hash,
            control_hash=control_hash,
            traffic_split=traffic_split,
            min_samples_per_variant=min_samples_per_variant,
        )

    async def analyze_prompt_rollout_experiment_async(
        self,
        *,
        section_name: str,
        provider: str,
        treatment_hash: str,
    ) -> Optional[Dict[str, Any]]:
        """Async version of analyze_prompt_rollout_experiment."""
        return await asyncio.to_thread(
            self.analyze_prompt_rollout_experiment,
            section_name=section_name,
            provider=provider,
            treatment_hash=treatment_hash,
        )

    async def apply_prompt_rollout_recommendation_async(
        self,
        *,
        section_name: str,
        provider: str,
        treatment_hash: str,
        dry_run: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Async version of apply_prompt_rollout_recommendation."""
        return await asyncio.to_thread(
            self.apply_prompt_rollout_recommendation,
            section_name=section_name,
            provider=provider,
            treatment_hash=treatment_hash,
            dry_run=dry_run,
        )

    async def process_prompt_candidate_evaluation_suite_async(
        self,
        suite: Any,
        *,
        min_pass_rate: float = 0.5,
        promote_best: bool = False,
        create_rollout: bool = False,
        rollout_control_hash: Optional[str] = None,
        rollout_traffic_split: float = 0.1,
        rollout_min_samples_per_variant: int = 100,
        analyze_rollout: bool = False,
        apply_rollout_decision: bool = False,
        rollout_decision_dry_run: bool = False,
    ) -> PromptCandidateSuiteWorkflowResult:
        """Async version of process_prompt_candidate_evaluation_suite."""
        return await asyncio.to_thread(
            self.process_prompt_candidate_evaluation_suite,
            suite,
            min_pass_rate=min_pass_rate,
            promote_best=promote_best,
            create_rollout=create_rollout,
            rollout_control_hash=rollout_control_hash,
            rollout_traffic_split=rollout_traffic_split,
            rollout_min_samples_per_variant=rollout_min_samples_per_variant,
            analyze_rollout=analyze_rollout,
            apply_rollout_decision=apply_rollout_decision,
            rollout_decision_dry_run=rollout_decision_dry_run,
        )

    async def export_metrics_async(self) -> Dict[str, Any]:
        """Async version of export_metrics.

        Returns:
            Dictionary with metrics from all learners
        """
        return await asyncio.to_thread(self.export_metrics)

    async def get_stats_async(self) -> Dict[str, Any]:
        """Async version of get_stats.

        Returns:
            Dictionary with coordinator statistics
        """
        return await asyncio.to_thread(self.get_stats)

    def export_metrics(self) -> Dict[str, Any]:
        """Export all learned values and metrics for monitoring.

        Returns:
            Dictionary with metrics from all learners
        """
        metrics = {"coordinator": {"db_path": str(self.db_path), "learners": {}}}

        for name, learner in self._learners.items():
            try:
                metrics["coordinator"]["learners"][name] = learner.export_metrics()
            except Exception as e:
                logger.error(f"RL: Failed to export metrics for {name}: {e}")
                metrics["coordinator"]["learners"][name] = {"error": str(e)}

        # Add global stats
        cursor = self.db.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {Tables.RL_OUTCOME}")
        metrics["coordinator"]["total_outcomes"] = cursor.fetchone()[0]

        return metrics

    def get_all_recommendations(
        self, provider: str, model: str, task_type: str
    ) -> Dict[str, RLRecommendation]:
        """Get recommendations from all learners for given context.

        Useful for displaying all learned values for a specific provider/model/task.

        Args:
            provider: Provider name
            model: Model name
            task_type: Task type

        Returns:
            Dictionary mapping learner name to recommendation
        """
        recommendations = {}

        for name in [
            "continuation_patience",
            "continuation_prompts",
            "semantic_threshold",
        ]:
            rec = self.get_recommendation(name, provider, model, task_type)
            if rec:
                recommendations[name] = rec

        return recommendations

    def list_learners(self) -> list[str]:
        """List all registered learner names.

        Returns:
            List of learner names
        """
        return list(self._learners.keys())

    def export_all_learner_data(self) -> Dict[str, Any]:
        """Export all learner data for training/analysis.

        Returns:
            Dictionary mapping learner name to its exported data
        """
        data = {}
        for name, learner in self._learners.items():
            try:
                if hasattr(learner, "export_data"):
                    data[name] = learner.export_data()
                elif hasattr(learner, "get_q_table"):
                    data[name] = {"q_table": learner.get_q_table()}
                elif hasattr(learner, "export_metrics"):
                    data[name] = learner.export_metrics()
                else:
                    data[name] = {"type": type(learner).__name__}
            except Exception as e:
                logger.error(f"RL: Failed to export data for {name}: {e}")
                data[name] = {"error": str(e)}
        return data

    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator-level statistics.

        Returns:
            Dictionary with coordinator statistics
        """
        # Count outcomes from database
        cursor = self.db.cursor()
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {Tables.RL_OUTCOME}")
            total_outcomes = cursor.fetchone()[0]
        except Exception:
            total_outcomes = 0

        # Gather learner sample counts
        learner_samples = {}
        for name, learner in self._learners.items():
            try:
                if hasattr(learner, "get_sample_count"):
                    learner_samples[name] = learner.get_sample_count()
                elif hasattr(learner, "_sample_count"):
                    learner_samples[name] = getattr(learner, "_sample_count", 0)
                else:
                    learner_samples[name] = 0
            except Exception:
                learner_samples[name] = 0

        return {
            "total_outcomes": total_outcomes,
            "learner_count": len(self._learners),
            "learners": list(self._learners.keys()),
            "learner_samples": learner_samples,
            "db_path": str(self.db_path),
        }

    # =========================================================================
    # Async Writer Queue Management
    # =========================================================================

    async def enable_writer_queue(
        self,
        batch_size: int = 50,
        flush_interval: float = 5.0,
        max_queue_size: int = 1000,
    ) -> None:
        """Enable async writer queue for non-blocking outcome recording.

        When enabled, record_outcome_async() will queue outcomes for
        background batched writing instead of committing immediately.

        This is recommended for high-frequency outcome recording
        (e.g., during agent loops) to avoid I/O bottlenecks.

        Args:
            batch_size: Outcomes per batch before flush
            flush_interval: Seconds between auto-flushes
            max_queue_size: Maximum queue depth (oldest dropped if exceeded)
        """
        if self._writer_queue and self._writer_queue_enabled:
            logger.debug("RL: Writer queue already enabled")
            return

        self._writer_queue = AsyncWriterQueue(
            coordinator=self,
            batch_size=batch_size,
            flush_interval=flush_interval,
            max_queue_size=max_queue_size,
        )
        await self._writer_queue.start()
        self._writer_queue_enabled = True
        logger.info(
            f"RL: Writer queue enabled (batch={batch_size}, "
            f"interval={flush_interval}s, max={max_queue_size})"
        )

    async def disable_writer_queue(self) -> None:
        """Disable async writer queue and flush remaining outcomes.

        After calling this, record_outcome_async() will commit immediately.
        """
        if not self._writer_queue:
            return

        self._writer_queue_enabled = False
        await self._writer_queue.stop()
        self._writer_queue = None
        logger.info("RL: Writer queue disabled")

    async def flush_writer_queue(self) -> int:
        """Manually flush the writer queue.

        Returns:
            Number of outcomes flushed
        """
        if not self._writer_queue:
            return 0
        return await self._writer_queue._do_flush()

    def get_writer_queue_metrics(self) -> Optional[Dict[str, Any]]:
        """Get writer queue metrics.

        Returns:
            Dictionary with queue metrics, or None if queue not enabled
        """
        if not self._writer_queue:
            return None
        return self._writer_queue.get_metrics()

    def is_writer_queue_enabled(self) -> bool:
        """Check if writer queue is enabled.

        Returns:
            True if writer queue is active
        """
        return self._writer_queue_enabled and self._writer_queue is not None

    @property
    def is_closed(self) -> bool:
        """Check if coordinator is closed.

        Returns:
            True if coordinator has been closed
        """
        return self._is_closed

    def close(self) -> None:
        """Close database connection and mark coordinator as closed."""
        self._is_closed = True
        if self.db:
            self.db.close()
            logger.debug("RL: Database connection closed")


# Global singleton
_rl_coordinator: Optional[RLCoordinator] = None


def get_rl_coordinator() -> RLCoordinator:
    """Get global RL coordinator (lazy init).

    For sync callers. Use get_rl_coordinator_async() from async code.

    Returns:
        Global coordinator singleton
    """
    global _rl_coordinator
    if _rl_coordinator is None or _rl_coordinator.is_closed:
        _rl_coordinator = RLCoordinator()
    else:
        _rl_coordinator._ensure_db_connection()
    return _rl_coordinator


def reset_rl_coordinator() -> None:
    """Reset the global RL coordinator singleton."""
    global _rl_coordinator
    _rl_coordinator = None


async def get_rl_coordinator_async() -> RLCoordinator:
    """Get global RL coordinator with async-safe initialization.

    Use this from async code to avoid blocking the event loop during
    coordinator initialization (which involves SQLite table creation).

    Returns:
        Global coordinator singleton
    """
    global _rl_coordinator
    if _rl_coordinator is None or _rl_coordinator.is_closed:
        # Initialize in thread to avoid blocking event loop
        _rl_coordinator = await asyncio.to_thread(RLCoordinator)
    else:
        await asyncio.to_thread(_rl_coordinator._ensure_db_connection)
    return _rl_coordinator
