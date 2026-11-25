# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Per-session registry of background subagent runs (FOU-A1 phase 2).

``SubAgentRunRegistry`` is the authoritative record of the background
children a session spawned through ``spawn_subagent(background=True)``:
one entry per run (``subagent_id`` -> status, child session id, timestamps
and the ``SubAgentResult`` fields), bounded retention (oldest evicted past
``max_runs``), and a terminal notification on the event stream mirroring the
``BackgroundAgentManager`` conventions (``victor.agent.background_agent``):

- an optional direct ``event_callback`` (``set_event_callback``), tagged with
  ``owner_client_id`` when the run has one, delivering ``subagent_completed``
  / ``subagent_failed`` / ``subagent_cancelled``; and
- a fire-and-forget ``subagent.terminal`` event on the observability event
  bus (``victor.core.events``), emitted through ``emit_event_sync`` so it
  works from both async and sync contexts.

The registry is the ONLY source of run ids for the ``subagent_status`` /
``subagent_result`` companion tools — an id the LLM fabricates can never
resolve (structured not-found). Thread safety: a single ``threading.RLock``
guards the run map; the detached child runs as asyncio tasks in the parent's
loop, but completion may also be reported from worker threads.

Placement: runtime layer, beside the ``SubAgentOrchestrator`` it tracks. The
tool layer never imports this module — the factory binds the instance into
the tool-executor context (``subagent_run_registry``) and the spawn runner
closure, mirroring the phase-1 ``spawn_subagent_runner`` wiring.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default bounded retention: enough history for a long session without
# unbounded growth; the oldest run is evicted past the cap.
DEFAULT_MAX_RUNS = 100

# Event-stream topic for the observability bus (the "existing event stream").
SUBAGENT_TERMINAL_TOPIC = "subagent.terminal"

EventCallback = Callable[[str, Dict[str, Any]], None]


class SubAgentRunStatus(str, Enum):
    """Lifecycle status of one background subagent run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset(
    {
        SubAgentRunStatus.COMPLETED,
        SubAgentRunStatus.FAILED,
        SubAgentRunStatus.CANCELLED,
    }
)

# Terminal event names, mirroring BackgroundAgentManager's
# agent_completed / agent_error / agent_cancelled convention.
_TERMINAL_EVENT_NAMES: Dict[SubAgentRunStatus, str] = {
    SubAgentRunStatus.COMPLETED: "subagent_completed",
    SubAgentRunStatus.FAILED: "subagent_failed",
    SubAgentRunStatus.CANCELLED: "subagent_cancelled",
}


def _swallow_task_error(task: Any) -> None:
    """Done-callback for fire-and-forget bus emissions."""
    if task.cancelled():
        return
    if task.exception() is not None:
        logger.debug("Subagent terminal bus emission failed: %s", task.exception())


@dataclass
class SubAgentRunRecord:
    """One background subagent run as exposed to the companion tools."""

    subagent_id: str
    child_session_id: str
    prompt: str
    parent_session_id: Optional[str] = None
    #: Resolved principal that owns this run (owner-scoped event delivery).
    owner_client_id: Optional[str] = None
    status: SubAgentRunStatus = SubAgentRunStatus.RUNNING
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    #: The substrate's ``SubAgentResult.to_dict()`` when terminal.
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    #: Strong reference to the detached asyncio task (dropped at terminal) —
    #: asyncio keeps only weak references to running tasks on its own.
    task: Optional[Any] = field(default=None, repr=False, compare=False)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def to_dict(self) -> Dict[str, Any]:
        """Status/metadata dict for ``subagent_status`` and notifications."""
        raw = self.result or {}
        summary = raw.get("summary") or ""
        duration: Optional[float] = None
        if self.finished_at is not None:
            duration = self.finished_at - self.started_at
        return {
            "subagent_id": self.subagent_id,
            "child_session_id": self.child_session_id,
            "prompt": self.prompt[:200],
            "parent_session_id": self.parent_session_id,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": duration,
            "success": raw.get("success"),
            "summary": summary[:500] or None,
            "error": self.error,
            "owner_client_id": self.owner_client_id,
        }

    def result_dict(self) -> Dict[str, Any]:
        """The ``SubAgentResult`` fields the ``subagent_result`` tool exposes."""
        raw = self.result or {}
        details = raw.get("details") or {}
        if not isinstance(details, dict):
            details = {}
        return {
            "success": bool(raw.get("success")),
            "content": raw.get("summary") or "",
            "tool_calls_used": int(raw.get("tool_calls_used") or 0),
            "duration_seconds": float(raw.get("duration_seconds") or 0.0),
            "error": raw.get("error") or self.error,
            "child_session_id": self.child_session_id or details.get("child_session_id"),
            "role": details.get("role"),
        }


class SubAgentRunRegistry:
    """Bounded, thread-safe registry of background subagent runs.

    Attributes:
        max_runs: Retention cap; the OLDEST run is evicted past it.

    Example:
        registry = SubAgentRunRegistry()
        record = registry.start_run(prompt="t", child_session_id="sess-1")
        registry.complete_run(record.subagent_id, result.to_dict())
    """

    def __init__(
        self,
        max_runs: int = DEFAULT_MAX_RUNS,
        event_callback: Optional[EventCallback] = None,
        event_bus: Optional[Any] = None,
    ):
        if max_runs < 1:
            raise ValueError(f"max_runs must be >= 1, got {max_runs}")
        self._max_runs = max_runs
        self._event_callback = event_callback
        self._event_bus = event_bus
        self._runs: "OrderedDict[str, SubAgentRunRecord]" = OrderedDict()
        self._lock = threading.RLock()
        logger.debug("SubAgentRunRegistry initialized (max_runs=%d)", max_runs)

    def set_event_callback(self, callback: Optional[EventCallback]) -> None:
        """Set (or clear) the direct terminal-event callback."""
        self._event_callback = callback

    def mint_subagent_id(self) -> str:
        """Mint a unique run id (BackgroundAgentManager's ``agent-`` pattern)."""
        return f"subagent-{uuid.uuid4().hex[:8]}"

    def start_run(
        self,
        *,
        prompt: str,
        child_session_id: str,
        parent_session_id: Optional[str] = None,
        owner_client_id: Optional[str] = None,
        subagent_id: Optional[str] = None,
    ) -> SubAgentRunRecord:
        """Register a new RUNNING run and return its record."""
        with self._lock:
            record = SubAgentRunRecord(
                subagent_id=subagent_id or self.mint_subagent_id(),
                child_session_id=child_session_id,
                prompt=prompt,
                parent_session_id=parent_session_id,
                owner_client_id=owner_client_id,
            )
            self._runs[record.subagent_id] = record
            self._evict_overflow_locked()
            return record

    def attach_task(self, subagent_id: str, task: Any) -> bool:
        """Keep a strong reference to the detached run's asyncio task."""
        with self._lock:
            record = self._runs.get(subagent_id)
            if record is None:
                return False
            record.task = task
            return True

    def complete_run(self, subagent_id: str, result: Dict[str, Any]) -> Optional[SubAgentRunRecord]:
        """Report the substrate's ``SubAgentResult.to_dict()`` for a run.

        Maps ``result["success"]`` to COMPLETED/FAILED. Returns the updated
        record, or ``None`` when the id is unknown or the run was already
        terminal (a late duplicate report never overwrites the first state).
        """
        success = bool((result or {}).get("success"))
        error = (result or {}).get("error")
        return self._finish(
            subagent_id,
            SubAgentRunStatus.COMPLETED if success else SubAgentRunStatus.FAILED,
            result=result,
            error=error if not success else None,
        )

    def fail_run(self, subagent_id: str, error: str) -> Optional[SubAgentRunRecord]:
        """Mark a run FAILED (substrate crash, launcher failure)."""
        return self._finish(subagent_id, SubAgentRunStatus.FAILED, result=None, error=error)

    def cancel_run(self, subagent_id: str) -> Optional[SubAgentRunRecord]:
        """Mark a run CANCELLED (parent shutdown, explicit cancellation)."""
        return self._finish(subagent_id, SubAgentRunStatus.CANCELLED, result=None, error=None)

    def get(self, subagent_id: str) -> Optional[SubAgentRunRecord]:
        with self._lock:
            return self._runs.get(subagent_id)

    def list_runs(
        self,
        status: Optional[SubAgentRunStatus] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Runs, newest first, optionally filtered by status."""
        with self._lock:
            records = list(self._runs.values())
        if status is not None:
            records = [r for r in records if r.status == status]
        records.reverse()
        return [r.to_dict() for r in records[:limit]]

    def __len__(self) -> int:
        with self._lock:
            return len(self._runs)

    # -- internals ---------------------------------------------------------

    def _finish(
        self,
        subagent_id: str,
        status: SubAgentRunStatus,
        *,
        result: Optional[Dict[str, Any]],
        error: Optional[str],
    ) -> Optional[SubAgentRunRecord]:
        with self._lock:
            record = self._runs.get(subagent_id)
            if record is None:
                logger.warning("SubAgentRunRegistry: unknown run id %s", subagent_id)
                return None
            if record.is_terminal:
                logger.debug(
                    "SubAgentRunRegistry: run %s already %s; ignoring %s report",
                    subagent_id,
                    record.status.value,
                    status.value,
                )
                return None
            record.status = status
            record.result = result
            record.error = error
            record.finished_at = time.time()
            record.task = None  # drop the strong task reference
            self._runs.move_to_end(subagent_id)
            self._evict_overflow_locked()
        self._notify_terminal(record)
        return record

    def _evict_overflow_locked(self) -> None:
        while len(self._runs) > self._max_runs:
            evicted_id, _ = self._runs.popitem(last=False)
            logger.debug("SubAgentRunRegistry: evicted oldest run %s", evicted_id)

    def _notify_terminal(self, record: SubAgentRunRecord) -> None:
        """Emit the terminal event on the event stream (BAM conventions)."""
        event_name = _TERMINAL_EVENT_NAMES[record.status]
        data = record.to_dict()
        if record.owner_client_id:
            # Owner-scoped delivery parity with BackgroundAgentManager.
            data.setdefault("owner_client_id", record.owner_client_id)
        logger.info(
            "Background subagent %s finished (%s): %s",
            record.subagent_id,
            record.status.value,
            (data.get("summary") or data.get("error") or "")[:200],
        )
        if self._event_callback is not None:
            try:
                self._event_callback(event_name, data)
            except Exception as exc:
                logger.warning("Subagent run event callback error: %s", exc)
        self._emit_bus_event(SUBAGENT_TERMINAL_TOPIC, data)

    def _emit_bus_event(self, topic: str, data: Dict[str, Any]) -> None:
        """Fire-and-forget emission on the observability event bus.

        Notification failures must never break run bookkeeping: any error in
        resolving the bus or scheduling the emit is swallowed. With a running
        loop (the normal case — completion is reported from the detached
        child's task) the emit is scheduled on it directly so the backend's
        lazy auto-connect runs; without one, ``emit_event_sync`` falls back
        to the shared background emit loop.
        """
        try:
            if self._event_bus is None:
                from victor.core.events import get_observability_bus

                self._event_bus = get_observability_bus()
            if self._event_bus is None:
                return
            bus = self._event_bus
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                task = loop.create_task(bus.emit(topic, data))
                task.add_done_callback(_swallow_task_error)
            else:
                from victor.core.events.emit_helper import emit_event_sync

                emit_event_sync(bus, topic, data, use_background_loop=True)
        except Exception as exc:
            logger.debug("Subagent terminal bus emission skipped: %s", exc)


__all__ = [
    "SubAgentRunRegistry",
    "SubAgentRunRecord",
    "SubAgentRunStatus",
    "TERMINAL_STATUSES",
    "DEFAULT_MAX_RUNS",
    "SUBAGENT_TERMINAL_TOPIC",
]
