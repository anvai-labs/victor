# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""In-memory store of durably-paused single-agent runs (FEP-0029, Phase 1).

When a turn parks on a policy ASK, the turn boundary records a :class:`PausedRun` here keyed by an
opaque ``run_id`` and surfaces that token on the ``awaiting_approval`` result. This Phase-1 store is
process-local (a module singleton); Phase 2 replaces it with a ``project.db``-backed ``paused_run``
table (same interface) so pauses survive a restart. The transcript itself is *not* stored here — it
already lives durably in ``ConversationStore`` keyed by ``session_id``.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PausedRun:
    """A single durable pause point awaiting a human approval decision (FEP-0029)."""

    run_id: str
    session_id: Optional[str]
    agent_id: Optional[str]
    approval_request: Dict[str, Any]
    # Best-effort record of the gated action (tool_name + arguments), read from the approval
    # request context — Phase 3 uses this to faithfully replay the persisted call on resume.
    pending_tool: Optional[Dict[str, Any]] = None
    status: str = "awaiting_approval"  # 'awaiting_approval' | 'resumed' | 'cancelled'
    created_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PausedRunStore:
    """Process-local registry of paused runs (Phase 1). Thread-safe."""

    def __init__(self) -> None:
        self._runs: Dict[str, PausedRun] = {}
        self._lock = threading.Lock()

    def save(
        self,
        *,
        session_id: Optional[str],
        agent_id: Optional[str],
        approval_request: Dict[str, Any],
        pending_tool: Optional[Dict[str, Any]] = None,
        created_at: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record a pause and return its opaque ``run_id``."""
        run_id = uuid.uuid4().hex
        run = PausedRun(
            run_id=run_id,
            session_id=session_id,
            agent_id=agent_id,
            approval_request=approval_request,
            pending_tool=pending_tool,
            created_at=created_at,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._runs[run_id] = run
        return run_id

    def get(self, run_id: str) -> Optional[PausedRun]:
        with self._lock:
            return self._runs.get(run_id)

    def mark_resumed(self, run_id: str) -> bool:
        """Mark a run resumed (single-use). Returns False if unknown or already resumed."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.status != "awaiting_approval":
                return False
            run.status = "resumed"
            return True

    def list_pending(self) -> List[PausedRun]:
        with self._lock:
            return [r for r in self._runs.values() if r.status == "awaiting_approval"]

    def clear(self) -> None:
        """Test hook: drop all records."""
        with self._lock:
            self._runs.clear()


_store: Optional[PausedRunStore] = None
_store_lock = threading.Lock()


def get_paused_run_store() -> PausedRunStore:
    """Return the process-wide paused-run store (Phase 1 in-memory singleton)."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = PausedRunStore()
    return _store
