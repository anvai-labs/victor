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

"""End-of-session memory consolidation hook (FEP-0048, FOU-A2b).

Proposes episodes from the session ledger and stores them ONLY after human
confirmation by default — wrong memories are worse than missing ones, so
aggressive auto-capture is opt-in via ``MemorySettings.memory_auto_capture``
(default ``False``).

Lifecycle: ``propose`` derives validated, unstored episode candidates from a
session ledger (user-side messages); ``commit`` persists them through
``PreferenceEpisodicStore.record_episode`` (the store re-validates fail-closed
per item); ``discard`` drops staged proposals without any write. The Session
service (``victor/agent/services/session_service.py``) binds a consolidator as
a runtime collaborator and exposes ``consolidate_memory`` /
``commit_memory_proposals`` / ``discard_memory_proposals``; the confirm
surface (TUI prompt / API) that drives the confirm path lands with A2d per
the FEP sequencing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from victor.storage.memory.preference_episodic import (
    Episode,
    MemoryScope,
    MemoryWriteRefused,
    PreferenceEpisodicStore,
)

logger = logging.getLogger(__name__)

#: Hard bound on proposals per consolidation run. A session ledger can be
#: long; unbounded proposal floods would drown the confirm surface (and the
#: auto-capture path could silently write hundreds of episodes).
DEFAULT_MAX_PROPOSALS = 10

#: Proposals are episode summaries, not transcripts. Long user messages are
#: truncated to this many characters before validation/storage.
PROPOSAL_SUMMARY_MAX_CHARS = 280

#: Ledger roles eligible for episode proposals. The human's side of the
#: session carries the durable facts ("we renamed the service to X");
#: assistant output is derivative and is not proposed.
_SUMMARY_ROLES = ("user",)

_PROPOSAL_SCOPE = MemoryScope.PROJECT


@dataclass(frozen=True)
class ConsolidationResult:
    """Outcome of one consolidation run.

    Attributes:
        proposed: Validated proposals (stored only if auto-capture committed
            them or a later confirm path commits them).
        stored: Episodes actually persisted by this run (auto-capture only;
            the confirm path's commits are reported by ``commit_staged``).
        staged: Proposals parked for the confirm path (confirm-by-default).
    """

    proposed: Tuple[Episode, ...] = ()
    stored: Tuple[Episode, ...] = ()
    staged: Tuple[Episode, ...] = ()

    @property
    def nothing_stored(self) -> bool:
        """True when this run persisted nothing (confirm-by-default path)."""
        return not self.stored


class MemoryConsolidator:
    """Propose/commit/discard episodes from a session ledger.

    Args:
        store: The FOU-A2a preference/episodic store proposals commit into.
        auto_capture: When True, ``run`` commits proposals immediately
            (opt-in; default False — confirm-by-default).
        max_proposals: Per-run proposal bound.
    """

    def __init__(
        self,
        store: PreferenceEpisodicStore,
        *,
        auto_capture: bool = False,
        max_proposals: int = DEFAULT_MAX_PROPOSALS,
        now_fn: Any = time.time,
    ):
        self._store = store
        self._auto_capture = auto_capture
        self._max_proposals = max(1, max_proposals)
        self._now_fn = now_fn
        self._staged: List[Episode] = []

    @property
    def auto_capture(self) -> bool:
        """Whether ``run`` commits without confirmation."""
        return self._auto_capture

    def propose(self, session_ledger: Sequence[Any]) -> List[Episode]:
        """Derive validated episode candidates from the session ledger.

        Candidates are the human-side entries of the ledger (roles in
        ``_SUMMARY_ROLES``), summarized and deduplicated. Candidates failing
        the store's write-path policy (secret-shaped, oversized) are dropped
        here — a confirm surface must never offer proposals that cannot be
        stored. Nothing is written by this method.

        Args:
            session_ledger: Ledger entries as ``{"role", "content"}`` dicts
                or objects exposing ``role``/``content`` attributes.

        Returns:
            Unstored ``Episode`` candidates (project scope), ledger order.
        """
        proposals: List[Episode] = []
        seen: set = set()
        dropped = 0
        now = self._now_fn()

        for entry in session_ledger:
            if len(proposals) >= self._max_proposals:
                break

            role, content = _ledger_entry(entry)
            if role not in _SUMMARY_ROLES:
                continue

            summary = _summarize(content)
            if not summary or summary in seen:
                continue

            try:
                self._store.validate_content(summary)
            except MemoryWriteRefused as exc:
                dropped += 1
                logger.debug("Consolidation candidate dropped (%s)", type(exc).__name__)
                continue

            seen.add(summary)
            proposals.append(
                Episode(
                    id=f"proposed_{len(proposals)}",  # placeholder; store assigns the real id on commit
                    scope=_PROPOSAL_SCOPE,
                    summary=summary,
                    timestamp=now,
                    metadata={"proposed": True},
                )
            )

        if dropped:
            logger.info("Consolidation dropped %d unfit candidates", dropped)
        return proposals

    def commit(self, proposals: Iterable[Episode]) -> List[Episode]:
        """Store proposals as episodes; refusals skip the item, never abort.

        The store re-validates fail-closed, so a proposal that slipped past
        propose-time validation still cannot persist secret-shaped or
        oversized content.

        Returns:
            The stored episodes (with real store-assigned ids).
        """
        stored: List[Episode] = []
        for proposal in proposals:
            try:
                stored.append(
                    self._store.record_episode(
                        proposal.summary,
                        session_id=proposal.session_id,
                        thread_id=proposal.thread_id,
                        metadata={**proposal.metadata, "proposed": False},
                        timestamp=proposal.timestamp or None,
                    )
                )
            except MemoryWriteRefused as exc:
                logger.warning(
                    "Consolidation commit refused (%s); proposal skipped",
                    type(exc).__name__,
                )
        return stored

    def discard(self, proposals: Optional[Iterable[Episode]] = None) -> int:
        """Drop staged proposals without writing anything.

        Args:
            proposals: Proposals carry no state outside the consolidator, so
                this argument is accepted for call-site symmetry with
                ``commit`` and ignored; staged proposals are what gets
                discarded.

        Returns:
            Number of staged proposals discarded.
        """
        discarded = len(self._staged)
        self._staged.clear()
        return discarded

    def run(self, session_ledger: Sequence[Any]) -> ConsolidationResult:
        """Propose, then either commit (auto-capture) or stage for confirm.

        Confirm-by-default: with ``auto_capture`` off (the default), the
        proposals are staged and nothing is written until ``commit_staged``
        runs.
        """
        proposals = self.propose(session_ledger)
        if not proposals:
            return ConsolidationResult()

        if self._auto_capture:
            stored = self.commit(proposals)
            return ConsolidationResult(proposed=tuple(proposals), stored=tuple(stored))

        self._staged.extend(proposals)
        return ConsolidationResult(proposed=tuple(proposals), staged=tuple(proposals))

    # ------------------------------------------------------------------
    # Confirm path
    # ------------------------------------------------------------------

    def staged_proposals(self) -> List[Episode]:
        """Proposals parked by the last confirm-by-default ``run``."""
        return list(self._staged)

    def commit_staged(self) -> List[Episode]:
        """Commit (and clear) the staged proposals."""
        stored = self.commit(self._staged)
        self._staged.clear()
        return stored

    def discard_staged(self) -> int:
        """Discard (and clear) the staged proposals without writing."""
        return self.discard()


# ------------------------------------------------------------------------------
# Shared process handle (FOU-A2d)
# ------------------------------------------------------------------------------
#
# The confirm surfaces (the TUI /memory command, the /memory/proposals API
# routes) must drive the SAME consolidator the Session-service end_session
# hook stages onto, or staged proposals would be invisible to the confirm
# path. One lazily-constructed store + consolidator per process, injection
# points for tests/embedders; client layers reach it through the sanctioned
# framework seam (``victor.framework.memory``) because they may not import
# ``victor.agent.*`` directly.

_shared_store: Optional[PreferenceEpisodicStore] = None
_shared_consolidator: Optional[MemoryConsolidator] = None


def get_shared_memory_store() -> PreferenceEpisodicStore:
    """The process-wide preference/episodic store (lazy; injectable).

    Constructs against the database singletons on first call, which honor
    ``VICTOR_HOME`` / project-path overrides — tests redirect storage via
    ``set_shared_memory_store`` or the database resets, not this module.
    """
    global _shared_store
    if _shared_store is None:
        _shared_store = PreferenceEpisodicStore()
    return _shared_store


def set_shared_memory_store(store: PreferenceEpisodicStore) -> None:
    """Pin the shared store (tests / embedders with their own databases)."""
    global _shared_store
    _shared_store = store


def get_shared_memory_consolidator(*, auto_capture: bool = False) -> MemoryConsolidator:
    """The process-wide consolidator the confirm surfaces drive (lazy).

    Confirm-by-default: ``auto_capture`` only takes effect when this call
    constructs the instance (first call wins; later calls return the
    existing consolidator regardless of the flag).
    """
    global _shared_consolidator
    if _shared_consolidator is None:
        _shared_consolidator = MemoryConsolidator(
            get_shared_memory_store(),
            auto_capture=auto_capture,
        )
    return _shared_consolidator


def set_shared_memory_consolidator(consolidator: MemoryConsolidator) -> None:
    """Pin the shared consolidator (tests / embedders)."""
    global _shared_consolidator
    _shared_consolidator = consolidator


def reset_shared_memory() -> None:
    """Drop both shared handles (tests re-point storage between cases)."""
    global _shared_store, _shared_consolidator
    _shared_store = None
    _shared_consolidator = None


def _ledger_entry(entry: Any) -> Tuple[str, str]:
    """Extract (role, content) from a dict- or object-shaped ledger entry."""
    if isinstance(entry, dict):
        return str(entry.get("role", "")), str(entry.get("content", ""))
    return str(getattr(entry, "role", "")), str(getattr(entry, "content", ""))


def _summarize(content: str) -> str:
    """First meaningful line, bounded to the proposal summary length."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:PROPOSAL_SUMMARY_MAX_CHARS]
    return ""


__all__ = [
    "ConsolidationResult",
    "DEFAULT_MAX_PROPOSALS",
    "MemoryConsolidator",
    "PROPOSAL_SUMMARY_MAX_CHARS",
    "get_shared_memory_consolidator",
    "get_shared_memory_store",
    "reset_shared_memory",
    "set_shared_memory_consolidator",
    "set_shared_memory_store",
]
