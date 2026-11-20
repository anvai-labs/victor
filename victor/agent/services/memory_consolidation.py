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
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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

#: Retention bound on TOTAL staged proposals across sessions (Codex finding
#: 15): the shared consolidator is a process-wide handle, so unbounded
#: staging would grow pending state for the life of the process. When a
#: run would exceed this, the OLDEST staged proposals (earliest proposed
#: timestamp) are evicted first.
DEFAULT_MAX_STAGED = 100

#: Proposals are episode summaries, not transcripts. Long user messages are
#: truncated to this many characters before validation/storage.
PROPOSAL_SUMMARY_MAX_CHARS = 280

#: Ledger roles eligible for episode proposals. The human's side of the
#: session carries the durable facts ("we renamed the service to X");
#: assistant output is derivative and is not proposed.
_SUMMARY_ROLES = ("user",)

_PROPOSAL_SCOPE = MemoryScope.PROJECT

#: Deterministic proposal-ID namespace (Codex finding 15): the same
#: session + summary always derives the same proposal id, so a re-run of
#: the same ledger (retry after a mid-commit failure) is idempotent.
_PROPOSAL_NS = uuid.uuid5(uuid.NAMESPACE_URL, "victor:memory:consolidation:proposal")


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

    Proposals carry STABLE IDs (deterministic per session + summary) and
    session/owner membership (Codex finding 15): the shared process-wide
    consolidator stages proposals from many sessions, so each proposal
    records the session (and optional owner principal) it belongs to and
    the confirm/discard paths can be scoped to one session without
    touching another's pending proposals.

    Args:
        store: The FOU-A2a preference/episodic store proposals commit into.
        auto_capture: When True, ``run`` commits proposals immediately
            (opt-in; default False — confirm-by-default).
        max_proposals: Per-run proposal bound.
        max_staged: Retention bound on total staged proposals across
            sessions; the oldest staged proposals are evicted first.
    """

    def __init__(
        self,
        store: PreferenceEpisodicStore,
        *,
        auto_capture: bool = False,
        max_proposals: int = DEFAULT_MAX_PROPOSALS,
        max_staged: int = DEFAULT_MAX_STAGED,
        now_fn: Any = time.time,
    ):
        self._store = store
        self._auto_capture = auto_capture
        self._max_proposals = max(1, max_proposals)
        self._max_staged = max(1, max_staged)
        self._now_fn = now_fn
        # Staging is keyed by SESSION membership (Codex finding 15): the
        # process-wide shared consolidator must not let session B's confirm
        # path commit session A's proposals implicitly. Key "" is the
        # unattributed bucket (callers that pass no session).
        self._staged: Dict[str, List[Episode]] = {}
        # Stable proposal IDs already committed this process: a retry (or a
        # re-run of the same session ledger) never duplicates an episode
        # that was already persisted.
        self._committed_ids: set = set()

    @property
    def auto_capture(self) -> bool:
        """Whether ``run`` commits without confirmation."""
        return self._auto_capture

    @auto_capture.setter
    def auto_capture(self, value: bool) -> None:
        """Update the auto-capture posture for later runs.

        The shared consolidator is constructed lazily; a later session
        binding with a different settings-derived flag must take effect
        instead of being silently frozen to the first construction."""
        self._auto_capture = bool(value)

    def propose(
        self,
        session_ledger: Sequence[Any],
        *,
        session_id: str = "",
        owner_client_id: Optional[str] = None,
    ) -> List[Episode]:
        """Derive validated episode candidates from the session ledger.

        Candidates are the human-side entries of the ledger (roles in
        ``_SUMMARY_ROLES``), summarized and deduplicated. Candidates failing
        the store's write-path policy (secret-shaped, oversized) are dropped
        here — a confirm surface must never offer proposals that cannot be
        stored. Nothing is written by this method.

        Each proposal gets a STABLE id (uuid5 over session + summary) plus
        its session and owner membership, so confirmation is idempotent and
        session-scoped paths can select their own proposals.

        Args:
            session_ledger: Ledger entries as ``{"role", "content"}`` dicts
                or objects exposing ``role``/``content`` attributes.
            session_id: Owning session for the proposals (membership).
            owner_client_id: Optional owner principal stamped into the
                proposal metadata (membership).

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
                    id=_proposal_id(session_id, summary),
                    scope=_PROPOSAL_SCOPE,
                    summary=summary,
                    session_id=session_id,
                    timestamp=now,
                    metadata={
                        "proposed": True,
                        "proposal_id": _proposal_id(session_id, summary),
                        "owner_client_id": owner_client_id,
                    },
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

        Idempotency (Codex finding 15): proposals whose stable id was
        already committed (earlier confirm, or an earlier retry that
        persisted a prefix before failing) are skipped, so a retried
        commit never duplicates an already-committed episode.

        Returns:
            The stored episodes (with real store-assigned ids).
        """
        stored: List[Episode] = []
        for proposal in proposals:
            if proposal.id in self._committed_ids:
                logger.debug("Consolidation proposal %s already committed; skipped", proposal.id)
                continue
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
                self._committed_ids.add(proposal.id)
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
        discarded = sum(len(bucket) for bucket in self._staged.values())
        self._staged.clear()
        return discarded

    def run(
        self,
        session_ledger: Sequence[Any],
        *,
        session_id: str = "",
        owner_client_id: Optional[str] = None,
    ) -> ConsolidationResult:
        """Propose, then either commit (auto-capture) or stage for confirm.

        Confirm-by-default: with ``auto_capture`` off (the default), the
        proposals are staged under the session's membership key and nothing
        is written until ``commit_staged`` runs.

        Args:
            session_ledger: The session's conversation ledger.
            session_id: Owning session (proposal membership + staging key).
            owner_client_id: Optional owner principal stamped on proposals.
        """
        proposals = self.propose(
            session_ledger,
            session_id=session_id,
            owner_client_id=owner_client_id,
        )
        if not proposals:
            return ConsolidationResult()

        if self._auto_capture:
            stored = self.commit(proposals)
            return ConsolidationResult(proposed=tuple(proposals), stored=tuple(stored))

        self._stage(session_id, proposals)
        return ConsolidationResult(proposed=tuple(proposals), staged=tuple(proposals))

    # ------------------------------------------------------------------
    # Staging (per-session membership + retention bound)
    # ------------------------------------------------------------------

    def _stage(self, session_id: str, proposals: List[Episode]) -> None:
        """Append proposals to the session's staged bucket, honoring the
        retention bound (oldest staged proposals evicted first)."""
        bucket = self._staged.setdefault(session_id, [])
        bucket.extend(proposals)
        while sum(len(b) for b in self._staged.values()) > self._max_staged:
            oldest_key = min(
                self._staged,
                key=lambda key: (
                    self._staged[key][0].timestamp if self._staged[key] else float("inf")
                ),
            )
            evicted = self._staged[oldest_key].pop(0)
            if not self._staged[oldest_key]:
                del self._staged[oldest_key]
            logger.info(
                "Consolidation staged retention bound (%d) reached; evicted oldest "
                "proposal %s from session %r",
                self._max_staged,
                evicted.id,
                oldest_key,
            )

    # ------------------------------------------------------------------
    # Confirm path
    # ------------------------------------------------------------------

    def staged_proposals(self, session_id: Optional[str] = None) -> List[Episode]:
        """Proposals parked by confirm-by-default runs.

        Args:
            session_id: When given, only that session's proposals; ``None``
                (the default) returns every staged proposal across
                sessions, in staging order.
        """
        if session_id is None:
            return [proposal for bucket in self._staged.values() for proposal in bucket]
        return list(self._staged.get(session_id, []))

    def commit_staged(self, session_id: Optional[str] = None) -> List[Episode]:
        """Confirm path: store staged proposals as episodes.

        Session-scoped when ``session_id`` is given: only that session's
        proposals are committed and cleared — another session's pending
        proposals are untouched (Codex finding 15: one process-wide
        consolidator serves many sessions).

        Idempotent + non-duplicating: each proposal is removed from its
        staging bucket the moment its write succeeds, so a database error
        midway leaves only the UNCOMMITTED remainder staged — a retried
        commit cannot duplicate the already-committed prefix. (Refused
        proposals are also removed: they can never be stored, and
        retrying them is futile.)
        """
        keys = [session_id] if session_id is not None else list(self._staged.keys())
        stored: List[Episode] = []
        for key in keys:
            bucket = self._staged.get(key, [])
            while bucket:
                proposal = bucket[0]
                try:
                    stored.extend(self.commit([proposal]))
                except Exception:
                    # Unexpected store failure: this proposal and anything
                    # behind it stay staged; the already-committed prefix
                    # was removed bucket-head-first below and is never
                    # re-written on retry.
                    raise
                # Success or refusal: the head proposal is resolved
                # (committed, or not storable) — advance past it. On an
                # exception the pop above never runs, so the failed
                # proposal and its tail stay staged for retry.
                bucket.pop(0)
                if not bucket:
                    del self._staged[key]
        return stored

    def discard_staged(self, session_id: Optional[str] = None) -> int:
        """Discard staged proposals without writing.

        Session-scoped when ``session_id`` is given; otherwise discards all
        staged proposals.
        """
        if session_id is None:
            return self.discard()
        bucket = self._staged.pop(session_id, [])
        return len(bucket)


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

    Confirm-by-default: ``auto_capture`` reflects the caller's
    settings-derived flag on EVERY call — when the instance already
    exists, the flag is updated on it instead of being silently ignored.
    Without this, the first construction would freeze the auto-capture
    posture for every later session (Codex finding 15: a consolidator
    built before a settings change would pin the old behavior for the
    life of the process).
    """
    global _shared_consolidator
    if _shared_consolidator is None:
        _shared_consolidator = MemoryConsolidator(
            get_shared_memory_store(),
            auto_capture=auto_capture,
        )
    elif _shared_consolidator.auto_capture != bool(auto_capture):
        logger.debug(
            "Shared memory consolidator auto-capture updated: %s -> %s "
            "(settings changed after first construction)",
            _shared_consolidator.auto_capture,
            bool(auto_capture),
        )
        _shared_consolidator.auto_capture = bool(auto_capture)
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


def _proposal_id(session_id: str, summary: str) -> str:
    """Stable proposal id: deterministic per session + summary content.

    The same session re-proposing the same fact derives the SAME id, so
    idempotent confirmation is content-addressed, not run-addressed."""
    return "prop_" + uuid.uuid5(_PROPOSAL_NS, f"{session_id}|{summary}").hex[:16]


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
    "DEFAULT_MAX_STAGED",
    "MemoryConsolidator",
    "PROPOSAL_SUMMARY_MAX_CHARS",
    "get_shared_memory_consolidator",
    "get_shared_memory_store",
    "reset_shared_memory",
    "set_shared_memory_consolidator",
    "set_shared_memory_store",
]
