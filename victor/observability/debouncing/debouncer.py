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

"""Adaptive event debouncer for high-frequency observability events.

This module provides the SessionStartDebouncer which implements adaptive
debouncing to prevent log bloat from duplicate session_start events.

Features:
- Time-window based deduplication (default: 5 seconds)
- Session ID tracking with metadata fingerprinting
- Burst limiting (max N events per window)
- Thread-safe implementation for concurrent access
- Configurable debouncing strategies

Design Pattern: Strategy + Decorator
The debouncer wraps event emission with filtering logic, using different
strategies based on configuration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

from victor.observability.debouncing.strategies import (
    DebounceConfig,
    WindowType,
)

logger = logging.getLogger(__name__)


@dataclass
class EventRecord:
    """Record of an emitted event.

    Attributes:
        timestamp: When the event was emitted.
        metadata_hash: Hash of the event metadata.
        count: Number of times this event was attempted.
    """

    timestamp: datetime
    metadata_hash: str
    count: int = 1


class SessionStartDebouncer:
    """
    Adaptive debouncer for session_start events.

    This debouncer prevents log bloat by filtering duplicate session_start
    events that occur within a short time window.

    Features:
    - Time-window based deduplication (default: 5 seconds)
    - Session ID tracking with metadata fingerprinting
    - Burst limiting (max N events per window)
    - Metadata-based semantic deduplication
    - Thread-safe implementation

    Example:
        >>> from victor.observability.debouncing import SessionStartDebouncer, DebounceConfig
        >>> config = DebounceConfig(window_seconds=5, max_events_per_window=3)
        >>> debouncer = SessionStartDebouncer(config)
        >>>
        >>> metadata = {"session_id": "abc-123", "provider": "anthropic"}
        >>> if debouncer.should_emit("abc-123", metadata):
        ...     emit_session_start(metadata)
        ...     debouncer.record("abc-123", metadata)
    """

    def __init__(self, config: Optional[DebounceConfig] = None) -> None:
        """Initialize the debouncer.

        Args:
            config: Debounce configuration. If None, uses defaults.
        """
        self.config = config or DebounceConfig()

        # Event tracking: event_key -> List[EventRecord]
        self._events: Dict[str, List[EventRecord]] = {}

        # Session metadata: session_id -> set of metadata_hashes seen
        self._session_metadata: Dict[str, Set[str]] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            "total_checks": 0,
            "emitted": 0,
            "debounced": 0,
            "metadata_deduplicated": 0,
        }

        logger.debug(
            f"SessionStartDebouncer initialized: {self.config.window_type.value}, "
            f"window={self.config.window_seconds}s, "
            f"max_events={self.config.max_events_per_window}"
        )

    def should_emit(self, session_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Determine if session_start event should be emitted.

        This method implements the debouncing logic:
        1. Cleans old events outside the time window (per session)
        2. For TIME_BASED: Checks metadata deduplication first (block duplicates)
        3. For COUNT_BASED: Only checks count limit (allows duplicate metadata)
        4. Checks count-based limit (all events for session combined)

        The debouncer tracks all events for a session together, regardless of
        metadata fingerprinting. Metadata fingerprinting behavior depends on
        the window type:
        - TIME_BASED: Duplicate metadata is blocked immediately
        - COUNT_BASED: Duplicate metadata is allowed up to max_events_per_window

        Args:
            session_id: Session identifier.
            metadata: Event metadata (provider, model, vertical, mode, etc.).

        Returns:
            True if event should be emitted, False if debounced.
        """
        self._stats["total_checks"] += 1

        with self._lock:
            now = datetime.now(tz=timezone.utc)
            window_start = now - timedelta(seconds=self.config.window_seconds)

            # Clean old events outside the time window (for all keys of this session)
            self._cleanup_old_events_for_session(session_id, window_start)

            # Get all recent events for this session (across all metadata variants)
            all_recent_events = self._get_all_events_for_session(session_id)

            # Check count-based limit first
            if len(all_recent_events) >= self.config.max_events_per_window:
                # At count limit, apply additional deduplication logic
                if self.config.enable_metadata_fingerprinting:
                    # With fingerprinting: block duplicate metadata at limit
                    metadata_hash = self._compute_metadata_hash(metadata)
                    seen_hashes = self._session_metadata.get(session_id, set())

                    if metadata_hash in seen_hashes:
                        # Exact duplicate metadata at limit - block
                        self._stats["debounced"] += 1
                        self._stats["metadata_deduplicated"] += 1
                        logger.debug(
                            f"Debounced session_start: {session_id} "
                            f"(at limit with duplicate metadata)"
                        )
                        return False

                # At limit without matching metadata or fingerprinting disabled
                self._stats["debounced"] += 1
                logger.debug(
                    f"Debounced session_start: {session_id} "
                    f"(count limit: {len(all_recent_events)}/{self.config.max_events_per_window})"
                )
                return False

            # Under count limit, apply TIME_BASED specific logic without fingerprinting
            if (
                self.config.window_type == WindowType.TIME_BASED
                and not self.config.enable_metadata_fingerprinting
            ):
                # Without fingerprinting: allow only 1 event per session (session-based deduplication)
                if all_recent_events:
                    self._stats["debounced"] += 1
                    logger.debug(
                        f"Debounced session_start: {session_id} "
                        f"(session already has event in time-based mode without fingerprinting)"
                    )
                    return False

            # Event should be emitted
            self._stats["emitted"] += 1
            return True

    def record(self, session_id: str, metadata: Dict[str, Any]) -> None:
        """
        Record that a session_start event was emitted.

        This should be called immediately after emitting the event
        to update the debouncing state.

        Args:
            session_id: Session identifier.
            metadata: Event metadata that was emitted.
        """
        event_key = self._compute_event_key(session_id, metadata)
        timestamp = datetime.now(tz=timezone.utc)
        metadata_hash = self._compute_metadata_hash(metadata)

        with self._lock:
            # Create event record
            record = EventRecord(
                timestamp=timestamp,
                metadata_hash=metadata_hash,
                count=1,
            )

            # Add to events list
            if event_key not in self._events:
                self._events[event_key] = []

            self._events[event_key].append(record)

            # Update session metadata (add to set of seen hashes)
            if session_id not in self._session_metadata:
                self._session_metadata[session_id] = set()
            self._session_metadata[session_id].add(metadata_hash)

    def reset(self) -> None:
        """Reset all debouncing state (useful for testing)."""
        with self._lock:
            self._events.clear()
            self._session_metadata.clear()
            self._stats = {
                "total_checks": 0,
                "emitted": 0,
                "debounced": 0,
                "metadata_deduplicated": 0,
            }
        logger.debug("SessionStartDebouncer reset")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get debouncing statistics.

        Returns:
            Dictionary with debouncing statistics.
        """
        with self._lock:
            return {
                **self._stats,
                "active_sessions": len(self._session_metadata),
                "active_keys": len(self._events),
                "debounce_rate": (
                    self._stats["debounced"] / self._stats["total_checks"]
                    if self._stats["total_checks"] > 0
                    else 0.0
                ),
            }

    def _compute_event_key(self, session_id: str, metadata: Dict[str, Any]) -> str:
        """
        Compute event key for deduplication.

        If metadata fingerprinting is enabled, includes relevant metadata
        in the key. This allows different session types to be tracked separately.

        Args:
            session_id: Session identifier.
            metadata: Event metadata.

        Returns:
            Event key string.
        """
        if self.config.enable_metadata_fingerprinting:
            # Include key metadata in fingerprint
            relevant_fields = {"provider", "model", "vertical", "mode"}
            relevant = {k: v for k, v in metadata.items() if k in relevant_fields}

            if relevant:
                fingerprint = hashlib.md5(
                    json.dumps(relevant, sort_keys=True).encode()
                ).hexdigest()[:8]
                return f"{session_id}:{fingerprint}"

        return session_id

    def _compute_metadata_hash(self, metadata: Dict[str, Any]) -> str:
        """
        Compute hash of metadata for deduplication.

        Args:
            metadata: Event metadata.

        Returns:
            MD5 hash of metadata.
        """
        # Sort keys for consistent hashing
        normalized = json.dumps(metadata, sort_keys=True)
        return hashlib.md5(normalized.encode()).hexdigest()

    def _cleanup_old_events(self, event_key: str, window_start: datetime) -> None:
        """
        Remove events outside the time window.

        Args:
            event_key: Event key to clean.
            window_start: Start of the current window.
        """
        if event_key in self._events:
            self._events[event_key] = [
                record for record in self._events[event_key] if record.timestamp > window_start
            ]

            # Clean up empty lists
            if not self._events[event_key]:
                del self._events[event_key]

    def _cleanup_old_events_for_session(self, session_id: str, window_start: datetime) -> None:
        """
        Remove old events for all event keys belonging to a session.

        Args:
            session_id: Session ID to clean events for.
            window_start: Start of the current window.
        """
        keys_to_clean = [
            key for key in self._events.keys() if key.startswith(session_id) or key == session_id
        ]

        for key in keys_to_clean:
            self._cleanup_old_events(key, window_start)

    def _get_all_events_for_session(self, session_id: str) -> List[EventRecord]:
        """
        Get all recent events for a session across all event keys.

        Args:
            session_id: Session ID to get events for.

        Returns:
            List of all event records for this session.
        """
        all_events = []

        # Check all keys that might belong to this session
        for key, events in self._events.items():
            if key == session_id or key.startswith(f"{session_id}:"):
                all_events.extend(events)

        return all_events
