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

"""Debouncing strategies and configuration.

This module defines the protocols and configuration for adaptive debouncing.
Strategies can be plugged in to customize debouncing behavior based on:
- Time windows
- Event counts
- Metadata fingerprinting
- Adaptive rate adjustment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Protocol, Optional


class WindowType(Enum):
    """Debouncing window types.

    TIME_BASED: Fixed time window (e.g., 5 seconds)
    COUNT_BASED: Max N events per window, regardless of time
    ADAPTIVE: Adjusts window size based on event rate
    """

    TIME_BASED = "time_based"
    COUNT_BASED = "count_based"
    ADAPTIVE = "adaptive"


class DebounceStrategy(Protocol):
    """Protocol for debounce strategies.

    A debouncing strategy determines whether an event should be emitted
    based on historical event data and configuration.
    """

    def should_emit(self, event_id: str, metadata: Dict[str, Any]) -> bool:
        """Determine if event should be emitted.

        Args:
            event_id: Unique identifier for the event (e.g., session_id).
            metadata: Event metadata for semantic deduplication.

        Returns:
            True if event should be emitted, False if debounced.
        """
        ...

    def record(self, event_id: str, metadata: Dict[str, Any]) -> None:
        """Record that an event was emitted.

        Args:
            event_id: Unique identifier for the event.
            metadata: Event metadata that was emitted.
        """
        ...

    def reset(self) -> None:
        """Reset debouncing state (useful for testing)."""
        ...


@dataclass
class DebounceConfig:
    """Configuration for debouncing behavior.

    This configuration controls how events are debounced to prevent
    log bloat from high-frequency duplicate events.

    Attributes:
        window_type: Type of debouncing window to use.
        window_seconds: Time window for debouncing (TIME_BASED, ADAPTIVE).
        max_events_per_window: Maximum events per window (COUNT_BASED).
        enable_metadata_fingerprinting: Use metadata hash for semantic deduplication.
        track_session_lifecycle: Track session lifecycle for better deduplication.
        adaptive_min_window: Minimum window size for ADAPTIVE mode.
        adaptive_max_window: Maximum window size for ADAPTIVE mode.
        adaptive_rate_threshold: Events per second to trigger window expansion.
    """

    window_type: WindowType = WindowType.TIME_BASED
    window_seconds: int = 5
    max_events_per_window: int = 3
    enable_metadata_fingerprinting: bool = True
    track_session_lifecycle: bool = True

    # Adaptive mode settings
    adaptive_min_window: int = 2
    adaptive_max_window: int = 30
    adaptive_rate_threshold: float = 0.5  # events per second

    @classmethod
    def from_settings(cls, settings: Any) -> "DebounceConfig":
        """Create config from Victor settings.

        Args:
            settings: Victor Settings instance.

        Returns:
            DebounceConfig instance.
        """
        # Try to get debouncing settings
        try:
            debouncing_settings = settings.event_debouncing
            return cls(
                window_type=WindowType(debouncing_settings.session_start_window_type),
                window_seconds=debouncing_settings.session_start_window_seconds,
                max_events_per_window=debouncing_settings.session_start_max_per_window,
                enable_metadata_fingerprinting=debouncing_settings.session_start_metadata_fingerprinting,
                track_session_lifecycle=debouncing_settings.session_start_track_lifecycle,
            )
        except (AttributeError, TypeError, ValueError):
            # Fallback to defaults
            return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of config.
        """
        return {
            "window_type": self.window_type.value,
            "window_seconds": self.window_seconds,
            "max_events_per_window": self.max_events_per_window,
            "enable_metadata_fingerprinting": self.enable_metadata_fingerprinting,
            "track_session_lifecycle": self.track_session_lifecycle,
            "adaptive_min_window": self.adaptive_min_window,
            "adaptive_max_window": self.adaptive_max_window,
            "adaptive_rate_threshold": self.adaptive_rate_threshold,
        }
