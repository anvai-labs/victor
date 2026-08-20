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

"""Event debouncing configuration to prevent log bloat.

This module provides settings for adaptive event debouncing, which filters
high-frequency duplicate events to prevent log bloat and improve performance.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EventDebouncingSettings(BaseModel):
    """Event debouncing configuration.

    These settings control adaptive debouncing of high-frequency events
    like session_start to prevent log bloat.

    Attributes:
        session_start_enabled: Enable debouncing for session_start events.
        session_start_window_type: Type of debouncing window (time_based, count_based, adaptive).
        session_start_window_seconds: Time window in seconds for debouncing.
        session_start_max_per_window: Maximum events per time window.
        session_start_metadata_fingerprinting: Use metadata hash for semantic deduplication.
        session_start_track_lifecycle: Track session lifecycle for better deduplication.
        tool_intent_enabled: Enable debouncing for tool intent events.
        tool_intent_window_seconds: Time window for tool intent debouncing.
        tool_intent_max_per_window: Maximum tool intent events per window.
    """

    # The model_request_* fields below collide with Pydantic's protected
    # "model_" namespace; these are debouncing knobs, not Pydantic model config,
    # so disable the guard to silence the spurious UserWarning at import time.
    model_config = ConfigDict(protected_namespaces=())

    # Session-start debouncing (CRITICAL: prevents log bloat from duplicate session_start events)
    session_start_enabled: bool = True
    session_start_window_type: str = "time_based"
    session_start_window_seconds: int = 5
    session_start_max_per_window: int = 3
    session_start_metadata_fingerprinting: bool = True
    session_start_track_lifecycle: bool = True

    # Tool execution intent debouncing (prevents duplicate tool intent logging)
    tool_intent_enabled: bool = True
    tool_intent_window_seconds: int = 2
    tool_intent_max_per_window: int = 5

    # Model request debouncing (prevents duplicate model request logging)
    model_request_enabled: bool = False
    model_request_window_seconds: int = 1
    model_request_max_per_window: int = 10
