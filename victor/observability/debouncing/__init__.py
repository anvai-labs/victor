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

"""Event debouncing framework for Victor observability.

This module provides adaptive debouncing strategies for high-frequency events
to prevent log bloat and improve performance.

Design Patterns:
- Strategy Pattern: Pluggable debouncing strategies
- Decorator Pattern: Debouncer wraps event emission
- Event-Driven Architecture: Time-based and count-based filtering

Example:
    >>> from victor.observability.debouncing import SessionStartDebouncer, DebounceConfig
    >>> config = DebounceConfig(window_seconds=5, max_events_per_window=3)
    >>> debouncer = SessionStartDebouncer(config)
    >>> if debouncer.should_emit(session_id, metadata):
    ...     emit_event()
    ...     debouncer.record(session_id, metadata)
"""

from victor.observability.debouncing.strategies import (
    DebounceConfig,
    DebounceStrategy,
    WindowType,
)
from victor.observability.debouncing.debouncer import SessionStartDebouncer

__all__ = [
    "DebounceConfig",
    "DebounceStrategy",
    "WindowType",
    "SessionStartDebouncer",
]
