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

"""Runtime diagnostics snapshot for vertical/plugin loading subsystems."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from victor.core.verticals.vertical_loader import VerticalLoader, get_vertical_loader


def get_entry_point_loader_stats() -> Dict[str, Any]:
    """Lazy delegate for :func:`victor.framework.entry_point_loader.get_entry_point_loader_stats`.

    victor.core must not import victor.framework at module import time
    (framework._api imports victor.core.verticals back; the cycle was unmasked
    by the lazy PEP 562 victor/__init__). A module-level delegate keeps the
    import deferred while preserving the patch seam tests rely on.
    """
    from victor.framework.entry_point_loader import (
        get_entry_point_loader_stats as _impl,
    )

    return _impl()


def get_vertical_runtime_diagnostics(
    loader: Optional[VerticalLoader] = None,
) -> Dict[str, Any]:
    """Return a consolidated runtime diagnostics snapshot for vertical integration.

    Note: tool_dependency_loader is imported lazily to avoid circular import
    with victor.core.verticals.__init__.

    The snapshot is best-effort. Individual sections report ``{"error": "..."}`
    if their telemetry source is unavailable.
    """
    active_vertical: Optional[str] = None
    loader_stats: Dict[str, Any]

    try:
        loader_instance = loader or get_vertical_loader()
        active_vertical = loader_instance.active_vertical_name
        loader_stats = loader_instance.get_discovery_stats()
    except Exception as exc:
        loader_stats = {"error": str(exc)}

    try:
        from victor.core.tool_dependency_loader import (
            get_tool_dependency_resolution_stats,
        )

        tool_dependency_stats = get_tool_dependency_resolution_stats()
    except Exception as exc:
        tool_dependency_stats = {"error": str(exc)}

    try:
        entry_point_loader_stats = get_entry_point_loader_stats()
    except Exception as exc:
        entry_point_loader_stats = {"error": str(exc)}

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "active_vertical": active_vertical,
        "vertical_loader": loader_stats,
        "tool_dependency_loader": tool_dependency_stats,
        "framework_entry_point_loader": entry_point_loader_stats,
    }


__all__ = ["get_vertical_runtime_diagnostics"]
