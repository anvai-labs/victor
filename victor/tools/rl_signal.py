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

"""Per-execution channel for tool-supplied RL outcome metadata (RLM-2/RLM-3).

Tools compute structured outcome metadata (error categories, counts, timings)
after running; the RL event is emitted centrally by the tool executor, which
has no other access to tool results' internals. This module bridges the two
with a context-local holder:

- the executor seeds a fresh holder before awaiting the tool
  (``seed_tool_outcome_metadata``);
- the tool attaches its structured fields inside the same execution context
  (``attach_tool_outcome_metadata``) — a no-op when nothing was seeded, so
  tools can call it unconditionally (direct calls, tests, non-executor paths);
- the executor drains the accumulated fields at event-emission time
  (``drain_tool_outcome_metadata``) and merges them into ``RLEvent.metadata``.

Field values should stay flat and small (strings/numbers) so they merge
cleanly into event metadata.
"""

from __future__ import annotations

import contextvars
from typing import Any, Dict, Optional

_holder: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "victor_tool_outcome_metadata", default=None
)


def seed_tool_outcome_metadata() -> None:
    """Start a fresh metadata holder for this execution context (executor side)."""
    _holder.set({})


def attach_tool_outcome_metadata(**fields: Any) -> None:
    """Attach structured outcome fields for the current execution (tool side).

    Silently no-ops when the executor has not seeded a holder — attaching must
    never become a tool failure mode.
    """
    holder = _holder.get()
    if holder is None:
        return
    for key, value in fields.items():
        if value is None:
            continue
        holder[key] = value


def drain_tool_outcome_metadata() -> Dict[str, Any]:
    """Return and clear the accumulated metadata (executor side, at emit)."""
    holder = _holder.get()
    _holder.set(None)
    if holder is None:
        return {}
    return dict(holder)
