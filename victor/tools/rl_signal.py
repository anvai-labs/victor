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

"""Structured tool-outcome side-channel for RL events (RLM-2, spec Phase 1).

`ToolExecutor._emit_rl_tool_event` only ever saw `execution_time` — no
exit code, no error class, no timing detail — so 44%-failure-rate questions
("is shell broken or are these expected failures?") were unanswerable from
`rl_outcome` (see `docs/analysis/rl-data-summary-2026-11-15.md`).

Tools attach structured fields here; the executor drains them into the
``TOOL_EXECUTED`` event metadata. The channel is a contextvar holding a
mutable dict that the executor seeds *before* awaiting the tool: asyncio
copies the context on task creation, so plain ``ContextVar.set`` inside the
tool coroutine would be invisible to the executor — mutating the shared
dict is visible, which is exactly the propagation we need.

Contract:
- ``seed`` once per tool execution (executor); ``attach`` any number of
  times (tool); ``drain`` once at emit (executor). Draining clears state so
  a stale payload can never leak into a later event.
- All functions are defensive: a missing seed makes ``attach`` a no-op and
  ``drain`` return ``{}`` — the channel must never break tool execution.
- One holder per ``execute()`` call; nested executions replace it (documented
  limitation — tools do not nest executions today).
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_OUTCOME_HOLDER: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "rl_tool_outcome_metadata", default=None
)

__all__ = [
    "seed_tool_outcome_metadata",
    "attach_tool_outcome_metadata",
    "drain_tool_outcome_metadata",
]


def seed_tool_outcome_metadata() -> None:
    """Open a fresh outcome payload for the current tool execution (executor)."""
    _OUTCOME_HOLDER.set({})


def attach_tool_outcome_metadata(**fields: Any) -> None:
    """Attach structured outcome fields from inside a tool implementation.

    A no-op when the executor has not seeded a payload (direct tool calls,
    tests) — structured RL metadata is best-effort and must never raise.
    """
    holder = _OUTCOME_HOLDER.get()
    if holder is None:
        return
    try:
        holder.update(fields)
    except Exception:  # pragma: no cover - defensive
        logger.debug("rl_signal: failed to attach %s", list(fields))


def drain_tool_outcome_metadata() -> Dict[str, Any]:
    """Return and clear the pending outcome payload (executor, at emit time)."""
    holder = _OUTCOME_HOLDER.get()
    _OUTCOME_HOLDER.set(None)
    if not holder:
        return {}
    return dict(holder)
