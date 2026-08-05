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

"""UI-ephemeral progress channel for long-running tools.

Long-running tools (shell, code_search, graph) can stream partial stdout/stderr
to an active live renderer *while they run* so the user sees output progressively
instead of a final blob. This output is purely for human visibility — it is
**never** added to the conversation, persisted, or sent to the model, so it is
decoupled from the LLM message stream and from ``OutputPruner`` entirely.

Design:
- A single process-global sink, intended for the single-active-turn interactive
  CLI. The rendering layer registers a sink around a turn; tools emit through it.
- When no sink is registered (the default — API/server path, tests, headless),
  ``emit_tool_progress`` is a cheap no-op, so this has zero behavioural effect.
- Emission is best-effort: a failing sink never propagates into tool execution.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Sink signature: (name, stdout, stderr, progress, is_final) -> None
ProgressSink = Callable[..., None]

_sink: Optional[ProgressSink] = None

__all__ = [
    "set_progress_sink",
    "clear_progress_sink",
    "has_progress_sink",
    "emit_tool_progress",
]


def set_progress_sink(sink: Optional[ProgressSink]) -> None:
    """Register the active progress sink (rendering layer owns the lifecycle)."""
    global _sink
    _sink = sink


def clear_progress_sink() -> None:
    """Remove the active progress sink."""
    global _sink
    _sink = None


def has_progress_sink() -> bool:
    """Return whether a sink is currently registered.

    Tools can use this to skip the (slightly more expensive) streaming read path
    when nobody is listening.
    """
    return _sink is not None


def emit_tool_progress(
    name: str,
    stdout: str = "",
    stderr: str = "",
    progress: float = 0.0,
    is_final: bool = False,
) -> None:
    """Emit a partial tool-output chunk to the active sink (no-op if unset).

    Args:
        name: Name of the running tool.
        stdout: Partial stdout produced since the last emission.
        stderr: Partial stderr produced since the last emission.
        progress: Optional 0.0–1.0 progress estimate.
        is_final: Whether this is the last chunk before the tool result.
    """
    sink = _sink
    if sink is None:
        return
    try:
        sink(
            name=name,
            stdout=stdout,
            stderr=stderr,
            progress=progress,
            is_final=is_final,
        )
    except Exception:  # pragma: no cover - defensive; UI must never break a tool
        logger.debug("tool progress sink raised; ignoring", exc_info=True)
