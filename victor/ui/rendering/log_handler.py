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

"""Live-display-aware console log handler.

During interactive chat a raw ``StreamHandler(sys.stderr)`` competes with the
Rich ``Live`` region: a log record emitted mid-frame tears the display, and
tool-execution errors end up rendered twice (raw log line + the renderer's
✗ error line). This handler routes console log output through the active Rich
console when a live display owns the screen — Rich prints it cleanly above
the live region — and behaves exactly like a plain stream handler otherwise
(headless, API server, tests).

The active live console is registered by :class:`LiveDisplayRenderer` around
its display lifecycle. Records tagged ``victor_category="tool_execution"``
(see ``ErrorHandler._log_error``) are dropped while a live display is active,
because the renderer already surfaces those as error cards; with no live
display they still reach the stream unchanged.
"""

from __future__ import annotations

import logging
import sys
from typing import IO, Optional

from rich.text import Text

# The registry itself lives in victor.runtime so agent-layer consumers can
# read it without importing UI modules; re-exported here for UI callers.
from victor.runtime.live_console import (  # noqa: F401
    get_live_console,
    live_display_active,
    register_live_console,
    unregister_live_console,
)

__all__ = [
    "LiveAwareLogHandler",
    "register_live_console",
    "unregister_live_console",
    "get_live_console",
    "live_display_active",
]


class LiveAwareLogHandler(logging.Handler):
    """Console log handler that cooperates with an active Rich Live display."""

    def __init__(self, stream: Optional[IO[str]] = None) -> None:
        super().__init__()
        self._stream = stream

    def emit(self, record: logging.LogRecord) -> None:
        try:
            console = get_live_console()
            if console is not None:
                # Already surfaced as a renderer error card — don't duplicate.
                if getattr(record, "victor_category", "") == "tool_execution":
                    return
                style = "red" if record.levelno >= logging.ERROR else "yellow"
                console.print(Text(self.format(record), style=f"dim {style}"))
                return
            stream = self._stream if self._stream is not None else sys.stderr
            stream.write(self.format(record) + "\n")
            stream.flush()
        except Exception:  # pragma: no cover - defensive, mirrors logging.Handler
            self.handleError(record)
