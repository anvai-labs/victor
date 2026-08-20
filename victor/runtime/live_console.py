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

"""Process-global registry for the console owning an active live display.

The interactive CLI's ``LiveDisplayRenderer`` registers its Rich console
around the display lifecycle. Consumers on both sides of the UI/agent
boundary read it:

- ``victor.ui.rendering.log_handler.LiveAwareLogHandler`` routes console log
  records through the live console so they print above the Live region.
- ``victor.agent.services.tool_service`` suppresses its direct console error
  print while a live renderer is active (the renderer already surfaces the
  failure as an error card).

Living in ``victor.runtime`` keeps the agent layer free of UI imports: the
registry stores the console as an opaque object and never imports rich.
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = [
    "register_live_console",
    "unregister_live_console",
    "get_live_console",
    "live_display_active",
]

_live_console: Optional[Any] = None


def register_live_console(console: Any) -> None:
    """Mark ``console`` as owning the screen (a live display is running)."""
    global _live_console
    _live_console = console


def unregister_live_console() -> None:
    """Clear the live-console registration (display stopped)."""
    global _live_console
    _live_console = None


def get_live_console() -> Optional[Any]:
    """Return the console owning an active live display, if any."""
    return _live_console


def live_display_active() -> bool:
    """True when an interactive live renderer owns console output."""
    return _live_console is not None
