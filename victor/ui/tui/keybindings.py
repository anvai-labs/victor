# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""User-overridable keybindings for the TUI (ADR-021).

Ships sane defaults; an optional ``~/.victor/keybindings.json`` remaps any action
to a different key. The file maps *action name* → *key* so users rebind by intent
(e.g. ``{"toggle_sidebar": "f2", "interrupt": "escape"}``) without needing to know
Textual's binding tuples. A missing or malformed file falls back to the defaults.

Pure logic; imports nothing from Textual and is unit-testable standalone.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: A Textual binding: (key, action, human description).
Binding = Tuple[str, str, str]

#: Default bindings, keyed by action. Order is the display order in the footer.
DEFAULT_BINDINGS: Tuple[Binding, ...] = (
    ("ctrl+q", "quit", "Quit"),
    ("escape", "interrupt", "Interrupt"),
    ("ctrl+p", "command_palette", "Commands"),
    ("f1", "help", "Help"),
    ("f2", "toggle_sidebar", "Sidebar"),
    ("f3", "toggle_diff", "Diff"),
    ("f4", "diff_next", "Next edit"),
    ("f6", "cycle_theme", "Theme"),
    ("ctrl+l", "clear", "Clear"),
    ("ctrl+c", "copy", "Copy"),
)

#: Where the user override file lives, relative to the home directory.
_OVERRIDE_RELPATH = Path(".victor") / "keybindings.json"


def default_override_path() -> Path:
    """Return the default ``~/.victor/keybindings.json`` path."""
    return Path.home() / _OVERRIDE_RELPATH


def load_keybindings(path: Optional[Path] = None) -> List[Binding]:
    """Return the effective bindings, applying user overrides when present.

    Args:
        path: Override-file location; defaults to :func:`default_override_path`.

    Returns:
        The default bindings with any ``action → key`` overrides applied. Unknown
        actions in the file are ignored (logged at debug); a missing or malformed
        file yields the defaults unchanged.
    """
    overrides = _read_overrides(path if path is not None else default_override_path())
    result: List[Binding] = []
    for key, action, description in DEFAULT_BINDINGS:
        new_key = overrides.get(action)
        if isinstance(new_key, str) and new_key.strip():
            result.append((new_key.strip(), action, description))
        else:
            result.append((key, action, description))
    return result


def _read_overrides(path: Path) -> Dict[str, str]:
    """Read the override JSON as an ``action → key`` map, or ``{}`` on any error."""
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        logger.debug("keybindings override is not valid JSON: %s", path)
        return {}
    if not isinstance(data, dict):
        logger.debug("keybindings override must be a JSON object: %s", path)
        return {}
    # Keep only string→string entries; ignore the rest defensively.
    return {
        str(action): value
        for action, value in data.items()
        if isinstance(action, str) and isinstance(value, str)
    }
