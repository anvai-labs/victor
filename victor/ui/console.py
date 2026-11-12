# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""The process-wide themed Rich console seam for Victor's UI layer.

Before this module there were ~48 bare ``Console()`` instances across
``victor/ui`` against exactly one themed console (``cli.py``), which meant the
capability-detected theme tokens (``victor/ui/theme.py``) reached essentially
none of the user-visible output. Every module that prints to the terminal
should take its console from here instead of constructing one.

Not covered by the seam (intentionally):
- ``Console(stderr=True)`` diagnostic consoles (deliberate stream separation),
- file-bound consoles (``OutputFormatter`` writes to a possibly-captured
  stdout — a mode-aware surface, not a styling choice),
- capture consoles writing into buffers (TUI slash capture, tests),
- ``victor/core/*`` (the core layer must not import ``victor.ui``).

The Textual theme registry (``victor/ui/tui/themes.py``) is a separate,
healthy system for the Textual app and is deliberately not merged here.
"""

from __future__ import annotations

from rich.console import Console

from victor.ui.theme import victor_theme

_console: Console | None = None
_stderr_console: Console | None = None


def get_console() -> Console:
    """Return the process-wide themed stdout console (idempotent).

    First call wins: the console binds once with the capability-detected
    ``victor_theme`` so every consumer shares one set of theme tokens.
    """
    global _console
    if _console is None:
        _console = Console(theme=victor_theme)
    return _console


def get_stderr_console() -> Console:
    """Return the shared themed stderr console for diagnostics and noise.

    Stream separation (keeping diagnostics off stdout, e.g. under JSON mode)
    is preserved; only the styling is shared with the main console.
    """
    global _stderr_console
    if _stderr_console is None:
        _stderr_console = Console(theme=victor_theme, stderr=True)
    return _stderr_console
