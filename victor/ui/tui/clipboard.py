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

"""Read the real OS clipboard for the TUI paste action.

Textual's ``Input.action_paste`` (Ctrl+V) only reads Textual's *in-app* clipboard
buffer — text copied inside the app via OSC 52 — not the operating-system
clipboard. A terminal app cannot read the OS clipboard portably, so this module
shells out to the platform's clipboard tool (``pbpaste`` on macOS,
``powershell.exe Get-Clipboard`` under WSL/Windows interop, ``wl-paste``/``xclip``/
``xsel`` on Linux), falling back to :mod:`pyperclip` when present.

Backend selection is pure and unit-testable: pass a ``runner`` and/or explicit
``candidates`` to avoid touching a real clipboard.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from typing import Callable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: A command runner: argv -> captured stdout. Raises on non-zero exit / timeout.
Runner = Callable[[Sequence[str]], str]

#: A backend candidate: (argv, strip_one_trailing_newline).
Candidate = Tuple[Sequence[str], bool]

#: Max seconds to wait on a clipboard tool before giving up (paste must feel instant).
_TIMEOUT_S = 2.0


def _default_runner(argv: Sequence[str]) -> str:
    """Run ``argv`` and return its stdout, raising on failure or timeout."""
    proc = subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
        check=True,
    )
    return proc.stdout


def _candidates() -> List[Candidate]:
    """Return the clipboard-read backends to try, best-first, for this platform.

    Only tools actually present on ``PATH`` are included, so an unavailable
    backend never contributes a failing attempt.
    """
    cands: List[Candidate] = []
    if sys.platform == "darwin" and shutil.which("pbpaste"):
        # pbpaste emits the clipboard verbatim, no trailing newline added.
        cands.append((["pbpaste"], False))
    # WSL / Windows interop: Get-Clipboard appends a trailing CRLF.
    if shutil.which("powershell.exe"):
        cands.append((["powershell.exe", "-NoProfile", "-Command", "Get-Clipboard"], True))
    # Wayland, then X11. --no-newline / -o keep the payload intact.
    if shutil.which("wl-paste"):
        cands.append((["wl-paste", "--no-newline"], False))
    if shutil.which("xclip"):
        cands.append((["xclip", "-selection", "clipboard", "-o"], False))
    if shutil.which("xsel"):
        cands.append((["xsel", "--clipboard", "--output"], False))
    return cands


def _normalize(text: str, strip_trailing: bool) -> str:
    """Normalize line endings to ``\\n`` and drop a tool-appended trailing newline."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if strip_trailing and text.endswith("\n"):
        text = text[:-1]
    return text


def read_clipboard(
    runner: Optional[Runner] = None,
    candidates: Optional[Sequence[Candidate]] = None,
) -> Optional[str]:
    """Return the OS clipboard contents, or ``None`` if it can't be read.

    Tries each platform backend in order and returns the first non-empty result;
    when no CLI backend succeeds, falls back to :mod:`pyperclip` if installed.

    Args:
        runner: Command runner override (for tests). Defaults to a subprocess
            runner with a short timeout.
        candidates: Explicit backend list (for tests). Defaults to the
            platform-detected set.

    Returns:
        The clipboard text with normalized ``\\n`` line endings, or ``None`` when
        no backend is available or the clipboard is empty.
    """
    run = runner if runner is not None else _default_runner
    for argv, strip_trailing in candidates if candidates is not None else _candidates():
        try:
            out = run(argv)
        except Exception:  # noqa: BLE001 - any backend failure just tries the next
            logger.debug("clipboard backend failed: %s", argv[0] if argv else argv)
            continue
        if out:
            return _normalize(out, strip_trailing)
    return _read_pyperclip()


def _read_pyperclip() -> Optional[str]:
    """Last-resort clipboard read via pyperclip, or ``None`` if unusable."""
    try:
        import pyperclip  # noqa: PLC0415 - optional, imported lazily

        text = pyperclip.paste()
    except Exception:  # noqa: BLE001 - not installed or no backend on this host
        return None
    return _normalize(text, False) if text else None
