"""Process-level fault/hang diagnostics for the Victor CLI.

Installs an in-process :mod:`faulthandler` so that a wedged session can be
introspected *without* root — important on macOS, where the hardened runtime on
framework Python builds blocks ``task_for_pid`` and therefore py-spy, lldb and
dtrace cannot attach to a running interpreter.

Capabilities (all in-process, so no external attach is required):

* ``faulthandler.enable()`` — dump a C+Python traceback to stderr on a fatal
  error (segfault, abort, bus error).
* ``SIGUSR1`` → dump every thread's Python stack to stderr on demand. Diagnose a
  hang with ``kill -USR1 <pid>``; the stacks print straight to the session's
  terminal, pinpointing the exact frame the process is stuck in.
* ``VICTOR_FAULTHANDLER_TIMEOUT=<seconds>`` (optional) — also auto-dump at that
  interval, repeating, to catch hangs with no manual signalling. This is a
  wall-clock timer; it also fires during healthy idle sessions.

Gated by ``VICTOR_FAULTHANDLER`` (default ``"1"``; set to ``"0"`` to disable).
The ``SIGUSR1`` dump does not chain to the previous handler: the default
disposition terminates the process, which would destroy the session being
diagnosed immediately after printing its first dump.
"""

from __future__ import annotations

import faulthandler
import os
import signal
import sys
from typing import Optional, TextIO

__all__ = ["install_fault_diagnostics"]

_installed = False


def _diagnostics_enabled() -> bool:
    return os.environ.get("VICTOR_FAULTHANDLER", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def install_fault_diagnostics(stream: Optional[TextIO] = None) -> bool:
    """Install faulthandler-based hang/crash diagnostics.

    Safe to call multiple times (idempotent) and from the CLI entrypoint only —
    signal registration must happen on the main thread, and any failure (e.g. a
    non-main thread, a closed stderr, or an unsupported platform) is swallowed so
    diagnostics never break the actual command.

    Args:
        stream: Destination for traceback dumps. Defaults to ``sys.stderr``.

    Returns:
        ``True`` if at least the on-demand dump handler was installed.
    """
    global _installed
    if _installed or not _diagnostics_enabled():
        return False

    out = stream if stream is not None else sys.stderr
    # faulthandler needs a real file descriptor; captured/StringIO streams (e.g.
    # under pytest capture) have no fileno() and must be skipped gracefully.
    try:
        out.fileno()
    except (AttributeError, OSError, ValueError):
        return False

    installed_any = False

    try:
        faulthandler.enable(file=out, all_threads=True)
        installed_any = True
    except (RuntimeError, ValueError, OSError):
        pass

    sigusr1 = getattr(signal, "SIGUSR1", None)
    if sigusr1 is not None:
        try:
            faulthandler.register(sigusr1, file=out, all_threads=True, chain=False)
            installed_any = True
        except (RuntimeError, ValueError, OSError):
            pass

    timeout_raw = os.environ.get("VICTOR_FAULTHANDLER_TIMEOUT")
    if timeout_raw:
        try:
            timeout = float(timeout_raw)
        except (TypeError, ValueError):
            timeout = 0.0
        if timeout > 0:
            try:
                faulthandler.dump_traceback_later(timeout, repeat=True, file=out)
                installed_any = True
            except (RuntimeError, ValueError, OSError):
                pass

    _installed = installed_any
    return installed_any
