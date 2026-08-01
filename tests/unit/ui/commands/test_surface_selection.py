"""ADR-020: the interactive TUI is the default on capable terminals (opt-out via --repl).

The line-oriented REPL remains the fallback for dumb terminals / pipes / CI, guarded by
``_tui_capable()``. These tests pin the tri-state flag mapping and the critical safety
property that a non-TTY / test / CI environment never selects the Textual TUI.
"""

from __future__ import annotations

from victor.ui.commands.chat import _resolve_surface, _tui_capable

# ── tri-state flag → surface ──────────────────────────────────────


def test_no_flag_is_auto() -> None:
    # Neither --tui nor --repl → capability-gated default.
    assert _resolve_surface(None) == "auto"


def test_tui_flag_forces_tui() -> None:
    assert _resolve_surface(True) == "tui"


def test_repl_flag_forces_repl() -> None:
    assert _resolve_surface(False) == "repl"


# ── safety: never launch the TUI in a non-interactive environment ──


def test_not_tui_capable_under_pytest() -> None:
    # pytest runs without an interactive full-capability TTY (and CI sets CI=true),
    # so the auto default must resolve to the REPL, never the Textual TUI. This guards
    # the pipe/CI regression the tri-state default could otherwise introduce.
    assert _tui_capable() is False
