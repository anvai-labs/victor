# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Guard tests for the process-wide themed console seam (audit P1).

Before victor/ui/console.py there were ~48 bare ``Console()`` instances in
``victor/ui`` against one themed console, so the capability-detected theme
tokens reached almost none of the user-visible output. The seam is now the
only sanctioned way to get a console; these tests pin identity, idempotency,
and forbid regressions to bare construction.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from victor.ui.console import get_console, get_stderr_console
from victor.ui.theme import victor_theme

UI_DIR = Path(__file__).resolve().parents[3] / "victor" / "ui"


class TestConsoleSeam:
    def test_get_console_is_idempotent(self):
        first = get_console()
        second = get_console()
        assert first is second

    def test_get_console_binds_victor_theme(self, monkeypatch):
        """The seam must construct with the capability-detected theme — the
        whole point of consolidating the ~48 bare consoles (Rich 14 does not
        retain the theme on the instance, so spy on construction instead)."""
        import victor.ui.console as console_mod

        captured: dict = {}
        real_init = Console.__init__

        def spy(self, *args, **kwargs):
            captured.update(kwargs)
            return real_init(self, *args, **kwargs)

        monkeypatch.setattr(console_mod.Console, "__init__", spy)
        monkeypatch.setattr(console_mod, "_console", None)
        console_mod.get_console()
        assert captured.get("theme") is victor_theme

    def test_stderr_console_is_separate_and_on_stderr(self):
        err = get_stderr_console()
        out = get_console()
        assert err is not out
        assert err.stderr
        assert err.file is out.file or err is get_stderr_console()

    def test_errors_module_uses_the_seam(self):
        import victor.ui.errors as errors

        assert isinstance(errors._default_console, Console)
        # One definition only: the historical duplicate at the module tail
        # re-bound a fresh unthemed console over the seam (audit P1).
        source = (UI_DIR / "errors.py").read_text()
        assert source.count("_default_console = ") == 1
        assert "Console()" not in source

    def test_commands_modules_share_one_console_instance(self):
        """The mechanical sweep: every migrated module resolves to the same
        console object — the property the theme tokens depend on."""
        import victor.ui.commands.chat as chat_cmd
        import victor.ui.commands.keys as keys_cmd
        import victor.ui.commands.models as models_cmd
        import victor.ui.cli as cli

        assert chat_cmd.console is cli.console
        assert keys_cmd.console is cli.console
        assert models_cmd.console is cli.console


class TestNoBareConsoleRegression:
    def test_no_bare_console_construction_in_victor_ui(self):
        """Grep guard: bare ``Console()`` (no kwargs) is forbidden in victor/ui
        outside the seam module. Sanctioned exceptions carry explicit kwargs
        (file= capture consoles, stderr=True diagnostics) or live in the
        core-layer (which must not import victor.ui)."""
        offenders: list[str] = []
        for path in sorted(UI_DIR.rglob("*.py")):
            if path.name == "console.py":
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "Console()" in line and "file=" not in line and "stderr" not in line:
                    offenders.append(f"{path.relative_to(UI_DIR.parent)}:{lineno}: {stripped}")
        assert not offenders, "bare Console() regression:\n" + "\n".join(offenders)
