# Copyright 2026 Vijaykumar Singh <vijaykumar@anvaiops.com>
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

"""Lazy slash-command registration (co-design review item 21).

`victor.ui.slash.__init__` used to eagerly `from victor.ui.slash import
commands`, importing all 17 command modules at package-import time —
dominated by commands/bayesian.py pulling victor.framework.rl.monitoring
-> victor.agent (measured ~0.9s, 205 victor.agent submodules). Commands
now register lazily: CommandRegistry._ensure_discovered() runs
discover_commands() on first read (get/has/list_commands/iter_commands/
categories/list_by_category), memoized; SlashCommandHandler.__init__
still discovers eagerly at handler-construction time as before.

The sys.modules probe MUST run in a fresh subprocess: any earlier test
in the same pytest session may have already imported victor.agent for
unrelated reasons, which would make the probe pass even with the
eager-import bug reintroduced.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from victor.ui.slash.registry import CommandRegistry

# Repo root: parents[3] = tests/unit/ui/<file> -> tests/unit -> tests -> root.
# Explicit cwd matters — a subprocess's sys.path[0] is derived from cwd for
# `-c` scripts, and this repo's local dev workflow can have an editable
# `victor` install shadowing the checkout unless cwd is pinned to it.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_probe(snippet: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, f"probe failed:\n{result.stdout}\n{result.stderr}"
    return result.stdout.strip()


class TestImportDoesNotPullAgent:
    def test_importing_slash_package_does_not_import_agent(self):
        """The regression this PR fixes: importing victor.ui.slash alone
        must not drag in victor.agent or victor.framework.rl."""
        output = _run_probe(
            """
            import sys
            import victor.ui.slash
            agent_mods = [m for m in sys.modules if m.startswith("victor.agent")]
            rl_mods = [m for m in sys.modules if "framework.rl" in m]
            print(len(agent_mods), len(rl_mods))
            """
        )
        agent_count, rl_count = (int(x) for x in output.split())
        assert agent_count == 0, "victor.ui.slash import must not pull victor.agent"
        assert rl_count == 0, "victor.ui.slash import must not pull framework.rl"

    def test_slash_commands_submodule_still_directly_importable(self):
        """Tests/tooling that want the heavy chain explicitly can still get
        it — laziness is about the DEFAULT path, not removing the module."""
        output = _run_probe(
            """
            import victor.ui.slash.commands.bayesian  # noqa: F401
            print("ok")
            """
        )
        assert output == "ok"


class TestFirstDispatchStillWorks:
    def test_handler_construction_discovers_commands(self):
        """SlashCommandHandler.__init__ must still populate the registry
        (its own auto_discover path), independent of package-import time."""
        from io import StringIO

        from rich.console import Console

        from victor.ui.slash.handler import SlashCommandHandler

        console = Console(file=StringIO())
        registry = CommandRegistry()
        handler = SlashCommandHandler(console=console, settings=None, registry=registry)
        assert handler.registry.has("help")
        assert handler.registry.has("clear")

    def test_help_command_executes_after_lazy_discovery(self):
        from io import StringIO

        from rich.console import Console

        from victor.ui.slash.handler import SlashCommandHandler

        console = Console(file=StringIO())
        handler = SlashCommandHandler(console=console, settings=None, registry=CommandRegistry())
        assert handler.is_command("/help")


class TestRegistryLazyDiscovery:
    """The gap the original plan missed: callers that read the registry
    WITHOUT going through SlashCommandHandler (e.g. the TUI palette)."""

    def test_get_triggers_discovery_on_empty_registry(self):
        registry = CommandRegistry()
        assert registry.get("help") is not None

    def test_has_triggers_discovery_on_empty_registry(self):
        registry = CommandRegistry()
        assert registry.has("clear") is True

    def test_list_commands_triggers_discovery_on_empty_registry(self):
        registry = CommandRegistry()
        assert len(registry.list_commands()) > 0

    def test_iter_commands_triggers_discovery_on_empty_registry(self):
        registry = CommandRegistry()
        assert any(registry.iter_commands())

    def test_categories_triggers_discovery_on_empty_registry(self):
        registry = CommandRegistry()
        assert len(registry.categories()) > 0

    def test_discovery_happens_exactly_once(self):
        """A second read must not re-scan the commands package."""
        registry = CommandRegistry()
        registry.get("help")
        first_count = len(registry._commands)
        registry.has("clear")
        registry.list_commands()
        assert len(registry._commands) == first_count
        assert registry._auto_discover_attempted is True

    def test_manually_populated_registry_is_not_rediscovered(self):
        """A registry with explicit test commands must not be clobbered
        by auto-discovery — matches the existing `not any(iter_commands())`
        convention used elsewhere in this package."""
        from victor.ui.slash.protocol import BaseSlashCommand, CommandMetadata

        class _Probe(BaseSlashCommand):
            @property
            def metadata(self) -> CommandMetadata:
                return CommandMetadata(name="probe_only", description="d", usage="/probe_only")

        registry = CommandRegistry()
        registry.register(_Probe())
        assert registry.has("probe_only")
        assert registry.has("help") is False, "must not auto-discover once non-empty"

    def test_tui_palette_works_without_prior_handler_construction(self):
        """victor/ui/tui/palette.py reads get_command_registry() directly —
        the one production call site that bypassed both
        SlashCommandHandler's and chat.py's own discovery guards."""
        output = _run_probe(
            """
            from victor.ui.tui.palette import load_commands
            rows = load_commands()
            print(len(rows))
            """
        )
        assert int(output) > 0
