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

"""UX P2: CLI verb consolidation — nested verbs, hidden aliases, hints.

Satellite command apps nest under their owning verb (``config``/``index``/
``tool`` grew children; ``observe`` and ``data`` are new groups). The old flat
names stay fully invocable but hidden, and resolving one prints a one-line
migration hint (``SuggestingGroup.legacy_paths``). No command loses
functionality.
"""

from __future__ import annotations

import click
import pytest
from typer.main import get_command
from typer.testing import CliRunner

from victor.ui.cli import _LEGACY_VERB_PATHS, app
from victor.ui.cli_group import SuggestingGroup

runner = CliRunner()


NEW_VERB_PATHS = [
    ["config", "profiles"],
    ["config", "provider"],
    ["index", "graph"],
    ["index", "embedding"],
    ["tool", "capability"],
    ["tool", "examples"],
    ["observe", "benchmark"],
    ["observe", "experiment"],
    ["observe", "ab"],
    ["observe", "ml"],
    ["observe", "dashboard"],
    ["observe", "bayesian"],
    ["observe", "observability"],
    ["observe", "gateway"],
    ["data", "session"],
    ["data", "db"],
]


class TestVerbNesting:
    @pytest.mark.parametrize("path", NEW_VERB_PATHS, ids=" ".join)
    def test_new_verb_path_resolves(self, path):
        result = runner.invoke(app, [*path, "--help"])
        assert result.exit_code == 0, result.output


class TestLegacyAliases:
    @pytest.mark.parametrize("name", sorted(_LEGACY_VERB_PATHS), ids=str)
    def test_legacy_flat_name_still_works(self, name):
        """Old invocations keep working — no command loses functionality."""
        result = runner.invoke(app, [name, "--help"])
        assert result.exit_code == 0, result.output

    def test_legacy_names_are_hidden_from_top_level_help(self):
        cmd = get_command(app)
        ctx = click.Context(cmd, info_name="victor")
        visible = {
            name
            for name in cmd.list_commands(ctx)
            if not getattr(cmd.get_command(ctx, name), "hidden", False)
        }
        leaked = visible & set(_LEGACY_VERB_PATHS)
        assert not leaked, f"legacy flat names visible in top-level help: {leaked}"

    def test_verbs_are_visible(self):
        cmd = get_command(app)
        ctx = click.Context(cmd, info_name="victor")
        visible = {
            name
            for name in cmd.list_commands(ctx)
            if not getattr(cmd.get_command(ctx, name), "hidden", False)
        }
        assert {"config", "index", "tool", "observe", "data"} <= visible

    def test_legacy_resolution_prints_migration_hint(self):
        result = runner.invoke(app, ["db", "--help"])
        combined = (result.stderr or "") + result.output
        assert "is moving to 'victor data db'" in combined

    def test_new_paths_do_not_print_hint(self):
        result = runner.invoke(app, ["data", "db", "--help"])
        combined = (result.stderr or "") + result.output
        assert "is moving to" not in combined

    def test_hint_map_matches_registered_legacy_names(self):
        """Every hint key must be a real (hidden) top-level command."""
        cmd = get_command(app)
        ctx = click.Context(cmd, info_name="victor")
        registered = set(cmd.list_commands(ctx))
        missing = set(_LEGACY_VERB_PATHS) - registered
        assert not missing, f"hint map references unregistered commands: {missing}"
        assert SuggestingGroup.legacy_paths == _LEGACY_VERB_PATHS

    @pytest.mark.parametrize("name", sorted(_LEGACY_VERB_PATHS), ids=str)
    def test_hint_target_resolves(self, name):
        """Every 'moving to' target must itself be a working invocation."""
        target = _LEGACY_VERB_PATHS[name].split()
        result = runner.invoke(app, [*target, "--help"])
        assert result.exit_code == 0, f"hint target 'victor {_LEGACY_VERB_PATHS[name]}' broken"
