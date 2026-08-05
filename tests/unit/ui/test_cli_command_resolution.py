# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Tests for helpful command resolution on the Victor CLI.

Guards against regressions of the UX where unknown / mistyped commands emitted
a bare "No such command" and where no ``help`` subcommand existed.
"""

from typer.testing import CliRunner

from victor.ui.cli import app

runner = CliRunner()


def _all_output(result) -> str:
    """Combine stdout and stderr so assertions are robust to Click routing/version.

    Click <8.2's ``CliRunner`` mixes stderr into stdout and raises
    ``ValueError("stderr not separately captured")`` on ``result.stderr``;
    Click >=8.2 captures it separately. Tolerate both.
    """
    output = result.output or ""
    try:
        output += getattr(result, "stderr", "") or ""
    except ValueError:
        pass  # stderr not separately captured (Click <8.2 with mixed streams)
    return output


def test_unknown_command_suggests_nested_subcommand():
    """`victor vacuum` should point at the real `victor db vacuum`."""
    result = runner.invoke(app, ["vacuum"])

    assert result.exit_code != 0
    assert "victor db vacuum" in _all_output(result)


def test_typo_suggests_help():
    """`victor hwlp` should suggest the `help` command."""
    result = runner.invoke(app, ["hwlp"])

    assert result.exit_code != 0
    assert "victor help" in _all_output(result)


def test_help_command_exists_and_shows_top_level_help():
    """`victor help` should behave like `--help` (exit 0)."""
    result = runner.invoke(app, ["help"])

    assert result.exit_code == 0
    out = _all_output(result)
    assert "Victor - Open-source agentic AI framework" in out
    # A previously-unknown command is now discoverable.
    assert "help" in out


def test_help_command_descends_into_nested_group():
    """`victor help db` should render the `db` group help."""
    result = runner.invoke(app, ["help", "db"])

    assert result.exit_code == 0
    assert "Database maintenance" in _all_output(result)


def test_help_command_descends_into_nested_command():
    """`victor help db vacuum` should render the `db vacuum` command help."""
    result = runner.invoke(app, ["help", "db", "vacuum"])

    assert result.exit_code == 0
    assert "SQLite VACUUM" in _all_output(result)


def test_help_command_suggests_on_nested_typo():
    """A typo under `help` should still suggest the real nested command."""
    result = runner.invoke(app, ["help", "db", "vacum"])

    assert result.exit_code != 0
    assert "victor db vacuum" in _all_output(result)


def test_normal_command_resolution_is_unaffected():
    """A real top-level command must still resolve (no false suggestion error)."""
    result = runner.invoke(app, ["doctor", "--help"])

    assert result.exit_code == 0
    assert "Run system diagnostics" in _all_output(result)


def test_truly_unknown_command_still_errors():
    """Something with no close match must still fail (no silent success)."""
    result = runner.invoke(app, ["zzqqxx-not-a-command"])

    assert result.exit_code != 0
    assert "No such command" in _all_output(result)
