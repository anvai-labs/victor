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

"""Unit tests for the unified ``gh`` tool.

Covers the three contract points that matter for a passthrough-over-parser
design: the parser validates the subcommand shape, unmodeled flags survive,
and the argv handed to the shell tool is shlex-safe (a title/body with
spaces, backticks or ``$`` must reach ``gh`` verbatim, not the shell).
"""

import shlex
from unittest.mock import AsyncMock, patch

import pytest

from victor.agent.shared_tool_registry import SharedToolRegistry
from victor.tools.unified.gh_tool import create_gh_parser, gh_tool

SHELL_RESULT = {"success": True, "return_code": 0, "stdout": "ok", "stderr": ""}


@pytest.fixture
def mock_shell():
    with patch("victor.tools.unified.gh_tool.shell", new_callable=AsyncMock) as m:
        m.return_value = dict(SHELL_RESULT)
        yield m


@pytest.fixture
def gh_installed():
    with patch("victor.tools.unified.gh_tool.shutil.which", return_value="/usr/bin/gh"):
        yield


class TestGhParser:
    def test_known_subcommands_parse(self):
        parser = create_gh_parser()
        parsed = parser.parse_known_args(["pr", "list", "--base", "develop"])[0]
        assert parsed.subcommand == "pr"

    def test_no_subcommand_detected(self):
        # gh_tool() reports this after parsing; parser itself yields subcommand=None.
        parsed = create_gh_parser().parse_known_args([])[0]
        assert parsed.subcommand is None


class TestGhTool:
    @pytest.mark.asyncio
    async def test_unknown_subcommand_rejected_without_shelling_out(self, mock_shell, gh_installed):
        # Unlisted subcommands (gh gist, gh codespace, ...) are out of this
        # tool's contract: parse_known_args leaves them as subcommand=None and
        # gh_tool must refuse rather than execute them.
        result = await gh_tool("gh gist list")
        assert "### ❌ ERROR" in result
        assert "choose from" in result
        mock_shell.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_binary_returns_install_hint_without_shelling_out(self, mock_shell):
        with patch("victor.tools.unified.gh_tool.shutil.which", return_value=None):
            result = await gh_tool("gh pr list")
        assert "brew install gh" in result
        mock_shell.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("gh_installed")
    async def test_no_subcommand_returns_error(self, mock_shell):
        result = await gh_tool("gh")
        assert "### ❌ ERROR" in result
        mock_shell.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("gh_installed")
    async def test_argv_is_shell_quoted(self, mock_shell):
        # split_command strips the quoting; _run_gh must re-quote so spaces,
        # backticks and $ in a title/body go to gh, not to the shell.
        raw = 'gh pr create --title "Fix X" --body "uses `gh` in $HOME"'
        tokens = shlex.split(raw)[1:]
        await gh_tool(raw)
        cmd = mock_shell.call_args.kwargs["cmd"]
        assert shlex.split(cmd) == ["gh", *tokens]
        assert shlex.split(cmd)[1 + tokens.index("uses `gh` in $HOME")] == "uses `gh` in $HOME"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("gh_installed")
    async def test_unmodeled_flags_pass_through(self, mock_shell):
        # --squash / --label are not modeled by the parser; execution must still
        # run the raw command (validation covers the subcommand shape only).
        await gh_tool("gh pr merge 123 --squash")
        assert mock_shell.call_args.kwargs["cmd"] == "gh pr merge 123 --squash"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("gh_installed")
    async def test_nonzero_exit_surfaces_stderr(self, mock_shell):
        mock_shell.return_value = {
            "success": False,
            "return_code": 4,
            "stdout": "",
            "stderr": "not logged in",
        }
        result = await gh_tool("gh pr list")
        assert "exit 4" in result
        assert "not logged in" in result


class TestGhDemandWiring:
    """gh must be reachable: registered on demand, hydrated by GitHub keywords."""

    def test_gh_has_a_demand_spec_pointing_at_the_decorated_tool(self):
        from victor.agent.shared_tool_registry import DEMAND_TOOL_SPECS

        module_name, member_name = DEMAND_TOOL_SPECS["gh"]
        assert module_name == "victor.tools.unified.gh_tool"
        import victor.tools.unified.gh_tool as gh_module

        member = getattr(gh_module, member_name)
        assert getattr(member, "_is_tool", False), (
            "DEMAND spec must point at the @tool-decorated function"
        )

    def test_infer_demand_tools_hydrates_gh_on_github_mentions(self):
        registry = SharedToolRegistry.get_instance()
        text = "please open a pull request for this branch on github"
        assert "gh" in registry.infer_demand_tools(text)

    def test_infer_demand_tools_does_not_fire_on_plain_text(self):
        registry = SharedToolRegistry.get_instance()
        # "pr" alone must not match (it is a substring of prompt/print/process).
        assert "gh" not in registry.infer_demand_tools("print the prompt status")
        assert "gh" not in registry.infer_demand_tools("refactor the parser module")

    def test_demand_load_registers_the_tool(self):
        registry = SharedToolRegistry.get_instance()
        loaded = registry.get_tools_for_names(["gh"])
        assert loaded, "gh demand spec failed to load"
        assert registry.get_tool_classes().get("gh") is not None
