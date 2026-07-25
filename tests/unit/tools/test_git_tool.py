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

"""Tests for the unified ``git`` command tool.

Covers the bash-style parser, delegation to ``victor_devops.git`` when the
package is present, the shell fallback when it is absent, AI commit message
generation, and PR creation.
"""

import argparse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.tools.unified.git_tool import (
    _augment_push_hint,
    _build_gh_pr_argv,
    _fallback_command,
    _git_kwarg_map,
    create_git_parser,
    git_tool,
)
from victor.tools.unified.parser import split_command


def _parse(cmd: str) -> argparse.Namespace:
    """Parse using the real shlex-aware splitter (handles quoted messages)."""
    parser = create_git_parser()
    parts = split_command(cmd)
    if parts and parts[0] == "git":
        parts = parts[1:]
    return parser.parse_args(parts)


class TestGitParser:
    """Subcommand parsing."""

    def test_status(self):
        assert _parse("git status").subcommand == "status"

    def test_diff_staged_with_files(self):
        ns = _parse("git diff --staged a.py b.py")
        assert ns.subcommand == "diff"
        assert ns.staged is True
        assert ns.files == ["a.py", "b.py"]

    def test_log_limit(self):
        ns = _parse("git log -n 25")
        assert ns.subcommand == "log"
        assert ns.limit == 25

    def test_stage_and_add_alias(self):
        assert _parse("git stage a.py").subcommand == "stage"
        assert _parse("git add a.py").subcommand == "add"

    def test_commit_message_and_ai(self):
        ns = _parse('git commit -m "fix it"')
        assert ns.subcommand == "commit"
        assert ns.message == "fix it"
        assert ns.ai is False
        ns_ai = _parse("git commit --ai")
        assert ns_ai.ai is True

    def test_push_options(self):
        ns = _parse("git push origin main --force --tags")
        assert ns.subcommand == "push"
        assert ns.remote == "origin"
        assert ns.branch == "main"
        assert ns.force is True
        assert ns.tags is True

    def test_branch_optional_name(self):
        assert _parse("git branch").name is None
        assert _parse("git branch feat/x").name == "feat/x"

    def test_pr_subcommand(self):
        ns = _parse("git pr --title T --base develop")
        assert ns.subcommand == "pr"
        assert ns.title == "T"
        assert ns.base == "develop"


class TestKwargMap:
    """Bash args -> ``victor_devops.git(operation=...)`` kwargs."""

    def test_status(self):
        op, kwargs = _git_kwarg_map(_parse("git status"))
        assert op == "status" and kwargs == {}

    def test_diff_maps_staged_and_files(self):
        op, kwargs = _git_kwarg_map(_parse("git diff --staged a.py"))
        assert op == "diff"
        assert kwargs == {"staged": True, "files": ["a.py"]}

    def test_log_maps_limit(self):
        op, kwargs = _git_kwarg_map(_parse("git log -n 5"))
        assert op == "log" and kwargs == {"limit": 5}

    def test_add_alias_normalizes_to_stage(self):
        op, kwargs = _git_kwarg_map(_parse("git add a.py b.py"))
        assert op == "stage"
        assert kwargs == {"files": ["a.py", "b.py"]}

    def test_commit_maps_message_and_author(self):
        op, kwargs = _git_kwarg_map(
            _parse('git commit -m "msg" --author-name Bot --author-email b@x.io')
        )
        assert op == "commit"
        assert kwargs["message"] == "msg"
        assert kwargs["author_name"] == "Bot"
        assert kwargs["author_email"] == "b@x.io"

    def test_push_maps_options(self):
        op, kwargs = _git_kwarg_map(_parse("git push origin main --force"))
        assert op == "push"
        assert kwargs["branch"] == "main"
        assert kwargs["options"] == {"remote": "origin", "force": True}


class TestFallbackCommand:
    """Shell fallback command construction (used when devops is absent)."""

    def test_status_is_readonly(self):
        sub, argv, ro = _fallback_command(_parse("git status"))
        assert sub == "status" and ro is True

    def test_stage_uses_add_and_dot(self):
        sub, argv, ro = _fallback_command(_parse("git stage"))
        assert sub == "add" and argv == ["."] and ro is False

    def test_stage_specific_files(self):
        sub, argv, ro = _fallback_command(_parse("git stage a.py b.py"))
        assert sub == "add" and argv == ["a.py", "b.py"] and ro is False

    def test_commit_requires_message(self):
        with pytest.raises(ValueError, match="commit message required"):
            _fallback_command(_parse("git commit"))

    def test_commit_message_built(self):
        sub, argv, ro = _fallback_command(_parse('git commit -m "fix"'))
        assert sub == "commit" and argv == ["-m", "fix"] and ro is False

    def test_ai_only_ops_require_devops(self):
        with pytest.raises(ValueError, match="victor-devops"):
            _fallback_command(_parse("git commit_msg"))
        with pytest.raises(ValueError, match="victor-devops"):
            _fallback_command(_parse("git pr --title t"))


class TestDelegationPath:
    """When ``victor_devops.git`` is resolvable, the dispatcher delegates."""

    @pytest.mark.asyncio
    async def test_status_delegates_to_devops(self):
        mock_git = AsyncMock(return_value={"success": True, "output": "clean tree"})
        with patch(
            "victor.tools.unified.git_tool.resolve_vertical_callable",
            return_value=(mock_git, "victor_devops.tools.git_tool"),
        ):
            result = await git_tool("git status")
        mock_git.assert_awaited_once_with(operation="status")
        assert result == "clean tree"

    @pytest.mark.asyncio
    async def test_commit_delegates_with_message(self, monkeypatch):
        # Agent-layer attribution proactively appends the Victor co-author trailer to
        # an explicit commit message (hook-free, Claude-Code-style). Default ON.
        monkeypatch.delenv("VICTOR_COMMIT_ATTRIBUTION", raising=False)
        mock_git = AsyncMock(return_value={"success": True, "output": "Committed"})
        with patch(
            "victor.tools.unified.git_tool.resolve_vertical_callable",
            return_value=(mock_git, "victor_devops.tools.git_tool"),
        ):
            result = await git_tool('git commit -m "fix: x"')
        second_call = mock_git.await_args
        assert second_call.kwargs["operation"] == "commit"
        assert second_call.kwargs["message"].startswith("fix: x")
        assert "Co-authored-by: victor-code-ai" in second_call.kwargs["message"]
        assert second_call.kwargs["author_name"] is None
        assert second_call.kwargs["author_email"] is None
        assert "Committed" in result

    @pytest.mark.asyncio
    async def test_commit_attribution_opt_out(self, monkeypatch):
        # VICTOR_COMMIT_ATTRIBUTION=0 leaves the message untouched.
        monkeypatch.setenv("VICTOR_COMMIT_ATTRIBUTION", "0")
        mock_git = AsyncMock(return_value={"success": True, "output": "Committed"})
        with patch(
            "victor.tools.unified.git_tool.resolve_vertical_callable",
            return_value=(mock_git, "victor_devops.tools.git_tool"),
        ):
            await git_tool('git commit -m "fix: x"')
        assert mock_git.await_args.kwargs["message"] == "fix: x"

    @pytest.mark.asyncio
    async def test_failed_devops_result_is_formatted_as_error(self):
        mock_git = AsyncMock(return_value={"success": False, "error": "no upstream"})
        with patch(
            "victor.tools.unified.git_tool.resolve_vertical_callable",
            return_value=(mock_git, "victor_devops.tools.git_tool"),
        ):
            result = await git_tool("git push")
        assert "### ❌ ERROR" in result
        assert "no upstream" in result


class TestFallbackPath:
    """When devops is absent, the dispatcher falls back to the shell tool."""

    @pytest.mark.asyncio
    async def test_status_uses_shell_when_devops_absent(self):
        mock_shell = AsyncMock(return_value={"success": True, "stdout": "## develop"})
        with (
            patch(
                "victor.tools.unified.git_tool.resolve_vertical_callable",
                return_value=(None, None),
            ),
            patch("victor.tools.bash.shell", mock_shell),
        ):
            result = await git_tool("git status")
        mock_shell.assert_awaited_once()
        called_cmd = mock_shell.call_args.kwargs["cmd"]
        assert called_cmd.startswith("git status")
        assert mock_shell.call_args.kwargs["readonly"] is True
        assert "## develop" in result

    @pytest.mark.asyncio
    async def test_commit_msg_without_devops_returns_error(self):
        with patch(
            "victor.tools.unified.git_tool.resolve_vertical_callable",
            return_value=(None, None),
        ):
            result = await git_tool("git commit_msg")
        assert "### ❌ ERROR" in result
        assert "victor-devops" in result


class TestAiCommit:
    """AI commit message generation flow."""

    @pytest.mark.asyncio
    async def test_ai_commit_generates_then_commits(self, monkeypatch):
        # The AI-generated message is also attributed (hook-free) by default.
        monkeypatch.delenv("VICTOR_COMMIT_ATTRIBUTION", raising=False)
        mock_git = AsyncMock(
            side_effect=[
                {"success": True, "message": "feat: add x"},  # commit_msg
                {"success": True, "output": "Committed"},  # commit
            ]
        )
        with patch(
            "victor.tools.unified.git_tool.resolve_vertical_callable",
            return_value=(mock_git, "victor_devops.tools.git_tool"),
        ):
            result = await git_tool("git commit --ai")
        assert mock_git.await_count == 2
        # First call generates the message.
        first_call = mock_git.await_args_list[0]
        assert first_call.kwargs["operation"] == "commit_msg"
        # Second call commits with the generated message + attribution trailer.
        second_call = mock_git.await_args_list[1]
        assert second_call.kwargs["operation"] == "commit"
        assert second_call.kwargs["message"].startswith("feat: add x")
        assert "Co-authored-by: victor-code-ai" in second_call.kwargs["message"]
        assert second_call.kwargs["author_name"] is None
        assert second_call.kwargs["author_email"] is None
        assert "feat: add x" in result
        assert "Committed" in result

    @pytest.mark.asyncio
    async def test_ai_commit_requires_devops(self):
        with patch(
            "victor.tools.unified.git_tool.resolve_vertical_callable",
            return_value=(None, None),
        ):
            result = await git_tool("git commit --ai")
        assert "### ❌ ERROR" in result
        assert "victor-devops" in result


class TestPrAndEdgeCases:
    """PR creation and edge-case handling."""

    @pytest.mark.asyncio
    async def test_pr_delegates_to_pr_callable(self):
        mock_pr = AsyncMock(return_value={"success": True, "output": "PR #42 created"})
        with patch(
            "victor.tools.unified.git_tool.resolve_vertical_callable",
            return_value=(mock_pr, "victor_devops.tools.git_tool"),
        ):
            result = await git_tool('git pr --title "Add auth" --base develop')
        mock_pr.assert_awaited_once_with(pr_title="Add auth", base_branch="develop")
        assert "PR #42 created" in result

    @pytest.mark.asyncio
    async def test_pr_without_devops_falls_back_to_gh(self):
        # Behavior change: without victor-devops, PR creation now falls back to the
        # gh CLI instead of hard-failing. (shell is mocked so no real PR is created.)
        mock_shell = AsyncMock(return_value={"success": True, "stdout": "pull/1 created"})
        with (
            patch(
                "victor.tools.unified.git_tool.resolve_vertical_callable",
                return_value=(None, None),
            ),
            patch("victor.tools.bash.shell", mock_shell),
        ):
            result = await git_tool("git pr --title T")
        assert mock_shell.call_args.kwargs["cmd"].startswith("gh pr create")
        assert "pull/1 created" in result

    @pytest.mark.asyncio
    async def test_no_subcommand_returns_help(self):
        result = await git_tool("git")
        assert "### ❌ ERROR" in result
        assert "usage" in result.lower()

    @pytest.mark.asyncio
    async def test_unknown_subcommand_returns_error(self):
        result = await git_tool("git bogus")
        assert "### ❌ ERROR" in result

    @pytest.mark.asyncio
    async def test_tool_registered_with_canonical_name(self):
        assert git_tool.Tool.name == "git"


class TestPorcelainFlags:
    """P3: common porcelain flags accepted; rejections teach the shell fallback.

    Measured (13-day telemetry): git had a 42.9% error rate dominated by
    ``--oneline``/``--stat``/``--show-current`` and unsupported subcommands
    (``fetch``, ``worktree``).
    """

    # -- parser acceptance --

    def test_log_oneline_flag_parses(self):
        ns = _parse("git log --oneline -n 5")
        assert ns.subcommand == "log"
        assert ns.oneline is True
        assert ns.limit == 5

    def test_log_stat_flag_parses(self):
        assert _parse("git log --stat").stat is True

    def test_diff_stat_flag_parses(self):
        assert _parse("git diff --stat").stat is True

    def test_diff_cached_is_alias_of_staged(self):
        assert _parse("git diff --cached").staged is True
        assert _parse("git diff --staged").staged is True

    def test_status_short_flag_absorbed(self):
        assert _parse("git status --short").short is True
        assert _parse("git status -s").short is True

    # -- fallback command construction --

    def test_fallback_log_stat_replaces_oneline(self):
        sub, argv, ro = _fallback_command(_parse("git log --stat -n 3"))
        assert sub == "log" and ro is True
        assert "--stat" in argv
        assert "--oneline" not in argv
        assert "-3" in argv

    def test_fallback_diff_cached_stat_combo(self):
        # The exact live-failure combo from the telemetry.
        sub, argv, ro = _fallback_command(_parse("git diff --cached --stat"))
        assert sub == "diff" and ro is True
        assert argv == ["--staged", "--stat"]

    def test_fallback_branch_show_current(self):
        sub, argv, ro = _fallback_command(_parse("git branch --show-current"))
        assert sub == "branch"
        assert argv == ["--show-current"]
        assert ro is True

    # -- routing: format flags bypass the devops delegate --

    @pytest.mark.asyncio
    async def test_format_flags_bypass_devops_delegate(self):
        mock_resolve = MagicMock()
        mock_shell = AsyncMock(return_value={"success": True, "stdout": "2 files changed"})
        with (
            patch("victor.tools.unified.git_tool.resolve_vertical_callable", mock_resolve),
            patch("victor.tools.bash.shell", mock_shell),
        ):
            result = await git_tool("git log --stat")
        mock_resolve.assert_not_called()
        called_cmd = mock_shell.call_args.kwargs["cmd"]
        assert "--stat" in called_cmd
        assert "2 files changed" in result

    # -- rejection hints teach the shell fallback --

    @pytest.mark.asyncio
    async def test_fetch_rejection_teaches_shell_fallback(self):
        result = await git_tool("git fetch origin")
        assert "### ❌ ERROR" in result
        assert "This tool supports:" in result
        assert "shell(cmd='git ...', action='exec')" in result

    @pytest.mark.asyncio
    async def test_worktree_and_cd_rejection_teaches_shell_fallback(self):
        for cmd in ("git worktree add ../wt", "git cd .."):
            result = await git_tool(cmd)
            assert "### ❌ ERROR" in result
            assert "shell(cmd='git ...', action='exec')" in result

    @pytest.mark.asyncio
    async def test_unknown_flag_rejection_teaches_shell_fallback(self):
        result = await git_tool("git log --graph")
        assert "### ❌ ERROR" in result
        assert "This tool supports:" in result
        assert "shell(cmd='git ...', action='exec')" in result


class TestPushSetUpstream:
    """`git push -u` (set-upstream) — the standard first push of a new branch.

    Live-failure: ``git push -u origin <branch>`` errored with "unrecognized
    arguments: -u", forcing a raw shell fallback. ``-u``/``--set-upstream`` is
    now a first-class push flag on both the devops and shell paths.
    """

    def test_push_u_short_flag_parses(self):
        ns = _parse("git push -u origin feat/x")
        assert ns.subcommand == "push"
        assert ns.set_upstream is True
        assert ns.remote == "origin"
        assert ns.branch == "feat/x"

    def test_push_set_upstream_long_flag_parses(self):
        assert _parse("git push --set-upstream origin main").set_upstream is True

    def test_push_without_u_defaults_false(self):
        assert _parse("git push origin main").set_upstream is False

    def test_kwarg_map_includes_set_upstream(self):
        op, kwargs = _git_kwarg_map(_parse("git push -u origin feat/x"))
        assert op == "push"
        assert kwargs["options"]["set_upstream"] is True
        assert kwargs["options"]["remote"] == "origin"

    def test_fallback_push_emits_u_flag(self):
        sub, argv, ro = _fallback_command(_parse("git push -u origin feat/x"))
        assert sub == "push" and ro is False
        assert argv == ["-u", "origin", "feat/x"]

    def test_fallback_push_u_with_force_order(self):
        sub, argv, _ = _fallback_command(_parse("git push -u origin feat/x --force"))
        assert argv[0] == "-u"
        assert "--force-with-lease" in argv

    @pytest.mark.asyncio
    async def test_push_u_shell_fallback_end_to_end(self):
        mock_shell = AsyncMock(return_value={"success": True, "stdout": "branch pushed"})
        with (
            patch(
                "victor.tools.unified.git_tool.resolve_vertical_callable",
                return_value=(None, None),
            ),
            patch("victor.tools.bash.shell", mock_shell),
        ):
            result = await git_tool("git push -u origin feat/x")
        called = mock_shell.call_args.kwargs["cmd"]
        assert "git push -u origin feat/x" in called
        assert "branch pushed" in result


class TestPushUpstreamHint:
    """A plain push that fails for lack of an upstream teaches ``push -u``."""

    def test_hint_appended_on_upstream_error(self):
        out = _augment_push_hint(
            "### ❌ ERROR\nfatal: The current branch feat/x has no upstream branch.",
            _parse("git push"),
        )
        assert "### 💡 HINT" in out
        assert "git push -u origin" in out

    def test_no_hint_on_success(self):
        out = _augment_push_hint("Everything up-to-date", _parse("git push"))
        assert "HINT" not in out

    def test_no_hint_when_u_already_used(self):
        out = _augment_push_hint(
            "### ❌ ERROR\nfatal: ... no upstream ...", _parse("git push -u origin feat/x")
        )
        assert "HINT" not in out


class TestPrGhFallback:
    """PR creation falls back to ``gh pr create`` when victor-devops is absent."""

    def test_pr_new_flags_parse(self):
        ns = _parse('git pr --title T --head feat/x --body "b" --draft --web --fill')
        assert ns.title == "T"
        assert ns.head == "feat/x"
        assert ns.body == "b"
        assert ns.draft is True
        assert ns.web is True
        assert ns.fill is True

    def test_pr_base_defaults_to_none(self):
        # None lets gh use the repo default / gh-merge-base rather than hardcoding main.
        assert _parse("git pr --title T").base is None

    def test_build_gh_argv_autofills_body_when_absent(self):
        argv = _build_gh_pr_argv(_parse("git pr --title T --base develop"))
        assert argv[:3] == ["gh", "pr", "create"]
        assert "--title" in argv and "T" in argv
        assert "--fill" in argv  # body autofilled → non-interactive
        assert "--base" in argv and "develop" in argv

    def test_build_gh_argv_uses_explicit_body_without_fill(self):
        argv = _build_gh_pr_argv(_parse('git pr --title T --body "hello"'))
        assert "--body" in argv and "hello" in argv
        assert "--fill" not in argv

    def test_build_gh_argv_strips_needless_backtick_escapes(self):
        # argv is shlex.quoted, so `\`` was never needed for shell safety — but
        # shipped to GitHub it is a markdown escape and breaks every code span.
        argv = _build_gh_pr_argv(
            _parse(r'git pr --title "promote \`main\`" --body "runs \`cargo test\` for \$HOME"')
        )
        assert "promote `main`" in argv
        assert "runs `cargo test` for $HOME" in argv
        assert not any("\\`" in a for a in argv)

    def test_build_gh_argv_keeps_deliberate_markdown_escapes(self):
        argv = _build_gh_pr_argv(_parse(r'git pr --title T --body "a \*literal\* star"'))
        assert r"a \*literal\* star" in argv

    def test_build_gh_argv_web_skips_fill(self):
        argv = _build_gh_pr_argv(_parse("git pr --web"))
        assert "--web" in argv
        assert "--fill" not in argv

    @pytest.mark.asyncio
    async def test_pr_uses_gh_when_devops_absent(self):
        mock_shell = AsyncMock(
            return_value={"success": True, "stdout": "https://github.com/o/r/pull/7"}
        )
        with (
            patch(
                "victor.tools.unified.git_tool.resolve_vertical_callable",
                return_value=(None, None),
            ),
            patch("victor.tools.bash.shell", mock_shell),
        ):
            result = await git_tool('git pr --title "Add x" --base develop')
        called = mock_shell.call_args.kwargs["cmd"]
        assert called.startswith("gh pr create")
        assert "--title" in called
        assert "pull/7" in result

    @pytest.mark.asyncio
    async def test_pr_gh_missing_teaches_install(self):
        mock_shell = AsyncMock(return_value={"success": False, "stderr": "gh: command not found"})
        with (
            patch(
                "victor.tools.unified.git_tool.resolve_vertical_callable",
                return_value=(None, None),
            ),
            patch("victor.tools.bash.shell", mock_shell),
        ):
            result = await git_tool("git pr --title T")
        assert "### ❌ ERROR" in result
        assert "gh auth login" in result

    @pytest.mark.asyncio
    async def test_pr_still_prefers_devops_when_present(self):
        mock_pr = AsyncMock(return_value={"success": True, "output": "PR #9 created"})
        with patch(
            "victor.tools.unified.git_tool.resolve_vertical_callable",
            return_value=(mock_pr, "victor_devops.tools.git_tool"),
        ):
            result = await git_tool('git pr --title "Add auth" --base develop')
        mock_pr.assert_awaited_once_with(pr_title="Add auth", base_branch="develop")
        assert "PR #9 created" in result
