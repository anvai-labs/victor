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

"""Unified ``gh`` command tool — GitHub CLI surface (bash-style).

The GitHub CLI is a separate binary from ``git`` and need not be installed
when ``git`` is. This tool owns that dependency honestly: it checks for
``gh`` availability first and returns an actionable install hint on missing
binary instead of surfacing a confusing shell error mid-pipeline.

``gh`` operations are network/mutating by nature (creating PRs, merging,
commenting). Every subcommand in this tool runs through
``victor.tools.bash.shell`` with ``action='exec'`` so intent classification
stays truthful — a read-only GitHub query still opens a network socket, so
pinning it to ``readonly`` would be a lie.

Example commands:
    gh pr list --base develop
    gh pr view 1051
    gh pr create --title "Fix X" --base develop
    gh run list --limit 5
"""

from __future__ import annotations

import argparse
import shutil
import sys
from typing import List

from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.bash import shell
from victor.tools.decorators import tool
from victor.tools.unified.parser import split_command


class UnifiedGhParser(argparse.ArgumentParser):
    def error(self, message):  # type: ignore[override]
        self.print_usage(sys.stderr)
        raise ValueError(f"Argument parsing error: {message}")


def create_gh_parser() -> UnifiedGhParser:
    parser = UnifiedGhParser(
        prog="gh", description="GitHub CLI operations.", exit_on_error=False
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="The operation to perform")

    pr = subparsers.add_parser("pr", help="Pull requests")
    pr.add_argument("--state", default="open", help="open|closed|merged|all")
    pr.add_argument("--base", default=None, help="Filter by base branch")
    pr.add_argument("--head", default=None, help="Filter by head branch")
    pr.add_argument("--limit", type=int, default=None, help="Max results")
    pr.add_argument("--json", default=None, help="Comma-separated fields for JSON output")
    pr.add_argument("--title", default=None, help="Title (create)")
    pr.add_argument("--body", default=None, help="Body (create)")
    pr.add_argument("--draft", action="store_true", help="Create as draft")
    pr.add_argument("--web", action="store_true", help="Open in browser")
    pr.add_argument("--fill", action="store_true", help="Use commit info for title/body")
    pr.add_argument("--reviewer", default=None, help="Request reviewers (create)")
    pr.add_argument("args", nargs="*", help="Subcommand: list|view|create|merge|close")

    issue = subparsers.add_parser("issue", help="Issues")
    issue.add_argument("--state", default="open", help="open|closed|all")
    issue.add_argument("--limit", type=int, default=None, help="Max results")
    issue.add_argument("--json", default=None, help="Comma-separated fields for JSON output")
    issue.add_argument("--title", default=None, help="Title (create)")
    issue.add_argument("--body", default=None, help="Body (create)")
    issue.add_argument("--web", action="store_true", help="Open in browser")
    issue.add_argument("args", nargs="*", help="Subcommand: list|view|create|close")

    run = subparsers.add_parser("run", help="Actions runs")
    run.add_argument("--limit", type=int, default=None, help="Max results")
    run.add_argument("--json", default=None, help="Comma-separated fields for JSON output")
    run.add_argument("--status", default=None, help="Filter by status")
    run.add_argument("--branch", default=None, help="Filter by branch")
    run.add_argument("args", nargs="*", help="Subcommand: list|view|watch")

    release = subparsers.add_parser("release", help="Releases")
    release.add_argument("--limit", type=int, default=None, help="Max results")
    release.add_argument("args", nargs="*", help="Subcommand: list|view|create|download")

    repo = subparsers.add_parser("repo", help="Repository")
    repo.add_argument("--json", default=None, help="Comma-separated fields for JSON output")
    repo.add_argument("args", nargs="*", help="Subcommand: list|view|clone|fork")

    api = subparsers.add_parser("api", help="Raw GitHub REST/GraphQL API call")
    api.add_argument("endpoint", nargs="*", help="API endpoint, e.g. /repos/{owner}/{repo}")
    api.add_argument("--method", default=None, help="HTTP method")
    api.add_argument("--field", action="append", help="KV params (repeatable)")
    api.add_argument("--input", default=None, help="Read body from file")
    api.add_argument("--jq", default=None, help="jq expression")

    auth = subparsers.add_parser("auth", help="Authentication")
    auth.add_argument("args", nargs="*", help="Subcommand: status|login|logout|token")

    return parser


_GH_INSTALL_HINT = (
    "GitHub CLI (`gh`) is not installed. Install it, then authenticate:\n"
    "  brew install gh && gh auth login   # macOS\n"
    "  sudo apt install gh && gh auth login   # Debian/Ubuntu\n"
    "  https://cli.github.com/ for other platforms.\n"
    "Then retry this command."
)


async def _run_gh(argv: List[str]) -> str:
    """Run ``gh <argv>`` through the shell tool with exec intent."""
    result = await shell(cmd=f"gh {' '.join(argv)}", action="exec")
    if isinstance(result, dict):
        rc = result.get("return_code")
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        if rc == 0:
            return stdout.strip() or "(no output)"
        return (
            f"### ❌ ERROR\ngh returned exit {rc}:\n{stderr.strip() or stdout.strip()}"
        )
    return str(result)


async def gh_tool(cmd: str) -> str:
    """GitHub CLI (bash-style): pr · issue · run · release · repo · api · auth.

    Runs the real `gh` binary via shell(action='exec'). `gh` is NOT `git` — it
    is a separate install. If missing, the tool returns an install hint rather
    than a confusing shell error. All gh subcommands not shown here can be run
    via shell(cmd='gh <subcommand>', action='exec').

    Backticks in --body are NOT escaped (unlike the old git-tool `pr` path):
    pass markdown verbatim.

    Args:
        cmd: Bash-style command string, e.g. "gh pr list --base develop".
    """
    if shutil.which("gh") is None:
        return f"### ❌ ERROR\n{_GH_INSTALL_HINT}"

    parser = create_gh_parser()
    try:
        args_list = split_command(cmd)
        if args_list and args_list[0] == "gh":
            args_list = args_list[1:]
        parsed = parser.parse_args(args_list)
    except ValueError as e:
        return f"### ❌ ERROR\n{e}"
    except Exception as e:
        return f"### ❌ ERROR\nUnexpected error parsing command: {e}"

    if not parsed.subcommand:
        return (
            "### ❌ ERROR\nNo gh subcommand given. Use: "
            "gh pr|issue|run|release|repo|api|auth"
        )

    # Rebuild the gh argv from the original tokens: passthrough keeps flags we
    # do not model in the parser (gh's surface is large) while the parser still
    # validates structure and rejects unknown top-level subcommands.
    tokens = split_command(cmd)
    if tokens and tokens[0] == "gh":
        tokens = tokens[1:]
    return await _run_gh(tokens)


@tool(
    name="gh",
    category="github",
    access_mode=AccessMode.MIXED,
    danger_level=DangerLevel.MEDIUM,
    execution_category=ExecutionCategory.MIXED,
    priority=Priority.LOW,
    keywords=[
        "github",
        "gh",
        "pull request",
        "pr",
        "issue",
        "actions",
        "release",
        "cicd",
        "review",
    ],
    task_types=["action", "collaboration", "release"],
)
async def gh_tool_registered(cmd: str) -> str:
    """GitHub CLI (bash-style): pr · issue · run · release · repo · api · auth.

    Runs the real `gh` binary via shell(action='exec'). `gh` is NOT `git` — it
    is a separate install. If missing, the tool returns an install hint rather
    than a confusing shell error. All gh subcommands not shown here can be run
    via shell(cmd='gh <subcommand>', action='exec').

    Backticks in --body are NOT escaped (unlike the old git-tool `pr` path):
    pass markdown verbatim.
    """
    return await gh_tool(cmd)


__all__ = ["gh_tool", "gh_tool_registered", "create_gh_parser"]
