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

"""Unified ``git`` command tool with bash-like ``domain action args`` syntax.

The ``git`` tool is the canonical LLM-facing surface for version control. It
parses a bash-style command (``git status``, ``git commit -m "..."``) and
**delegates** to the richer ``victor_devops`` git implementation when that
package is importable (AI commit messages, conflict analysis, PR creation),
falling back to a shell-driven ``git`` invocation otherwise. Either way the
LLM sees a single ``git`` tool, which resolves the "shell used for git"
problem by giving git a first-class domain.

Example commands:
    git status
    git diff --staged
    git log -n 20
    git stage src/app.py tests/test_app.py
    git commit -m "fix: handle empty input"
    git commit --ai
    git branch feature/auth
    git push origin main
    git push -u origin feature/auth
    git conflicts
    git commit_msg
    git pr --title "Add auth" --base develop
"""

from __future__ import annotations

import argparse
import shlex
import sys
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.decorators import tool
from victor.tools.unified._vertical_resolver import resolve_vertical_callable
from victor.tools.unified.parser import split_command

# Subcommands that only read repo state (safe for readonly shell fallback).
_READ_ONLY_SUBCOMMANDS = {"status", "diff", "log", "conflicts"}


class UnifiedGitParser(argparse.ArgumentParser):
    """Custom parser that raises instead of exiting on error."""

    def error(self, message):  # type: ignore[override]
        self.print_usage(sys.stderr)
        raise ValueError(f"Argument parsing error: {message}")


def create_git_parser() -> UnifiedGitParser:
    """Create the parser for the git tool."""
    parser = UnifiedGitParser(
        prog="git",
        description="Unified git version-control operations.",
        exit_on_error=False,
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="The operation to perform")

    status_parser = subparsers.add_parser("status", help="Show working tree status")
    status_parser.add_argument(
        "--short", "-s", action="store_true", help="(absorbed) short format is always used"
    )
    status_parser.add_argument("files", nargs="*", help="(ignored) path filter")

    diff_parser = subparsers.add_parser("diff", help="Show changes")
    diff_parser.add_argument(
        "--staged", "--cached", action="store_true", dest="staged", help="Show staged changes"
    )
    diff_parser.add_argument("--stat", action="store_true", help="Show diffstat instead of patch")
    diff_parser.add_argument("files", nargs="*", help="Paths to diff")

    log_parser = subparsers.add_parser("log", help="Show commit history")
    log_parser.add_argument("-n", "--limit", type=int, default=10, help="Number of commits")
    log_parser.add_argument(
        "--oneline", action="store_true", help="(absorbed) one line per commit is the default"
    )
    log_parser.add_argument("--stat", action="store_true", help="Show diffstat per commit")
    log_parser.add_argument("files", nargs="*", help="(ignored) path filter")

    stage_parser = subparsers.add_parser("stage", help="Stage files (or all if none given)")
    stage_parser.add_argument("files", nargs="*", help="Paths to stage")
    add_parser = subparsers.add_parser("add", help="Alias of 'stage'")
    add_parser.add_argument("files", nargs="*", help="Paths to stage")

    commit_parser = subparsers.add_parser("commit", help="Commit staged changes")
    commit_parser.add_argument("-m", "--message", help="Commit message")
    commit_parser.add_argument(
        "--ai", action="store_true", help="Generate the commit message with AI"
    )
    commit_parser.add_argument(
        "--author-name",
        dest="author_name",
        default=None,
        help="Override commit author name",
    )
    commit_parser.add_argument(
        "--author-email",
        dest="author_email",
        default=None,
        help="Override commit author email",
    )

    subparsers.add_parser("commit_msg", help="Generate an AI commit message from the staged diff")

    branch_parser = subparsers.add_parser("branch", help="List branches or switch to one")
    branch_parser.add_argument("name", nargs="?", default=None, help="Branch to create/switch to")
    branch_parser.add_argument(
        "--show-current",
        action="store_true",
        dest="show_current",
        help="Print only the current branch name",
    )

    push_parser = subparsers.add_parser("push", help="Push commits to remote")
    push_parser.add_argument("remote", nargs="?", default=None, help="Remote name (default origin)")
    push_parser.add_argument("branch", nargs="?", default=None, help="Branch to push")
    push_parser.add_argument(
        "-u",
        "--set-upstream",
        action="store_true",
        dest="set_upstream",
        help="Set the pushed branch as its upstream (git push -u) — the standard "
        "first push for a new branch",
    )
    push_parser.add_argument("--force", action="store_true", help="Use --force-with-lease")
    push_parser.add_argument("--tags", action="store_true", help="Push tags")
    push_parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="Dry run")

    subparsers.add_parser("conflicts", help="Analyze merge conflicts")

    pr_parser = subparsers.add_parser("pr", help="Create a pull request (gh CLI)")
    pr_parser.add_argument("--title", default=None, help="PR title")
    pr_parser.add_argument(
        "--base",
        default=None,
        help="Base branch (default: repo default / gh-merge-base)",
    )
    pr_parser.add_argument("--head", default=None, help="Head branch (default: current)")
    pr_parser.add_argument("--body", default=None, help="PR body text")
    pr_parser.add_argument("--draft", action="store_true", help="Create the PR as a draft")
    pr_parser.add_argument(
        "--web", action="store_true", help="Open the PR create page in a browser"
    )
    pr_parser.add_argument(
        "--fill", action="store_true", help="Autofill title/body from commit messages"
    )

    return parser


def _format_result(result: Any) -> str:
    """Normalize a git operation result (dict or str) into a display string."""
    if isinstance(result, dict):
        if result.get("success") is False:
            return f"### ❌ ERROR\n{result.get('error') or result.get('output') or 'git operation failed'}"
        output = result.get("output")
        if output is None:
            # Surface a compact key=value view for atypical result dicts.
            return (
                "\n".join(f"**{k}**: {v}" for k, v in result.items() if k != "success") or "Done."
            )
        return str(output)
    return str(result)


async def _shell_git(subcommand: str, args: List[str], *, readonly: bool) -> str:
    """Fallback: run a plain ``git`` command through the production shell surface."""
    from victor.tools.bash import shell

    cmd_parts = ["git", subcommand, *args]
    result = await shell(cmd=" ".join(shlex.quote(p) for p in cmd_parts), readonly=readonly)
    if isinstance(result, dict):
        stdout = (result.get("stdout") or result.get("output") or "").strip()
        stderr = (result.get("stderr") or result.get("error") or "").strip()
        if result.get("success") is False:
            return f"### ❌ ERROR\n{stderr or stdout or 'git command failed'}"
        return stdout or stderr or "Done."
    return str(result)


def _git_kwarg_map(args: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    """Map parsed bash subcommand args to ``victor_devops.git(operation=...)`` kwargs."""
    sub = args.subcommand
    if sub == "add":  # normalize alias
        sub = "stage"

    if sub == "status":
        return "status", {}
    if sub == "diff":
        return "diff", {"staged": args.staged, "files": args.files or None}
    if sub == "log":
        return "log", {"limit": args.limit}
    if sub == "stage":
        return "stage", {"files": args.files or None}
    if sub == "commit":
        return "commit", {
            "message": args.message,
            "author_name": args.author_name,
            "author_email": args.author_email,
        }
    if sub == "commit_msg":
        return "commit_msg", {}
    if sub == "branch":
        return "branch", {"branch": args.name}
    if sub == "push":
        opts: Dict[str, Any] = {}
        if args.remote:
            opts["remote"] = args.remote
        if getattr(args, "set_upstream", False):
            opts["set_upstream"] = True
        if args.force:
            opts["force"] = True
        if args.tags:
            opts["tags"] = True
        if args.dry_run:
            opts["dry_run"] = True
        return "push", {"branch": args.branch, "options": opts or None}
    if sub == "conflicts":
        return "conflicts", {}
    raise ValueError(f"No devops mapping for git subcommand '{sub}'")


def _fallback_command(args: argparse.Namespace) -> Tuple[str, List[str], bool]:
    """Build (subcommand, argv, readonly) for the shell fallback path."""
    sub = args.subcommand
    if sub == "add":
        sub = "stage"

    if sub == "status":
        return "status", ["--short", "--branch"], True
    if sub == "diff":
        cmd_args = ["--staged"] if getattr(args, "staged", False) else []
        if getattr(args, "stat", False):
            cmd_args.append("--stat")
        return "diff", [*cmd_args, *(args.files or [])], True
    if sub == "log":
        fmt = "--stat" if getattr(args, "stat", False) else "--oneline"
        return "log", [f"-{args.limit}", fmt], True
    if sub == "stage":
        return "add", list(args.files) if args.files else ["."], False
    if sub == "commit":
        if not args.message:
            raise ValueError("commit message required (-m) for the shell fallback")
        return "commit", ["-m", args.message], False
    if sub == "branch":
        if getattr(args, "show_current", False):
            return "branch", ["--show-current"], True
        if args.name:
            return "checkout", [args.name], False
        return "branch", ["-a"], True
    if sub == "push":
        cmd_args = ["-u"] if getattr(args, "set_upstream", False) else []
        if getattr(args, "force", False):
            cmd_args.append("--force-with-lease")
        if getattr(args, "dry_run", False):
            cmd_args.append("--dry-run")
        remote = args.remote or "origin"
        cmd_args.append(remote)
        if args.branch:
            cmd_args.append(args.branch)
        return "push", cmd_args, False
    if sub == "conflicts":
        return "status", [], True  # best-effort read
    # commit_msg and pr are AI/gh-only — no plain-git equivalent.
    raise ValueError(
        f"git '{args.subcommand}' requires the victor-devops package, which is not installed."
    )


@tool(
    name="git",
    category="git",
    access_mode=AccessMode.MIXED,
    danger_level=DangerLevel.MEDIUM,
    execution_category=ExecutionCategory.MIXED,
    priority=Priority.HIGH,
    keywords=[
        "git",
        "commit",
        "push",
        "branch",
        "diff",
        "status",
        "log",
        "merge",
        "version control",
    ],
    task_types=["action", "analysis"],
)
async def git_tool(cmd: str) -> str:
    """Git tool (bash-style). Subcommands: status [--short] · diff [--staged|--cached] [--stat] [paths]
    · log [-n N] [--oneline] [--stat] · stage/add [paths] · commit -m "msg" | --ai · commit_msg
    · branch [name | --show-current] · push [remote] [branch] [-u/--set-upstream] [--force]
    [--tags] [--dry-run] · conflicts · pr --title "t" [--base b] [--head h] [--body ...]
    [--draft] [--web] [--fill].
    Delegates to victor-devops git when installed; falls back to shell git (and to the
    `gh` CLI for `pr`), so push -u and PR creation work without victor-devops.
    Anything else (fetch, pull, rebase, stash, worktree, ...): use shell(cmd='git ...', action='exec').
    """
    parser = create_git_parser()

    try:
        args_list = split_command(cmd)
        if args_list and args_list[0] == "git":
            args_list = args_list[1:]
        parsed = parser.parse_args(args_list)
    except (ValueError, argparse.ArgumentError) as e:
        # 42.9% of git tool calls errored (13-day telemetry) — mostly porcelain
        # flags and subcommands outside this surface. Teach the escape hatch.
        return (
            "### ❌ ERROR\n"
            f"{e}\n"
            "\n"
            "This tool supports: status [--short], diff [--staged|--cached] [--stat], "
            "log [-n N] [--oneline] [--stat], stage/add, commit (-m | --ai), commit_msg, "
            "branch [name | --show-current], push [-u] [--force] [--tags] [--dry-run], "
            "conflicts, pr [--title] [--base] [--body] [--draft] [--web].\n"
            "For anything else use shell(cmd='git ...', action='exec')."
        )
    except Exception as e:
        return f"### ❌ ERROR\nUnexpected error parsing command: {e}"

    if not parsed.subcommand:
        old_stdout = sys.stdout
        sys.stdout = capture = StringIO()
        parser.print_help()
        sys.stdout = old_stdout
        return f"### ❌ ERROR\nNo git subcommand given.\n\n```text\n{capture.getvalue()}```"

    # PR is a separate vertical callable (gh-based); handle before git() mapping.
    if parsed.subcommand == "pr":
        return await _handle_pr(parsed)

    # Agent-layer attribution: proactively append the Victor AI co-author trailer to
    # an *explicit* commit message (``-m``), Claude-Code-style — injected into the
    # message itself so it applies in any repo without a git hook. The ``--ai`` path
    # is attributed inside ``_handle_ai_commit``. Idempotent; opt out with
    # ``VICTOR_COMMIT_ATTRIBUTION=0``.
    if parsed.subcommand == "commit" and not getattr(parsed, "ai", False):
        parsed.message = _attributed_commit_message(parsed.message)

    # AI commit: generate message first, then commit.
    if parsed.subcommand == "commit" and getattr(parsed, "ai", False):
        return await _handle_ai_commit(parsed)

    # Format flags (--stat, --show-current) have no devops kwarg mapping — route
    # straight to plain git so the requested output format actually takes effect.
    if getattr(parsed, "stat", False) or getattr(parsed, "show_current", False):
        try:
            subcommand, argv, readonly = _fallback_command(parsed)
            return await _shell_git(subcommand, argv, readonly=readonly)
        except Exception as e:
            return f"### ❌ ERROR\ngit {parsed.subcommand} failed: {e}"

    git_fn, _src = resolve_vertical_callable(
        "git", fallback_module="victor_devops.tools.git_tool", fallback_attr="git"
    )
    if git_fn is not None:
        try:
            operation, kwargs = _git_kwarg_map(parsed)
            result = await git_fn(operation=operation, **kwargs)
            out = _format_result(result)
            if parsed.subcommand == "push":
                out = _augment_push_hint(out, parsed)
            return out
        except Exception as e:
            return f"### ❌ ERROR\ngit {parsed.subcommand} failed: {e}"

    # Fallback: plain git via shell.
    try:
        subcommand, argv, readonly = _fallback_command(parsed)
    except ValueError as e:
        return f"### ❌ ERROR\n{e}"
    try:
        out = await _shell_git(subcommand, argv, readonly=readonly)
        if parsed.subcommand == "push":
            out = _augment_push_hint(out, parsed)
        return out
    except Exception as e:
        return f"### ❌ ERROR\ngit {parsed.subcommand} failed: {e}"


def _augment_push_hint(output: str, parsed: argparse.Namespace) -> str:
    """Append a corrective hint when a push failed for lack of an upstream.

    A plain ``git push`` on a fresh branch fails with "no upstream". Rather than
    leave the agent to rediscover the fix, surface the exact ``push -u`` command.
    No-op on success, and skipped when ``-u`` was already used.
    """
    if not isinstance(output, str) or "❌" not in output:
        return output
    if getattr(parsed, "set_upstream", False):
        return output
    low = output.lower()
    if "upstream" in low or "set-upstream" in low or "no configured push destination" in low:
        remote = parsed.remote or "origin"
        branch = parsed.branch or "<branch>"
        return (
            f"{output}\n\n### 💡 HINT\nThis branch has no upstream. Set one while pushing:\n"
            f"`git push -u {remote} {branch}`"
        )
    return output


async def _handle_ai_commit(parsed: argparse.Namespace) -> str:
    """Generate a commit message via commit_msg, then commit it."""
    git_fn, _src = resolve_vertical_callable(
        "git", fallback_module="victor_devops.tools.git_tool", fallback_attr="git"
    )
    if git_fn is None:
        return (
            "### ❌ ERROR\nAI commit message generation requires the victor-devops package. "
            'Provide an explicit message with `git commit -m "..."` instead.'
        )
    try:
        msg_result = await git_fn(operation="commit_msg")
        message = _extract_generated_message(msg_result)
        if not message:
            return (
                f"### ❌ ERROR\nCould not generate a commit message.\n{_format_result(msg_result)}"
            )
        # Agent-layer attribution on the AI-generated message too (hook-free).
        message = _attributed_commit_message(message)
        commit_result = await git_fn(
            operation="commit",
            message=message,
            author_name=getattr(parsed, "author_name", None),
            author_email=getattr(parsed, "author_email", None),
        )
        prefix = f"### 💡 SYSTEM HINT\nGenerated commit message:\n```\n{message}\n```\n\n"
        return prefix + _format_result(commit_result)
    except Exception as e:
        return f"### ❌ ERROR\nAI commit failed: {e}"


def _extract_generated_message(result: Any) -> Optional[str]:
    """Best-effort extraction of a commit message from a commit_msg result."""
    if isinstance(result, dict):
        for key in ("message", "commit_message", "output", "suggestion"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(result, str) and result.strip():
        return result.strip()
    return None


def _attributed_commit_message(message: Optional[str]) -> Optional[str]:
    """Proactively append the Victor AI co-author trailer to a commit message.

    Claude-Code-style agent-layer attribution: injected into the message text
    itself (not a git hook), so it applies in any repo without setup. Idempotent;
    opt out by setting ``VICTOR_COMMIT_ATTRIBUTION=0``.
    """
    import os

    if not message:
        return message
    disabled = os.getenv("VICTOR_COMMIT_ATTRIBUTION", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }
    if disabled:
        return message
    from victor.core.attribution import append_victor_commit_attribution

    return append_victor_commit_attribution(message)


async def _handle_pr(parsed: argparse.Namespace) -> str:
    """Create a pull request.

    Prefers the richer ``victor-devops`` ``pr`` callable when installed; otherwise
    falls back to the GitHub CLI (``gh pr create``) directly. The ``gh`` fallback
    means PR creation works out-of-the-box on any machine with an authenticated
    ``gh`` — no ``victor-devops`` required — instead of hard-failing.
    """
    pr_fn, _src = resolve_vertical_callable(
        "pr", fallback_module="victor_devops.tools.git_tool", fallback_attr="pr"
    )
    if pr_fn is not None:
        try:
            result = await pr_fn(pr_title=parsed.title, base_branch=parsed.base or "main")
            return _format_result(result)
        except Exception as e:
            return f"### ❌ ERROR\nPR creation failed: {e}"

    # Fallback: create the PR via the GitHub CLI (no victor-devops needed).
    return await _gh_pr_create(parsed)


def _unescape_shell_metachars(text: str) -> str:
    r"""Drop backslash escapes a caller added for shell safety they don't need.

    The argv is ``shlex.quote``d before it reaches the shell, so backticks and
    ``$`` inside a title/body are already inert. A caller that escapes them
    anyway — models routinely emit ``\`main\``` out of caution — ships those
    backslashes to GitHub verbatim, where they are markdown escapes and render
    as literal backslashes, so every inline code span in the PR body breaks.
    Neither ``\```` nor ``\$`` is meaningful markdown, so undoing them is safe;
    every other escape (``\*``, ``\_``, ``\\``) is left alone because it may be
    deliberate.
    """
    return text.replace("\\`", "`").replace("\\$", "$")


def _build_gh_pr_argv(parsed: argparse.Namespace) -> List[str]:
    """Build the ``gh pr create`` argv from parsed PR options.

    ``gh`` needs a non-interactive body, so ``--fill`` is added whenever a body
    is not explicitly supplied (and the browser flow is not requested); ``gh``
    lets an explicit ``--title`` take precedence over autofilled content.
    """
    argv: List[str] = ["gh", "pr", "create"]
    if parsed.title:
        argv += ["--title", _unescape_shell_metachars(parsed.title)]
    if parsed.body is not None:
        argv += ["--body", _unescape_shell_metachars(parsed.body)]
    if getattr(parsed, "web", False):
        argv.append("--web")
    elif getattr(parsed, "fill", False) or parsed.body is None:
        # Autofill the body from commits so the command is non-interactive.
        argv.append("--fill")
    if parsed.base:
        argv += ["--base", parsed.base]
    if getattr(parsed, "head", None):
        argv += ["--head", parsed.head]
    if getattr(parsed, "draft", False):
        argv.append("--draft")
    return argv


async def _gh_pr_create(parsed: argparse.Namespace) -> str:
    """Create a PR through the ``gh`` CLI via the shell surface."""
    from victor.tools.bash import shell

    argv = _build_gh_pr_argv(parsed)
    cmd = " ".join(shlex.quote(p) for p in argv)
    try:
        result = await shell(cmd=cmd, readonly=False)
    except Exception as e:
        return f"### ❌ ERROR\ngh pr create failed: {e}"

    if isinstance(result, dict):
        stdout = (result.get("stdout") or result.get("output") or "").strip()
        stderr = (result.get("stderr") or result.get("error") or "").strip()
        if result.get("success") is False:
            blob = f"{stderr}\n{stdout}".lower()
            if "not found" in blob or "no such file" in blob or "command not found" in blob:
                return (
                    "### ❌ ERROR\nPR creation needs either the victor-devops package or the "
                    "GitHub CLI (`gh`). Install one — e.g. `brew install gh && gh auth login`."
                )
            return f"### ❌ ERROR\ngh pr create failed: {stderr or stdout or 'unknown error'}"
        return stdout or stderr or "Pull request created."
    return str(result)


__all__ = ["git_tool", "create_git_parser"]
