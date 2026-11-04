from __future__ import annotations

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

"""Bash command execution tool with readonly mode support."""

import contextvars
import logging
import os
import platform
import re
import shlex
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

from victor.config.timeouts import ProcessTimeouts
from victor.security.shell_commands import (
    FIND_VALUE_TESTS,
    GIT_READONLY_SUBCOMMANDS as GIT_READONLY_SUBCOMMANDS,
    _validate_git_readonly as _validate_git_readonly,
    _validate_sed_readonly as _validate_sed_readonly,
    PURE_OUTPUT_COMMANDS,
    cacheable_inspection,
    dynamic_arguments_safe,
    execution_environment_assignment,
    inspection_command,
)
from victor.security.shell_syntax import (
    MAX_SHELL_NESTING,
    ShellSyntaxError,
    scan_file_writes,
    scan_substitutions,
    split_shell_commands,
    strip_shell_prefix,
    shell_words,
    shell_assignments,
)
from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.decorators import tool

# Dangerous commands that should be blocked
# Consolidated dangerous command detection — single source of truth.
from victor.security.command_safety import (
    is_dangerous_command as _is_dangerous_consolidated,
)

# FEP-0013: damage-scoped shell safety policy. The default (legacy) policy is a
# no-op here — the inline allowlist gate below runs unchanged. A non-legacy
# policy installed per-session via SessionConfig.shell_safety replaces it.
from victor.security.shell_safety_policy import (
    ShellCommandContext as _ShellSafetyCtx,
    SafetyVerdict as _SafetyVerdict,
    get_shell_safety_policy as _get_shell_safety_policy,
)

# Platform-specific readonly commands - safe for exploration/analysis
# These commands cannot modify state, only read
READONLY_COMMANDS_UNIX: Set[str] = {
    # Navigation & listing
    "pwd",
    "ls",
    "ll",
    "la",
    "cd",  # Directory navigation (read-only, doesn't modify files)
    "tree",
    "find",
    "locate",
    "which",
    "whereis",
    "type",
    # Path manipulation (safe)
    "basename",
    "dirname",
    "realpath",
    "readlink",
    "mktemp",  # Creates temp files/dirs but safe for exploration
    # File content viewing
    "cat",
    "head",
    "tail",
    "less",
    "more",
    "wc",
    "file",
    "stat",
    "md5sum",
    "sha256sum",
    "shasum",
    # Binary analysis (truly readonly - only read/analyze)
    "strings",
    "hexdump",
    "xxd",
    "od",
    # Document/media inspection. Commands with output-capable modes receive
    # additional argument validation in _validate_single_readonly_command.
    "pdfinfo",
    "pdftotext",
    "pdftocairo",
    "tesseract",
    "ffprobe",
    "ffmpeg",
    # Text search & processing (readonly)
    "grep",
    "egrep",
    "fgrep",
    "rg",
    "ag",
    "awk",
    "sed",  # Only with -n (no in-place edit) - validated separately
    "cut",
    "sort",
    "uniq",
    "diff",
    "cmp",
    "tr",
    "paste",
    "join",
    "tac",  # Reverse cat
    "nl",  # Line numbers
    # Compressed file viewing (read-only operations)
    "zcat",
    "zgrep",
    "zegrep",
    "zfgrep",
    "zless",
    "zmore",
    "gzcat",
    # Archive listing (read-only only - validated separately)
    "tar",
    "zipinfo",
    "unzip",
    # Math/expression (read-only - no file I/O)
    "expr",
    "bc",
    "test",
    "[",  # test alias
    # System info
    "uname",
    "hostname",
    "whoami",
    "id",
    "date",
    "uptime",
    "df",
    "du",
    "free",
    "top",
    "ps",
    "pgrep",
    "lsof",
    "env",
    "printenv",
    "echo",
    "printf",
    # Network info (read-only)
    "netstat",
    "ss",
    "ip",
    "nslookup",
    "dig",
    "host",
    "ping",  # Network testing (harmless)
    # Process monitoring
    "pstree",
    "htop",
    "atop",
    # Git (readonly commands - validated separately)
    "git",
    # Package info (readonly - validated separately)
    "pip",
    "pip3",
    "npm",
    "cargo",
    "go",
    "make",
    # Development (readonly - validated separately)
    "python",
    "python3",
    "pytest",  # Testing is read-heavy exploration
    "tox",
    "node",
    "yarn",
    "pnpm",
    # CLI tools (readonly - validated separately)
    "gh",
    "az",
    "kubectl",
    "helm",
    "docker",
    "podman",
    # Research tools
    "arxiv",
    "web_search",
    # Code quality (readonly - validated separately)
    "flake8",
    "pylint",
    "mypy",
    "black",
    "ruff",
    "eslint",
    # Dependency tree visualization (readonly)
    "pipdeptree",
    # Infrastructure as Code inspection (readonly)
    "terraform",
}

READONLY_COMMANDS_WINDOWS: Set[str] = {
    # Navigation & listing
    "cd",
    "dir",
    "tree",
    "where",
    "type",
    # File content viewing
    "more",
    "find",
    "findstr",
    # System info
    "hostname",
    "whoami",
    "date",
    "time",
    "ver",
    "systeminfo",
    "set",
    "echo",
    # Git (readonly)
    "git",
}

# Git subcommands that are readonly
# pip subcommands that are readonly
PIP_READONLY_SUBCOMMANDS: Set[str] = {
    "list",
    "show",
    "freeze",
    "check",
    "config",  # readonly by default
    "debug",
    "help",
    "search",
    "index",
    "wheel",  # Building wheels is usually safe (readonly source)
    "build",  # Building is usually safe
    "inspect",
}

# npm subcommands that are readonly
NPM_READONLY_SUBCOMMANDS: Set[str] = {
    "list",
    "ls",
    "view",
    "show",
    "info",
    "search",
    "help",
    "config",  # readonly by default
    "outdated",
    "audit",
    "explain",
    "pkg",
    "query",
    "version",
    "why",
}

# GitHub CLI (gh) readonly subcommands
GH_READONLY_SUBCOMMANDS: Set[str] = {
    "view",
    "list",
    "search",
    "repo",
    "issue",
    "pr",
    "release",
    "workflow",
    "run",
    "actions",
    "auth",
    "config",
    "secret",
    "variable",
    "environment",
}

# Azure CLI (az) readonly subcommands
AZ_READONLY_SUBCOMMANDS: Set[str] = {
    "list",
    "show",
    "find",
    "account",
    "config",
    "monitor",
    "log",
    "metrics",
}

# Kubernetes (kubectl) readonly subcommands
KUBECTL_READONLY_SUBCOMMANDS: Set[str] = {
    "get",
    "describe",
    "logs",
    "top",
    "api-resources",
    "api-versions",
    "cluster-info",
    "version",
    "auth",
    "certificate",
    "cp",
    "diff",
    "explain",
}


def _get_readonly_commands() -> Set[str]:
    """Get platform-specific readonly commands."""
    if platform.system() == "Windows":
        return READONLY_COMMANDS_WINDOWS
    return READONLY_COMMANDS_UNIX


def _extract_base_command(cmd: str) -> str:
    """Extract the base command from a command string."""
    try:
        parts = shlex.split(cmd.strip())
        if parts:
            return parts[0].lower()
    except ValueError:
        # Handle shlex parsing errors (unbalanced quotes, etc)
        parts = cmd.strip().split()
        if parts:
            return parts[0].lower()
    return ""


def _extract_subcommand(cmd: str, base_cmd: str) -> Optional[str]:
    """Extract subcommand for commands like git, pip, npm."""
    try:
        parts = shlex.split(cmd.strip())
        if len(parts) >= 2 and parts[0].lower() == base_cmd:
            # Skip options to find subcommand
            for part in parts[1:]:
                if not part.startswith("-"):
                    return part.lower()
    except ValueError:
        pass
    return None


def _split_compound_command(cmd: str) -> List[str]:
    """Compatibility port for the shared compound-command lexer."""
    return split_shell_commands(cmd)


def _validate_readonly_redirections(cmd: str) -> tuple[bool, str]:
    """Allow only non-file redirections and literal /dev/null suppression."""
    for write in scan_file_writes(cmd):
        if write.dynamic or write.target != "/dev/null":
            return False, f"redirection ({write.operator})"
    return True, ""


# Read-only shell builtins that aren't external binaries in the allowlist.
_SHELL_SAFE_BUILTINS = frozenset(
    {
        "read",
        "echo",
        "printf",
        "test",
        "true",
        "false",
        ":",
        "pwd",
        "shift",
        "break",
        "continue",
        "return",
        "local",
    }
)
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# Remote-exec commands that need payload/direction analysis for readonly mode.
# These are NOT in READONLY_COMMANDS_UNIX — they require special handling because
# their readonly-ness depends on what they do remotely, not the command name alone.
_REMOTE_EXEC_COMMANDS: Set[str] = {"ssh", "scp", "rsync"}

# find actions that mutate: -delete removes files, the -fprint family writes
# named files (bare -print/-ls/-print0 stay stdout-only and readonly).
_FIND_WRITE_ACTIONS: frozenset = frozenset({"-delete", "-fprint", "-fprintf", "-fprint0", "-fls"})
# find actions that execute a command per matched file. The wrapped command is
# validated recursively; {} stays in as an opaque argument, so base-command
# allowlisting carries the decision.
_FIND_EXEC_ACTIONS: frozenset = frozenset({"-exec", "-execdir", "-ok", "-okdir"})
# find tests/options that consume exactly one argument — skipping the value
# keeps a flag-shaped argument (e.g. `find . -name -delete`, matching files
# NAMED "-delete") from being misread as an action flag.
_FIND_VALUE_TESTS = FIND_VALUE_TESTS


def _validate_find_readonly(tokens: list[str]) -> tuple[bool, str]:
    """Validate a find invocation's action flags for readonly mode.

    ``-delete`` and the ``-fprint`` family are always rejected; ``-exec`` /
    ``-execdir`` / ``-ok`` / ``-okdir`` run a command per match, so that
    command (tokens up to the ``;`` / ``+`` terminator) is validated
    recursively through the full readonly pipeline.
    """
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok in _FIND_WRITE_ACTIONS:
            return False, f"find {tok}"
        if tok in _FIND_EXEC_ACTIONS:
            j = i + 1
            while j < len(tokens) and tokens[j] not in (";", "+"):
                j += 1
            wrapped = tokens[i + 1 : j]
            if wrapped:
                valid, failing = _validate_readonly_command(shlex.join(wrapped))
                if not valid:
                    return False, f"find {tok}: {failing}"
            # Continue past the terminator: multiple -exec clauses may follow.
            i = j + 1
            continue
        if tok in _FIND_VALUE_TESTS:
            i += 2  # skip the test plus its argument
            continue
        i += 1
    return True, ""


def _scan_substitutions(text: str) -> tuple[list[str], str]:
    """Compatibility port for the shared shell substitution lexer."""
    return scan_substitutions(text)


def _strip_structural_prefix(component: str) -> Optional[str]:
    """Keep executable environment changes visible to the permission gate."""
    try:
        words = shell_assignments(component)
    except ValueError:
        return component
    for word in words:
        if execution_environment_assignment(word):
            return "__victor_opaque_environment__"
    return strip_shell_prefix(component)


_READONLY_VALIDATION_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "readonly_shell_validation_depth", default=0
)


def _validate_readonly_command(cmd: str, *, _depth: int = 0) -> tuple[bool, str]:
    depth = _READONLY_VALIDATION_DEPTH.get()
    if depth > MAX_SHELL_NESTING:
        return False, "shell nesting limit exceeded"
    token = _READONLY_VALIDATION_DEPTH.set(depth + 1)
    try:
        return _validate_readonly_command_impl(cmd, _depth=_depth)
    except ShellSyntaxError as exc:
        return False, str(exc)
    finally:
        _READONLY_VALIDATION_DEPTH.reset(token)


def _validate_readonly_command_impl(cmd: str, *, _depth: int = 0) -> tuple[bool, str]:
    """Validate if a command is readonly and return (is_valid, failing_cmd).

    Shell control-flow (``for``/``while``/``if`` ... ``do``/``then``/``done``) is
    supported: the structural keywords are stripped and every *actual* command —
    loop bodies, command substitutions, and assignment RHS — is validated against
    the read-only allowlist. A read-only loop is allowed, but a mutation in a loop
    body (``rm``) or a header substitution (``$(curl ...)``) is still rejected.
    """
    # 1. Quote-aware substitution scan: extract genuine $(...) / backtick subs
    #    (validated below) and strip them so their inner spaces/operators don't
    #    confuse structural parsing. Literal backticks inside quotes are kept.
    if _depth > MAX_SHELL_NESTING:
        return False, "shell nesting limit exceeded"
    try:
        subs, sanitized = scan_substitutions(cmd, depth=_depth)
    except ShellSyntaxError as exc:
        return False, str(exc)
    for sub in subs:
        sub = sub.strip()
        if not sub:
            continue
        valid, failing = _validate_readonly_command(sub, _depth=_depth + 1)
        if not valid:
            return False, failing

    # 3. Split compound commands and validate each simple command.
    components = _split_compound_command(sanitized)
    if len(components) > 1:
        for comp in components:
            valid, failing = _validate_single_readonly_command(comp.strip())
            if not valid:
                return False, failing
        return True, ""

    return _validate_single_readonly_command(sanitized.strip())


def _validate_remote_exec_readonly(cmd: str, base_cmd: str) -> tuple[bool, str]:
    """Validate SSH payloads; file transfers write state in either direction."""
    redirect_valid, redirect_reason = _validate_readonly_redirections(cmd)
    if not redirect_valid:
        return False, redirect_reason
    if base_cmd in {"scp", "rsync"}:
        return False, f"{base_cmd} (file transfer writes state)"
    try:
        tokens = shlex.split(cmd.strip())
    except ValueError:
        return False, f"{base_cmd} (malformed arguments)"
    if not tokens:
        return False, base_cmd
    if base_cmd == "ssh":
        if not dynamic_arguments_safe(base_cmd, shell_words(cmd)):
            return False, "ssh (opaque dynamic arguments)"
        return _validate_ssh_readonly(tokens)
    return False, base_cmd


def _validate_ssh_readonly(tokens: list[str]) -> tuple[bool, str]:
    """Allow connection options with known semantics, then validate the payload.

    SSH options can run local commands, load executable configuration, write
    logs or allocate forwarding sockets independently of the remote payload.
    Unknown options therefore require the normal non-readonly execution path.
    """
    boolean_flags = frozenset("46aCknqTtvx")
    value_flags = frozenset("bBcilmop")
    safe_options = frozenset(
        {
            "addressfamily",
            "batchmode",
            "compression",
            "connectionattempts",
            "connecttimeout",
            "globalknownhostsfile",
            "hostname",
            "identitiesonly",
            "identityfile",
            "loglevel",
            "passwordauthentication",
            "port",
            "preferredauthentications",
            "pubkeyauthentication",
            "requesttty",
            "serveralivecountmax",
            "serveraliveinterval",
            "stricthostkeychecking",
            "user",
            "userknownhostsfile",
        }
    )
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--":
            i += 1
            break
        if not tok.startswith("-") or tok == "-":
            break
        j = 1
        while j < len(tok):
            flag = tok[j]
            if flag in boolean_flags:
                j += 1
                continue
            if flag not in value_flags:
                return False, f"ssh (unsupported option -{flag})"
            value = tok[j + 1 :]
            if not value:
                i += 1
                if i >= len(tokens):
                    return False, f"ssh (missing value for -{flag})"
                value = tokens[i]
            if flag == "o":
                parts = re.split(r"[=\s]+", value.strip(), maxsplit=1)
                if len(parts) != 2 or parts[0].lower() not in safe_options or not parts[1]:
                    return False, "ssh (unsupported configuration option)"
                if parts[0].lower() == "userknownhostsfile" and parts[1] != "/dev/null":
                    return False, "ssh (known-host output file)"
            # The rest of this token is a value, not another option cluster.
            break
        i += 1

    if i >= len(tokens) or i + 1 >= len(tokens):
        return False, "ssh (interactive session)"
    # SSH itself joins remote arguments with spaces and sends the result to a
    # remote shell, so shell operators inside one argument are meaningful.
    remote_cmd = " ".join(tokens[i + 1 :])
    valid, failing = _validate_readonly_command(remote_cmd)
    if not valid:
        return False, f"ssh remote: {failing}"
    return True, ""


def _validate_single_readonly_command(cmd: str) -> tuple[bool, str]:
    """Validate a single (already split, substitution-free) shell command."""
    # Strip leading control-flow keywords, loop headers, and assignments. What
    # remains is the actual command to validate (or nothing, for a structural
    # fragment like `do` / `done` / `for x in ...`).
    redirect_valid, redirect_reason = _validate_readonly_redirections(cmd)
    if not redirect_valid:
        return False, redirect_reason
    residual = _strip_structural_prefix(cmd)
    if residual is None:
        return True, ""
    cmd = residual

    readonly_commands = _get_readonly_commands()
    base_cmd = _extract_base_command(cmd)

    if not base_cmd:
        return False, cmd

    # Read-only shell builtins (read, printf, test, true, ...) aren't external
    # binaries in the allowlist; accept them past the allowlist gate but still
    # apply the redirect / pipe-to-shell checks below (so `echo x > f` is caught).
    if base_cmd not in readonly_commands and base_cmd not in _SHELL_SAFE_BUILTINS:
        # Remote-exec commands (ssh, scp, rsync) need payload/direction analysis
        # rather than a blanket reject. The readonly-ness depends on what the
        # remote command does (ssh) or the transfer direction (scp/rsync).
        if base_cmd in _REMOTE_EXEC_COMMANDS:
            return _validate_remote_exec_readonly(cmd, base_cmd)
        return False, base_cmd

    # Allow bare --version / --help / -V / -h for any command (they're always readonly)
    # This must come BEFORE subcommand handlers because _extract_subcommand
    # returns None for bare flags, which would cause false rejections.
    _bare_info = {"--version", "-v", "-V", "--help", "-h", "version", "help"}
    try:
        _tokens = shlex.split(cmd.strip())
    except ValueError:
        _tokens = cmd.strip().split()

    redirect_valid, redirect_reason = _validate_readonly_redirections(cmd)
    if not redirect_valid:
        return False, redirect_reason

    if not dynamic_arguments_safe(base_cmd, shell_words(cmd)):
        return False, f"{base_cmd} (opaque dynamic arguments)"

    # env is a generic exec wrapper: `env rm -f x` runs rm. Validate the
    # wrapped command recursively instead of trusting the env allowlist entry
    # (which exists for bare `env` / `env VAR=x <readonly-cmd>` usage).
    if base_cmd == "env":
        j = 1
        while j < len(_tokens):
            t = _tokens[j]
            if t == "--":
                j += 1
                break
            if t in {"-u", "--unset", "-C", "--chdir"}:
                if j + 1 >= len(_tokens):
                    return False, "env (missing option value)"
                j += 2
                continue
            if t.startswith(("--unset=", "--chdir=")) or (
                t.startswith(("-u", "-C")) and len(t) > 2
            ):
                j += 1
                continue
            if t in {"-i", "--ignore-environment", "-0", "--null", "-"}:
                j += 1
                continue
            if t.startswith("-"):
                # In particular -S/--split-string executes a second argument
                # language; do not confuse it with an inert wrapper flag.
                return False, "env (unsupported option)"
            if _ASSIGNMENT_RE.match(t):
                if execution_environment_assignment(t):
                    return False, "env (executable environment change)"
                j += 1
                continue
            break
        wrapped = _tokens[j:]
        if not wrapped:
            return True, ""
        wrapped_valid, wrapped_failing = _validate_readonly_command(shlex.join(wrapped))
        if not wrapped_valid:
            return False, f"env: {wrapped_failing}"
        return True, ""

    inspection = inspection_command(_tokens) if base_cmd != "find" else None
    if inspection is not None:
        return (True, "") if inspection else (False, f"{base_cmd} (execution or write operation)")

    # find's action flags can execute (-exec family) or delete/write
    # (-delete, -fprint*) even though plain find is readonly.
    if base_cmd == "find":
        find_valid, find_reason = _validate_find_readonly(_tokens)
        if not find_valid:
            return False, find_reason

    if len(_tokens) <= 2 and any(t in _bare_info for t in _tokens[1:]):
        return True, ""

    return (
        (True, "")
        if base_cmd in PURE_OUTPUT_COMMANDS
        else (False, f"{base_cmd} (unsupported inspection)")
    )


def _is_readonly_command(cmd: str) -> bool:
    """Check if command is a readonly command.

    Returns True if the command is safe for read-only operations.
    """
    valid, _ = _validate_readonly_command(cmd)
    return valid


def get_allowed_readonly_commands() -> List[str]:
    """Return list of allowed readonly commands for LLM reference."""
    commands = list(_get_readonly_commands())
    commands.sort()
    return commands


# =========================================================================
# Command Optimizer Pipeline
# =========================================================================
# Pluggable chain of command optimizers applied before execution.
# Each optimizer is a callable (str) -> str that may rewrite the command
# for better performance. Optimizers are applied in registration order.
# =========================================================================

_command_optimizers: List[Any] = []


def register_command_optimizer(optimizer: Any) -> Any:
    """Register a command optimizer. Can be used as a decorator."""
    _command_optimizers.append(optimizer)
    return optimizer


def optimize_command(cmd: str) -> str:
    """Apply all registered command optimizers to a shell command."""
    for opt in _command_optimizers:
        cmd = opt(cmd)
    return cmd


@register_command_optimizer
def _strip_shell_comments(cmd: str) -> str:
    r"""Strip shell comments before command validation.

    LLMs often include comments in shell commands (e.g., '# Count files').
    These are harmless in readonly mode but cause validation failures.
    This optimizer removes comments while preserving the actual command.

    Handles:
    - Leading comments on lines: '# Comment' -> removed
    - Inline comments: 'cmd # comment' -> 'cmd'
    - Preserves comments in quoted strings: echo "# not a comment"
    - Preserves escaped hashes: echo \# not a comment
    """
    i = 0
    in_quote = None  # None, '"', or "'"
    in_escape = False

    while i < len(cmd):
        char = cmd[i]

        # Handle escape sequences
        if in_escape:
            in_escape = False
            i += 1
            continue

        if char == "\\":
            in_escape = True
            i += 1
            continue

        # Handle quoted strings
        if char in ('"', "'"):
            if in_quote is None:
                in_quote = char
            elif in_quote == char:
                in_quote = None
            i += 1
            continue

        # Handle comments outside quotes
        if char == "#" and in_quote is None:
            # Skip to end of line
            while i < len(cmd) and cmd[i] != "\n":
                i += 1
            continue

        i += 1

    # After stripping comments, clean up empty lines and trailing whitespace
    result = cmd[:i]  # This is wrong - need to rebuild properly

    # Better approach: process line by line
    result_lines = []
    for line in cmd.split("\n"):
        processed = []
        i = 0
        in_quote = None
        in_escape = False

        while i < len(line):
            char = line[i]

            if in_escape:
                processed.append(char)
                in_escape = False
                i += 1
                continue

            if char == "\\":
                processed.append(char)
                in_escape = True
                i += 1
                continue

            if char in ('"', "'"):
                if in_quote is None:
                    in_quote = char
                elif in_quote == char:
                    in_quote = None
                processed.append(char)
                i += 1
                continue

            if char == "#" and in_quote is None:
                # Skip rest of line
                break

            processed.append(char)
            i += 1

        stripped = "".join(processed).rstrip()
        if stripped:
            result_lines.append(stripped)

    result = "\n".join(result_lines)

    if result != cmd:
        logger.debug("Shell optimizer: stripped comments from command")

    return result


@register_command_optimizer
def _optimize_grep_to_rg(cmd: str) -> str:
    """Replace slow recursive grep with ripgrep (rg) when available.

    grep -r/-R on large repos can hang for minutes. ripgrep is 10-100x faster
    because it respects .gitignore, uses memory-mapped I/O, and parallelizes.
    Basic grep flags (-n, -i, -l, -c, -w, -e) are compatible with rg.
    """
    if not re.match(r"^grep\s+.*-[rR]", cmd) and not re.match(r"^grep\s+-[a-zA-Z]*[rR]", cmd):
        return cmd

    if not shutil.which("rg"):
        return cmd

    # Replace 'grep' with 'rg' and remove -r/-R (rg is recursive by default)
    optimized = re.sub(r"^grep\b", "rg", cmd)
    optimized = re.sub(
        r"\s-([a-zA-Z]*)r([a-zA-Z]*)",
        lambda m: (f" -{m.group(1)}{m.group(2)}" if m.group(1) or m.group(2) else ""),
        optimized,
    )
    optimized = re.sub(
        r"\s-([a-zA-Z]*)R([a-zA-Z]*)",
        lambda m: (f" -{m.group(1)}{m.group(2)}" if m.group(1) or m.group(2) else ""),
        optimized,
    )
    # Clean up empty flag groups and extra whitespace
    optimized = re.sub(r"\s-\s", " ", optimized)
    optimized = re.sub(r"\s+", " ", optimized).strip()

    if optimized != cmd:
        logger.info(f"Shell optimizer: grep→rg rewrite: {cmd!r} → {optimized!r}")

    return optimized


_PLACEHOLDER_CMDS = frozenset(
    {
        "...",
        "…",
        "cmd",
        "command",
        "<cmd>",
        "<command>",
        "your_command",
        "your command",
        "none",
        "null",
        "todo",
    }
)


def _is_placeholder_cmd(command: str) -> bool:
    """True when ``cmd`` is a doc placeholder the model copied verbatim.

    Prompt guidance used to read ``shell(cmd='...', action='exec')``; models
    emitted the literal ``...``, /bin/sh reported ``...: command not found``
    (exit 127), and the session concluded the shell tool was unusable. Naming
    the mistake is far more recoverable than a shell error about a missing
    binary.
    """
    return command.strip().strip("'\"").casefold() in _PLACEHOLDER_CMDS


def _is_dangerous(command: str) -> bool:
    """Check if command is potentially dangerous.

    Delegates to the consolidated command safety module.

    Args:
        command: Command to check

    Returns:
        True if dangerous, False otherwise
    """
    return _is_dangerous_consolidated(command)


_PROGRESS_TOOL_NAME = "shell"


# --- Shell git-commit attribution interceptor -------------------------------------------
# Closes the bypass where an LLM issues ``git commit -m "..."`` via the raw
# ``shell`` tool instead of the domain ``git`` tool (which already attributes).
# BEST-EFFORT and deliberately conservative: it only rewrites a *plain, single*
# ``git commit ... -m <constant>`` / ``--message=<constant>`` and bails (leaving
# the command untouched) on any compound form that would make robust rewriting
# fragile: chaining (&&, ||, |, ;), redirections, command substitution,
# heredocs, message-file (-F), or amend/`-c`/`-C` flags. Idempotent and gated by
# the same ``VICTOR_COMMIT_ATTRIBUTION`` env flag as the ``git`` tool path.
# Optional but proactive-by-default, Claude-Code-style.
_SHELL_ATTRIBUTION_BAIL_TOKENS = (
    "&&",
    "||",
    "|",
    ";",
    ">",
    "<",
    "`",
    "$(",
    "<<",
    "-F",
    "--amend",
    "-c ",
    "-C",
)


def _attribution_disabled() -> bool:
    """Return True if the user opted out of commit attribution via env."""
    return os.getenv("VICTOR_COMMIT_ATTRIBUTION", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }


def _maybe_attribute_shell_git_commit(cmd: str) -> str:
    """Best-effort: append the Victor co-author trailer to a plain ``git commit``.

    Returns ``cmd`` unchanged whenever the command is not a *plain, single*
    ``git commit`` with a constant ``-m``/``--message`` argument, or when
    attribution is disabled. Never raises; never fails the command.
    """
    if not cmd or _attribution_disabled():
        return cmd
    stripped = cmd.strip()
    # Bail on anything compound/dynamic that would make rewriting fragile.
    if any(tok in stripped for tok in _SHELL_ATTRIBUTION_BAIL_TOKENS):
        return cmd
    try:
        tokens = shlex.split(stripped, posix=True)
    except ValueError:
        # Unbalanced quotes / escapes: do not risk corrupting the command.
        return cmd
    if len(tokens) < 2 or tokens[0] != "git" or tokens[1] != "commit":
        return cmd

    # Locate the -m / --message constant value (support ``-m msg``, ``-m=msg``,
    # ``--message msg``, ``--message=msg``). Only act on a literal string value.
    msg_index = None
    msg_value = None
    combined = False  # True for ``-m=...`` / ``--message=...`` form
    for i, tok in enumerate(tokens):
        if tok in ("-m", "--message"):
            if i + 1 < len(tokens):
                msg_index = i + 1
                msg_value = tokens[i + 1]
            break
        if tok.startswith("--message="):
            msg_index, msg_value, combined = i, tok.split("--message=", 1)[1], True
            break
        if tok.startswith("-m="):
            msg_index, msg_value, combined = i, tok.split("-m=", 1)[1], True
            break
    if msg_value is None or not msg_value:
        return cmd

    try:
        from victor.core.attribution import append_victor_commit_attribution
    except Exception:
        return cmd
    attributed = append_victor_commit_attribution(msg_value)
    if attributed == msg_value:
        return cmd  # nothing to add (already attributed)

    # Rebuild the located message token. ``--message=v`` / ``-m=v`` keep the
    # prefix; the separate-argument form replaces the value token in place.
    quoted = shlex.quote(attributed)
    if combined:
        prefix = "--message=" if tokens[msg_index].startswith("--message=") else "-m="
        tokens[msg_index] = f"{prefix}{quoted}"
    else:
        tokens[msg_index] = quoted
    return " ".join(tokens)


@tool(
    category="execution",
    priority=Priority.CRITICAL,  # Always available
    access_mode=AccessMode.EXECUTE,  # Executes external commands
    danger_level=DangerLevel.HIGH,  # Arbitrary command execution is risky
    # Registry-driven metadata for tool selection and loop detection
    signature_params=["cmd"],  # Different commands indicate progress, not loops
    stages=["execution", "verification"],  # Conversation stages where relevant
    task_types=["action", "analysis"],  # Task types for classification-aware selection
    execution_category=ExecutionCategory.EXECUTE,  # Cannot run safely in parallel
    mandatory_keywords=[
        "run command",
        "execute",
        "shell",
        # Git diff/compare operations (from MANDATORY_TOOL_KEYWORDS)
        "diff",
        "show changes",
        "git diff",
        "show diff",
        "compare",
        # Running/executing (from MANDATORY_TOOL_KEYWORDS)
        "run",
        "install",
        "test",
        # Count operations (from MANDATORY_TOOL_KEYWORDS)
        "count",
        "how many",
        # Database operations
        "database",
        "sqlite",
        "query",
        "sql",
    ],  # Force inclusion
    keywords=[
        "bash",
        "shell",
        "command",
        "run",
        "execute",
        "terminal",
        "cli",
        "sqlite3",
        "database",
        "sql",
    ],
)
async def shell(
    cmd: str,
    cwd: str = ".",
    timeout: Optional[int] = None,
    dangerous: bool = False,
    readonly: bool = False,
    action: str = "read",
    stdout_limit: Optional[int] = None,
    stderr_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute a shell command from a working directory.

    Mutating/network commands: pass action='exec' (or 'write'/'network') — on
    read-intent turns plain calls are pinned readonly and will be blocked.

    The `cmd` parameter is required. The `cwd` parameter sets the working
    directory the command runs from — its canonical path. Default is `"."`
    (the current/present working directory). Prefer passing `cwd` over
    embedding `cd dir && cmd` in the command string, since `cwd` is validated,
    logged, and preserved across the execution pipeline.

    Args:
        cmd: The shell command to execute.
        cwd: Canonical working directory to run the command from.
            Defaults to `"."` (present working directory). Must exist.
        timeout: Max seconds before the command is killed.
        dangerous: Override the dangerous-command blocklist (use sparingly).
        readonly: When True, validate the command against the readonly
            allowlist. Defaults to False — the dangerous-command check and
            ShellSafetyPolicy are the primary safety floor. Pass readonly=True
            to opt INTO the allowlist for commands you know are read-only.
        action: "read" (default) or "write"/"network"/"exec" — the caller's
            intent. Non-"read" actions bypass the allowlist (same as
            readonly=False).
        stdout_limit: Max stdout lines to return.
        stderr_limit: Max stderr lines to return.

    Examples:
        shell(cmd="ls -la")                       # runs in present working dir
        shell(cmd="pytest -q", cwd="tests/unit")  # run from a subdirectory
        shell(cmd="git status")                   # git via shell (no git tool needed)
        shell(cmd='sqlite3 data.db ".tables"')    # Database operations
        shell(cmd='sqlite3 data.db "SELECT * FROM users LIMIT 10"')

    For database files (SQLite, PostgreSQL, MySQL):
    - Use shell(cmd='sqlite3 file.db ".tables"') to list tables
    - Use shell(cmd='sqlite3 file.db "SELECT * FROM table LIMIT 10"') to query

    Feasible CLI commands (use shell for these instead of dedicated tools):
        git status, git diff, git log, git branch, git checkout, git push
        docker ps, docker build, docker run, docker exec, docker images
        pip install, pip list, pip show, pipdeptree
        pytest, make test, npm test, cargo test
        gh pr create, gh pr list, gh issue list
        terraform validate, terraform plan, docker-compose config
        make, npm, cargo, go build, yarn, pnpm
    - Use shell(cmd='psql -h localhost -U user -d db -c "SELECT * FROM table"') for PostgreSQL
    - Use shell(cmd='mysql -u user -p db -e "SHOW TABLES"') for MySQL

    For multiline scripts or quote-heavy payloads, prefer a heredoc inside
    `cmd` instead of deeply nested shell escaping. Example:
        shell(cmd="python - <<'PY'\\nprint('hello')\\nPY")

    Args:
        cmd: The shell command string to execute (required)
        cwd: Working directory for the command
        timeout: Maximum seconds before timeout
        dangerous: Set true only for destructive commands (rm, kill, etc.)
        readonly: Defaults to False. Pass True to validate the command against
            the readonly allowlist (opt-in for purely read-only commands).
        stdout_limit: Max lines for stdout (None=unlimited, default: 10000)
        stderr_limit: Max lines for stderr (None=unlimited, default: 2000)

    Returns:
        Dict with stdout, stderr, return_code keys
    """
    if not cmd:
        return {
            "success": False,
            "error": "Missing required parameter: cmd",
            "stdout": "",
            "stderr": "",
            "return_code": -1,
        }

    if _is_placeholder_cmd(cmd):
        return {
            "success": False,
            "error": (
                f"'{cmd.strip()}' is a documentation placeholder, not a command. "
                "Pass the real command text, e.g. "
                "shell(cmd='ls -la /path/to/dir', action='read') or "
                "shell(cmd='pytest -q', action='exec')."
            ),
            "stdout": "",
            "stderr": "",
            "return_code": -1,
        }

    # Apply default timeout from centralized config
    if timeout is None:
        timeout = ProcessTimeouts.BASH_DEFAULT

    # Apply command optimizer pipeline (grep→rg, etc.)
    cmd = optimize_command(cmd)

    # Best-effort agent-layer attribution for raw ``git commit -m "..."`` issued
    # through the shell escape hatch (the domain ``git`` tool is attributed in its
    # own module). Conservative: only touches a plain single git commit; leaves
    # compound/dynamic commands untouched. Gated by VICTOR_COMMIT_ATTRIBUTION.
    cmd = _maybe_attribute_shell_git_commit(cmd)

    # Guard against filesystem-wide searches (find / ...) that timeout.
    if re.match(r"^\s*find\s+/", cmd.strip()):
        return {
            "success": False,
            "error": (
                "Filesystem-wide searches are not allowed. "
                "Use `code search` or `find . -name ...` within the workspace."
            ),
            "stdout": "",
            "stderr": "",
            "return_code": -1,
        }

    # Check for dangerous commands
    if not dangerous and _is_dangerous(cmd):
        return {
            "success": False,
            "error": f"Dangerous command blocked: {cmd}",
            "stdout": "",
            "stderr": "",
            "return_code": -1,
        }

    # Map explicit `action` intent to a readonly override so the model can
    # declare what KIND of command it is running (read|write|network|exec)
    # instead of guessing the inverted `readonly` boolean.
    #   action="read"    -> readonly stays as passed (enforce allowlist)
    #   action="network" -> allow curl/wget/ping/ssh (network class)
    #   action="write"   -> mutate filesystem/git state
    #   action="exec"    -> arbitrary exec
    _ACTION_EFFECTIVE_READONLY = {
        "read": None,  # honor the `readonly` arg as-is
        "network": False,  # network ops are never readonly-allowlisted
        "write": False,  # mutations must bypass the allowlist
        "exec": False,  # arbitrary exec must bypass the allowlist
    }
    _eff = _ACTION_EFFECTIVE_READONLY.get(action)
    if _eff is not None:
        readonly = _eff

    # --- FEP-0013: consult the session-scoped shell safety policy. The default
    # ``legacy`` policy preserves the inline allowlist gate below unchanged; a
    # non-legacy (damage-scoped) policy replaces it with invariant-based allow/
    # deny/ask. ``readonly`` is honoured as a *hint* and overridden by the
    # policy's effective_readonly on ALLOW.
    _shell_policy = _get_shell_safety_policy()
    _use_shell_policy = _shell_policy.name != "legacy-allowlist"
    if _use_shell_policy:
        _decision = _shell_policy.evaluate(
            _ShellSafetyCtx(command=cmd, cwd=cwd, readonly_hint=readonly, action_hint=action)
        )
        if _decision.verdict is _SafetyVerdict.DENY:
            _inv = f"/{_decision.invariant}" if _decision.invariant else ""
            return {
                "success": False,
                "error": (
                    f"Shell command denied ({_decision.category}{_inv}): "
                    f"{_decision.reason}. Keep writes inside the working directory "
                    f"({cwd}) and avoid protected paths."
                ),
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "cwd": cwd,
            }
        if _decision.verdict is _SafetyVerdict.ASK:
            # Phase 1 surfaces the approval need as an actionable error; routing
            # ASK through the governance PolicyEngine (FEP-0005) is Phase 2.
            return {
                "success": False,
                "error": (
                    f"Shell command requires approval ({_decision.category}): "
                    f"{_decision.reason}"
                ),
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "cwd": cwd,
            }
        # ALLOW: adopt the policy's effective readonly and skip the legacy gate.
        readonly = _decision.effective_readonly

    # Check readonly mode restrictions (legacy inline allowlist gate; skipped
    # when a non-legacy policy authorized the command above).
    if readonly and not _use_shell_policy:
        is_valid, failing_cmd = _validate_readonly_command(cmd)
        if not is_valid:
            return {
                "success": False,
                "error": (
                    f"Command '{failing_cmd}' is not allowed in readonly mode "
                    "(the turn may have been classified read-only). "
                    f"Allowed commands: {', '.join(sorted(get_allowed_readonly_commands())[:15])}... "
                    "Re-run with readonly=False (or action='network'/'write'/'exec') "
                    "to run mutating or network commands."
                ),
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "cwd": os.getcwd(),
            }

    # Validate working directory exists before execution
    if cwd:
        if not os.path.isdir(cwd):
            return {
                "success": False,
                "error": f"Working directory does not exist: {cwd}",
                "stdout": "",
                "stderr": "",
                "return_code": -1,
            }

    # Frugal flag injection: rewrite recognized commands to quieter forms so
    # less output exists in the first place (e.g. pytest -q --tb=short -rxX).
    original_command: Optional[str] = None
    try:
        from victor.config.tool_settings import get_tool_settings as _get_ts
        from victor.tools.output_condenser import rewrite_frugal_command

        _ts = _get_ts()
        if _ts.shell_output_condensation_enabled and _ts.shell_frugal_flags_enabled:
            _rewritten = rewrite_frugal_command(cmd)
            if _rewritten:
                original_command = cmd
                cmd = _rewritten
                logger.debug("Frugal flag rewrite: %r → %r", original_command, cmd)
    except Exception:
        logger.debug("Frugal flag rewrite skipped", exc_info=True)

    def _condense_streams(
        stdout_text: str, stderr_text: str, returncode: int
    ) -> tuple[str, str, Optional[Dict[str, Any]]]:
        """Apply command-aware output condensation (fail-open passthrough)."""
        try:
            from victor.config.tool_settings import get_tool_settings
            from victor.tools.output_condenser import condense_shell_output

            settings = get_tool_settings()
            if not settings.shell_output_condensation_enabled:
                return stdout_text, stderr_text, None
            condensed = condense_shell_output(
                cmd,
                stdout_text,
                stderr_text,
                returncode,
                tee_enabled=settings.shell_output_raw_tee_enabled,
            )
            if condensed is None:
                return stdout_text, stderr_text, None
            return (
                condensed.stdout,
                condensed.stderr,
                {
                    "condenser": condensed.condenser,
                    "original_chars": condensed.original_chars,
                    "condensed_chars": condensed.condensed_chars,
                    "raw_log_path": condensed.raw_log_path,
                },
            )
        except Exception:
            logger.debug("Shell output condensation skipped", exc_info=True)
            return stdout_text, stderr_text, None

    try:
        # Cache lookup never executes. Every miss has the same async process
        # owner, OS sandbox, progress callbacks, timeout and cancellation path.
        from victor.tools.shell_command_cache import get_shell_cache
        from victor.tools.subprocess_executor import _resolve_default_sandbox, run_managed_process

        _sandbox = _resolve_default_sandbox()
        cache_eligible = (
            _sandbox.type_name == "none"
            and not dangerous
            and _is_readonly_command(cmd)
            and cacheable_inspection(cmd)
            and (not _use_shell_policy or readonly)
        )
        cache = None
        cached_result = None
        if cache_eligible:
            try:
                cache = get_shell_cache()
                cached_result = cache.get(cmd, cwd=cwd)
            except Exception as cache_error:
                logger.warning("Cache lookup failed: %s", cache_error)
        cache_hit = cached_result is not None
        if cached_result is not None:
            returncode, stdout_str, stderr_str = cached_result
            process = SimpleNamespace(returncode=returncode)
            raw_stdout_str, raw_stderr_str = stdout_str, stderr_str
            _capped = False
        else:
            # Create subprocess and run it to completion or timeout via the shared
            # runner (co-design review item 15): spawn/kill/timeout are
            # centralized in run_managed_process, which spawns in its own POSIX
            # process group and kills the WHOLE group on timeout — a child the
            # shell command backgrounded (e.g. ``sleep 100 &``) is no longer
            # orphaned past the timeout the way a plain process.kill() left it.
            # (apply OS sandbox if active; default off = no change).
            _wrapped = (
                _sandbox.wrap_argv(["/bin/sh", "-c", cmd], Path(cwd) if cwd else None)
                if _sandbox.type_name != "none"
                else None
            )

            # When a live progress sink is active, stream stdout/stderr chunks to
            # it as they arrive so the UI shows output as it is produced.
            from victor.core.tool_progress import emit_tool_progress, has_progress_sink

            _streaming_progress = has_progress_sink()

            def _emit_chunk(is_stderr: bool, chunk: bytes) -> None:
                text = chunk.decode("utf-8", "replace")
                emit_tool_progress(
                    name=_PROGRESS_TOOL_NAME,
                    stdout="" if is_stderr else text,
                    stderr=text if is_stderr else "",
                )

            stdout, stderr, _return_code, _timed_out, _capped = await run_managed_process(
                command=cmd if _wrapped is None else None,
                argv=_wrapped,
                cwd=cwd,
                timeout=timeout,
                on_chunk=_emit_chunk if _streaming_progress else None,
            )
            # Lightweight stand-in preserving the process.returncode reads below
            # (condensation, cache write, error summary) without a real Process
            # object — run_managed_process already reaped the real one.
            process = SimpleNamespace(returncode=_return_code)

            if _timed_out:
                return {
                    "success": False,
                    "error": f"Command timed out after {timeout} seconds",
                    "stdout": stdout.decode("utf-8", "replace") if stdout else "",
                    "stderr": stderr.decode("utf-8", "replace") if stderr else "",
                    "return_code": -1,
                }

            if _streaming_progress:
                emit_tool_progress(name=_PROGRESS_TOOL_NAME, is_final=True)

            stdout_str = stdout.decode("utf-8", "replace") if stdout else ""
            stderr_str = stderr.decode("utf-8", "replace") if stderr else ""
            raw_stdout_str, raw_stderr_str = stdout_str, stderr_str

        # Condense recognized command output before generic truncation so the
        # diagnostic tail (failure summaries, error counts) survives.
        stdout_str, stderr_str, condensation_info = _condense_streams(
            stdout_str, stderr_str, process.returncode or 0
        )

        # Apply defaults: 10K stdout lines, 2K stderr lines (None=unlimited)
        final_stdout_limit = stdout_limit if stdout_limit is not None else 10000
        final_stderr_limit = stderr_limit if stderr_limit is not None else 2000

        # Truncate stdout
        from victor.tools.subprocess_executor import _truncate_output_by_lines

        stdout_str, stdout_truncated, stdout_lines = _truncate_output_by_lines(
            stdout_str,
            final_stdout_limit,
            max_bytes=None,  # Use internal 1MB default
            stream_name="stdout",
        )

        # Truncate stderr
        stderr_str, stderr_truncated, stderr_lines = _truncate_output_by_lines(
            stderr_str,
            final_stderr_limit,
            max_bytes=None,  # Use internal 1MB default
            stream_name="stderr",
        )

        was_truncated = stdout_truncated or stderr_truncated

        # Invalidate file content cache if command was NOT readonly (may have modified files)
        if not readonly:
            try:
                from victor.tools.filesystem import (
                    clear_file_content_cache,
                    is_file_cache_enabled,
                )

                if is_file_cache_enabled():
                    clear_file_content_cache(reset_stats=False)
                    logger.debug("Cleared file content cache after non-readonly shell command")
            except (ImportError, Exception):
                pass

        result = {
            "success": process.returncode == 0,
            "stdout": stdout_str,
            "stderr": stderr_str,
            "return_code": process.returncode,
            "command": cmd,
            "working_dir": cwd,
            "cached": cache_hit,
            "truncated": was_truncated,
            "stdout_lines": stdout_lines,
            "stderr_lines": stderr_lines,
        }
        if condensation_info:
            result["condensed"] = condensation_info
        if original_command:
            result["original_command"] = original_command

        # Cache successful read-only command results (raw streams — the cached
        # path re-applies condensation so hits and misses stay consistent)
        if process.returncode == 0 and cache_eligible and not cache_hit and not _capped:
            try:
                if cache is None:
                    cache = get_shell_cache()
                cache.set(cmd, (process.returncode, raw_stdout_str, raw_stderr_str), cwd)
            except Exception as cache_error:
                logger.warning(f"Failed to cache command result: {cache_error}")

        # Include informative error message when command fails
        if process.returncode != 0:
            error_parts = []
            # Always attribute the failure to the command that ran (log
            # analysis showed exit-code errors reach the file log with no
            # tool/args context, making 22/30 failures in a 3h window
            # unattributable). Keep it bounded so long commands don't bloat
            # logs; stderr preview below carries the diagnosis.
            cmd_preview = cmd if len(cmd) <= 200 else cmd[:200] + "..."
            error_parts.append(f"Command failed with exit code {process.returncode}: {cmd_preview}")
            if stderr_str.strip():
                # Truncate stderr if too long, keeping first and last parts
                stderr_preview = stderr_str.strip()
                if len(stderr_preview) > 500:
                    stderr_preview = stderr_preview[:250] + "\n...\n" + stderr_preview[-250:]
                error_parts.append(f"stderr: {stderr_preview}")
            elif stdout_str.strip():
                # Some commands output errors to stdout
                stdout_preview = stdout_str.strip()
                if len(stdout_preview) > 300:
                    stdout_preview = stdout_preview[:150] + "..." + stdout_preview[-150:]
                error_parts.append(f"output: {stdout_preview}")
            result["error"] = "\n".join(error_parts)

        return result

    except FileNotFoundError:
        return {
            "success": False,
            "error": f"Working directory not found: {cwd}",
            "stdout": "",
            "stderr": "",
            "return_code": -1,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to execute command: {str(e)}",
            "stdout": "",
            "stderr": "",
            "return_code": -1,
        }
