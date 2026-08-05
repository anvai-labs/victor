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

import asyncio
import logging
import os
import platform
import re
import shlex
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

from victor.config.timeouts import ProcessTimeouts
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
    "source",  # Shell environment (venv activation)
    ".",  # Shell environment (venv activation)
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
    "source",
    ".",
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
GIT_READONLY_SUBCOMMANDS: Set[str] = {
    "status",
    "log",
    "show",
    "diff",
    "branch",
    "tag",
    "remote",
    "ls-files",
    "ls-tree",
    "rev-parse",
    "rev-list",
    "describe",
    "shortlog",
    "blame",
    "grep",
    "config",  # readonly by default (no --global, --unset)
    "reflog",
    "stash",  # list only
    "worktree",  # list only
    "cat-file",
    "ls-remote",
    "check-ignore",
    "check-attr",
    "name-rev",
    "verify-commit",
    "verify-tag",
    "for-each-ref",
}

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
    """Split a compound command into individual command segments.

    Handles ``&&``, ``||``, ``;`` and pipeline ``|`` operators while respecting
    quoted strings. Every returned segment is validated independently for
    readonly mode, so a safe command cannot hide a mutating command later in a
    chain or pipeline.
    """
    components = []
    current = []
    i = 0
    in_quote = None  # None, '"', or "'"
    in_escape = False

    # Use variables to avoid quote escaping issues
    single_quote = "'"
    double_quote = '"'
    quote_chars = {single_quote, double_quote}

    while i < len(cmd):
        char = cmd[i]

        if in_escape:
            current.append(char)
            in_escape = False
            i += 1
            continue

        if char == "\\":
            current.append(char)
            in_escape = True
            i += 1
            continue

        if char in quote_chars and (not in_quote or in_quote == char):
            if in_quote == char:
                in_quote = None
            else:
                in_quote = char
            current.append(char)
            i += 1
            continue

        # Check for compound operators (only when not in quotes)
        if not in_quote:
            # Check for && (must be & followed by &, not part of other text)
            if char == "&" and i + 1 < len(cmd) and cmd[i + 1] == "&":
                component = "".join(current).strip()
                if component:
                    components.append(component)
                current = []
                i += 2
                continue
            # Check for ||
            if char == "|" and i + 1 < len(cmd) and cmd[i + 1] == "|":
                component = "".join(current).strip()
                if component:
                    components.append(component)
                current = []
                i += 2
                continue
            # Check for single pipeline |. A pipeline is readonly only if every
            # segment is readonly; `cat f | tee out` must fail on the tee segment.
            if char == "|":
                component = "".join(current).strip()
                if component:
                    components.append(component)
                current = []
                i += 1
                continue
            # Check for ; (not part of ;; or other constructs)
            if char == ";":
                # Skip if it's ;; (used in some shells like case statements)
                if i + 1 < len(cmd) and cmd[i + 1] == ";":
                    current.append(char)
                    i += 1
                else:
                    component = "".join(current).strip()
                    if component:
                        components.append(component)
                    current = []
                i += 1
                continue

        current.append(char)
        i += 1

    # Add the last component
    component = "".join(current).strip()
    if component:
        components.append(component)

    return components if components else [cmd.strip()]


def _redirection_target(cmd: str, start: int) -> tuple[str, int]:
    """Return the target token after a redirection operator.

    ``start`` points just after the operator. The target is read until shell
    whitespace or another unquoted shell operator. Quotes are stripped by
    ``shlex`` later; here we only need enough lexical awareness to distinguish
    ``>/dev/null`` from ``> output.txt``.
    """
    n = len(cmd)
    i = start
    while i < n and cmd[i].isspace():
        i += 1
    out: List[str] = []
    in_quote = None
    escape = False
    while i < n:
        ch = cmd[i]
        if escape:
            out.append(ch)
            escape = False
            i += 1
            continue
        if ch == "\\":
            escape = True
            i += 1
            continue
        if ch in {"'", '"'} and (in_quote is None or in_quote == ch):
            in_quote = None if in_quote == ch else ch
            i += 1
            continue
        if in_quote is None and (ch.isspace() or ch in {";", "|", "&", "<", ">"}):
            break
        out.append(ch)
        i += 1
    return "".join(out), i


def _validate_readonly_redirections(cmd: str) -> tuple[bool, str]:
    """Reject output redirection that can write files in readonly mode.

    Allowed:
      * stderr/stdout suppression to ``/dev/null``: ``>/dev/null``, ``2>/dev/null``
      * descriptor duplication/closure: ``2>&1``, ``1>&2``, ``2>&-``
      * input-only redirection: ``less < file``

    Rejected:
      * file writes/appends: ``> out``, ``>> out``, ``2> err``, ``&> out``
    """
    i, n = 0, len(cmd)
    in_single = False
    in_double = False
    escape = False
    while i < n:
        ch = cmd[i]
        if escape:
            escape = False
            i += 1
            continue
        if ch == "\\" and not in_single:
            escape = True
            i += 1
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            i += 1
            continue
        if in_single or in_double:
            i += 1
            continue

        # Input redirection is readonly; skip `< file` and here-strings/docs.
        if ch == "<":
            i += 2 if i + 1 < n and cmd[i + 1] == "<" else 1
            continue

        # stdout/stderr redirection. Include optional leading fd and &>.
        op_start = i
        if ch.isdigit():
            j = i
            while j < n and cmd[j].isdigit():
                j += 1
            if j >= n or cmd[j] != ">":
                i += 1
                continue
            op_start = j
        elif ch == "&" and i + 1 < n and cmd[i + 1] == ">":
            op_start = i + 1
        elif ch != ">":
            i += 1
            continue

        # Do not confuse comparison operators in test/[ with redirection; this
        # scanner is conservative, but `-gt`/`>` comparisons are usually quoted
        # or spaced inside test. Redirection requires an operator token here.
        if op_start > 0 and cmd[op_start - 1] not in " \t\n\r&0123456789":
            i += 1
            continue

        j = op_start + 1
        if j < n and cmd[j] == ">":
            j += 1
        if j < n and cmd[j] == "&":
            target, next_i = _redirection_target(cmd, j + 1)
            if target in {"1", "2", "-"}:
                i = next_i
                continue
            return False, "redirection (>&)"

        target, next_i = _redirection_target(cmd, j)
        if target == "/dev/null":
            i = next_i
            continue
        return False, "redirection (>)"

    return True, ""


# Shell control-flow keywords are structural — they run no command themselves.
_SHELL_CONTROL_FLOW_KEYWORDS = frozenset(
    {
        "for",
        "while",
        "until",
        "if",
        "then",
        "elif",
        "else",
        "fi",
        "do",
        "done",
        "case",
        "esac",
        "select",
        "time",
        "function",
        "!",
        "{",
        "}",
    }
)
# Keywords that introduce a word-list/header (not a command) after them.
_SHELL_LOOP_HEADER_KEYWORDS = frozenset({"for", "select", "case"})
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


def _scan_substitutions(text: str) -> "tuple[List[str], str]":
    """Quote-aware scan for command substitutions.

    Returns ``(subs, stripped)`` where ``subs`` are the inner command strings of
    genuine ``$(...)`` / backtick substitutions — which execute during expansion
    and must be validated against the read-only allowlist — and ``stripped`` is
    ``text`` with those substitutions replaced by the benign ``true`` builtin so
    the surrounding command can be parsed without their spaces/operators leaking
    into structural parsing.

    Shell quoting is respected, which is the whole point: inside **single quotes**
    everything is literal, so backticks/``$()`` in a grep pattern such as
    ``'^```mermaid'`` or ``'```\\|```'`` are NOT substitutions and are left intact.
    Inside double quotes and unquoted text, ``$(...)`` and backticks are active,
    matching shell semantics. (The previous regex-based extractor ignored quoting
    and mis-read literal backticks as substitutions, rejecting valid commands.)
    """
    subs: List[str] = []
    out: List[str] = []
    i, n = 0, len(text)
    in_single = False
    in_double = False
    while i < n:
        ch = text[i]
        # Backslash escapes the next char (not inside single quotes).
        if ch == "\\" and i + 1 < n and not in_single:
            out.append(ch)
            out.append(text[i + 1])
            i += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            out.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            out.append(ch)
            i += 1
            continue
        if not in_single:
            # $(...) — depth-balanced so nested parens don't end it early.
            if ch == "$" and i + 1 < n and text[i + 1] == "(":
                depth, j = 1, i + 2
                start = j
                while j < n and depth:
                    if text[j] == "(":
                        depth += 1
                    elif text[j] == ")":
                        depth -= 1
                    j += 1
                if depth == 0:
                    subs.append(text[start : j - 1])
                    out.append("true")
                    i = j
                    continue
                out.append(text[i:])  # unbalanced — keep the rest literal
                break
            # `...` backtick substitution
            if ch == "`":
                j = text.find("`", i + 1)
                if j != -1:
                    subs.append(text[i + 1 : j])
                    out.append("true")
                    i = j + 1
                    continue
                # unbalanced backtick — treat literally
        out.append(ch)
        i += 1
    return subs, "".join(out)


def _strip_structural_prefix(component: str) -> Optional[str]:
    """Strip leading control-flow keywords and variable assignments.

    Returns the residual simple command, or ``None`` if the component is purely
    structural (a loop/case header, a bare keyword like ``do``/``done``/``then``/
    ``fi``, or only ``VAR=value`` assignments) and runs no command itself.
    """
    s = component.strip()
    if not s:
        return None
    try:
        tokens = shlex.split(s)
    except ValueError:
        tokens = s.split()
    if not tokens:
        return None
    # A loop/case header (`for VAR in WORDS`, `case WORD in`) runs no command.
    if tokens[0].lower() in _SHELL_LOOP_HEADER_KEYWORDS:
        return None
    idx = 0
    while idx < len(tokens) and tokens[idx].lower() in _SHELL_CONTROL_FLOW_KEYWORDS:
        idx += 1
    while idx < len(tokens) and _ASSIGNMENT_RE.match(tokens[idx]):
        idx += 1
    if idx >= len(tokens):
        return None
    return " ".join(tokens[idx:])


def _validate_readonly_command(cmd: str) -> tuple[bool, str]:
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
    subs, sanitized = _scan_substitutions(cmd)
    for sub in subs:
        sub = sub.strip()
        if not sub:
            continue
        valid, failing = _validate_readonly_command(sub)
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


def _validate_single_readonly_command(cmd: str) -> tuple[bool, str]:
    """Validate a single (already split, substitution-free) shell command."""
    # Strip leading control-flow keywords, loop headers, and assignments. What
    # remains is the actual command to validate (or nothing, for a structural
    # fragment like `do` / `done` / `for x in ...`).
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

    # sed is readonly unless it edits in place. Tokenize this check so a pattern
    # containing the text "-i" is not misclassified.
    if base_cmd == "sed" and any(t == "-i" or t.startswith("-i") for t in _tokens[1:]):
        return False, "sed -i"

    if len(_tokens) <= 2 and any(t in _bare_info for t in _tokens[1:]):
        return True, ""

    # Special handling for commands with subcommands
    if base_cmd == "git":
        subcommand = _extract_subcommand(cmd, "git")
        if not subcommand or subcommand not in GIT_READONLY_SUBCOMMANDS:
            return False, f"git {subcommand or ''}"
        return True, ""

    if base_cmd in {"pip", "pip3"}:
        subcommand = _extract_subcommand(cmd, base_cmd)
        if not subcommand or subcommand not in PIP_READONLY_SUBCOMMANDS:
            return False, f"{base_cmd} {subcommand or ''}"
        return True, ""

    if base_cmd == "npm":
        subcommand = _extract_subcommand(cmd, "npm")
        # Allow test, list, view, show, etc.
        if subcommand in NPM_READONLY_SUBCOMMANDS or subcommand == "test":
            return True, ""
        return False, f"npm {subcommand or ''}"

    if base_cmd == "cargo":
        subcommand = _extract_subcommand(cmd, "cargo")
        # Allow test, list, metadata, check, verify, etc.
        if subcommand in {"test", "check", "metadata", "list", "verify", "tree"}:
            return True, ""
        return False, f"cargo {subcommand or ''}"

    if base_cmd == "go":
        subcommand = _extract_subcommand(cmd, "go")
        # Allow test, list, version, etc.
        if subcommand in {"test", "list", "version", "help", "doc"}:
            return True, ""
        return False, f"go {subcommand or ''}"

    if base_cmd == "make":
        # Allow make test, make check, make help
        if any(target in cmd for target in ["test", "check", "help", "--version"]):
            return True, ""
        # 'make' alone often defaults to a build/test target, but we'll be slightly restrictive
        return False, "make (non-test target)"

    if base_cmd == "gh":
        subcommand = _extract_subcommand(cmd, "gh")
        if not subcommand or subcommand not in GH_READONLY_SUBCOMMANDS:
            return False, f"gh {subcommand or ''}"
        return True, ""

    if base_cmd == "az":
        subcommand = _extract_subcommand(cmd, "az")
        if not subcommand or subcommand not in AZ_READONLY_SUBCOMMANDS:
            return False, f"az {subcommand or ''}"
        return True, ""

    if base_cmd == "kubectl":
        subcommand = _extract_subcommand(cmd, "kubectl")
        if not subcommand or subcommand not in KUBECTL_READONLY_SUBCOMMANDS:
            return False, f"kubectl {subcommand or ''}"
        return True, ""

    # tar handling: only allow listing operations
    if base_cmd == "tar":
        # Allow -t, -tzf, -tf for listing (write operations: -c, -x blocked)
        if "-t" in cmd and "-c" not in cmd and "-x" not in cmd:
            return True, ""
        return False, "tar (non-list operation)"

    # unzip handling: only allow listing
    if base_cmd == "unzip":
        # Allow -l, -Z for listing
        if "-l" in cmd or "-Z" in cmd:
            return True, ""
        return False, "unzip (non-list operation)"

    # ip handling: only allow read-only subcommands
    if base_cmd == "ip":
        # Allow addr show, route show, link show, etc.
        readonly_ip_verbs = {"show", "list", "get"}
        for verb in readonly_ip_verbs:
            if f"ip {verb}" in cmd or f"ip.{verb}" in cmd:
                return True, ""
        return False, "ip (write operation)"

    # docker handling: only allow read-only subcommands
    if base_cmd == "docker":
        # Allow images, ps, inspect, network ls, volume ls, etc.
        readonly_docker = {
            "images",
            "ps",
            "inspect",
            "network",
            "volume",
            "search",
            "info",
            "version",
            "system",
            "stats",
            "history",
            "context",
            "buildx",
            "compose",
            "manifest",
            "image",
            "container",
        }
        for sub in readonly_docker:
            if f"docker {sub}" in cmd:
                # For network and volume, only allow ls/inspect
                if sub in {"network", "volume"}:
                    if "ls" in cmd or "inspect" in cmd:
                        return True, ""
                elif sub == "system":
                    # Allow df, info, events but not prune
                    if "prune" in cmd:
                        return False, "docker system prune"
                    return True, ""
                elif sub == "image":
                    # Allow ls, inspect, history, prune is write
                    if "ls" in cmd or "inspect" in cmd or "history" in cmd:
                        return True, ""
                    return False, "docker image (write op)"
                elif sub == "container":
                    if "ls" in cmd or "inspect" in cmd or "stats" in cmd:
                        return True, ""
                    return False, "docker container (write op)"
                elif sub == "compose":
                    # Allow ps, logs, config, top, images
                    _compose_ro = {"ps", "logs", "config", "top", "images", "port"}
                    _compose_sub = _extract_subcommand(
                        cmd.replace("docker compose", "compose"), "compose"
                    )
                    if _compose_sub in _compose_ro:
                        return True, ""
                    return False, f"docker compose {_compose_sub or ''}"
                else:
                    return True, ""
        return False, "docker (write operation)"

    # podman handling: same as docker
    if base_cmd == "podman":
        readonly_podman = {
            "images",
            "ps",
            "inspect",
            "network",
            "volume",
            "search",
            "info",
            "version",
            "system",
        }
        for sub in readonly_podman:
            if f"podman {sub}" in cmd:
                if sub in {"network", "volume"}:
                    if "ls" in cmd or "inspect" in cmd:
                        return True, ""
                else:
                    return True, ""
        if "podman" in cmd and ("--version" in cmd or "version" in cmd):
            return True, ""
        return False, "podman (write operation)"

    # helm handling: only allow read-only subcommands
    if base_cmd == "helm":
        readonly_helm = {
            "list",
            "ls",
            "status",
            "history",
            "get",
            "search",
            "info",
            "version",
            "repo",
        }
        for sub in readonly_helm:
            if f"helm {sub}" in cmd or f"helm {sub}" in cmd.replace("helm ", ""):
                return True, ""
        if "helm" in cmd and ("--version" in cmd or "version" in cmd):
            return True, ""
        return False, "helm (write operation)"

    # yarn handling: similar to npm
    if base_cmd == "yarn":
        readonly_yarn = {
            "list",
            "ls",
            "info",
            "why",
            "version",
            "help",
            "cache",
            "check",
        }
        for sub in readonly_yarn:
            if f"yarn {sub}" in cmd:
                return True, ""
        return False, "yarn (write operation)"

    # pnpm handling: similar to npm
    if base_cmd == "pnpm":
        readonly_pnpm = {
            "list",
            "ls",
            "view",
            "show",
            "info",
            "why",
            "version",
            "help",
            "outdated",
        }
        for sub in readonly_pnpm:
            if f"pnpm {sub}" in cmd:
                return True, ""
        return False, "pnpm (write operation)"

    # npm handling: similar to pip — only allow readonly subcommands
    if base_cmd == "npm":
        subcommand = _extract_subcommand(cmd, "npm")
        if subcommand and subcommand in NPM_READONLY_SUBCOMMANDS:
            return True, ""
        return False, f"npm {subcommand or ''}"

    # Code quality tools: allow --check, --version, help
    if base_cmd in {"flake8", "pylint", "mypy", "ruff", "eslint"}:
        if (
            "--check" in cmd
            or "--version" in cmd
            or "version" in cmd
            or "-h" in cmd
            or "--help" in cmd
        ):
            return True, ""
        return False, f"{base_cmd} (non-check operation)"

    # black: allow --check, --diff, --version
    if base_cmd == "black":
        if "--check" in cmd or "--diff" in cmd or "--version" in cmd or "version" in cmd:
            return True, ""
        return False, "black (write operation)"

    # Python handling: allow -m pytest, -c, --version, -V, -h, --help
    if base_cmd in {"python", "python3"}:
        # Check for dangerous flags like -i (interactive) or -u
        if "-m pytest" in cmd:
            return True, ""
        if "-c" in cmd or "--version" in cmd or "-V" in cmd or "-h" in cmd or "--help" in cmd:
            return True, ""
        # If it's just 'python script.py', it depends on the script, but we allow it
        # as python is in the readonly set. We could be stricter here.
        return True, ""

    # Check for pipe to shell
    if any(p in cmd for p in ["| sh", "| bash", "|sh", "|bash"]):
        return False, "pipe to shell"

    return True, ""


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


async def _stream_subprocess_output(process: Any, tool_name: str) -> tuple[bytes, bytes]:
    """Read a subprocess's stdout/stderr concurrently, emitting live progress.

    Returns the full accumulated (stdout, stderr) bytes so the existing result
    contract (truncation, caching) is unchanged. Used only when a UI progress
    sink is active; otherwise the caller uses the cheaper ``communicate()``.
    """
    from victor.framework.tool_progress import emit_tool_progress

    out_buf = bytearray()
    err_buf = bytearray()

    async def _drain(stream: Any, buf: bytearray, is_stderr: bool) -> None:
        if stream is None:
            return
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
            buf.extend(chunk)
            text = chunk.decode("utf-8", "replace")
            emit_tool_progress(
                name=tool_name,
                stdout="" if is_stderr else text,
                stderr=text if is_stderr else "",
            )

    await asyncio.gather(
        _drain(process.stdout, out_buf, False),
        _drain(process.stderr, err_buf, True),
    )
    await process.wait()
    return bytes(out_buf), bytes(err_buf)


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

    try:
        # Check cache for read-only commands (CI/CD queries, git log, etc.)
        from victor.tools.shell_command_cache import get_shell_cache, execute_with_cache

        # Use cache for read-only commands
        if not dangerous and _is_readonly_command(cmd):
            try:
                returncode, stdout_str, stderr_str = execute_with_cache(
                    cmd,
                    cwd=cwd,
                    shell=True,
                    timeout=timeout,
                )

                # Apply truncation to cached results too
                final_stdout_limit = stdout_limit if stdout_limit is not None else 10000
                final_stderr_limit = stderr_limit if stderr_limit is not None else 2000

                from victor.tools.subprocess_executor import _truncate_output_by_lines

                stdout_str, stdout_truncated, stdout_lines = _truncate_output_by_lines(
                    stdout_str, final_stdout_limit, max_bytes=None, stream_name="stdout"
                )

                stderr_str, stderr_truncated, stderr_lines = _truncate_output_by_lines(
                    stderr_str, final_stderr_limit, max_bytes=None, stream_name="stderr"
                )

                return {
                    "success": returncode == 0,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "return_code": returncode,
                    "command": cmd,
                    "working_dir": cwd,
                    "cached": True,
                    "truncated": stdout_truncated or stderr_truncated,
                    "stdout_lines": stdout_lines,
                    "stderr_lines": stderr_lines,
                }
            except Exception as cache_error:
                # If caching fails, fall through to normal execution
                logger.warning(f"Cache lookup failed, executing directly: {cache_error}")

        # Create subprocess (apply OS sandbox if active; default off = no change).
        from victor.tools.subprocess_executor import _resolve_default_sandbox

        _sandbox = _resolve_default_sandbox()
        if _sandbox.type_name != "none":
            _wrapped = _sandbox.wrap_argv(["/bin/sh", "-c", cmd], Path(cwd) if cwd else None)
            process = await asyncio.create_subprocess_exec(
                *_wrapped,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        else:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

        # Wait for completion with timeout. When a live progress sink is active,
        # stream stdout/stderr incrementally so the UI shows output as it is
        # produced; otherwise use the cheaper single communicate() call.
        from victor.framework.tool_progress import emit_tool_progress, has_progress_sink

        _streaming_progress = has_progress_sink()
        try:
            if _streaming_progress:
                stdout, stderr = await asyncio.wait_for(
                    _stream_subprocess_output(process, _PROGRESS_TOOL_NAME),
                    timeout=timeout,
                )
            else:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds",
                "stdout": "",
                "stderr": "",
                "return_code": -1,
            }

        if _streaming_progress:
            emit_tool_progress(name=_PROGRESS_TOOL_NAME, is_final=True)

        stdout_str = stdout.decode("utf-8") if stdout else ""
        stderr_str = stderr.decode("utf-8") if stderr else ""

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
            "truncated": was_truncated,
            "stdout_lines": stdout_lines,
            "stderr_lines": stderr_lines,
        }

        # Cache successful read-only command results
        if process.returncode == 0 and not dangerous and _is_readonly_command(cmd):
            try:
                cache = get_shell_cache()
                cache.set(cmd, (process.returncode, stdout_str, stderr_str), cwd)
            except Exception as cache_error:
                logger.warning(f"Failed to cache command result: {cache_error}")

        # Include informative error message when command fails
        if process.returncode != 0:
            error_parts = []
            error_parts.append(f"Command failed with exit code {process.returncode}")
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
