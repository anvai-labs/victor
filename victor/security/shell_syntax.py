# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Shared lexical recognition of shell file writes.

This scanner recognizes redirection operators and shell quoting; it does not
execute expansions or claim to parse arbitrary shell programs. Dynamic targets
stay explicit so callers can reject them or require approval.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

MAX_SHELL_NESTING = 32
SUBSTITUTION_VALUE = "__victor_shell_substitution__"


class ShellSyntaxError(ValueError):
    """Shell input cannot be analyzed within the supported lexical bounds."""


@dataclass(frozen=True)
class ShellWrite:
    """A file-opening redirection, with its unexpanded destination."""

    operator: str
    target: str
    dynamic: bool = False


def _target(command: str, start: int) -> tuple[str, bool, int]:
    i = start
    while i < len(command) and command[i] in " \t":
        i += 1
    chars: list[str] = []
    quote = ""
    dynamic = False
    while i < len(command):
        char = command[i]
        if not quote and (char.isspace() or char in ";|&<>()"):
            break
        if char == "\\" and quote != "'" and i + 1 < len(command):
            chars.append(command[i + 1])
            i += 2
            continue
        if char in "'\"" and (not quote or char == quote):
            quote = "" if quote else char
        else:
            if quote != "'" and (char in "$`" or (not quote and char in "*?[{")):
                dynamic = True
            chars.append(char)
        i += 1
    target = "".join(chars)
    return target, dynamic or bool(quote) or SUBSTITUTION_VALUE in target, i


def _scan_plain_file_writes(command: str) -> list[ShellWrite]:
    """Find file-writing redirects, including adjacent operators and ``<>``.

    Quoted or escaped operator text and descriptor duplication/closure do not
    open files. Targets are decoded for quotes/escapes but never expanded.
    """
    writes: list[ShellWrite] = []
    i = 0
    quote = ""
    while i < len(command):
        char = command[i]
        if char == "\\" and quote != "'":
            i += 2
            continue
        if char in "'\"" and (not quote or char == quote):
            quote = "" if quote else char
            i += 1
            continue
        if quote:
            i += 1
            continue
        operator = ""
        for candidate in ("&>>", "&>", ">>", ">|", ">&", "<>", ">"):
            if command.startswith(candidate, i):
                operator = candidate
                break
        if not operator:
            i += 1
            continue
        target, dynamic, end = _target(command, i + len(operator))
        if not (operator == ">&" and not dynamic and (target.isdigit() or target == "-")):
            writes.append(ShellWrite(operator, target, dynamic))
        i = max(end, i + len(operator))
    return writes


def _backtick_body(command: str, start: int) -> tuple[str, int]:
    """Decode the backslash layer removed by legacy backtick substitution."""
    body: list[str] = []
    i = start + 1
    while i < len(command):
        char = command[i]
        if char == "`":
            return "".join(body), i + 1
        if char == "\\" and i + 1 < len(command) and command[i + 1] in "$`\\\n":
            if command[i + 1] != "\n":
                body.append(command[i + 1])
            i += 2
            continue
        body.append(char)
        i += 1
    raise ShellSyntaxError("unclosed backtick substitution")


def _parenthesized_body(command: str, start: int, depth: int) -> tuple[str, int]:
    if depth > MAX_SHELL_NESTING:
        raise ShellSyntaxError("shell substitution nesting limit exceeded")
    i = start + 1
    parens = 1
    quote = ""
    while i < len(command):
        char = command[i]
        if (
            char == "#"
            and not quote
            and (i == 0 or command[i - 1].isspace() or command[i - 1] in ";|&()")
        ):
            newline = command.find("\n", i)
            if newline < 0:
                raise ShellSyntaxError("unclosed parenthesized substitution")
            i = newline + 1
            continue
        if char == "\\" and quote != "'":
            i += 2
            continue
        if quote == "'":
            if char == "'":
                quote = ""
            i += 1
            continue
        if char == "`":
            _, i = _backtick_body(command, i)
            continue
        if command.startswith("$(", i) or (not quote and command[i : i + 2] in {"<(", ">("}):
            _, i = _parenthesized_body(command, i + 1, depth + 1)
            continue
        if char in "'\"" and (not quote or char == quote):
            quote = "" if quote else char
        elif not quote:
            if char == "(":
                parens += 1
            elif char == ")":
                parens -= 1
                if parens == 0:
                    return command[start + 1 : i], i + 1
        i += 1
    raise ShellSyntaxError("unclosed parenthesized substitution")


def scan_substitutions(command: str, *, depth: int = 0) -> tuple[list[str], str]:
    """Extract executable substitutions while preserving literal shell words.

    Each direct child is returned as shell source for recursive effect analysis.
    Sanitization uses an unknown word, never an executable allowed builtin: a
    substitution in command position cannot inherit permission from its output.
    """
    if depth > MAX_SHELL_NESTING:
        raise ShellSyntaxError("shell substitution nesting limit exceeded")
    command = strip_shell_comments(command)
    children: list[str] = []
    sanitized: list[str] = []
    quote = ""
    i = 0
    while i < len(command):
        char = command[i]
        if char == "\\" and quote != "'" and i + 1 < len(command):
            sanitized.extend(command[i : i + 2])
            i += 2
            continue
        if char in "'\"" and (not quote or char == quote):
            quote = "" if quote else char
            sanitized.append(char)
            i += 1
            continue
        if quote != "'":
            if command.startswith("$(", i) or (not quote and command[i : i + 2] in {"<(", ">("}):
                body, i = _parenthesized_body(command, i + 1, depth + 1)
                children.append(body)
                sanitized.append(SUBSTITUTION_VALUE)
                continue
            if char == "`":
                body, i = _backtick_body(command, i)
                children.append(body)
                sanitized.append(SUBSTITUTION_VALUE)
                continue
        sanitized.append(char)
        i += 1
    if quote:
        raise ShellSyntaxError("unclosed shell quote")
    return children, "".join(sanitized)


def scan_file_writes(command: str, *, depth: int = 0) -> list[ShellWrite]:
    """Find writes in both the outer command and every executable substitution."""
    children, sanitized = scan_substitutions(command, depth=depth)
    writes = _scan_plain_file_writes(sanitized)
    for child in children:
        writes.extend(scan_file_writes(child, depth=depth + 1))
    return writes


def split_shell_commands(cmd: str) -> list[str]:
    """Split a compound command into individual command segments.

    Handles ``&&``, ``||``, ``;`` and pipeline ``|`` operators while respecting
    quoted strings. Every returned segment is validated independently for
    readonly mode, so a safe command cannot hide a mutating command later in a
    chain or pipeline.
    """
    cmd = strip_shell_comments(cmd)
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

        if char == "\\" and in_quote != "'":
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
            # Newlines and background separators are command boundaries too.
            if char == "\n" or (
                char == "&" and (i == 0 or cmd[i - 1] not in "<>") and not cmd.startswith("&>", i)
            ):
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


_STRUCTURAL_WORDS = frozenset(
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
_LOOP_HEADERS = frozenset({"for", "select", "case"})
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def strip_shell_prefix(component: str) -> str | None:
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
        # Preserve malformed input for the command gate rather than attempting
        # to strip prefixes from a different tokenization.
        return s
    if not tokens:
        return None
    # A loop/case header (`for VAR in WORDS`, `case WORD in`) runs no command.
    if tokens[0].lower() in _LOOP_HEADERS:
        return None
    idx = 0
    while idx < len(tokens) and tokens[idx].lower() in _STRUCTURAL_WORDS:
        idx += 1
    while idx < len(tokens) and _ASSIGNMENT.match(tokens[idx]):
        idx += 1
    if idx >= len(tokens):
        return None
    if idx == 0:
        return s
    # Preserve the residual shell text: rebuilding from split tokens erases
    # quotes in SSH options, redirects and literal arguments to exec wrappers.
    lexer = shlex.shlex(s, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    for _ in range(idx):
        lexer.get_token()
    return s[lexer.instream.tell() :]


@dataclass(frozen=True)
class ShellWord:
    """One lexical word, retaining whether its value/arity comes from expansion."""

    value: str
    dynamic: bool = False
    field_split: bool = False
    fixed_prefix: str = ""


def shell_words(command: str) -> list[ShellWord]:
    """Decode words while retaining quote/expansion provenance for argv checks."""
    words: list[ShellWord] = []
    chars: list[str] = []
    quote = ""
    dynamic = False
    field_split = False
    fixed_prefix: str | None = None
    started = False
    i = 0
    while i < len(command):
        char = command[i]
        if not quote and char.isspace():
            if started:
                words.append(ShellWord("".join(chars), dynamic, field_split, fixed_prefix or ""))
            chars, dynamic, field_split, fixed_prefix, started = [], False, False, None, False
            i += 1
            continue
        started = True
        if char == "\\" and quote != "'" and i + 1 < len(command):
            chars.append(command[i + 1])
            i += 2
            continue
        if char in "'\"" and (not quote or char == quote):
            quote = "" if quote else char
            i += 1
            continue
        marker = command.startswith(SUBSTITUTION_VALUE, i)
        brace_expansion = (
            char == "{"
            and "}" in command[i:]
            and any(separator in command[i : command.find("}", i)] for separator in (",", ".."))
        )
        if marker or (
            quote != "'" and (char == "$" or (not quote and (char in "*?[" or brace_expansion)))
        ):
            if fixed_prefix is None:
                fixed_prefix = "".join(chars)
            dynamic = True
            field_split |= not bool(quote)
        if marker:
            chars.extend(SUBSTITUTION_VALUE)
            i += len(SUBSTITUTION_VALUE)
        else:
            chars.append(char)
            i += 1
    if quote:
        raise ShellSyntaxError("unclosed shell quote")
    if started:
        words.append(ShellWord("".join(chars), dynamic, field_split, fixed_prefix or ""))
    return words


def shell_assignments(command: str) -> list[str]:
    """Expose assignment prefixes and loop bindings before structural stripping."""
    words = shlex.split(command)
    if words[:1] in [["for"], ["select"]] and len(words) > 1:
        return [words[1] + "="]
    index = 0
    while index < len(words) and words[index] in _STRUCTURAL_WORDS:
        index += 1
    assignments: list[str] = []
    while index < len(words) and _ASSIGNMENT.match(words[index]):
        assignments.append(words[index])
        index += 1
    return assignments


def strip_shell_comments(command: str) -> str:
    """Remove only unquoted comments beginning at a shell word boundary."""
    output: list[str] = []
    quote = ""
    i = 0
    while i < len(command):
        char = command[i]
        if char == "\\" and quote != "'" and i + 1 < len(command):
            output.extend(command[i : i + 2])
            i += 2
            continue
        if char in "'\"" and (not quote or char == quote):
            quote = "" if quote else char
        if (
            char == "#"
            and not quote
            and (i == 0 or command[i - 1].isspace() or command[i - 1] in ";|&()")
        ):
            newline = command.find("\n", i)
            if newline < 0:
                break
            i = newline
            continue
        output.append(char)
        i += 1
    return "".join(output)
