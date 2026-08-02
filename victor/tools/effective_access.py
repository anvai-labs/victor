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

"""Per-invocation effective access-mode resolution.

Static :class:`~victor.tools.metadata.ToolMetadata` declares a tool's
*capability envelope* (shell CAN execute, code CAN run tests). Multi-command
tools are therefore classified write-capable even when a specific invocation
is a pure read (``shell readonly=True``, ``code grep …``) — which over-fires
pre-tool checkpoints and renders misleading write badges in the CLI.

This module resolves the access mode of one concrete invocation from its
arguments. Resolvers may only **narrow** the declared envelope (to READONLY),
never widen it, and any resolver failure falls back to the static metadata —
so safety-relevant consumers can trust the result is at most as permissive as
the declaration.

Consumers: agent-side tool categorization (checkpoint gating,
parallelization) and the CLI renderer (access badges).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

from victor.tools.base import AccessMode

__all__ = [
    "resolve_effective_access",
    "register_effective_access_resolver",
]

# A resolver inspects one invocation's arguments and returns an AccessMode to
# narrow to, or None when the static metadata should stand.
AccessResolver = Callable[[Dict[str, Any]], Optional[AccessMode]]

# Argument strings that could smuggle side effects past a "read-only"
# subcommand. The unified parsers reject shell operators themselves; this is a
# cheap independent guard so a resolver bug can't grant READONLY to a
# compound command.
_SUSPICIOUS_TOKENS = ("|", ";", "&&", "||", ">", "<", "`", "$(")


def _shell_resolver(arguments: Dict[str, Any]) -> Optional[AccessMode]:
    """``shell``: only ``readonly=True`` narrows — it is enforced against the
    readonly allowlist by the tool itself, so it is a trustworthy signal.
    (``action="read"`` alone is declared intent, not enforcement.)"""
    if arguments.get("readonly") is True:
        return AccessMode.READONLY
    return None


# ``code`` subcommands that only read the tree. test/execute/python stay on
# the static (MIXED) classification.
_CODE_READONLY_SUBCOMMANDS = frozenset({"grep", "search", "metrics"})


def _code_resolver(arguments: Dict[str, Any]) -> Optional[AccessMode]:
    """``code``: grep/search/metrics subcommands are pure reads."""
    cmd = arguments.get("cmd")
    if not isinstance(cmd, str):
        return None
    if any(token in cmd for token in _SUSPICIOUS_TOKENS):
        return None
    tokens = cmd.split()
    if tokens and tokens[0] == "code":
        tokens = tokens[1:]
    if tokens and tokens[0] in _CODE_READONLY_SUBCOMMANDS:
        return AccessMode.READONLY
    return None


_RESOLVERS: Dict[str, AccessResolver] = {
    "shell": _shell_resolver,
    "bash": _shell_resolver,
    "code": _code_resolver,
}


def register_effective_access_resolver(tool_name: str, resolver: AccessResolver) -> None:
    """Register (or replace) the per-invocation resolver for a tool."""
    _RESOLVERS[tool_name] = resolver


def resolve_effective_access(
    tool_name: str, arguments: Optional[Mapping[str, Any]]
) -> Optional[AccessMode]:
    """Resolve the effective access mode of one invocation.

    Args:
        tool_name: Canonical tool name.
        arguments: The invocation's arguments (may be None/empty).

    Returns:
        ``AccessMode.READONLY`` when this specific invocation is provably a
        pure read; ``None`` when the static metadata should stand. Resolvers
        can only narrow — any other resolver output is discarded.
    """
    resolver = _RESOLVERS.get(tool_name)
    if resolver is None:
        return None
    try:
        mode = resolver(dict(arguments or {}))
    except Exception:
        return None
    return mode if mode is AccessMode.READONLY else None
