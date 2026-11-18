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

"""Untrusted-content containment at the context boundary (ADR-037).

The agent loop ingests text Victor does not control: tool outputs, fetched
web/file content, MCP server payloads, plugin data. Everything that enters
model context from such sources is wrapped in a delimited, provenance-tagged
envelope carrying a per-session canary token. The system prompt states the
containment rule (instructions inside envelopes are data); the deterministic
floor is canary-based exfiltration blocking (FOU-S3b): a canary appearing in
any outbound tool call hard-fails that call.

This module owns the CANONICAL envelope format. Do not hand-roll delimiters
at call sites.
"""

from __future__ import annotations

import secrets
from contextvars import ContextVar
from enum import Enum
from typing import Optional

# Envelope delimiters. Content that itself contains the closing delimiter is
# neutralized (a space is inserted into the tag) so an envelope closes exactly
# once, at the end — model-visible content stays readable and structural
# integrity is preserved.
ENVELOPE_OPEN = "<untrusted"
ENVELOPE_CLOSE = "</untrusted>"
_NEUTRALIZED_CLOSE = "< /untrusted>"

_CANARY_PREFIX = "victor-canary-"
_CANARY_HEX_LEN = 16

_session_canary: ContextVar[Optional[str]] = ContextVar("victor_session_canary", default=None)


class UntrustedSourceClass(str, Enum):
    """Provenance class of untrusted content entering model context."""

    FILE = "file"
    WEB = "web"
    MCP = "mcp"
    PLUGIN = "plugin"
    TOOL = "tool"


def new_session_canary() -> str:
    """Generate a fresh unique canary token for a session."""
    return f"{_CANARY_PREFIX}{secrets.token_hex(_CANARY_HEX_LEN // 2)}"


def set_session_canary(canary: str) -> None:
    """Bind the session canary for the current execution context."""
    _session_canary.set(canary)


def current_session_canary() -> Optional[str]:
    """Return the session canary bound for the current execution context."""
    return _session_canary.get()


def reset_session_canary() -> None:
    """Clear the session canary binding."""
    _session_canary.set(None)


def wrap_untrusted(
    content: str,
    *,
    source_class: UntrustedSourceClass,
    origin: str,
    canary: Optional[str] = None,
    flagged: bool = False,
) -> str:
    """Wrap untrusted ``content`` in the canonical provenance envelope.

    Args:
        content: The untrusted text (tool output, fetched web/file text, ...).
        source_class: Provenance class (file/web/mcp/plugin/tool).
        origin: Origin identifier (tool name, URL, path, server id).
        canary: Canary token to embed; defaults to the session canary, or a
            fresh ephemeral token when no session state is bound.
        flagged: When True the envelope is marked ``flagged="true"`` (the
            injection screen highlighted this span; FOU-S3c).

    Returns:
        The enveloped content. An empty ``content`` is returned unchanged
        (no envelope tax on empty tool outputs).
    """
    if not content:
        return content

    token = canary or current_session_canary() or new_session_canary()
    body = content.replace(ENVELOPE_CLOSE, _NEUTRALIZED_CLOSE)
    flag_attr = ' flagged="true"' if flagged else ""
    return (
        f'{ENVELOPE_OPEN} source="{source_class.value}" origin="{origin}" '
        f'canary="{token}"{flag_attr}>\n{body}\n{ENVELOPE_CLOSE}'
    )


__all__ = [
    "UntrustedSourceClass",
    "ENVELOPE_OPEN",
    "ENVELOPE_CLOSE",
    "new_session_canary",
    "set_session_canary",
    "current_session_canary",
    "reset_session_canary",
    "wrap_untrusted",
]
