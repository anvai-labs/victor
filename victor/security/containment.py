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
from contextlib import contextmanager
from enum import Enum
from typing import Any, Iterator, Optional

# Envelope delimiters. Content that itself contains the closing delimiter is
# neutralized (a space is inserted into the tag) so an envelope closes exactly
# once, at the end — model-visible content stays readable and structural
# integrity is preserved.
ENVELOPE_OPEN = "<untrusted"
ENVELOPE_CLOSE = "</untrusted>"
_NEUTRALIZED_CLOSE = "< /untrusted>"

_CANARY_PREFIX = "victor-canary-"
_CANARY_HEX_LEN = 16

# Fail-closed publication: when the containment envelope itself cannot be
# built (enforcement configured), the model-bound output is replaced by this
# inert notice instead of silently delivering unenveloped untrusted text.
CONTAINMENT_FAILURE_PLACEHOLDER = (
    "[untrusted-content containment failure: tool output withheld from model "
    "context (ADR-037 fail-closed).]"
)

_session_canary: ContextVar[Optional[str]] = ContextVar("victor_session_canary", default=None)

# Session-owner binding (Codex review fix wave 1, finding 1): the owning
# session object holds one stable SessionContainmentState and binds it around
# each turn; dispatch reads the canary from the bound state. The legacy
# ``_session_canary`` var below remains the transport-level override used by
# direct callers and tests.
_session_containment: ContextVar[Optional["SessionContainmentState"]] = ContextVar(
    "victor_session_containment", default=None
)

# Evidence bound: flagged injection families remembered per session are a
# bounded set (the classifier family vocabulary is small; the cap exists so a
# pathological future family source cannot grow owner state without limit).
MAX_FLAGGED_FAMILIES = 32


class SessionContainmentState:
    """Session-owned containment state: one stable canary + bounded evidence.

    The session owner (``RuntimeIdentity``) creates one instance per session
    id and binds it around every turn via :func:`bind_containment_state`. The
    canary is stable for the lifetime of the session and regenerated when the
    owner moves to a different session (resume with a new id, member bind), so
    child sessions never share a canary with their parent.
    """

    __slots__ = ("session_id", "canary", "_flagged_families")

    def __init__(self, session_id: str = "") -> None:
        self.session_id = session_id
        self.canary = new_session_canary()
        self._flagged_families: frozenset = frozenset()

    @property
    def flagged_families(self) -> frozenset:
        """Injection families the screen flagged in this session's output."""
        return self._flagged_families

    def record_flagged_families(self, families: Any) -> None:
        """Record flagged families, bounded by :data:`MAX_FLAGGED_FAMILIES`."""
        incoming = frozenset(families or ())
        if not incoming:
            return
        merged = self._flagged_families | incoming
        if len(merged) > MAX_FLAGGED_FAMILIES:
            merged = frozenset(sorted(merged)[:MAX_FLAGGED_FAMILIES])
        self._flagged_families = merged

    def clear_flagged_families(self) -> None:
        """Drop remembered flagged families (session exit / explicit reset)."""
        self._flagged_families = frozenset()


def current_containment_state() -> Optional[SessionContainmentState]:
    """Return the session-owned containment state bound for this context."""
    return _session_containment.get()


@contextmanager
def bind_containment_state(state: SessionContainmentState) -> Iterator[None]:
    """Bind a session's containment state for the current execution context.

    The session owner calls this around each turn (and around each resumed
    stream step); on exit the previous binding is restored, so a fresh request
    task never inherits another session's canary or flagged-family evidence.
    """
    token = _session_containment.set(state)
    try:
        yield
    finally:
        _session_containment.reset(token)


class UntrustedSourceClass(str, Enum):
    """Provenance class of untrusted content entering model context."""

    FILE = "file"
    WEB = "web"
    MCP = "mcp"
    PLUGIN = "plugin"
    TOOL = "tool"
    #: Long-term preference/episode memory recalled into the prompt
    #: (FEP-0048): user-authored free-form content, so it enters context as
    #: provenance-tagged data under the same envelope contract (ADR-037),
    #: never as raw system-prompt input.
    MEMORY = "memory"


def new_session_canary() -> str:
    """Generate a fresh unique canary token for a session."""
    return f"{_CANARY_PREFIX}{secrets.token_hex(_CANARY_HEX_LEN // 2)}"


def set_session_canary(canary: str) -> None:
    """Bind the session canary for the current execution context (transport override).

    Production code binds the whole :class:`SessionContainmentState` through
    :func:`bind_containment_state`; this setter remains for direct callers
    and tests that need the canary alone.
    """
    _session_canary.set(canary)


def current_session_canary() -> Optional[str]:
    """Return the session canary bound for the current execution context.

    Prefers the session-owned containment state when one is bound; falls back
    to the transport-level setter value.
    """
    state = _session_containment.get()
    if state is not None:
        return state.canary
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


def find_canary_in_value(value: Any, canary: str, _path: str = "") -> Optional[str]:
    """Return the dotted path of the first occurrence of ``canary`` in ``value``.

    Deterministic exfiltration check (ADR-037 decision 3): walks the FULL
    serialized tool-argument structure looking for the exact canary token —
    string values, mapping KEYS, nested containers, and the string form of
    non-JSON-able leaves (arguments are serialized with ``default=str`` on
    the wire, so a custom object's ``str()`` is part of the outbound
    structure). Returns ``None`` when absent.

    Args:
        value: The structure to scan (tool arguments).
        canary: The exact session canary token to look for.
        _path: Internal accumulator for the dotted report path.

    Returns:
        A dotted path such as ``command`` or ``files[0].content`` naming
        where the canary appears, or ``None``. Key hits are reported with a
        redacted ``<key>`` path component so the (model-visible) skip reason
        never echoes the canary token itself.
    """
    if isinstance(value, str):
        if canary and canary in value:
            return _path or "<root>"
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{_path}.{key}" if _path else str(key)
            if canary and canary in (key if isinstance(key, str) else str(key)):
                # Redact the key text: the path flows into model-visible
                # refusal copy and must not carry the canary back to the model.
                redacted = f"{_path}.<key>" if _path else "<key>"
                return redacted
            hit = find_canary_in_value(item, canary, child)
            if hit is not None:
                return hit
        return None
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{_path}[{index}]" if _path else f"[{index}]"
            hit = find_canary_in_value(item, canary, child)
            if hit is not None:
                return hit
        return None
    # Non-container, non-string leaf: the serialized argument structure
    # includes str() of such objects, so scan that form too.
    if canary and canary in str(value):
        return _path or "<root>"
    return None


__all__ = [
    "UntrustedSourceClass",
    "ENVELOPE_OPEN",
    "ENVELOPE_CLOSE",
    "CONTAINMENT_FAILURE_PLACEHOLDER",
    "MAX_FLAGGED_FAMILIES",
    "SessionContainmentState",
    "new_session_canary",
    "set_session_canary",
    "current_session_canary",
    "reset_session_canary",
    "current_containment_state",
    "bind_containment_state",
    "wrap_untrusted",
    "find_canary_in_value",
]
