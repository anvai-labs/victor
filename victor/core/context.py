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

"""Context utilities for Victor.

This module provides context-aware utilities for tracing, session management,
and other cross-cutting concerns using contextvars.
"""

from __future__ import annotations

from contextlib import contextmanager
import contextvars
import uuid
from typing import Iterator, Optional

# Trace ID for the current request/session
trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")

# Session ID for the current orchestrator session
session_id: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="")

# Turn ID for the current PERCEIVE->ACT->EVALUATE turn within a session. Together with
# session_id and request_id this forms the correlation spine that lets the capture axis
# (tool.supply offered -> tool.intent invoked -> rl_outcome resulted) be joined end-to-end.
turn_id: contextvars.ContextVar[str] = contextvars.ContextVar("turn_id", default="")

# Request ID for the current single operation (one tool call / decision) within a turn.
request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")

# Authenticated attribution identity (FEP-0020). Bound at the API-server auth
# seam (the resolved ``client_id`` becomes the subject) for the duration of a
# request so cost/usage records emitted downstream carry the authenticated
# subject/team. Unbound (None) for CLI/local use, where operator-level config
# defaults apply instead.
auth_subject_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "auth_subject_id",
    default=None,
)
auth_group_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "auth_group_id",
    default=None,
)

# Vertical name for the current operation
active_vertical: contextvars.ContextVar[str] = contextvars.ContextVar("active_vertical", default="")

# Vertical manifest version for the current operation
active_vertical_manifest_version: contextvars.ContextVar[str] = contextvars.ContextVar(
    "active_vertical_manifest_version",
    default="",
)

# Vertical plugin namespace for the current operation
active_vertical_namespace: contextvars.ContextVar[str] = contextvars.ContextVar(
    "active_vertical_namespace",
    default="",
)


def get_trace_id() -> str:
    """Get the current trace ID or generate a new one if not set."""
    tid = trace_id.get()
    if not tid:
        tid = str(uuid.uuid4())
        trace_id.set(tid)
    return tid


def set_trace_id(tid: str) -> contextvars.Token:
    """Set the current trace ID."""
    return trace_id.set(tid)


def get_session_id() -> str:
    """Get the current session ID."""
    return session_id.get()


def set_session_id(sid: str) -> contextvars.Token:
    """Set the current session ID."""
    return session_id.set(sid)


def get_turn_id() -> str:
    """Get the current turn ID (empty string if no turn is active)."""
    return turn_id.get()


def set_turn_id(tid: str) -> contextvars.Token:
    """Set the current turn ID."""
    return turn_id.set(tid)


def begin_turn() -> str:
    """Start a new turn: generate a fresh turn ID, set it, and return it.

    Call at each per-turn boundary (buffered ``execute_turn`` / streaming
    ``_stream_turn``) so capture records emitted during the turn share one
    ``turn_id``. Best-effort: callers should not let a failure here break the turn.
    """
    tid = uuid.uuid4().hex[:12]
    turn_id.set(tid)
    return tid


def get_request_id() -> str:
    """Get the current request/operation ID (empty string if none)."""
    return request_id.get()


def set_request_id(rid: str) -> contextvars.Token:
    """Set the current request/operation ID."""
    return request_id.set(rid)


def get_correlation() -> dict[str, str]:
    """Return the current correlation spine as a dict (omitting empty values).

    Convenience for stamping capture records (trace events, rl_outcome rows).
    """
    out: dict[str, str] = {}
    sid = session_id.get()
    tid = turn_id.get()
    rid = request_id.get()
    if sid:
        out["session_id"] = sid
    if tid:
        out["turn_id"] = tid
    if rid:
        out["request_id"] = rid
    return out


def get_auth_subject_id() -> Optional[str]:
    """Get the authenticated subject (user) bound to this execution context."""
    return auth_subject_id.get()


def get_auth_group_id() -> Optional[str]:
    """Get the authenticated group (team) bound to this execution context."""
    return auth_group_id.get()


@contextmanager
def bind_attribution(
    subject_id: Optional[str] = None,
    group_id: Optional[str] = None,
) -> Iterator[None]:
    """Bind an authenticated subject/group identity to the current context.

    FEP-0020 attribution join: called at the API-server auth seam with the
    ``client_id`` resolved from the presented API key, so every cost/usage
    record emitted while the request executes can attribute the spend to the
    authenticated user rather than the operator-level default.

    Only non-None values are bound — passing nothing is a no-op that leaves
    any outer binding intact. Bindings are restored on exit (including on
    error), so identity never leaks across requests.
    """
    subject_token = auth_subject_id.set(subject_id) if subject_id is not None else None
    group_token = auth_group_id.set(group_id) if group_id is not None else None
    try:
        yield
    finally:
        if group_token is not None:
            auth_group_id.reset(group_token)
        if subject_token is not None:
            auth_subject_id.reset(subject_token)


def get_active_vertical() -> str:
    """Get the current active vertical name."""
    return active_vertical.get()


def set_active_vertical(name: str) -> contextvars.Token:
    """Set the current active vertical name."""
    return active_vertical.set(name)


def get_active_vertical_manifest_version() -> str:
    """Get the current active vertical manifest version."""
    return active_vertical_manifest_version.get()


def set_active_vertical_manifest_version(version: str) -> contextvars.Token:
    """Set the current active vertical manifest version."""
    return active_vertical_manifest_version.set(version)


def get_active_vertical_namespace() -> str:
    """Get the current active vertical plugin namespace."""
    return active_vertical_namespace.get()


def set_active_vertical_namespace(namespace: str) -> contextvars.Token:
    """Set the current active vertical plugin namespace."""
    return active_vertical_namespace.set(namespace)


@contextmanager
def bind_active_vertical(
    name: str,
    *,
    manifest_version: str = "",
    namespace: str = "",
) -> Iterator[None]:
    """Bind vertical metadata to the current execution context."""

    name_token = active_vertical.set(name)
    version_token = active_vertical_manifest_version.set(manifest_version)
    namespace_token = active_vertical_namespace.set(namespace)
    try:
        yield
    finally:
        active_vertical.reset(name_token)
        active_vertical_manifest_version.reset(version_token)
        active_vertical_namespace.reset(namespace_token)
