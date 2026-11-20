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

"""Long-term memory write tools (FEP-0048, FOU-A2b).

``memory_set`` and ``memory_forget`` are the user-directed write paths into
the FOU-A2a ``PreferenceEpisodicStore``: stable directives ("always use uv,
never pip") at user or project scope, deletable by id or content match.

Safety: every write is validated by the store — secret-shaped content
(``victor.security.safety.secrets`` scan) and content over the documented
byte cap are refused with typed errors that name the REASON, never the
matched secret value. The refusal is surfaced as a structured error result
so the LLM can reformulate (e.g. store the rotation status instead of the
credential).

Permissions: both tools require workspace-write
(``AccessMode.WRITE`` -> ``PermissionMode.WORKSPACE_WRITE``; explicit
entries in ``DEFAULT_TOOL_PERMISSIONS``), so read-only sessions cannot
mutate long-term memory.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Union

from victor.storage.memory.preference_episodic import (
    MemoryScope,
    MemoryWriteRefused,
    PreferenceEpisodicStore,
)
from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.decorators import tool

logger = logging.getLogger(__name__)

_VALID_SCOPES = ("user", "project")

StoreLike = PreferenceEpisodicStore
StoreFactory = Callable[[], StoreLike]

_store: Union[StoreLike, None] = None
_store_factory: Union[StoreFactory, None] = None


def _get_store() -> PreferenceEpisodicStore:
    """Lazily construct the store against the process database singletons.

    The singletons honor ``VICTOR_HOME`` / project-path overrides, so tests
    and embedders redirect storage by resetting the database managers rather
    than touching this module. ``set_memory_write_store`` overrides this for
    explicit injection.
    """
    global _store
    if _store_factory is not None:
        return _store_factory()
    if _store is None:
        _store = PreferenceEpisodicStore()
    return _store


def set_memory_write_store(store: Union[StoreLike, StoreFactory, None]) -> None:
    """Inject a store (or factory) for the memory write tools.

    Pass ``None`` to restore the lazy default. Tests use this to pin the
    tmp-path databases instead of the process singletons.
    """
    global _store, _store_factory
    if store is None:
        _store = None
        _store_factory = None
    elif callable(store) and not isinstance(store, PreferenceEpisodicStore):
        _store_factory = store
        _store = None
    else:
        _store_factory = None
        _store = store


def _refusal_result(exc: MemoryWriteRefused) -> Dict[str, Any]:
    """Structured error for a store refusal.

    The message names the refusal reason (matched secret TYPES, size vs
    cap) — never the matched secret value itself.
    """
    refusal = type(exc).__name__
    logger.info("Memory write refused (%s)", refusal)
    return {
        "success": False,
        "error": str(exc),
        "refusal": refusal,
    }


def _scope_error(scope: Any) -> Dict[str, Any]:
    return {
        "success": False,
        "error": (f"unknown memory scope {scope!r}; valid scopes: user, project"),
    }


@tool(
    category="memory",
    priority=Priority.LOW,
    access_mode=AccessMode.WRITE,  # workspace-write per FEP-0048
    danger_level=DangerLevel.LOW,
    execution_category=ExecutionCategory.WRITE,
    keywords=["memory", "remember", "preference", "store", "recall", "persist"],
    task_types=["memory"],
)
async def memory_set(content: str, scope: str = "project", source: str = "user") -> Dict[str, Any]:
    """Store a durable long-term memory (a preference or directive).

    Persists a stable fact for future sessions: user-level preferences
    ("always use uv, never pip") or project conventions ("this project uses
    pytest, never unittest"). Secret-shaped content (credentials, API keys,
    private keys) and content over the size cap are REFUSED — store a
    reference or a rotation status instead of the secret itself.

    Args:
        content: The directive or fact to remember, phrased as a rule or
                 statement (not a document).
        scope: Memory scope: "project" (default, this repository only) or
               "user" (follows the user across projects).
        source: Provenance label, e.g. "user-correction" when the user
                explicitly asked to remember something.

    Returns:
        Dict with success flag; on success includes preference_id and scope,
        on refusal includes an actionable error naming the reason.
    """
    if scope not in _VALID_SCOPES:
        return _scope_error(scope)

    store = _get_store()
    try:
        preference = store.set_preference(
            content,
            scope=MemoryScope(scope),
            source=source,
        )
    except MemoryWriteRefused as exc:
        return _refusal_result(exc)

    return {
        "success": True,
        "preference_id": preference.id,
        "scope": preference.scope.value,
        "source": preference.source,
        "message": f"Stored {preference.scope.value} preference {preference.id}",
    }


@tool(
    category="memory",
    priority=Priority.LOW,
    access_mode=AccessMode.WRITE,  # workspace-write per FEP-0048
    danger_level=DangerLevel.LOW,
    execution_category=ExecutionCategory.WRITE,
    keywords=["forget", "memory", "unlearn", "delete", "remove", "preference"],
    task_types=["memory"],
)
async def memory_forget(
    preference_id: Optional[str] = None,
    content_match: Optional[str] = None,
    scope: Optional[str] = None,
) -> Dict[str, Any]:
    """Forget stored long-term memories by id or by content match.

    Deletion is permanent. Prefer forgetting by ``preference_id`` (exact
    row); ``content_match`` deletes every memory in ``scope`` whose content
    contains the given substring, so always pass ``scope`` with it.

    Args:
        preference_id: Exact preference id to delete (as returned by
                       memory_set or /memory).
        content_match: Case-sensitive substring; deletes all matching
                       preferences in ``scope``. Empty or whitespace-only
                       matches are REFUSED (they would contains-match every
                       memory in the scope) — pass a real substring or the
                       exact ``preference_id`` instead.
        scope: Restricts deletion: "user", "project", or omit when deleting
               by id. Required with ``content_match`` so a vague match can
               never sweep both scopes.

    Returns:
        Dict with success flag and the number of forgotten memories, or an
        actionable error.
    """
    if preference_id is None and content_match is None:
        return {
            "success": False,
            "error": "memory_forget requires preference_id or content_match",
        }

    if scope is not None and scope not in _VALID_SCOPES:
        return _scope_error(scope)

    memory_scope = MemoryScope(scope) if scope is not None else None
    store = _get_store()
    try:
        deleted = store.forget(
            preference_id=preference_id,
            content_match=content_match,
            scope=memory_scope,
        )
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    return {
        "success": True,
        "forgotten": deleted,
        "message": (
            f"Forgot {deleted} memor{'y' if deleted == 1 else 'ies'}"
            if deleted
            else "No matching memories found"
        ),
    }


__all__ = [
    "memory_set",
    "memory_forget",
    "set_memory_write_store",
]
