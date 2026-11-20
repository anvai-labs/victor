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

"""Memory routes: /memory list/set/delete + /memory/proposals confirm/discard.

The user-facing write/read surface for the FOU-A2a preference/episodic store
and the FOU-A2b consolidation confirm path (FEP-0048, FOU-A2d). Auth posture
mirrors the sibling management routes (config/rl): plain management API — the
server-level auth dependency attaches at include time, no per-route scheme.

Refusals (secret-shaped / oversized content) are structured 4xx responses
naming the typed reason; the matched secret value is never echoed. Storage
and the shared consolidator are reached through the framework seam
(``victor.framework.memory``) — this client layer may not import
``victor.agent.*`` directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from victor.integrations.api.operator_posture import posture_json_response, request_principal

if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

logger = logging.getLogger(__name__)

_VALID_SCOPES = ("user", "project")


class MemorySetRequest(BaseModel):
    """Store a stable directive at user or project scope."""

    content: str
    scope: str = "user"
    source: str = "user"


def _serialize_preference(preference: Any) -> Dict[str, Any]:
    return {
        "id": preference.id,
        "scope": preference.scope.value,
        "content": preference.content,
        "source": preference.source,
        "confidence": preference.confidence,
        "created_at": preference.created_at,
        "updated_at": preference.updated_at,
    }


def _serialize_episode(episode: Any) -> Dict[str, Any]:
    return {
        "id": episode.id,
        "scope": episode.scope.value,
        "summary": episode.summary,
        "session_id": episode.session_id,
        "thread_id": episode.thread_id,
        "timestamp": episode.timestamp,
    }


def _refusal_response(exc: Exception) -> JSONResponse:
    """Structured 4xx for a store refusal.

    The message names the refusal reason (matched secret TYPES, size vs
    cap) — never the matched secret value itself.
    """
    refusal = type(exc).__name__
    logger.info("Memory write refused (%s)", refusal)
    return JSONResponse(
        {"success": False, "error": str(exc), "refusal": refusal},
        status_code=400,
    )


def _scope_error(scope: str) -> JSONResponse:
    return JSONResponse(
        {
            "success": False,
            "error": f"unknown memory scope {scope!r}; valid scopes: {', '.join(_VALID_SCOPES)}",
        },
        status_code=400,
    )


def create_router(server: "VictorFastAPIServer") -> APIRouter:
    """Create memory routes bound to *server*."""
    router = APIRouter(tags=["Memory"])

    def _posture_refusal(surface: str, http_request: Request) -> Optional[JSONResponse]:
        """Enforced single-principal posture (Codex finding 4): the /memory
        surfaces address ONE shared store with no owner in the lookup, so
        under the multi-credential posture a second credential must never
        reach them (sticky first-principal claim)."""
        return posture_json_response(
            server.claim_shared_state(request_principal(http_request), surface), surface
        )

    @router.get("/memory")
    async def list_memories(
        http_request: Request,
        scope: Optional[str] = Query(None, description="Filter preferences: user | project"),
        episode_limit: int = Query(20, ge=1, le=100),
    ) -> JSONResponse:
        """List stored preferences (both scopes by default) and recent episodes."""
        from victor.framework.memory import get_shared_memory_store
        from victor.storage.memory.preference_episodic import MemoryScope

        refusal = _posture_refusal("memory", http_request)
        if refusal is not None:
            return refusal

        if scope is not None and scope not in _VALID_SCOPES:
            return _scope_error(scope)

        try:
            store = get_shared_memory_store()
            scopes = [MemoryScope(scope)] if scope else list(MemoryScope)
            preferences = [
                _serialize_preference(p)
                for memory_scope in scopes
                for p in store.get_preferences(memory_scope)
            ]
            episodes = (
                []
                if scope == "user"
                else [_serialize_episode(e) for e in store.recent_episodes(limit=episode_limit)]
            )
            return JSONResponse({"preferences": preferences, "episodes": episodes})
        except Exception:
            logger.exception("Memory list error")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.post("/memory")
    async def set_memory(request: MemorySetRequest, http_request: Request) -> JSONResponse:
        """Store a preference; secret-shaped/oversized content is refused."""
        from victor.framework.memory import get_shared_memory_store
        from victor.storage.memory.preference_episodic import (
            MemoryScope,
            MemoryWriteRefused,
        )

        refusal = _posture_refusal("memory", http_request)
        if refusal is not None:
            return refusal

        if request.scope not in _VALID_SCOPES:
            return _scope_error(request.scope)

        try:
            preference = get_shared_memory_store().set_preference(
                request.content,
                scope=MemoryScope(request.scope),
                source=request.source,
            )
        except MemoryWriteRefused as exc:
            return _refusal_response(exc)
        except Exception:
            logger.exception("Memory set error")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

        return JSONResponse(
            {
                "success": True,
                "preference_id": preference.id,
                "scope": preference.scope.value,
                "source": preference.source,
            }
        )

    @router.delete("/memory")
    async def forget_memory(
        http_request: Request,
        preference_id: Optional[str] = Query(None, description="Exact preference id"),
        content_match: Optional[str] = Query(None, description="Content substring"),
        scope: Optional[str] = Query(None, description="Required with content_match"),
    ) -> JSONResponse:
        """Forget preferences by id, or by content match within one scope."""
        from victor.framework.memory import get_shared_memory_store
        from victor.storage.memory.preference_episodic import MemoryScope

        refusal = _posture_refusal("memory", http_request)
        if refusal is not None:
            return refusal

        if preference_id is None and content_match is None:
            return JSONResponse(
                {
                    "success": False,
                    "error": "forget requires preference_id or content_match",
                },
                status_code=400,
            )
        if scope is not None and scope not in _VALID_SCOPES:
            return _scope_error(scope)

        try:
            memory_scope = MemoryScope(scope) if scope is not None else None
            deleted = get_shared_memory_store().forget(
                preference_id=preference_id,
                content_match=content_match,
                scope=memory_scope,
            )
        except ValueError as exc:
            return JSONResponse(
                {"success": False, "error": str(exc), "refusal": type(exc).__name__},
                status_code=400,
            )
        except Exception:
            logger.exception("Memory forget error")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

        return JSONResponse({"success": True, "forgotten": deleted})

    @router.get("/memory/proposals")
    async def list_proposals(http_request: Request) -> JSONResponse:
        """List consolidation proposals staged by the end-session hook."""
        from victor.framework.memory import get_shared_memory_consolidator

        refusal = _posture_refusal("memory", http_request)
        if refusal is not None:
            return refusal

        try:
            staged = get_shared_memory_consolidator().staged_proposals()
            return JSONResponse(
                {
                    "proposals": [_serialize_episode(p) for p in staged],
                    "count": len(staged),
                }
            )
        except Exception:
            logger.exception("Memory proposal list error")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.post("/memory/proposals/confirm")
    async def confirm_proposals(http_request: Request) -> JSONResponse:
        """Commit the staged proposals as episodes (the confirm path)."""
        from victor.framework.memory import get_shared_memory_consolidator

        refusal = _posture_refusal("memory", http_request)
        if refusal is not None:
            return refusal

        try:
            stored = get_shared_memory_consolidator().commit_staged()
            return JSONResponse(
                {
                    "success": True,
                    "count": len(stored),
                    "stored": [_serialize_episode(e) for e in stored],
                }
            )
        except Exception:
            logger.exception("Memory proposal confirm error")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.post("/memory/proposals/discard")
    async def discard_proposals(http_request: Request) -> JSONResponse:
        """Drop the staged proposals without writing anything."""
        from victor.framework.memory import get_shared_memory_consolidator

        refusal = _posture_refusal("memory", http_request)
        if refusal is not None:
            return refusal

        try:
            discarded = get_shared_memory_consolidator().discard_staged()
            return JSONResponse({"success": True, "discarded": discarded})
        except Exception:
            logger.exception("Memory proposal discard error")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    return router


__all__ = ["MemorySetRequest", "create_router"]
