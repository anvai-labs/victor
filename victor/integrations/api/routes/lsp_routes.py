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

"""LSP routes: /lsp/{completions,hover,definition,references,diagnostics}.

Ported from the legacy aiohttp server so the FastAPI server is the single canonical API
surface for the VS Code extension. The handlers resolve the LSP manager through the
``CapabilityRegistry`` (server-independent) and degrade gracefully to an empty result with
an ``error`` field when the LSP capability is not registered — matching the prior behavior.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

logger = logging.getLogger(__name__)


class LspPositionRequest(BaseModel):
    """A file position request shared by completions/hover/definition/references."""

    file: str = ""
    line: int = 0
    character: int = 0


class LspDiagnosticsRequest(BaseModel):
    file: str = ""


def _get_lsp_manager() -> Optional[Any]:
    """Resolve the registered LSP manager, or None when the capability is unavailable."""
    from victor.core.capability_registry import CapabilityRegistry
    from victor.framework.vertical_protocols import LSPManagerProtocol

    provider = CapabilityRegistry.get_instance().get(LSPManagerProtocol)
    if provider is None:
        return None
    return provider.get_lsp_manager()


def create_router(server: "VictorFastAPIServer") -> APIRouter:
    """Create LSP routes. Handlers are server-independent (capability-resolved)."""
    router = APIRouter()

    @router.post("/lsp/completions", tags=["LSP"])
    async def lsp_completions(req: LspPositionRequest) -> JSONResponse:
        try:
            manager = _get_lsp_manager()
            if manager is None:
                return JSONResponse({"completions": [], "error": "LSP not available"})
            completions = await manager.get_completions(req.file, req.line, req.character)
            return JSONResponse(
                {
                    "completions": [
                        {
                            "label": c.label,
                            "kind": c.kind,
                            "detail": c.detail,
                            "insert_text": c.insert_text,
                        }
                        for c in completions
                    ]
                }
            )
        except Exception as exc:  # noqa: BLE001 - graceful degradation matches prior behavior
            logger.exception("LSP completions error")
            return JSONResponse({"completions": [], "error": str(exc)})

    @router.post("/lsp/hover", tags=["LSP"])
    async def lsp_hover(req: LspPositionRequest) -> JSONResponse:
        try:
            manager = _get_lsp_manager()
            if manager is None:
                return JSONResponse({"contents": None, "error": "LSP not available"})
            hover = await manager.get_hover(req.file, req.line, req.character)
            return JSONResponse({"contents": hover.contents if hover else None})
        except Exception as exc:  # noqa: BLE001
            logger.exception("LSP hover error")
            return JSONResponse({"contents": None, "error": str(exc)})

    @router.post("/lsp/definition", tags=["LSP"])
    async def lsp_definition(req: LspPositionRequest) -> JSONResponse:
        try:
            manager = _get_lsp_manager()
            if manager is None:
                return JSONResponse({"locations": [], "error": "LSP not available"})
            locations = await manager.get_definition(req.file, req.line, req.character)
            return JSONResponse({"locations": locations})
        except Exception as exc:  # noqa: BLE001
            logger.exception("LSP definition error")
            return JSONResponse({"locations": [], "error": str(exc)})

    @router.post("/lsp/references", tags=["LSP"])
    async def lsp_references(req: LspPositionRequest) -> JSONResponse:
        try:
            manager = _get_lsp_manager()
            if manager is None:
                return JSONResponse({"locations": [], "error": "LSP not available"})
            locations = await manager.get_references(req.file, req.line, req.character)
            return JSONResponse({"locations": locations})
        except Exception as exc:  # noqa: BLE001
            logger.exception("LSP references error")
            return JSONResponse({"locations": [], "error": str(exc)})

    @router.post("/lsp/diagnostics", tags=["LSP"])
    async def lsp_diagnostics(req: LspDiagnosticsRequest) -> JSONResponse:
        try:
            manager = _get_lsp_manager()
            if manager is None:
                return JSONResponse({"diagnostics": [], "error": "LSP not available"})
            diagnostics = manager.get_diagnostics(req.file)
            return JSONResponse({"diagnostics": diagnostics})
        except Exception as exc:  # noqa: BLE001
            logger.exception("LSP diagnostics error")
            return JSONResponse({"diagnostics": [], "error": str(exc)})

    return router
