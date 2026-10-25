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

"""Code-graph explorer routes (ADR-033): the ONE explorer implementation.

Mounted by the main Victor server as a standard routes factory (the
``workflow_routes`` pattern - ``server.workspace_root`` locates the
project DB), and re-mounted by the standalone ``victor viz serve``
wrapper (``viz_server.py``). Every route is a read-only query; the
asset route is path-safety-guarded (resolve-and-prefix-check, the
observability ``serve_static`` guard) with a distinct ``/viz/assets/``
prefix so it cannot collide with observability's catch-all
``/static/{path}`` route where FastAPI match order is not guaranteed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from victor.core.graph_rag import viz_queries

_STATIC_VIZ_DIR = Path(__file__).parent.parent / "static" / "viz"
_VENDOR_DIR = Path(__file__).parent.parent / "templates" / "vendor"
# Vendor files reused from the workflow visualizer's wheel-shipped copy
# rather than duplicated (~1MB of JS).
_VENDOR_FILES = frozenset({"cytoscape.min.js", "cytoscape-dagre.min.js"})
_LOCAL_FILES = frozenset({"explorer.js", "explorer.css"})


def _guarded_file(base_dir: Path, file_name: str) -> Path:
    """Resolve ``file_name`` under ``base_dir``; reject path escapes."""
    candidate = (base_dir / file_name).resolve()
    base = base_dir.resolve()
    if base not in candidate.parents:
        raise HTTPException(status_code=404)
    if not candidate.is_file():
        raise HTTPException(status_code=404)
    return candidate


def create_router(server: Any) -> APIRouter:
    """Create the explorer routes bound to a server-shaped object.

    ``server.workspace_root`` (str or Path) locates the workspace whose
    ``.victor/project.db`` is explored.
    """
    router = APIRouter(tags=["visualization"])
    workspace_root = Path(getattr(server, "workspace_root", ".") or ".").resolve()
    db_path = workspace_root / ".victor" / "project.db"

    def require_db() -> Path:
        if not db_path.exists():
            raise HTTPException(
                status_code=409,
                detail=(
                    "No project graph found for this workspace - run "
                    "'victor index graph index' first"
                ),
            )
        return db_path

    @router.get("/viz", response_class=HTMLResponse)
    async def explorer_page() -> HTMLResponse:
        page = _STATIC_VIZ_DIR / "index.html"
        if not page.is_file():
            return HTMLResponse(
                content="<h1>Explorer not found</h1><p>Static files not installed.</p>",
                status_code=404,
            )
        return HTMLResponse(content=page.read_text(encoding="utf-8"))

    @router.get("/viz/assets/{file_path:path}")
    async def explorer_asset(file_path: str) -> FileResponse:
        # Local explorer assets under static/viz; the two vendored JS
        # files (referenced as assets/vendor/<name>) reuse templates/vendor.
        parts = file_path.split("/")
        if len(parts) == 2 and parts[0] == "vendor":
            if parts[1] not in _VENDOR_FILES:
                raise HTTPException(status_code=404)
            return FileResponse(_guarded_file(_VENDOR_DIR, parts[1]))
        if len(parts) == 1 and parts[0] in _LOCAL_FILES:
            return FileResponse(_guarded_file(_STATIC_VIZ_DIR, parts[0]))
        raise HTTPException(status_code=404)

    @router.get("/api/graph/package")
    async def package(top: int = 12) -> JSONResponse:
        return JSONResponse(viz_queries.package_graph(require_db(), top_packages=top))

    @router.get("/api/graph/module/{package_path:path}")
    async def module(package_path: str, top: int = 15) -> JSONResponse:
        return JSONResponse(viz_queries.module_graph(require_db(), package_path, top_modules=top))

    @router.get("/api/graph/class")
    async def klass(module: str, top: int = 40) -> JSONResponse:
        return JSONResponse(viz_queries.class_graph(require_db(), module, top_symbols=top))

    @router.get("/api/graph/search")
    async def search(q: str, limit: int = 25) -> JSONResponse:
        return JSONResponse({"results": viz_queries.search(require_db(), q, limit=limit)})

    return router
