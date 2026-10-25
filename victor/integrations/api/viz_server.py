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

"""Standalone interactive code-graph explorer server (ADR-033 phase 1).

A lightweight FastAPI app (the ``hitl_api.py`` standalone precedent, not
the full Victor server): serves the single-page Cytoscape explorer from
``static/viz/`` plus read-only JSON endpoints over ``viz_queries``.
Read-only by construction - every route is a query; there is no write
surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from victor.config.settings import get_project_paths
from victor.core.graph_rag import viz_queries

_STATIC_DIR = Path(__file__).parent / "static" / "viz"
_VENDOR_DIR = Path(__file__).parent / "templates" / "vendor"
# Same allowlist hardening as the workflow visualizer: no arbitrary path.
_STATIC_FILES = frozenset({"index.html", "explorer.js", "explorer.css"})
_VENDOR_FILES = frozenset({"cytoscape.min.js", "cytoscape-dagre.min.js"})


def create_viz_app(workspace_root: Optional[Path] = None) -> FastAPI:
    """Build the explorer app for a workspace root."""
    root = (workspace_root or Path.cwd()).resolve()
    db_path = get_project_paths(root).project_victor_dir / "project.db"

    app = FastAPI(title="Victor Code Explorer", docs_url=None, redoc_url=None)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/static/{file_name}")
    async def static_file(file_name: str) -> FileResponse:
        if file_name not in _STATIC_FILES:
            raise HTTPException(status_code=404)
        return FileResponse(_STATIC_DIR / file_name)

    @app.get("/static/vendor/{file_name}")
    async def vendor_file(file_name: str) -> FileResponse:
        # Reuse the wheel-shipped vendored Cytoscape (the workflow
        # visualizer's copy) rather than duplicating ~1MB of JS.
        if file_name not in _VENDOR_FILES:
            raise HTTPException(status_code=404)
        return FileResponse(_VENDOR_DIR / file_name)

    def _require_db() -> Path:
        if not db_path.exists():
            raise HTTPException(
                status_code=409,
                detail="No project graph found - run 'victor index graph index' first",
            )
        return db_path

    @app.get("/api/graph/package")
    async def api_package(top: int = 12) -> JSONResponse:
        return JSONResponse(viz_queries.package_graph(_require_db(), top_packages=top))

    @app.get("/api/graph/module/{package:path}")
    async def api_module(package: str, top: int = 15) -> JSONResponse:
        return JSONResponse(viz_queries.module_graph(_require_db(), package, top_modules=top))

    @app.get("/api/graph/class")
    async def api_class(module: str, top: int = 40) -> JSONResponse:
        return JSONResponse(viz_queries.class_graph(_require_db(), module, top_symbols=top))

    @app.get("/api/graph/search")
    async def api_search(q: str, limit: int = 25) -> JSONResponse:
        return JSONResponse({"results": viz_queries.search(_require_db(), q, limit=limit)})

    return app
