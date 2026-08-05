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

"""Tests for FastAPI router plugin registration."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.testclient import TestClient

from victor.integrations.api import fastapi_server
from victor.integrations.api.fastapi_server import collect_route_paths
from victor.integrations.api.router_plugins import APIRouterRegistration


def _route_paths(app) -> set[str]:
    # Use the production collector so the assertions are robust across Starlette
    # versions (newer Starlette stores lazily-included routers as wrappers in
    # app.routes rather than eagerly flattening their sub-routes).
    return collect_route_paths(app)


def _create_test_lsp_router() -> APIRouter:
    router = APIRouter(tags=["LSP"])

    @router.post("/lsp/completions")
    async def lsp_completions() -> dict[str, list]:
        return {"completions": []}

    @router.post("/lsp/hover")
    async def lsp_hover() -> dict[str, None]:
        return {"contents": None}

    @router.post("/lsp/definition")
    async def lsp_definition() -> dict[str, list]:
        return {"locations": []}

    @router.post("/lsp/references")
    async def lsp_references() -> dict[str, list]:
        return {"locations": []}

    @router.post("/lsp/diagnostics")
    async def lsp_diagnostics() -> dict[str, list]:
        return {"diagnostics": []}

    return router


def test_router_registration_can_define_expected_lsp_routes() -> None:
    """Router registrations can expose the public LSP surface."""
    router = _create_test_lsp_router()
    paths = _route_paths(router)
    assert "/lsp/completions" in paths
    assert "/lsp/hover" in paths
    assert "/lsp/definition" in paths
    assert "/lsp/references" in paths
    assert "/lsp/diagnostics" in paths


def test_fastapi_server_includes_core_lsp_routes_without_router_plugins(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Core server registers the LSP routes itself (consolidated from the old aiohttp server).

    Previously LSP was provided only by external router plugins; after consolidating the
    two HTTP servers, ``/lsp/*`` is a first-class core surface that degrades gracefully when
    no LSPManager capability is registered. The ``lsp`` capability is therefore reported even
    with no router plugins present.
    """
    monkeypatch.setattr(
        fastapi_server,
        "load_fastapi_router_registrations",
        lambda *, workspace_root: [],
    )

    server = fastapi_server.VictorFastAPIServer(
        workspace_root=str(tmp_path),
        enable_graphql=False,
    )

    assert "/lsp/completions" in _route_paths(server.app)
    with TestClient(server.app) as client:
        response = client.get("/status")
    assert response.status_code == 200
    capabilities = response.json().get("capabilities", [])
    assert "lsp" in capabilities


def test_fastapi_server_includes_lsp_routes_from_router_plugin(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """LSP routes should be provided through router plugin registrations."""
    registrations = [
        APIRouterRegistration(
            router=_create_test_lsp_router(),
            prefix="",
            entry_point_name="coding",
            entry_point_value="victor_coding.api.router_provider:get_fastapi_router_provider",
        )
    ]
    monkeypatch.setattr(
        fastapi_server,
        "load_fastapi_router_registrations",
        lambda *, workspace_root: registrations,
    )

    server = fastapi_server.VictorFastAPIServer(
        workspace_root=str(tmp_path),
        enable_graphql=False,
    )

    assert "/lsp/completions" in _route_paths(server.app)
    with TestClient(server.app) as client:
        response = client.get("/status")
    assert response.status_code == 200
    capabilities = response.json().get("capabilities", [])
    assert "lsp" in capabilities
