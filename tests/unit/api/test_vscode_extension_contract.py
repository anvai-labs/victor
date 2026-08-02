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

"""Contract test: the VS Code extension's REST client vs the FastAPI backend routes.

The extension (`vscode-victor/src/victorClient.ts`) is a thin HTTP client over the
victor FastAPI server. This test extracts every endpoint the extension calls and
verifies it is registered on the FastAPI app, so the two can't silently drift apart.

The router factories only touch the server at request time, so the full route table is
built with a mock server (no orchestrator / live server needed).
"""

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_VICTOR_CLIENT_TS = (
    Path(__file__).resolve().parents[3] / "vscode-victor" / "src" / "victorClient.ts"
)

# Endpoints the extension calls that the FastAPI backend does NOT expose. After
# consolidating the two servers into one (the legacy aiohttp server was removed and its
# /lsp/* routes ported to FastAPI, and /credentials/{set,delete,status} + /tools/cancel
# were added), this is EMPTY — every extension endpoint is served by the FastAPI app.
# Any future drift (a new extension endpoint without a backend route) fails the exact-match
# assertion below.
KNOWN_BACKEND_GAPS: set[tuple[str, str]] = set()

_HTTP_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH"}


def _normalize(path: str) -> str:
    """Collapse path params so backend `/x/{id}` matches the extension's `/x/${id}`."""
    path = re.sub(r"\$\{[^}]+\}", "{}", path)  # ${agentId} -> {}
    path = re.sub(r"\{[^}]+\}", "{}", path)  # {agent_id}  -> {}
    return path.rstrip("/") or "/"


def _backend_routes() -> set[tuple[str, str]]:
    from victor.integrations.api.routes import create_all_routers

    routes: set[tuple[str, str]] = set()
    for router in create_all_routers(MagicMock()):
        for route in getattr(router, "routes", []):
            path = getattr(route, "path", None)
            if not path:
                continue
            for method in getattr(route, "methods", set()) or set():
                if method in _HTTP_METHODS:
                    routes.add((method, _normalize(path)))
    return routes


def _extension_endpoints() -> set[tuple[str, str]]:
    text = _VICTOR_CLIENT_TS.read_text(encoding="utf-8")
    pattern = re.compile(
        r"this\.client\.(get|post|put|delete|patch)\(\s*[`'\"]([^`'\"]+)[`'\"]",
        re.IGNORECASE,
    )
    return {(m.upper(), _normalize(p)) for m, p in pattern.findall(text)}


@pytest.mark.skipif(not _VICTOR_CLIENT_TS.exists(), reason="vscode-victor extension not present")
class TestVSCodeExtensionFastAPIContract:
    def test_extension_endpoints_extracted(self):
        endpoints = _extension_endpoints()
        # Sanity: the client defines a substantial, recognizable surface.
        assert len(endpoints) >= 50
        assert ("POST", "/chat") in endpoints
        assert ("GET", "/models") in endpoints

    def test_backend_exposes_routes(self):
        routes = _backend_routes()
        assert len(routes) >= 50
        assert ("POST", "/chat") in routes

    def test_every_extension_endpoint_is_on_the_backend_except_tracked_gaps(self):
        backend = _backend_routes()
        extension = _extension_endpoints()
        missing = extension - backend

        # Exact match keeps the allowlist honest in both directions:
        #  - a NEW endpoint the backend lacks -> missing has an un-allowlisted entry -> fail
        #  - a gap the backend later fills    -> KNOWN_BACKEND_GAPS goes stale       -> fail
        assert missing == KNOWN_BACKEND_GAPS, (
            "VS Code extension <-> FastAPI contract drift.\n"
            f"  newly missing on backend: {sorted(missing - KNOWN_BACKEND_GAPS)}\n"
            f"  stale allowlist entries (now on backend): {sorted(KNOWN_BACKEND_GAPS - missing)}"
        )

    def test_majority_of_extension_endpoints_are_satisfied(self):
        backend = _backend_routes()
        extension = _extension_endpoints()
        satisfied = extension & backend
        # The contract should be overwhelmingly satisfied (only the tracked gaps remain).
        assert len(satisfied) >= len(extension) - len(KNOWN_BACKEND_GAPS)
