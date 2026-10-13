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

import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import httpx

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


@pytest.fixture
def compiled_client_node():
    """The HTTP smoke must execute the consumer, not a handwritten request copy.

    Python-only environments may skip this cross-language smoke; the CI Guards
    job installs/compiles the client and requires it explicitly.
    """
    node = shutil.which("node")
    compiled = _VICTOR_CLIENT_TS.parents[1] / "out" / "victorClient.js"
    if not node or not compiled.exists():
        message = "VS Code HTTP smoke needs Node and npm ci && npm run compile"
        if os.environ.get("VICTOR_REQUIRE_VSCODE_HTTP_SMOKE") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return node


@pytest.fixture(params=["core", "web"])
def chat_http_server(request, monkeypatch, tmp_path):
    """Real loopback HTTP and production routes; only agent execution is doubled."""
    import uvicorn

    from victor.integrations.api import fastapi_server
    from victor.observability.request_correlation import get_request_correlation_id

    calls = []
    requests = []
    created = []
    faults = []
    stream_tools = []

    class FakeClient:
        async def initialize(self):
            pass

        async def chat(self, message):
            calls.append((message, get_request_correlation_id(), id(self)))
            return SimpleNamespace(
                content="partial draft",
                tool_calls=[],
                status="awaiting_approval",
                run_id="run-http-approval",
                approval_request={"id": "approval-http", "metadata": {"hash": "abc"}},
            )

        async def stream_chat(self, message):
            calls.append((message, get_request_correlation_id(), id(self)))
            yield SimpleNamespace(content=f"echo:{message}", tool_calls=None)
            if stream_tools:
                yield SimpleNamespace(content="", tool_calls=list(stream_tools))

        async def stream(self, message):
            calls.append((message, get_request_correlation_id(), id(self)))
            yield SimpleNamespace(event_type="content", content=f"echo:{message}")

    def create_client(*args):
        client = FakeClient()
        created.append(client)
        return client

    if request.param == "core":
        monkeypatch.setattr(fastapi_server, "load_fastapi_router_registrations", lambda **_: [])
        monkeypatch.setattr(
            "victor.workflows.hitl_api.get_default_hitl_db_path", lambda: tmp_path / "hitl.db"
        )
        server = fastapi_server.VictorFastAPIServer(
            workspace_root=str(tmp_path),
            enable_graphql=False,
            api_keys={"contract-test-key": "contract-user"},
        )
        server._victor_client = create_client()
        server._pending_tool_approvals["smoke-pending"] = {
            "tool_name": "test_no_effect",
            "resolved": False,
        }
        app = server.app
    else:
        from web.server import session_store

        monkeypatch.setattr(session_store, "set_session_store", lambda store: store)
        from web.server import main
        from web.server.session_store import InMemorySessionStore

        monkeypatch.setattr(main, "API_KEY", "contract-test-key")
        monkeypatch.setattr(main, "SESSION_STORE", InMemorySessionStore(2))
        monkeypatch.setattr(main, "VictorClient", create_client)
        app = main.app

    async def capture(scope, receive, send):
        if scope["type"] == "http" and scope["path"] == "/chat/stream":
            requests.append(scope)

            async def fault_injection(message):
                if faults and message["type"] == "http.response.body":
                    body = message.get("body", b"")
                    if b"[DONE]" in body or b'"stream_end"' in body:
                        # Simulate a proxy losing the terminal frame, leaving
                        # a clean HTTP EOF. Production handlers still execute.
                        message = {**message, "body": b""}
                await send(message)

            await app(scope, receive, fault_injection)
        else:
            await app(scope, receive, send)

    # Bind port 0 once and retain the socket: no port-selection race or shared service.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    # App startup would initialize real agents/background cleanup; not part of this
    # transport contract. Production auth, validation, handlers and SSE remain real.
    http_server = uvicorn.Server(
        uvicorn.Config(capture, log_level="error", lifespan="off", loop="asyncio")
    )
    worker = threading.Thread(target=http_server.run, kwargs={"sockets": [sock]}, daemon=True)
    worker.start()
    try:
        deadline = time.monotonic() + 10
        while not http_server.started and worker.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert http_server.started, "ephemeral HTTP server did not start"
        yield SimpleNamespace(
            kind=request.param,
            url=url,
            calls=calls,
            requests=requests,
            created=created,
            faults=faults,
            stream_tools=stream_tools,
        )
    finally:
        http_server.should_exit = True
        worker.join(timeout=10)
        sock.close()
        assert not worker.is_alive(), "ephemeral HTTP server did not stop"


def test_actual_extension_streams_two_turns_over_http(compiled_client_node, chat_http_server):
    server = chat_http_server
    runner = _VICTOR_CLIENT_TS.parents[1] / "scripts" / "chat-contract-smoke.cjs"
    result = subprocess.run(
        [compiled_client_node, str(runner), server.url, server.kind],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["chunks"] == ["echo:latest 🧪\nline", "echo:second turn"]
    assert [call[0] for call in server.calls] == ["latest 🧪\nline", "second turn"]
    assert len(server.requests) == 3  # two valid turns, one rejected auth attempt; no retry
    assert len(server.created) == 1
    assert server.calls[0][2] == server.calls[1][2]
    if server.kind == "core":
        assert report["session_id"] is None  # core does not implement web session ownership
        assert report["request_ids"] == [call[1] for call in server.calls]
        assert len(set(report["request_ids"])) == 2
    else:
        assert report["session_id"]


def test_chat_rejects_malformed_body_before_agent_execution(chat_http_server):
    import httpx

    server = chat_http_server
    with httpx.Client(
        base_url=server.url, headers={"Authorization": "Bearer contract-test-key"}
    ) as client:
        # Old client body is invalid on core; it remains the supported web contract.
        invalid = {"message": "old singular-only"} if server.kind == "core" else {"message": {}}
        response = client.post("/chat/stream", json=invalid)
    assert response.status_code == 422
    assert server.calls == []


def test_actual_extension_rejects_lost_terminator_over_http(compiled_client_node, chat_http_server):
    server = chat_http_server
    server.faults.append("drop_terminator")
    runner = _VICTOR_CLIENT_TS.parents[1] / "scripts" / "chat-contract-smoke.cjs"
    result = subprocess.run(
        [compiled_client_node, str(runner), server.url, server.kind, "truncated"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["outcome"] == "interrupted"
    assert len(server.calls) == 1  # Never replay a POST after an unknown outcome.


@pytest.mark.parametrize("chat_http_server", ["core"], indirect=True)
def test_actual_extension_preserves_paused_chat_over_http(compiled_client_node, chat_http_server):
    server = chat_http_server
    runner = _VICTOR_CLIENT_TS.parents[1] / "scripts" / "chat-contract-smoke.cjs"
    result = subprocess.run(
        [compiled_client_node, str(runner), server.url, server.kind, "paused"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["outcome"] == "awaiting_approval"
    assert [call[0] for call in server.calls] == ["pause this turn"]
    assert server.calls[0][1]  # Existing request correlation still crosses the real route.


@pytest.mark.parametrize("chat_http_server", ["core"], indirect=True)
async def test_legacy_python_adapter_preserves_core_outcomes_over_http(chat_http_server):
    """Reuse the production-route fixture for the retained Python consumer."""
    from victor.integrations.protocol import ChatMessage, HTTPProtocolAdapter

    server = chat_http_server
    adapter = HTTPProtocolAdapter(server.url)
    adapter._client.headers["Authorization"] = "Bearer contract-test-key"
    messages = [ChatMessage(role="user", content="legacy caller")]
    try:
        paused = await adapter.chat(messages)
        assert paused.status == "awaiting_approval"
        assert paused.run_id == "run-http-approval"
        assert paused.approval_request["id"] == "approval-http"
        assert paused.finish_reason == "awaiting_approval"
        server.stream_tools.append({"name": "graph"})
        chunks = [chunk async for chunk in adapter.stream_chat(messages)]
        assert [chunk.content for chunk in chunks] == ["echo:legacy caller", ""]
        assert chunks[1].tool_call.name == "graph"
        server.faults.append("drop-terminator")
        with pytest.raises(ValueError, match="terminator"):
            _ = [chunk async for chunk in adapter.stream_chat(messages)]
        assert len(server.requests) == 2  # Neither stream was replayed.
        adapter._client.headers["Authorization"] = "Bearer incorrect-test-key"
        with pytest.raises(httpx.HTTPStatusError):
            await adapter.chat(messages)
        assert len(server.calls) == 3
        assert all(call[1] for call in server.calls)
    finally:
        await adapter.close()
