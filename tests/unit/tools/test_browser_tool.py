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

"""Tests for the browser tool (FOU-A5 phase 2).

Substrate (owner direction): the PRIMARY engine is the agentbrowser
service — ``victor.tools.browser_tool`` is a thin HTTP client of
``127.0.0.1:5709`` (session → page → navigate → extract → close). All
service interaction is mocked with respx against the protocol shapes from
agentbrowser's ``openapi.ts`` (SessionSummary / PageSummary /
NavigationStatus / ApiError / extract {data, evidence}); a sibling-checkout
contract test asserts the client's assumptions against the real
``openapi.json`` when available. NO real network and NO real browser here —
the playwright fallback engine has its own test module.

Every test asserts the guarantees the tool exists to provide:

1. Destination policy: the ``tools.web_fetch_allowed_hosts`` allowlist is
   enforced on the initial URL BEFORE any call is made (Victor's half of
   the policy division; agentbrowser's Cedar stack owns navigation
   admission), and the final URL is re-checked after navigation.
2. Structured errors: unreachable service (naming the startup command),
   ApiError envelopes, blocked navigation, timeout — all dicts, never
   raises.
3. Session lifecycle: the session Victor created is closed even when a
   later step fails; an explicit ``session_id`` is reused (no
   create/close) and never closed.
"""

import json
import os

import httpx
import pytest
import respx

from victor.tools import browser_tool
from victor.tools.base import ToolConfig
from victor.tools.web_search_tool import DestinationPolicyDenied

_ALLOWLIST_CTX = {"tool_config": ToolConfig(web_fetch_allowed_hosts=["allowed.test"])}

_BASE = "http://127.0.0.1:5709"

# Protocol-shaped fixtures (required fields per agentbrowser openapi.ts).
_SESSION_SUMMARY = {
    "sessionId": "sess-1",
    "status": "ready",
    "engine": "playwright-chromium",
    "createdAt": 0,
    "ttlMs": 60000,
    "idleTimeoutMs": 30000,
    "pages": [],
}
_PAGE_SUMMARY = {
    "pageId": "page-1",
    "sessionId": "sess-1",
    "status": "ready",
    "url": "https://allowed.test/page",
    "title": "Rendered Page",
}
_NAVIGATION_SUCCESS = {
    "status": "success",
    "url": "https://allowed.test/page",
    "redirectChain": [],
}


def _mock_service(
    *,
    navigation=None,
    page_summary=None,
    text_data=None,
    links_data=None,
):
    """Wire the full session → page → navigate → extract → close surface.

    Payloads use the protocol shapes from agentbrowser's openapi.ts
    (SessionSummary / PageSummary / NavigationStatus / extract
    {data, evidence} — required fields included)."""
    if text_data is None:
        text_data = {"text": "the rendered main text of the page"}
    if links_data is None:
        links_data = [{"text": "Next", "url": "https://allowed.test/next"}]
    respx.route(method="POST", path="/v1/sessions").mock(
        return_value=httpx.Response(201, json=dict(_SESSION_SUMMARY))
    )
    respx.route(method="POST", path="/v1/sessions/sess-1/pages").mock(
        return_value=httpx.Response(201, json=dict(page_summary or _PAGE_SUMMARY))
    )
    respx.route(method="POST", path="/v1/sessions/sess-1/pages/page-1/navigate").mock(
        return_value=httpx.Response(200, json=dict(navigation or _NAVIGATION_SUCCESS))
    )
    respx.route(method="GET", path="/v1/sessions/sess-1/pages/page-1").mock(
        return_value=httpx.Response(200, json=dict(page_summary or _PAGE_SUMMARY))
    )

    def _extract(request: httpx.Request) -> httpx.Response:
        fmt = json.loads(request.content or b"{}").get("format")
        data = text_data if fmt == "text" else links_data
        return httpx.Response(200, json={"data": data, "evidence": []})

    respx.route(method="POST", path="/v1/sessions/sess-1/pages/page-1/extract").mock(
        side_effect=_extract
    )
    close_route = respx.route(method="DELETE", path="/v1/sessions/sess-1").mock(
        return_value=httpx.Response(200, json={"sessionId": "sess-1", "status": "closed"})
    )
    return close_route


@pytest.fixture(autouse=True)
def _reset_browser_tool_state(monkeypatch):
    """Isolate the session seam and service env between tests."""
    from victor.tools.browser_playwright_fallback import set_browser_session_factory

    set_browser_session_factory(None)
    monkeypatch.delenv("AGENTBROWSER_BASE_URL", raising=False)
    monkeypatch.delenv("AGENTBROWSER_API_KEY", raising=False)
    yield
    set_browser_session_factory(None)


# ---------------------------------------------------------------------------
# Destination policy (Victor's half of the division — before ANY call)
# ---------------------------------------------------------------------------


class TestDestinationPolicy:
    @respx.mock
    async def test_denied_host_refused_before_any_service_call(self):
        called = []

        def _boom(request):
            called.append(request.url)
            return httpx.Response(200, json={})

        respx.route(host="127.0.0.1").mock(side_effect=_boom)

        result = await browser_tool.browser_tool(
            url="https://denied.test/page", _exec_ctx=_ALLOWLIST_CTX
        )

        assert result["success"] is False
        assert result["destination_policy_denied"] is True
        assert result["denied_host"] == "denied.test"
        assert result["denied_url"] == "https://denied.test/page"
        assert called == [], "policy must refuse before any agentbrowser call"

    async def test_file_url_refused_under_non_empty_allowlist(self):
        result = await browser_tool.browser_tool(
            url="file:///tmp/fixture.html", _exec_ctx=_ALLOWLIST_CTX
        )

        assert result["success"] is False
        assert result["destination_policy_denied"] is True

    async def test_allowlist_semantics_match_web_fetch(self):
        def _check(url):
            browser_tool._check_destination(url, ["allowed.test"])

        _check("https://allowed.test/page")  # exact host
        _check("https://www.allowed.test/deep/page")  # subdomain
        with pytest.raises(DestinationPolicyDenied):
            _check("https://notallowed.test/page")  # suffix trick
        with pytest.raises(DestinationPolicyDenied):
            _check("ftp://allowed.test/page")  # non-http scheme


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


class TestParameterValidation:
    async def test_missing_url(self):
        result = await browser_tool.browser_tool(url="")
        assert result["success"] is False

    async def test_invalid_engine(self):
        result = await browser_tool.browser_tool(url="https://allowed.test", engine="http")
        assert result["success"] is False
        assert "agentbrowser" in result["error"]

    async def test_invalid_wait_until(self):
        result = await browser_tool.browser_tool(url="https://allowed.test", wait_until="whenever")
        assert result["success"] is False
        assert "wait_until" in result["error"]

    async def test_commit_is_fallback_only(self):
        result = await browser_tool.browser_tool(url="https://allowed.test", wait_until="commit")
        assert result["success"] is False
        assert "playwright" in result["error"]


# ---------------------------------------------------------------------------
# agentbrowser primary: session → page → navigate → extract → close
# ---------------------------------------------------------------------------


class TestAgentbrowserPrimary:
    @respx.mock
    async def test_full_lifecycle_returns_structured_read(self):
        _mock_service(
            text_data={"text": "the rendered main text of the page"},
            links_data=[
                {"text": "Next", "url": "https://allowed.test/next"},
                {"text": "Again", "url": "https://allowed.test/next"},  # dup
                {"text": "Bare", "url": "/relative"},
                {"text": "Bad", "url": "javascript:void(0)"},
            ],
        )

        result = await browser_tool.browser_tool(url="https://allowed.test/page")

        assert result["success"] is True
        assert result["engine"] == "agentbrowser"
        assert result["title"] == "Rendered Page"
        assert result["url"] == "https://allowed.test/page"
        assert result["content"] == "the rendered main text of the page"
        assert result["session_id"] == "sess-1"
        hrefs = [link["href"] for link in result["links"]]
        assert "https://allowed.test/next" in hrefs
        assert "https://allowed.test/relative" in hrefs  # made absolute
        assert len(hrefs) == len(set(hrefs)), "links must be deduplicated"
        assert all("javascript:" not in h for h in hrefs)

    @respx.mock
    async def test_session_closed_after_successful_read(self):
        close_route = _mock_service(text_data={"text": "x"})

        result = await browser_tool.browser_tool(url="https://allowed.test/page")

        assert result["success"] is True
        assert close_route.called, "the created session must be closed"

    @respx.mock
    async def test_session_closed_when_extraction_fails(self):
        close_route = _mock_service(
            text_data={"text": "x"},
            links_data=[{"text": "a", "url": "https://allowed.test/a"}],
        )
        # Extraction raises an ApiError envelope (e.g. OUTPUT_TRUNCATED).
        respx.route(method="POST", path="/v1/sessions/sess-1/pages/page-1/extract").mock(
            return_value=httpx.Response(
                400,
                json={
                    "error": {
                        "code": "OUTPUT_TRUNCATED",
                        "message": "Extraction exceeds maxBytes",
                        "retryable": False,
                    }
                },
            )
        )

        result = await browser_tool.browser_tool(url="https://allowed.test/page")

        assert result["success"] is False
        assert result["agentbrowser_error"] is True
        assert result["agentbrowser_error_code"] == "OUTPUT_TRUNCATED"
        assert close_route.called, "session must close even when a step fails"

    @respx.mock
    async def test_explicit_session_id_is_reused_not_closed(self):
        def _no_create(request):
            raise AssertionError("session_id reuse must not create a session")

        respx.route(method="POST", path="/v1/sessions").mock(side_effect=_no_create)
        respx.route(method="POST", path="/v1/sessions/live-sess/pages").mock(
            return_value=httpx.Response(201, json={"pageId": "page-9", "sessionId": "live-sess"})
        )
        respx.route(method="POST", path="/v1/sessions/live-sess/pages/page-9/navigate").mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "success",
                    "url": "https://allowed.test/page",
                    "redirectChain": [],
                },
            )
        )
        respx.route(method="GET", path="/v1/sessions/live-sess/pages/page-9").mock(
            return_value=httpx.Response(
                200,
                json={
                    "pageId": "page-9",
                    "sessionId": "live-sess",
                    "status": "ready",
                    "url": "https://allowed.test/page",
                    "title": "Reused",
                },
            )
        )
        respx.route(method="POST", path="/v1/sessions/live-sess/pages/page-9/extract").mock(
            return_value=httpx.Response(200, json={"data": {"text": "body"}, "evidence": []})
        )
        close_route = respx.route(method="DELETE", path="/v1/sessions/live-sess").mock(
            return_value=httpx.Response(200, json={"sessionId": "live-sess", "status": "closed"})
        )

        result = await browser_tool.browser_tool(
            url="https://allowed.test/page", session_id="live-sess"
        )

        assert result["success"] is True
        assert result["session_id"] == "live-sess"
        assert close_route.called is False, "a caller-owned session is never closed"

    @respx.mock
    async def test_bearer_key_header_when_configured(self, monkeypatch):
        monkeypatch.setenv("AGENTBROWSER_API_KEY", "sekrit")
        seen_headers = []

        def _capture(request):
            seen_headers.append(request.headers.get("authorization"))
            return httpx.Response(201, json=dict(_SESSION_SUMMARY))

        respx.route(method="POST", path="/v1/sessions").mock(side_effect=_capture)
        respx.route(method="POST", path="/v1/sessions/sess-1/pages").mock(
            side_effect=httpx.ConnectError("connection refused")
        )

        result = await browser_tool.browser_tool(url="https://allowed.test/page")

        assert result["agentbrowser_unreachable"] is True
        assert seen_headers == ["Bearer sekrit"]

    @respx.mock
    async def test_base_url_env_override(self, monkeypatch):
        monkeypatch.setenv("AGENTBROWSER_BASE_URL", "http://127.0.0.1:9999")
        seen = []

        def _capture(request):
            seen.append(str(request.url))
            return httpx.Response(201, json=dict(_SESSION_SUMMARY))

        respx.route(method="POST", path="/v1/sessions").mock(side_effect=_capture)
        respx.route(method="POST", path="/v1/sessions/sess-1/pages").mock(
            side_effect=httpx.ConnectError("connection refused")
        )

        result = await browser_tool.browser_tool(url="https://allowed.test/page")

        assert seen and seen[0].startswith("http://127.0.0.1:9999")
        assert result["agentbrowser_unreachable"] is True
        assert result["service_url"] == "http://127.0.0.1:9999"

    @respx.mock
    async def test_bearer_auth_omits_body_tenant(self, monkeypatch):
        """The server derives the tenant from the bearer key (SHA-256(key) ->
        tenantId); a disagreeing body tenantId is refused FORBIDDEN (verified
        against the live service)."""
        monkeypatch.setenv("AGENTBROWSER_API_KEY", "sekrit")
        bodies = []

        def _capture(request):
            bodies.append(json.loads(request.content or b"{}"))
            return httpx.Response(201, json=dict(_SESSION_SUMMARY))

        respx.route(method="POST", path="/v1/sessions").mock(side_effect=_capture)
        respx.route(method="POST", path="/v1/sessions/sess-1/pages").mock(
            side_effect=httpx.ConnectError("connection refused")
        )

        result = await browser_tool.browser_tool(url="https://allowed.test/page")

        assert result["agentbrowser_unreachable"] is True
        assert bodies and "tenantId" not in bodies[0]
        assert bodies[0]["headless"] is True

    @respx.mock
    async def test_unauthenticated_local_deploy_declares_tenant(self):
        bodies = []

        def _capture(request):
            bodies.append(json.loads(request.content or b"{}"))
            return httpx.Response(201, json=dict(_SESSION_SUMMARY))

        respx.route(method="POST", path="/v1/sessions").mock(side_effect=_capture)
        respx.route(method="POST", path="/v1/sessions/sess-1/pages").mock(
            side_effect=httpx.ConnectError("connection refused")
        )

        result = await browser_tool.browser_tool(url="https://allowed.test/page")

        assert result["agentbrowser_unreachable"] is True
        assert bodies and bodies[0]["tenantId"] == "victor"

    @respx.mock
    async def test_unreachable_service_names_startup_command(self):
        respx.route(host="127.0.0.1").mock(side_effect=httpx.ConnectError("connection refused"))

        result = await browser_tool.browser_tool(url="https://allowed.test/page")

        assert result["success"] is False
        assert result["agentbrowser_unreachable"] is True
        assert browser_tool._SERVICE_START_COMMAND in result["error"]
        assert "playwright" in result["error"], "the fallback escape hatch is named"
        assert "AGENTBROWSER_POLICY_FILE" in result["error"]

    @respx.mock
    async def test_navigation_timeout_maps_to_structured_result(self):
        _mock_service(
            navigation={
                "status": "timeout",
                "url": "https://allowed.test/slow",
                "redirectChain": [],
            },
            text_data={"text": "x"},
        )

        result = await browser_tool.browser_tool(url="https://allowed.test/slow")

        assert result["success"] is False
        assert result["timeout"] is True

    @respx.mock
    async def test_blocked_navigation_surfaces_cedar_denial(self):
        _mock_service(
            navigation={
                "status": "blocked",
                "url": "https://allowed.test/locked",
                "redirectChain": [],
                "reason": "egress_policy",
                "scopeDenials": {"egress": 1},
            },
            text_data={"text": "x"},
        )

        result = await browser_tool.browser_tool(url="https://allowed.test/locked")

        assert result["success"] is False
        assert result["navigation_blocked"] is True
        assert result["blocked_reason"] == "egress_policy"
        assert result["scope_denials"] == {"egress": 1}

    @respx.mock
    async def test_final_url_allowlist_recheck_after_redirect(self):
        _mock_service(
            navigation={
                "status": "success",
                "url": "https://denied.test/escaped",
                "redirectChain": ["https://allowed.test/page"],
            },
            text_data={"text": "x"},
        )

        result = await browser_tool.browser_tool(
            url="https://allowed.test/page", _exec_ctx=_ALLOWLIST_CTX
        )

        assert result["success"] is False
        assert result["destination_policy_denied"] is True
        assert result["denied_host"] == "denied.test"

    @respx.mock
    async def test_max_links_caps_link_list(self):
        _mock_service(
            text_data={"text": "x"},
            links_data=[{"text": f"l{i}", "url": f"https://allowed.test/{i}"} for i in range(5)],
        )

        result = await browser_tool.browser_tool(url="https://allowed.test/page", max_links=2)

        assert result["success"] is True
        assert result["link_count"] <= 2


# ---------------------------------------------------------------------------
# Contract: the client's assumptions against agentbrowser's openapi.json
# ---------------------------------------------------------------------------

_AGENTBROWSER_OPENAPI = os.path.join(
    os.path.expanduser("~"), "code", "agentbrowser", "openapi.json"
)


class TestAgentbrowserContract:
    """Contract tests against the real openapi.json (skip when the sibling
    checkout is absent, e.g. CI)."""

    @pytest.fixture
    def openapi(self):
        if not os.path.exists(_AGENTBROWSER_OPENAPI):
            pytest.skip("agentbrowser sibling checkout not present")
        return json.load(open(_AGENTBROWSER_OPENAPI))

    def test_read_surface_paths_exist(self, openapi):
        for path in (
            "/v1/sessions",
            "/v1/sessions/{sessionId}/pages",
            "/v1/sessions/{sessionId}/pages/{pageId}/navigate",
            "/v1/sessions/{sessionId}/pages/{pageId}/extract",
            "/v1/sessions/{sessionId}",
        ):
            assert path in openapi["paths"], path

    def test_session_create_requires_tenant_id_and_headless(self, openapi):
        create = openapi["paths"]["/v1/sessions"]["post"]
        req = create["requestBody"]["content"]["application/json"]["schema"]
        assert req["required"] == ["tenantId"]
        props = req["properties"]
        assert "tenantId" in props and "headless" in props
        # Victor always passes headless: true — agentbrowser's Cedar
        # capability authorizer gates session.headed.
        assert props["headless"]["type"] == "boolean"

    def test_navigate_wait_until_enum_covers_client(self, openapi):
        navigate = openapi["paths"]["/v1/sessions/{sessionId}/pages/{pageId}/navigate"]["post"]
        req = navigate["requestBody"]["content"]["application/json"]["schema"]["properties"][
            "waitUntil"
        ]
        assert set(browser_tool._AGENTBROWSER_WAIT_UNTIL) <= set(req["enum"])

    def test_extract_formats_include_text_and_links(self, openapi):
        extract = openapi["paths"]["/v1/sessions/{sessionId}/pages/{pageId}/extract"]["post"]
        req = extract["requestBody"]["content"]["application/json"]["schema"]["properties"][
            "format"
        ]
        assert {"text", "links"} <= set(req["enum"])

    def test_navigation_status_shape(self, openapi):
        status = openapi["components"]["schemas"]["NavigationStatus"]
        assert {"status", "url", "redirectChain"} <= set(status["required"])
        states = status["properties"]["status"]
        values = states.get("enum") or [item["const"] for item in states.get("anyOf", [])]
        assert values == ["success", "timeout", "blocked"]

    def test_page_summary_carries_title(self, openapi):
        summary = openapi["components"]["schemas"]["PageSummary"]
        assert "title" in summary["properties"]

    def test_default_service_address(self):
        assert browser_tool._DEFAULT_SERVICE_URL == "http://127.0.0.1:5709"


# ---------------------------------------------------------------------------
# Registration and discovery
# ---------------------------------------------------------------------------


class TestRegistration:
    async def test_browser_tool_discoverable_via_shared_registry(self):
        from victor.agent.shared_tool_registry import SharedToolRegistry

        registry = SharedToolRegistry.get_instance()
        try:
            tools = registry.get_tool_classes()
            assert "browser_tool" in tools
        finally:
            SharedToolRegistry.reset_instance()

    async def test_demand_spec_loads_browser_tool(self):
        from victor.agent.shared_tool_registry import DEMAND_TOOL_SPECS, SharedToolRegistry

        assert "browser_tool" in DEMAND_TOOL_SPECS
        SharedToolRegistry.reset_instance()
        registry = SharedToolRegistry.get_instance()
        try:
            loaded = registry.get_tools_for_names(["browser_tool"])
            assert loaded, "demand spec must hydrate browser_tool"
            assert any(getattr(t, "name", "") == "browser_tool" for t in loaded)
        finally:
            SharedToolRegistry.reset_instance()

    async def test_demand_keywords_hydrate_browser_tool(self):
        from victor.agent.shared_tool_registry import SharedToolRegistry

        registry = SharedToolRegistry.get_instance()
        assert "browser_tool" in registry.infer_demand_tools(
            "please open this page in a browser and extract the table"
        )
        assert "browser_tool" not in registry.infer_demand_tools("list files")

    async def test_airgap_filter_covers_browser_tool(self):
        from victor.agent.shared_tool_registry import WEB_TOOL_NAMES

        assert "browser_tool" in WEB_TOOL_NAMES

    async def test_umbrella_web_browser_subcommand(self):
        from victor.tools.unified.web_tool import create_web_parser

        args = create_web_parser().parse_args(["browser", "https://allowed.test/page"])
        assert args.subcommand == "browser"
        assert args.url == "https://allowed.test/page"
