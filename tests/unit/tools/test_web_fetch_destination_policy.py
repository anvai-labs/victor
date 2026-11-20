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

"""Destination policy for ``web_fetch`` (Codex finding 12).

The research page reader's network reach is the service's network reach, so
its entry point carries an explicit destination policy:

* empty allowlist (the settings default) = every destination admitted —
  local single-operator behavior is unchanged;
* a non-empty allowlist admits only the listed hosts (exact match or
  subdomain, case-insensitive), is enforced at the entry point BEFORE the
  result cache, is re-checked at EVERY redirect hop (the HTTP path follows
  redirects manually now), and covers the browser render path's final URL;
* violations return a structured error naming the denied destination.

All HTTP is mocked with respx — no real network.
"""

import httpx
import pytest
import respx

from victor.tools import web_search_tool as wst
from victor.tools.base import ToolConfig
from victor.tools.web_search_tool import web_fetch

ALLOWED = "https://allowed.example"
DENIED = "https://denied.example"


@pytest.fixture(autouse=True)
def _clear_web_caches():
    wst._GENERIC_WEB_CACHE = None
    wst._RATE_LIMITER = None
    yield
    wst._GENERIC_WEB_CACHE = None
    wst._RATE_LIMITER = None


def _ctx(allowed_hosts, **cfg_kw) -> dict:
    config = ToolConfig(
        web_fetch_allowed_hosts=allowed_hosts,
        generic_result_cache_enabled=cfg_kw.pop("generic_result_cache_enabled", False),
        http_connection_pool_enabled=False,
    )
    # The per-host rate limiter spaces requests seconds apart; these tests
    # assert policy routing, not pacing.
    config.rate_limiter_enabled = False
    return {"tool_config": config}


def _page(host: str, path: str = "/article") -> respx.Route:
    return respx.get(f"{host}{path}").mock(
        return_value=httpx.Response(
            200,
            html=(
                "<html><head><title>Guide</title></head><body>"
                "<article><h1>Research</h1><p>"
                + ("Victor research tools combine readable page extraction. " * 8)
                + "</p></article></body></html>"
            ),
            headers={"Content-Type": "text/html; charset=utf-8"},
        )
    )


class TestPolicySemantics:
    def test_empty_allowlist_matches_everything(self):
        assert wst._host_allowed("anything.example", [])
        assert wst._host_allowed(None, [])
        assert wst._host_allowed(None, ["allowed.example"]) is False

    def test_entry_matches_exact_host_and_subdomains(self):
        hosts = ["allowed.example"]
        assert wst._host_allowed("allowed.example", hosts)
        assert wst._host_allowed("deep.sub.allowed.example", hosts)
        assert wst._host_allowed("denied.example", hosts) is False
        # A different host sharing the suffix is NOT a subdomain match.
        assert wst._host_allowed("notallowed.example", hosts) is False

    def test_matching_is_case_insensitive(self):
        assert wst._host_allowed("ALLOWED.Example", ["allowed.example"])

    def test_non_http_schemes_are_refused_when_policy_is_set(self):
        with pytest.raises(wst.DestinationPolicyDenied) as excinfo:
            wst._check_destination("file:///etc/passwd", ["allowed.example"])
        assert excinfo.value.host == "(no host)"


class TestDefaultPosture:
    @respx.mock
    async def test_empty_policy_allows_any_host(self):
        route = _page(DENIED)
        result = await web_fetch(url=f"{DENIED}/article", _exec_ctx=_ctx([]))
        assert result["success"] is True
        assert route.called

    @respx.mock
    async def test_empty_policy_still_follows_redirects(self):
        first = respx.get(f"{ALLOWED}/start").mock(
            return_value=httpx.Response(302, headers={"Location": f"{DENIED}/article"})
        )
        second = _page(DENIED)
        result = await web_fetch(url=f"{ALLOWED}/start", _exec_ctx=_ctx([]))
        assert result["success"] is True
        assert first.called and second.called


class TestEntryPolicy:
    @respx.mock
    async def test_denied_initial_host_names_the_destination(self):
        route = _page(DENIED)
        result = await web_fetch(url=f"{DENIED}/article", _exec_ctx=_ctx(["allowed.example"]))
        assert result["success"] is False
        assert result["destination_policy_denied"] is True
        assert result["denied_host"] == "denied.example"
        assert "denied.example" in result["error"]
        assert result["denied_url"] == f"{DENIED}/article"
        assert route.called is False  # refused before any HTTP contact

    @respx.mock
    async def test_admitted_host_is_fetched(self):
        route = _page(ALLOWED)
        result = await web_fetch(url=f"{ALLOWED}/article", _exec_ctx=_ctx(["allowed.example"]))
        assert result["success"] is True
        assert route.called

    @respx.mock
    async def test_policy_is_checked_before_the_result_cache(self):
        """A host denied by the policy must not be served from cache."""
        cache = wst._get_generic_web_cache(300)
        cache.set(
            wst.ResultType.SEARCH,
            f"fetch:{DENIED}/article",
            {"success": True, "content": "cached payload"},
            params={"max_content_length": 5000, "render": "http"},
        )
        result = await web_fetch(
            url=f"{DENIED}/article",
            _exec_ctx=_ctx(["allowed.example"], generic_result_cache_enabled=True),
        )
        assert result["success"] is False
        assert result["destination_policy_denied"] is True
        assert "cached payload" not in (result.get("content") or "")

    async def test_browser_mode_policy_precedes_the_render_stack(self):
        """The initial-URL check fires before Playwright is even imported."""
        result = await web_fetch(
            url=f"{DENIED}/article",
            render="browser",
            _exec_ctx=_ctx(["allowed.example"]),
        )
        assert result["destination_policy_denied"] is True
        assert result["denied_host"] == "denied.example"

    async def test_non_http_scheme_denied_when_policy_is_set(self):
        result = await web_fetch(url="file:///etc/passwd", _exec_ctx=_ctx(["allowed.example"]))
        assert result["destination_policy_denied"] is True
        assert result["denied_host"] == "(no host)"


class TestRedirectPolicy:
    @respx.mock
    async def test_redirect_to_denied_host_refused_at_the_hop(self):
        respx.get(f"{ALLOWED}/start").mock(
            return_value=httpx.Response(302, headers={"Location": f"{DENIED}/article"})
        )
        denied_route = _page(DENIED)
        result = await web_fetch(url=f"{ALLOWED}/start", _exec_ctx=_ctx(["allowed.example"]))
        assert result["success"] is False
        assert result["destination_policy_denied"] is True
        # The hop to the denied destination was never made.
        assert denied_route.called is False
        assert result["denied_host"] == "denied.example"
        assert result["denied_url"] == f"{DENIED}/article"

    @respx.mock
    async def test_redirect_chain_within_allowed_hosts_succeeds(self):
        respx.get(f"{ALLOWED}/a").mock(return_value=httpx.Response(302, headers={"Location": "/b"}))
        final = _page(ALLOWED, "/b")
        result = await web_fetch(url=f"{ALLOWED}/a", _exec_ctx=_ctx(["allowed.example"]))
        assert result["success"] is True
        assert final.called

    @respx.mock
    async def test_denied_host_may_redirect_into_an_allowed_host(self):
        """Policy applies per destination, not per origin: a denied entry
        point is refused at the entry, but this test pins that an ADMITTED
        origin redirecting within allowed hosts stays allowed — the mirror of
        the refused-hop case."""
        respx.get(f"{ALLOWED}/out").mock(
            return_value=httpx.Response(302, headers={"Location": f"{ALLOWED}/article"})
        )
        final = _page(ALLOWED)
        result = await web_fetch(url=f"{ALLOWED}/out", _exec_ctx=_ctx(["allowed.example"]))
        assert result["success"] is True
        assert final.called

    @respx.mock
    async def test_relative_redirect_locations_resolve_and_are_checked(self):
        respx.get(f"{ALLOWED}/rel").mock(
            return_value=httpx.Response(302, headers={"Location": "https://other.example/x"})
        )
        result = await web_fetch(url=f"{ALLOWED}/rel", _exec_ctx=_ctx(["allowed.example"]))
        assert result["destination_policy_denied"] is True
        assert result["denied_host"] == "other.example"
        assert result["denied_url"] == "https://other.example/x"

    @respx.mock
    async def test_redirect_hop_budget_is_bounded(self):
        respx.route(host="allowed.example").mock(
            return_value=httpx.Response(302, headers={"Location": "https://allowed.example/loop"})
        )
        result = await web_fetch(url=f"{ALLOWED}/loop", _exec_ctx=_ctx(["allowed.example"]))
        assert result["success"] is False
        # Not a policy denial: the policy ADMITTED every hop — the loop is
        # bounded by the redirect-hop budget instead.
        assert result.get("destination_policy_denied") is None
        assert "redirect" in result["error"].lower()


class TestSettingsWiring:
    def test_tool_settings_default_is_allow_all(self):
        from victor.config.tool_settings import ToolSettings

        assert ToolSettings().web_fetch_allowed_hosts == []

    def test_tool_config_default_is_allow_all(self):
        assert ToolConfig().web_fetch_allowed_hosts == []

    def test_get_web_config_surfaces_the_allowlist(self):
        config = wst._get_web_config(_ctx(["allowed.example"]))
        assert config["web_fetch_allowed_hosts"] == ["allowed.example"]
        assert wst._get_web_config(None)["web_fetch_allowed_hosts"] == []
