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

"""Tests for the browser tool's PLAYWRIGHT FALLBACK engine (FOU-A5 phase 2).

The primary substrate is the agentbrowser service (see ``test_browser_tool.py``);
this module covers the documented local fallback —
``browser_tool(engine="playwright")`` over
``victor.tools.browser_playwright_fallback`` — including the guarantees that
make it safe as a local engine: destination policy enforced on every request
inside a real browser (redirect hops followed manually inside the guard), the
download cap against a real loopback server, and the hard browser-kill
teardown probed against the live process table.

Network discipline: happy-path extraction uses file:// fixtures and a local
loopback test server — NO real network. Policy/failure paths use fakes and the
module's session-factory seam.
"""

import asyncio
import os
import sys

import pytest

from victor.tools import browser_tool
from victor.tools import browser_playwright_fallback as fallback
from victor.tools.browser_playwright_fallback import (
    _NavigationPolicy,
    set_browser_session_factory,
)
from victor.tools.base import ToolConfig
from victor.tools.web_search_tool import DestinationPolicyDenied

_ALLOWLIST_CTX = {"tool_config": ToolConfig(web_fetch_allowed_hosts=["allowed.test"])}


@pytest.fixture(autouse=True)
def _reset_browser_tool_state():
    """Isolate the module-level session-factory seam between tests."""
    set_browser_session_factory(None)
    yield
    set_browser_session_factory(None)


# ---------------------------------------------------------------------------
# Fakes
def _chromium_installed() -> bool:
    """(chromium_available, browsers_cache_dir), probed under the real HOME.

    Must run at import time: the session-scoped ``isolate_global_victor_db``
    conftest fixture redirects ``$HOME`` to a sandbox before any test runs,
    and the playwright driver resolves its browser cache from ``$HOME``. The
    autouse fixture below repoints ``PLAYWRIGHT_BROWSERS_PATH`` at the real
    cache so real-browser tests launch the developer-installed chromium.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return False, ""
    try:
        with sync_playwright() as p:
            exe = p.chromium.executable_path
    except Exception:
        return False, ""
    available = os.path.exists(exe)
    # <cache>/chromium-1208/chrome-linux64/chrome -> <cache>
    browsers_dir = os.path.dirname(os.path.dirname(os.path.dirname(exe)))
    return available, browsers_dir


_HAS_CHROMIUM, _BROWSERS_DIR = _chromium_installed()

_REAL_BROWSER = pytest.mark.skipif(
    not _HAS_CHROMIUM, reason="playwright chromium binary not installed"
)


@pytest.fixture(autouse=True)
def _playwright_cache_env():
    """Point the playwright driver at the real browsers cache when $HOME is
    redirected by the test-session conftest (see _chromium_installed)."""
    if _BROWSERS_DIR:
        saved = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = _BROWSERS_DIR
        try:
            yield
        finally:
            if saved is None:
                os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)
            else:
                os.environ["PLAYWRIGHT_BROWSERS_PATH"] = saved
    else:
        yield


# ---------------------------------------------------------------------------


class _FakeAPIResponse:
    def __init__(self, body: bytes = b"x", headers: dict | None = None, status: int = 200):
        self._body = body
        self.status = status
        self.headers = headers or {}

    async def body(self) -> bytes:
        return self._body


class _FakeRoute:
    def __init__(self, body: bytes = b"x", headers: dict | None = None):
        self.aborted = False
        self.fulfilled = False
        self.fetch_kwargs: dict | None = None
        self._response = _FakeAPIResponse(body, headers)

    async def abort(self):
        self.aborted = True

    async def fulfill(self, response=None, **kwargs):
        self.fulfilled = True

    async def fetch(self, **kwargs):
        self.fetch_kwargs = kwargs
        return self._response


class _FakeRequest:
    def __init__(self, url: str, resource_type: str = "document"):
        self.url = url
        self.resource_type = resource_type


class _FakeSession:
    """Minimal _BrowserSession stand-in for the tool-level factory seam."""

    def __init__(self, read=None, exc=None):
        self.started = False
        self.closed = False
        self._read = read
        self._exc = exc

    async def __aenter__(self):
        self.started = True
        return self

    async def __aexit__(self, *exc_info):
        self.closed = True
        return False

    async def read(self, url, **kwargs):
        if self._exc is not None:
            raise self._exc
        return await self._read(url, **kwargs)


# ---------------------------------------------------------------------------
# Destination policy (the #1990 policy class, fallback-browser edition)
# ---------------------------------------------------------------------------


class TestDestinationPolicy:
    async def test_allowlist_semantics_match_web_fetch(self):
        policy = _NavigationPolicy(allowed_hosts=["allowed.test"], byte_cap=1024)

        policy.check("https://allowed.test/page")  # exact host
        policy.check("https://www.allowed.test/deep/page")  # subdomain
        with pytest.raises(DestinationPolicyDenied):
            policy.check("https://notallowed.test/page")  # suffix trick
        with pytest.raises(DestinationPolicyDenied):
            policy.check("ftp://allowed.test/page")  # non-http scheme

    async def test_route_guard_denies_document_navigation_hop(self):
        policy = _NavigationPolicy(allowed_hosts=["allowed.test"], byte_cap=1024)
        route = _FakeRoute()

        await policy.handle_route(route, _FakeRequest("https://denied.test/redirect-target"))

        # A denied DOCUMENT is settled with a synthetic empty page (not an
        # abort) so read() maps the recorded denial deterministically.
        assert route.fulfilled is True
        assert policy.document_denied is not None
        assert policy.document_denied.host == "denied.test"

    async def test_route_guard_admits_allowed_hop(self):
        policy = _NavigationPolicy(allowed_hosts=["allowed.test"], byte_cap=1024)
        route = _FakeRoute(body=b"hello")

        await policy.handle_route(route, _FakeRequest("https://www.allowed.test/page"))

        assert route.aborted is False
        assert route.fulfilled is True
        assert policy.bytes_downloaded == 5

    async def test_subresource_to_denied_host_is_aborted_but_not_a_document_denial(self):
        policy = _NavigationPolicy(allowed_hosts=["allowed.test"], byte_cap=1024)
        route = _FakeRoute()

        await policy.handle_route(route, _FakeRequest("https://tracker.test/pixel", "image"))

        assert route.aborted is True
        assert policy.document_denied is None
        assert len(policy.denials) == 1


# ---------------------------------------------------------------------------
# Optional dependency: structured errors, never crashes
# ---------------------------------------------------------------------------


class TestOptionalDependency:
    async def test_playwright_missing_returns_structured_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "playwright", None)
        monkeypatch.setitem(sys.modules, "playwright.async_api", None)

        result = await browser_tool.browser_tool(
            url="https://allowed.test/page", engine="playwright"
        )

        assert result["success"] is False
        assert result["playwright_missing"] is True
        assert "victor-ai[web]" in result["error"]

    async def test_browser_binary_missing_returns_structured_error(self, monkeypatch):
        from playwright.async_api import Error as PWError
        from playwright.async_api import TimeoutError as PWTimeout

        class _FakeChromium:
            async def launch(self, **kwargs):
                raise PWError(
                    "Executable doesn't exist at /x/chrome. "
                    "Please run the following command to download new browsers: "
                    "playwright install chromium"
                )

        class _FakePW:
            chromium = _FakeChromium()

        class _FakePlaywrightCM:
            async def __aenter__(self):
                return _FakePW()

            async def __aexit__(self, *exc_info):
                return None

        monkeypatch.setattr(
            fallback,
            "_import_playwright",
            lambda: (lambda: _FakePlaywrightCM(), PWError, PWTimeout),
        )

        result = await browser_tool.browser_tool(
            url="https://allowed.test/page", engine="playwright"
        )

        assert result["success"] is False
        assert result["browser_binary_missing"] is True
        assert "playwright install chromium" in result["error"]


# ---------------------------------------------------------------------------
# Navigation failures: timeout mapping, download cap
# ---------------------------------------------------------------------------


class _FakePage:
    def __init__(self, goto_exc=None):
        self._goto_exc = goto_exc
        self.routes = []
        self.closed = False

    async def route(self, pattern, handler):
        self.routes.append((pattern, handler))

    async def goto(self, url, **kwargs):
        if self._goto_exc is not None:
            raise self._goto_exc

    async def close(self):
        self.closed = True


class _FakeContext:
    def __init__(self, page):
        self._page = page
        self.closed = False
        self.routes = []

    async def new_page(self, **kwargs):
        return self._page

    async def route(self, pattern, handler):
        self.routes.append((pattern, handler))

    async def close(self):
        self.closed = True


class _FakeBrowser:
    def __init__(self, page):
        self._page = page
        self.closed = False
        self.contexts = []

    async def new_context(self, **kwargs):
        context = _FakeContext(self._page)
        self.contexts.append(context)
        return context

    async def close(self):
        self.closed = True


class _FakeLaunch:
    def __init__(self, browser):
        self._browser = browser

    async def launch(self, **kwargs):
        return self._browser


class _FakePW:
    def __init__(self, browser):
        self.chromium = _FakeLaunch(browser)


class _FakePlaywrightCM:
    def __init__(self, browser):
        self._browser = browser
        self.stopped = False

    async def __aenter__(self):
        return _FakePW(self._browser)

    async def __aexit__(self, *exc_info):
        self.stopped = True
        return None


def _install_fake_playwright(monkeypatch, page):
    from playwright.async_api import Error as PWError
    from playwright.async_api import TimeoutError as PWTimeout

    browser = _FakeBrowser(page)
    cm = _FakePlaywrightCM(browser)
    monkeypatch.setattr(
        fallback,
        "_import_playwright",
        lambda: (lambda: cm, PWError, PWTimeout),
    )
    return browser, cm


class TestNavigationFailures:
    async def test_timeout_maps_to_structured_result_and_still_cleans_up(self, monkeypatch):
        from playwright.async_api import TimeoutError as PWTimeout

        page = _FakePage(goto_exc=PWTimeout("Timeout 30000ms exceeded"))
        browser, cm = _install_fake_playwright(monkeypatch, page)

        result = await browser_tool.browser_tool(
            url="https://allowed.test/slow", engine="playwright"
        )

        assert result["success"] is False
        assert result["timeout"] is True
        assert browser.closed is True, "timed-out navigation must still close the browser"
        assert cm.stopped is True
        # Negative test for the popup bypass: the policy guard MUST be
        # attached at the context level (page.route would leave popup pages
        # free to navigate outside the allowlist unguarded).
        assert page.routes == [], "the guard must NOT be page-scoped"
        assert browser.contexts and browser.contexts[-1].routes, "the guard must be context-scoped"

    async def test_route_fetch_pins_max_redirects_to_zero(self):
        """Negative test for the server-side redirect-chain bypass.

        route.fetch with its default maxRedirects=20 would follow the whole
        redirect chain inside the handler and fulfill the FINAL body —
        letting a redirect move the navigation to a host the operator never
        admitted without any policy check. The handler must pin 0 so every
        hop re-enters the guard as its own request.
        """
        policy = _NavigationPolicy(allowed_hosts=["allowed.test"], byte_cap=1024)
        route = _FakeRoute(body=b"x")

        await policy.handle_route(route, _FakeRequest("https://allowed.test/page"))

        assert route.fulfilled is True
        assert route.fetch_kwargs["max_redirects"] == 0
        assert route.fetch_kwargs["url"] == "https://allowed.test/page"

    async def test_download_cap_records_and_settles_requests(self):
        policy = _NavigationPolicy(allowed_hosts=[], byte_cap=10)

        first = _FakeRoute(body=b"abcdef")  # 6 bytes, admitted
        await policy.handle_route(first, _FakeRequest("https://any.test/a", "document"))
        assert policy.cap_exceeded is False
        assert first.fulfilled is True

        # A document that would blow the cap is settled with a synthetic
        # page (so read() maps the refusal deterministically)...
        second = _FakeRoute(body=b"ghijkl")  # total 12 > cap 10
        await policy.handle_route(second, _FakeRequest("https://any.test/b", "document"))
        assert policy.cap_exceeded is True
        assert second.fulfilled is True
        assert policy.bytes_downloaded == 6  # the oversized body is not admitted

        # ...while a subresource is simply aborted.
        third = _FakeRoute(body=b"ghijklmn")  # total 14 > cap 10
        await policy.handle_route(third, _FakeRequest("https://any.test/c", "image"))
        assert third.aborted is True

    @_REAL_BROWSER
    async def test_real_browser_download_cap_exceeded(self, tmp_path):
        """The network-level cap fires on a real (loopback) page load.

        file:// navigations bypass request interception, so the cap is
        exercised against a local test server — no external network.
        """
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        fixture = tmp_path / "big.html"
        fixture.write_text(
            "<html><head><title>Big</title></head><body>" + ("filler " * 512) + "</body></html>",
            encoding="utf-8",
        )

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = fixture.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/big.html"
            session = fallback._BrowserSession()
            async with session:
                result = await session.read(
                    url,
                    allowed_hosts=[],
                    byte_cap=8,
                    wait_until="load",
                    selector="",
                    timeout_seconds=15.0,
                    max_links=10,
                    max_content_length=5000,
                )
        finally:
            server.shutdown()
            server.server_close()

        assert result["success"] is False
        assert result["download_cap_exceeded"] is True


# ---------------------------------------------------------------------------
# Real redirect chains: the per-hop policy rule, end to end
# ---------------------------------------------------------------------------


def _redirect_server(target_path: str, redirect_to: str):
    """Local HTTP server that 302s ``target_path`` to ``redirect_to``."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == target_path:
                self.send_response(302)
                self.send_header("Location", redirect_to)
                self.end_headers()
                return
            body = b"<html><head><title>Landed</title></head><body>landed page</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class TestRedirectHops:
    @_REAL_BROWSER
    async def test_same_host_redirect_chain_succeeds(self, tmp_path):
        server = _redirect_server("/start", "/landed")
        try:
            url = f"http://127.0.0.1:{server.server_port}/start"
            session = fallback._BrowserSession()
            async with session:
                result = await session.read(
                    url,
                    allowed_hosts=["127.0.0.1"],  # redirect hop is re-admitted
                    byte_cap=fallback._PAGE_DOWNLOAD_CAP_BYTES,
                    wait_until="load",
                    selector="",
                    timeout_seconds=15.0,
                    max_links=10,
                    max_content_length=5000,
                )
        finally:
            server.shutdown()
            server.server_close()

        assert result["success"] is True
        assert result["title"] == "Landed"
        assert result["url"].endswith("/landed")

    @_REAL_BROWSER
    async def test_redirect_hop_to_denied_host_is_structurally_refused(self):
        server = _redirect_server("/start", "http://10.255.255.1:1/escaped")
        try:
            url = f"http://127.0.0.1:{server.server_port}/start"
            session = fallback._BrowserSession()
            async with session:
                result = await session.read(
                    url,
                    allowed_hosts=["127.0.0.1"],  # the redirect target is NOT admitted
                    byte_cap=fallback._PAGE_DOWNLOAD_CAP_BYTES,
                    wait_until="load",
                    selector="",
                    timeout_seconds=15.0,
                    max_links=10,
                    max_content_length=5000,
                )
        finally:
            server.shutdown()
            server.server_close()

        assert result["success"] is False
        assert result["destination_policy_denied"] is True
        assert result["denied_host"] == "10.255.255.1"


# ---------------------------------------------------------------------------
# Happy path: real headless chromium over a file:// fixture (no network)
# ---------------------------------------------------------------------------


def _write_fixture(path):
    paragraph = " ".join(
        ["The rendered page exposes content that plain HTTP extraction misses."] * 6
    )
    path.write_text(
        "<html><head><title>Fixture Page</title></head><body>"
        "<nav>ignored chrome</nav>"
        f"<main><h1>Rendered</h1><p>{paragraph}</p></main>"
        '<a href="https://example.com/absolute">Absolute</a>'
        '<a href="/relative">Relative</a>'
        '<a href="https://example.com/absolute">Duplicate</a>'
        "</body></html>",
        encoding="utf-8",
    )
    return path


class TestHappyPath:
    @_REAL_BROWSER
    async def test_file_url_extraction_returns_title_text_links(self, tmp_path):
        fixture = _write_fixture(tmp_path / "page.html")

        result = await browser_tool.browser_tool(url=fixture.as_uri(), engine="playwright")

        assert result["success"] is True
        assert result["title"] == "Fixture Page"
        assert "plainhttpextractionmisses" in result["content"].replace(" ", "").lower()
        assert result["render"] == "browser"
        assert result["url"] == fixture.as_uri()
        hrefs = [link["href"] for link in result["links"]]
        assert "https://example.com/absolute" in hrefs  # relative href made absolute
        assert len(hrefs) == len(set(hrefs)), "links must be deduplicated"

    @_REAL_BROWSER
    async def test_max_links_caps_link_list(self, tmp_path):
        fixture = _write_fixture(tmp_path / "page.html")

        result = await browser_tool.browser_tool(
            url=fixture.as_uri(), max_links=1, engine="playwright"
        )

        assert result["success"] is True
        assert result["link_count"] <= 1

    @_REAL_BROWSER
    async def test_clean_close_leaves_no_chromium_process(self, tmp_path):
        fixture = _write_fixture(tmp_path / "page.html")
        session = fallback._BrowserSession()
        async with session:
            live = fallback._chromium_pids(session._token)
            assert live, "a launched browser must be discoverable by its env token"
            result = await session.read(
                fixture.as_uri(),
                allowed_hosts=[],
                byte_cap=fallback._PAGE_DOWNLOAD_CAP_BYTES,
                wait_until="load",
                selector="",
                timeout_seconds=15.0,
                max_links=10,
                max_content_length=5000,
            )
            assert result["success"] is True

        assert fallback._chromium_pids(session._token) == set()
        assert session.killed_pids == set(), "a clean close must not need the hard kill"


# ---------------------------------------------------------------------------
# Hard browser-kill: no leaked chromium, even on abort
# ---------------------------------------------------------------------------


class TestHardKill:
    @_REAL_BROWSER
    async def test_cancelled_teardown_still_kills_the_browser(self, monkeypatch):
        session = fallback._BrowserSession()
        session._STEP_TIMEOUT_S = 0.2
        await session.__aenter__()
        live = fallback._chromium_pids(session._token)
        assert live, "precondition: live browser discoverable"

        # Make the graceful close hang so only the hard-kill path can reap.
        async def _slow_close():
            await asyncio.sleep(30)

        monkeypatch.setattr(session._browser, "close", _slow_close)

        closer = asyncio.create_task(session.aclose())
        await asyncio.sleep(0)
        closer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closer

        deadline = asyncio.get_event_loop().time() + 10.0
        while fallback._chromium_pids(session._token):
            assert asyncio.get_event_loop().time() < deadline, "browser process leaked"
            await asyncio.sleep(0.1)
        assert fallback._chromium_pids(session._token) == set()

    async def test_failed_close_still_hard_kills_leftover_pids(self, monkeypatch):
        session = fallback._BrowserSession()
        session._STEP_TIMEOUT_S = 0.2
        session._executable = "/fake/chrome"
        session._token = "victor-browser-test-token"

        class _BrokenBrowser:
            async def close(self):
                raise RuntimeError("close exploded")

        session._browser = _BrokenBrowser()

        killed: set = set()
        monkeypatch.setattr(
            fallback,
            "_chromium_pids",
            lambda tok: {4242} if tok == session._token else set(),
        )
        monkeypatch.setattr(fallback, "_kill_pids", lambda pids: killed.update(pids) or len(pids))

        await session.aclose()

        assert 4242 in killed
        assert session.killed_pids == {4242}

    async def test_aclose_is_idempotent(self, monkeypatch):
        session = fallback._BrowserSession()
        session._executable = "/fake/chrome"
        session._token = "victor-browser-test-token"

        calls: list = []
        monkeypatch.setattr(fallback, "_chromium_pids", lambda tok: set())
        monkeypatch.setattr(fallback, "_kill_pids", lambda pids: calls.append(tuple(pids)) or 0)

        await session.aclose()
        await session.aclose()

        assert calls == [()]


# ---------------------------------------------------------------------------
# Registration and discovery
# ---------------------------------------------------------------------------
