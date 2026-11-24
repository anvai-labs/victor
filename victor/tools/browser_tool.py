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

"""Headless-browser page read for JavaScript-required pages (FOU-A5 phase 2).

``browser_tool`` is the ``web_fetch`` companion: where ``web_fetch`` reads a
page over plain HTTP, this tool renders it in headless Chromium (Playwright)
and returns structured results — title, final URL, main text, and links —
for pages that need JS to produce content.

Guarantees (mirroring the web_fetch destination-policy hook that landed in
#1990, extended to the browser's request stream):

1. **Destination policy on EVERY request.** The
   ``tools.web_fetch_allowed_hosts`` allowlist (empty = admit everything)
   is checked on the initial URL BEFORE any browser process exists, then
   enforced inside the browser via request interception: every navigation
   hop (server redirects fire one request per hop) and every request the
   page makes (documents, XHR, subresources) is checked, and denied
   destinations are aborted. A denied document navigation returns the same
   structured ``destination_policy_denied`` result shape as web_fetch, and
   the final URL is re-checked after navigation (belt and braces).
2. **Download cap.** Every admitted response body is counted against a
   hard byte cap (2 MB, the ``web_fetch`` cap); the first response that
   would exceed it is aborted and the read fails with a structured
   ``download_cap_exceeded`` error. (``file://`` navigations bypass the
   network stack, so they cannot be intercepted or counted; the returned
   text is still size-bounded.)
3. **Timeout.** Navigation, selector waits, and extraction run under the
   per-navigation timeout; timeouts come back as structured
   ``timeout: true`` results.
4. **Hard browser-kill teardown.** The browser is launched with a unique
   env token, so its exact process tree can be found in ``/proc``. Teardown
   closes page/context/browser/driver under per-step timeouts, then
   SIGKILLs anything still alive under the token. The kill survives
   cancellation and close failures — a cancelled teardown still reaps
   (inline), and no other Chromium on the machine is touched.

Playwright is an OPTIONAL dependency (``pip install 'victor-ai[web]'`` plus
``playwright install chromium``); without it the tool returns a structured
``playwright_missing`` error instead of crashing.
"""

import asyncio
import logging
import os
import re
import signal
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin

from victor.tools.base import AccessMode, CostTier, DangerLevel, Priority
from victor.tools.decorators import tool
from victor.tools.web_search_tool import (
    _MAX_REDIRECT_HOPS,
    _PAGE_DOWNLOAD_CAP_BYTES,
    _check_destination,
    _destination_policy_error,
    _extract_content,
    _get_web_config,
    _pick_user_agent,
    DestinationPolicyDenied,
)

logger = logging.getLogger(__name__)

_VALID_WAIT_UNTIL = ("commit", "domcontentloaded", "load", "networkidle")

# Env var carrying the per-session token used to attribute chromium processes
# to THIS session precisely (process-tree ownership without collateral kills).
_BROWSER_ENV_TOKEN = "VICTOR_BROWSER_TOKEN"


class _NavigationRefused(Exception):
    """A navigation could not be admitted (redirect hop budget exhausted)."""


class _CapExceeded(Exception):
    """A response would push the session past its download cap."""


class BrowserDependencyError(Exception):
    """Playwright (package or browser binary) is not usable in this install.

    ``payload`` is the structured tool result to return.
    """

    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload
        super().__init__(payload.get("error", "playwright dependency missing"))


def _import_playwright() -> Tuple[Any, Any, Any]:
    """Import playwright lazily; raise BrowserDependencyError when absent.

    Returns ``(async_playwright, Error, TimeoutError)`` from
    ``playwright.async_api``.
    """
    try:
        from playwright.async_api import Error as pw_error
        from playwright.async_api import TimeoutError as pw_timeout
        from playwright.async_api import async_playwright
    except Exception as exc:
        raise BrowserDependencyError(
            {
                "success": False,
                "error": (
                    "Playwright is not installed. Install it with "
                    "`pip install 'victor-ai[web]'` and then run "
                    "`playwright install chromium` "
                    f"(import failed: {exc})."
                ),
                "playwright_missing": True,
            }
        ) from exc
    return async_playwright, pw_error, pw_timeout


def _chromium_pids(token: str) -> Set[int]:
    """Find live pids belonging to the browser session carrying ``token``.

    The token is a per-session UUID passed BOTH as a chromium launch
    argument (visible in the main process's ``/proc/<pid>/cmdline``) and in
    its launch environment (``/proc/<pid>/environ``). Child chromium
    processes don't carry the launch argument, but they live in the main
    process's detached process group, so :func:`_kill_pids` reaps them via
    ``killpg``. A concurrent browser session (or the user's own chrome) is
    never matched. Returns an empty set on any /proc access failure.
    """
    pids: Set[int] = set()
    if not token:
        return pids
    needle = token.encode()
    env_needle = f"{token}=".encode()
    try:
        entries = os.listdir("/proc")
    except OSError:
        return pids
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as fh:
                if needle in fh.read():
                    pids.add(int(entry))
                    continue
            with open(f"/proc/{entry}/environ", "rb") as fh:
                environ = fh.read()
        except (OSError, ValueError):
            continue
        if env_needle in environ:
            pids.add(int(entry))
    return pids


def _kill_pids(pids: Set[int]) -> int:
    """SIGKILL the session's browser processes; returns kills attempted.

    Playwright launches chromium detached (its own process-group leader),
    so killing the group takes the whole tree — child processes (zygote,
    GPU, renderer) don't carry the marker but share the group.
    """
    killed = 0
    for pid in pids:
        group_killed = False
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
            group_killed = True
        except (OSError, ValueError):
            pass
        try:
            os.kill(pid, signal.SIGKILL)
        except (OSError, ValueError):
            if not group_killed:
                continue
        killed += 1
    return killed


class _NavigationPolicy:
    """Per-session destination-policy + download-cap state.

    ``handle_route`` is the Playwright route handler. Redirects are followed
    MANUALLY inside the guard (``route.fetch(max_redirects=0)`` + Location
    resolution — the same structure as web_fetch's ``_fetch_page``): the
    destination policy is re-checked at EVERY hop and on EVERY request the
    page makes, and the browser never follows a redirect response itself
    (a fulfilled 3xx is followed by the network stack without firing
    another route event, which would bypass the guard). Denied destinations
    and download-cap overruns are answered with a synthetic empty page for
    document requests — so the navigation completes and ``read()`` maps the
    recorded denial deterministically — and with an abort for subresources.
    """

    def __init__(self, allowed_hosts: List[str], byte_cap: int):
        self.allowed_hosts = list(allowed_hosts or [])
        self.byte_cap = max(1, int(byte_cap))
        self.bytes_downloaded = 0
        self.denials: List[DestinationPolicyDenied] = []
        self.document_denied: Optional[DestinationPolicyDenied] = None
        self.cap_exceeded = False
        self.redirect_budget_exceeded: Optional[str] = None
        #: Last admitted hop of a document's redirect chain (the browser only
        #: sees the first URL when redirects are followed inside the guard).
        self.final_document_url: Optional[str] = None

    def check(self, url: str) -> None:
        """Admit one destination URL; raise DestinationPolicyDenied."""
        _check_destination(url, self.allowed_hosts)

    async def handle_route(self, route: Any, request: Any) -> None:
        url = request.url
        is_document = getattr(request, "resource_type", "") == "document"
        try:
            # Non-network schemes (file://) cannot be fetched/fulfilled
            # through the route API — pass them straight through. Policy
            # still applies: under a non-empty allowlist the scheme check in
            # _admit_and_fulfill already refuses them.
            if not url.lower().startswith(("http://", "https://")):
                await route.continue_()
                return
            await self._admit_and_fulfill(route, url, is_document)
        except DestinationPolicyDenied as denied:
            self.denials.append(denied)
            if is_document:
                self.document_denied = denied
            await self._settle(route, is_document)
        except _NavigationRefused as refused:
            if is_document:
                self.redirect_budget_exceeded = str(refused)
            await self._settle(route, is_document)
        except _CapExceeded:
            self.cap_exceeded = True
            await self._settle(route, is_document)
        except Exception:
            # The navigation was torn down (route aborted / page closed) or
            # the transport failed — never mask the original navigation error.
            try:
                await route.abort()
            except Exception:
                pass

    async def _admit_and_fulfill(self, route: Any, url: str, is_document: bool) -> None:
        """Policy-check + fetch one request, following redirects manually.

        Because redirects are followed inside the guard, the browser only
        ever sees the FIRST URL of a chain — ``final_document_url`` records
        the last admitted hop so ``read()`` can report and re-check the real
        destination.
        """
        current = url
        response = None
        for _ in range(_MAX_REDIRECT_HOPS):
            self.check(current)
            response = await route.fetch(url=current, max_redirects=0)
            if response.status in (301, 302, 303, 307, 308):
                location = response.headers.get("location") or ""
                if not location:
                    break
                current = urljoin(current, location)
                continue
            break
        else:
            raise _NavigationRefused(f"more than {_MAX_REDIRECT_HOPS} redirects: {url}")
        if is_document:
            self.final_document_url = current
        await self._count_and_fulfill(route, response)

    async def _count_and_fulfill(self, route: Any, response: Any) -> None:
        """Enforce the download cap on one admitted response, then fulfill."""
        declared = response.headers.get("content-length") or ""
        try:
            declared_bytes = int(declared) if declared else 0
        except ValueError:
            declared_bytes = 0
        if declared_bytes and self.bytes_downloaded + declared_bytes > self.byte_cap:
            raise _CapExceeded()
        body = await response.body()
        if self.bytes_downloaded + len(body) > self.byte_cap:
            raise _CapExceeded()
        self.bytes_downloaded += len(body)
        await route.fulfill(response=response)

    async def _settle(self, route: Any, is_document: bool) -> None:
        """Answer a refused request: synthetic page for documents (so the
        navigation completes and read() maps the recorded refusal
        deterministically), abort for subresources."""
        if is_document:
            try:
                await route.fulfill(
                    status=200,
                    content_type="text/html",
                    body="<html><body></body></html>",
                )
                return
            except Exception:
                pass
        try:
            await route.abort()
        except Exception:
            pass


class _BrowserSession:
    """One headless-chromium read with guaranteed hard cleanup.

    The browser is launched with a unique ``VICTOR_BROWSER_TOKEN`` env value;
    teardown (``aclose``) closes the page/context/browser/driver under
    per-step timeouts, then SIGKILLs any process still alive under the
    token. The kill is unconditional and survives cancellation: a cancelled
    ``aclose`` reaps inline before propagating, and the detached close task
    reaps again (idempotently) in the background.
    """

    _STEP_TIMEOUT_S: float = 4.0
    _TOTAL_TIMEOUT_S: float = 10.0

    def __init__(self) -> None:
        self._pw_cm: Any = None
        self._pw: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._executable = ""
        self._token = f"victor-browser-{uuid.uuid4().hex}"
        self._closed = False
        self._reap_task: Optional[asyncio.Task] = None
        #: Pids that had to be SIGKILLed (empty on a clean close).
        self.killed_pids: Set[int] = set()

    async def __aenter__(self) -> "_BrowserSession":
        async_playwright, pw_error, _ = _import_playwright()
        try:
            self._pw_cm = async_playwright()
            self._pw = await self._pw_cm.__aenter__()
            try:
                self._executable = self._pw.chromium.executable_path
            except Exception:
                self._executable = ""  # resolved at reap time only; hard kill degrades
            launch_env = {**os.environ, _BROWSER_ENV_TOKEN: self._token}
            self._browser = await self._pw.chromium.launch(
                headless=True,
                env=launch_env,
                # Chromium ignores unknown switches; this marks the main
                # process's cmdline for precise process-tree attribution.
                args=[f"--{_BROWSER_ENV_TOKEN}={self._token}"],
            )
        except pw_error as exc:
            message = str(exc)
            if "Executable doesn't exist" in message:
                raise BrowserDependencyError(
                    {
                        "success": False,
                        "error": (
                            "The playwright chromium browser is not downloaded. "
                            "Run `playwright install chromium` "
                            f"(launch failed: {message})."
                        ),
                        "browser_binary_missing": True,
                    }
                ) from exc
            raise BrowserDependencyError(
                {
                    "success": False,
                    "error": f"Failed to launch the headless browser: {message}",
                }
            ) from exc
        except Exception as exc:
            raise BrowserDependencyError(
                {
                    "success": False,
                    "error": f"Failed to launch the headless browser: {exc}",
                }
            ) from exc
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        await self.aclose()
        return False

    async def read(
        self,
        url: str,
        *,
        allowed_hosts: List[str],
        byte_cap: int = _PAGE_DOWNLOAD_CAP_BYTES,
        wait_until: str = "networkidle",
        selector: str = "",
        timeout_seconds: float = 30.0,
        max_links: int = 50,
        max_content_length: int = 5000,
    ) -> Dict[str, Any]:
        """Render one URL and return structured title/url/text/links.

        Never raises: policy denials, timeouts, download-cap hits, and
        navigation failures all come back as structured error dicts.
        """
        policy = _NavigationPolicy(allowed_hosts=allowed_hosts, byte_cap=byte_cap)
        timeout_ms = int(max(1.0, float(timeout_seconds)) * 1000)
        _, pw_error, pw_timeout = _import_playwright()

        context = await self._browser.new_context(user_agent=_pick_user_agent())
        self._context = context
        page = await context.new_page()
        try:
            # Route at the CONTEXT level, not the page: popup pages spawned by
            # the document would otherwise bypass the policy guard entirely.
            await context.route("**/*", policy.handle_route)
            try:
                await page.goto(url, wait_until=wait_until, timeout=timeout_ms)
            except pw_timeout:
                if policy.document_denied is not None:
                    return _destination_policy_error(policy.document_denied)
                return {
                    "success": False,
                    "error": (
                        f"Page load timed out after {timeout_seconds}s "
                        f"(wait_until={wait_until}): {url}"
                    ),
                    "timeout": True,
                    "url": url,
                }
            except pw_error:
                if policy.document_denied is not None:
                    return _destination_policy_error(policy.document_denied)
                if policy.cap_exceeded:
                    return self._cap_error(url)
                return {
                    "success": False,
                    "error": f"Navigation failed: {url}",
                    "url": url,
                }

            if policy.document_denied is not None:
                return _destination_policy_error(policy.document_denied)
            if policy.redirect_budget_exceeded is not None:
                return {
                    "success": False,
                    "error": (
                        f"Redirect hop budget exhausted while fetching {url}: "
                        f"{policy.redirect_budget_exceeded}"
                    ),
                    "redirect_budget_exceeded": True,
                    "url": url,
                }
            if policy.cap_exceeded:
                return self._cap_error(url)

            # With redirects followed inside the guard, the browser only sees
            # the first URL — report and re-check the guard's final hop.
            final_url = policy.final_document_url or page.url
            try:
                policy.check(final_url)
            except DestinationPolicyDenied as denied:
                # Belt and braces: a navigation that bypassed interception
                # (or a JS-driven redirect) still cannot escape the policy.
                return _destination_policy_error(denied)

            if selector:
                await page.wait_for_selector(selector, timeout=timeout_ms)

            title = (await page.title() or "").strip()
            html = await page.content()
            raw_links = await page.eval_on_selector_all(
                "a[href]",
                (
                    "els => els.map(e => ({"
                    "text: ((e.innerText || e.textContent || '') + '').trim().slice(0, 200),"
                    "href: e.href || e.getAttribute('href') || ''"
                    "}))"
                ),
            )
            links = self._normalize_links(raw_links, final_url, max_links)

            content = _extract_content(html, max_length=max_content_length)
            truncated = len(content) >= max_content_length
            if not content:
                content = (await page.locator("body").inner_text(timeout=timeout_ms) or "").strip()
                content = re.sub(r"\n\s*\n", "\n\n", content)
                content = re.sub(r" +", " ", content)
                content = content[:max_content_length]

            return {
                "success": True,
                "title": title,
                "url": final_url,
                "content": content,
                "links": links,
                "link_count": len(links),
                "html_length": len(html or ""),
                "bytes_downloaded": policy.bytes_downloaded,
                "denied_subresource_count": len(policy.denials)
                - (1 if policy.document_denied is not None else 0),
                "render": "browser",
                "truncated": truncated,
            }
        except pw_timeout:
            return {
                "success": False,
                "error": f"Page read timed out after {timeout_seconds}s: {url}",
                "timeout": True,
                "url": url,
            }
        finally:
            try:
                await asyncio.wait_for(page.close(), timeout=self._STEP_TIMEOUT_S)
            except Exception:
                pass

    @staticmethod
    def _cap_error(url: str) -> Dict[str, Any]:
        return {
            "success": False,
            "error": (
                f"Page exceeded the {_PAGE_DOWNLOAD_CAP_BYTES} byte download cap "
                "and cannot be read"
            ),
            "download_cap_exceeded": True,
            "url": url,
        }

    @staticmethod
    def _normalize_links(
        raw_links: List[Dict[str, Any]], base_url: str, max_links: int
    ) -> List[Dict[str, str]]:
        """Absolute-ize, filter, dedupe, and cap the extracted links."""
        links: List[Dict[str, str]] = []
        seen: Set[str] = set()
        for item in raw_links or []:
            href = str(item.get("href") or "").strip()
            if not href or href.startswith(("javascript:", "data:", "mailto:", "tel:")):
                continue
            absolute = urljoin(base_url, href)
            if absolute.startswith("file://") is False and not absolute.startswith(
                ("http://", "https://")
            ):
                continue
            if absolute in seen:
                continue
            seen.add(absolute)
            links.append({"text": str(item.get("text") or "")[:200], "href": absolute})
            if len(links) >= max(0, int(max_links)):
                break
        return links

    async def aclose(self) -> None:
        """Guaranteed teardown: graceful closes, then a precise hard kill.

        Cancellation-safe: if ``aclose`` itself is cancelled, the pids are
        reaped inline before the cancellation propagates, and the detached
        close task reaps again idempotently in the background.
        """
        if self._closed:
            return
        self._closed = True
        close_task = asyncio.ensure_future(self._close_and_reap())
        self._reap_task = close_task
        try:
            await asyncio.wait_for(asyncio.shield(close_task), timeout=self._TOTAL_TIMEOUT_S)
        except asyncio.CancelledError:
            # A cancelled teardown must not skip the kill.
            self._reap_now()
            raise
        except Exception:
            self._reap_now()

    async def _close_and_reap(self) -> None:
        if self._context is not None:
            await self._step(self._context.close)
        if self._browser is not None:
            await self._step(self._browser.close)
        if self._pw_cm is not None:
            await self._step(lambda: self._pw_cm.__aexit__(None, None, None))
        self._reap_now()

    async def _step(self, fn: Callable[[], Any]) -> None:
        try:
            await asyncio.wait_for(fn(), timeout=self._STEP_TIMEOUT_S)
        except Exception:
            pass  # hard kill below is the safety net

    def _reap_now(self) -> None:
        leftover = _chromium_pids(self._token)
        if leftover:
            logger.warning(
                "[browser_tool] hard-killing %d leftover chromium process(es) for session %s",
                len(leftover),
                self._token,
            )
        self.killed_pids |= leftover
        _kill_pids(leftover)


# Test/embedding seam: override the browser lifecycle wholesale.
_SESSION_FACTORY: Optional[Callable[[], _BrowserSession]] = None


def set_browser_session_factory(
    factory: Optional[Callable[[], _BrowserSession]],
) -> None:
    """Override the browser-session factory (tests pass ``None`` to reset)."""
    global _SESSION_FACTORY
    _SESSION_FACTORY = factory


def _get_session_factory() -> Callable[[], _BrowserSession]:
    return _SESSION_FACTORY or _BrowserSession


@tool(
    name="browser_tool",
    cost_tier=CostTier.HIGH,
    category="web",
    priority=Priority.LOW,  # companion to web_fetch; only for JS-required pages
    access_mode=AccessMode.NETWORK,
    danger_level=DangerLevel.SAFE,
    signature_params=["url"],  # different URLs indicate progress, not loops
    stages=["planning", "initial"],
    task_types=["research", "analysis"],
    execution_category="network",
    keywords=[
        "browser",
        "render page",
        "javascript",
        "headless",
        "spa",
        "dynamic page",
        "requires javascript",
        "js-rendered",
    ],
)
async def browser_tool(
    url: str,
    wait_until: str = "networkidle",
    selector: str = "",
    timeout_seconds: float = 30.0,
    max_links: int = 50,
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Render a JavaScript-required page in a headless browser and extract content.

    Purpose:
    - Read pages that need JS to produce content (SPAs, dynamic pages).
    - Return title, final URL, main text, and outbound links as structured data.
    - Companion to web_fetch: use web_fetch first; use this when extraction
      comes back empty or the page explicitly requires JavaScript.

    The destination policy applies identically to web_fetch: when
    ``tools.web_fetch_allowed_hosts`` is non-empty, only those hosts (and
    their subdomains) may be loaded — enforced on the initial URL before any
    browser launches, on every navigation hop and request the page makes
    (violations abort the request and return a structured
    ``destination_policy_denied`` error), and on the final URL. A hard
    download cap (2 MB) and a per-navigation timeout bound every read, and
    teardown SIGKILLs any surviving browser process, so nothing leaks.

    Args:
        url: URL to render and read.
        wait_until: When navigation is considered done: 'networkidle'
            (default), 'load', 'domcontentloaded', or 'commit'.
        selector: Optional CSS selector to wait for before extraction (e.g.
            '#results' for content that appears late).
        timeout_seconds: Per-navigation timeout (default: 30).
        max_links: Maximum number of outbound links to return (default: 50).
        _exec_ctx: Tool execution context containing ToolConfig.

    Returns:
        Dictionary containing:
        - success: Whether the page was rendered and read
        - title: Page title
        - url: Final URL after any redirects
        - content: Extracted main text
        - links: List of {text, href} dicts (absolute hrefs, deduplicated)
        - link_count: Number of links returned
        - bytes_downloaded / html_length / truncated: size accounting
        - render: Always 'browser'
        - error / destination_policy_denied / timeout /
          download_cap_exceeded / playwright_missing /
          browser_binary_missing: structured failure fields
    """
    if not url:
        return {"success": False, "error": "Missing required parameter: url"}
    wait_mode = (wait_until or "networkidle").lower().strip()
    if wait_mode not in _VALID_WAIT_UNTIL:
        return {
            "success": False,
            "error": (
                f"Invalid wait_until '{wait_until}'. "
                f"Use one of: {', '.join(_VALID_WAIT_UNTIL)}."
            ),
        }

    config = _get_web_config(_exec_ctx)
    allowed_hosts = list(config.get("web_fetch_allowed_hosts", []) or [])
    max_content_length = int(config.get("max_content_length", 5000) or 5000)

    # Destination policy BEFORE anything else — no browser process exists
    # while a denied URL is refused (same entry-point rule as web_fetch).
    try:
        _check_destination(url, allowed_hosts)
    except DestinationPolicyDenied as denied:
        return _destination_policy_error(denied)

    try:
        factory = _get_session_factory()
        session = factory()
        async with session:
            return await session.read(
                url,
                allowed_hosts=allowed_hosts,
                byte_cap=_PAGE_DOWNLOAD_CAP_BYTES,
                wait_until=wait_mode,
                selector=selector,
                timeout_seconds=timeout_seconds,
                max_links=max_links,
                max_content_length=max_content_length,
            )
    except BrowserDependencyError as dep:
        return dep.payload
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return {"success": False, "error": f"Browser read failed: {exc}"}
