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
page over plain HTTP, this tool renders it in a real browser session and
returns structured results — title, final URL, main text, and links — for
pages that need JS to produce content.

**Substrate (owner direction, FOU-A5 phase 2): the primary engine is the
agentbrowser service** (the owner's product; HTTP API on ``127.0.0.1:5709``,
override with ``AGENTBROWSER_BASE_URL``, authenticate with
``AGENTBROWSER_API_KEY``). This module is a thin HTTP CLIENT of that service:
per read it creates a session (``headless: true`` — agentbrowser's Cedar
capability authorizer gates ``session.headed``, so Victor never asks for a
headed session), creates a page, navigates, extracts (``text`` + ``links``),
reads the title, and closes the session in a ``finally`` (a failed close
never masks the read result). An explicit ``session_id`` reuses a live
session instead of creating/closing one.

Policy division, by design:

- **Victor** enforces the ``tools.web_fetch_allowed_hosts`` destination
  allowlist (the #1990 web_fetch policy class) on the INITIAL URL before
  any call is made — a denied host never reaches the service, and the
  final URL is re-checked after navigation (belt and braces).
- **agentbrowser** enforces its own Cedar capability policies and network
  admission on every navigation (its own denial traces); a ``blocked``
  navigation status is surfaced as a structured ``navigation_blocked``
  result carrying the service's reason.

Every failure is a structured dict, never a raise: service unreachable
(``agentbrowser_unreachable`` — names the startup command),
``agentbrowser_denied`` / ``navigation_blocked``, ``timeout``, policy
denial, and — for the documented playwright fallback engine
(``engine="playwright"``, ``victor.tools.browser_playwright_fallback``) —
``playwright_missing`` / ``browser_binary_missing`` /
``download_cap_exceeded``.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import httpx

from victor.tools.base import AccessMode, CostTier, DangerLevel, Priority
from victor.tools.decorators import tool
from victor.tools.web_search_tool import (
    _PAGE_DOWNLOAD_CAP_BYTES,
    _check_destination,
    _destination_policy_error,
    _get_web_config,
    DestinationPolicyDenied,
)

logger = logging.getLogger(__name__)

# agentbrowser service defaults (ADR-017: 127.0.0.1:5709; HOST/PORT env).
_DEFAULT_SERVICE_URL = "http://127.0.0.1:5709"
_SERVICE_ENV_URL = "AGENTBROWSER_BASE_URL"
_SERVICE_ENV_KEY = "AGENTBROWSER_API_KEY"
# agentbrowser derives the tenant from the bearer key when authenticated;
# this claim is used only on unauthenticated local deployments.
_TENANT_ID = "victor"

# agentbrowser's navigate waitUntil enum (openapi.ts SessionRequest /
# navigate route). The playwright fallback additionally accepts "commit".
_AGENTBROWSER_WAIT_UNTIL = ("networkidle", "load", "domcontentloaded")
_FALLBACK_ONLY_WAIT_UNTIL = ("commit",)

_SERVICE_UNREACHABLE_TEMPLATE = (
    "The agentbrowser service is not reachable at {base_url}. Start it with: "
    "`node packages/api/dist/bin.js` from the agentbrowser checkout "
    "(HOST/PORT env override the bind address; the Cedar capability policy "
    "file is configured via AGENTBROWSER_POLICY_FILE). Or deploy the "
    'documented local fallback with `engine="playwright"` '
    "(pip install 'victor-ai[browser-playwright]' plus "
    "`playwright install chromium`)."
)

# Startup hint surfaced verbatim in unreachable-service errors.
_SERVICE_START_COMMAND = "node packages/api/dist/bin.js"


class _AgentbrowserUnavailable(Exception):
    """The agentbrowser service could not be reached.

    ``payload`` is the structured tool result to return.
    """

    def __init__(self, base_url: str, detail: str = ""):
        self.base_url = base_url
        self.payload = {
            "success": False,
            "error": _SERVICE_UNREACHABLE_TEMPLATE.format(base_url=base_url)
            + (f" ({detail})" if detail else ""),
            "agentbrowser_unreachable": True,
            "service_url": base_url,
            "start_command": _SERVICE_START_COMMAND,
        }
        super().__init__(self.payload["error"])


class _AgentbrowserApiError(Exception):
    """An agentbrowser ApiError envelope (protocol-shaped refusal).

    ``payload`` is the structured tool result to return.
    """

    def __init__(self, code: str, message: str, status_code: int, retryable: bool):
        self.code = code
        self.payload = {
            "success": False,
            "error": f"agentbrowser {code}: {message}",
            "agentbrowser_error": True,
            "agentbrowser_error_code": code,
            "http_status": status_code,
            "retryable": retryable,
        }
        super().__init__(self.payload["error"])


def _service_base_url() -> str:
    """Resolve the agentbrowser service base URL (env-overridable)."""
    return (os.environ.get(_SERVICE_ENV_URL) or _DEFAULT_SERVICE_URL).rstrip("/")


class _AgentbrowserClient:
    """Thin async client for the agentbrowser HTTP API (openapi 1.0.0).

    Only the read surface Victor needs: session create/close, page
    create/get, navigate, extract (text + links). Every method maps
    protocol ApiError envelopes to :class:`_AgentbrowserApiError` and
    transport failures to :class:`_AgentbrowserUnavailable`.
    """

    def __init__(self, base_url: str, timeout_seconds: float):
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(max(1.0, float(timeout_seconds)))
        self._tenant_body = {"tenantId": _TENANT_ID}
        headers = {"Accept": "application/json"}
        api_key = os.environ.get(_SERVICE_ENV_KEY)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            # With a bearer key the server derives the tenant from the key
            # (SHA-256(key) -> tenantId); a body tenantId that disagrees is
            # refused FORBIDDEN. Only unauthenticated local deployments
            # declare the tenant in the body.
            self._tenant_body = {}
        self._headers = headers

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    @staticmethod
    async def _raise_for_envelope(response: httpx.Response) -> None:
        """Map a protocol ApiError envelope to _AgentbrowserApiError."""
        try:
            body = response.json()
        except Exception:
            body = {}
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict) and error.get("code"):
            raise _AgentbrowserApiError(
                code=str(error.get("code")),
                message=str(error.get("message") or "request refused"),
                status_code=response.status_code,
                retryable=bool(error.get("retryable")),
            )
        raise _AgentbrowserApiError(
            code="HTTP_ERROR",
            message=f"agentbrowser returned HTTP {response.status_code}",
            status_code=response.status_code,
            retryable=False,
        )

    async def _request(self, method: str, path: str, json_body: Any = None) -> Any:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.request(
                    method, self._url(path), json=json_body, headers=self._headers
                )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise _AgentbrowserUnavailable(self._base_url, str(exc)) from exc
        except httpx.TimeoutException as exc:
            raise _AgentbrowserUnavailable(
                self._base_url, f"timed out after {self._timeout.connect}s+"
            ) from exc
        except httpx.HTTPError as exc:
            raise _AgentbrowserUnavailable(self._base_url, str(exc)) from exc
        if response.status_code >= 400:
            await self._raise_for_envelope(response)
        if response.status_code == 204 or not response.content:
            return {}
        return response.json()

    async def create_session(self) -> Dict[str, Any]:
        """POST /v1/sessions — headless by default (Cedar gates headed)."""
        return await self._request(
            "POST",
            "/v1/sessions",
            {**self._tenant_body, "headless": True},
        )

    async def create_page(self, session_id: str, url: str = "") -> Dict[str, Any]:
        """POST /v1/sessions/{sessionId}/pages — page starts blank when
        ``url`` is omitted; the explicit navigate carries the wait policy."""
        body = {"url": url} if url else {}
        return await self._request("POST", f"/v1/sessions/{session_id}/pages", body)

    async def get_page(self, session_id: str, page_id: str) -> Dict[str, Any]:
        """GET /v1/sessions/{sessionId}/pages/{pageId} — PageSummary."""
        return await self._request("GET", f"/v1/sessions/{session_id}/pages/{page_id}")

    async def navigate(
        self, session_id: str, page_id: str, url: str, wait_until: str
    ) -> Dict[str, Any]:
        """POST .../pages/{pageId}/navigate — NavigationStatus."""
        return await self._request(
            "POST",
            f"/v1/sessions/{session_id}/pages/{page_id}/navigate",
            {"url": url, "waitUntil": wait_until},
        )

    async def extract(self, session_id: str, page_id: str, fmt: str) -> Any:
        """POST .../pages/{pageId}/extract — {data, evidence}."""
        body = await self._request(
            "POST",
            f"/v1/sessions/{session_id}/pages/{page_id}/extract",
            {"format": fmt},
        )
        data = body.get("data") if isinstance(body, dict) else None
        if data is None:
            raise _AgentbrowserApiError(
                code="EMPTY_EXTRACTION",
                message=f"extract format {fmt!r} returned no data",
                status_code=200,
                retryable=False,
            )
        return data

    async def close_session(self, session_id: str) -> None:
        """DELETE /v1/sessions/{sessionId} — best-effort teardown."""
        try:
            await self._request("DELETE", f"/v1/sessions/{session_id}")
        except Exception as exc:
            logger.debug("[browser_tool] session close failed: %s", exc)


def _normalize_links(raw_links: Any, base_url: str, max_links: int) -> List[Dict[str, str]]:
    """agentbrowser links extraction ([{text, url}]) → filtered, deduped,
    absolute-href, capped list (same shape as the playwright fallback)."""
    links: List[Dict[str, str]] = []
    seen: set = set()
    for item in raw_links or []:
        if not isinstance(item, dict):
            continue
        href = str(item.get("url") or item.get("href") or "").strip()
        if not href or href.startswith(("javascript:", "data:", "mailto:", "tel:")):
            continue
        absolute = urljoin(base_url, href)
        if not absolute.startswith(("http://", "https://", "file://")):
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        links.append({"text": str(item.get("text") or "")[:200], "href": absolute})
        if len(links) >= max(0, int(max_links)):
            break
    return links


async def _read_via_agentbrowser(
    url: str,
    *,
    client: _AgentbrowserClient,
    wait_until: str,
    timeout_seconds: float,
    max_links: int,
    max_content_length: int,
    allowed_hosts: List[str],
    existing_session_id: str,
) -> Dict[str, Any]:
    """One read over the agentbrowser surface: session → page → navigate →
    extract (text + links) → title → close. Closes the session it created
    even when a later step fails; never raises."""
    session: Optional[Dict[str, Any]] = None
    session_id = existing_session_id
    try:
        if not session_id:
            session = await client.create_session()
            session_id = str(session.get("sessionId") or "")
            if not session_id:
                return {
                    "success": False,
                    "error": "agentbrowser session creation returned no sessionId",
                    "agentbrowser_error": True,
                }
        page = await client.create_page(session_id)
        page_id = str(page.get("pageId") or "")
        if not page_id:
            return {
                "success": False,
                "error": "agentbrowser page creation returned no pageId",
                "agentbrowser_error": True,
                "session_id": session_id,
            }

        navigation = await client.navigate(session_id, page_id, url, wait_until)
        nav_status = str(navigation.get("status") or "")
        final_url = str(navigation.get("url") or url)
        if nav_status == "timeout":
            return {
                "success": False,
                "error": (
                    f"Page load timed out after {timeout_seconds}s "
                    f"(wait_until={wait_until}): {url}"
                ),
                "timeout": True,
                "url": final_url,
            }
        if nav_status == "blocked":
            # agentbrowser's Cedar admission stack refused the destination.
            return {
                "success": False,
                "error": (
                    "agentbrowser navigation blocked by its capability policy "
                    f"(reason: {navigation.get('reason') or 'policy'}): {final_url}"
                ),
                "navigation_blocked": True,
                "blocked_reason": navigation.get("reason"),
                "scope_denials": navigation.get("scopeDenials"),
                "url": final_url,
            }

        try:
            _check_destination(final_url, allowed_hosts)
        except DestinationPolicyDenied as denied:
            # Belt and braces: agentbrowser admitted it, but the Victor
            # destination allowlist still governs what we READ.
            return _destination_policy_error(denied)

        summary = await client.get_page(session_id, page_id)
        title = str(summary.get("title") or navigation.get("title") or "").strip()

        text_data = await client.extract(session_id, page_id, "text")
        text = text_data.get("text", "") if isinstance(text_data, dict) else str(text_data or "")
        text = str(text or "").strip()
        truncated = len(text) > max_content_length
        content = text[:max_content_length]

        raw_links = await client.extract(session_id, page_id, "links")
        links = _normalize_links(raw_links, final_url, max_links)

        return {
            "success": True,
            "title": title,
            "url": final_url,
            "content": content,
            "links": links,
            "link_count": len(links),
            "render": "browser",
            "engine": "agentbrowser",
            "session_id": session_id,
            "redirect_chain": navigation.get("redirectChain") or [],
            "truncated": truncated,
        }
    except _AgentbrowserUnavailable as unavailable:
        return unavailable.payload
    except _AgentbrowserApiError as api_error:
        payload = dict(api_error.payload)
        if session_id:
            payload["session_id"] = session_id
        return payload
    except httpx.TimeoutException:
        return {
            "success": False,
            "error": f"agentbrowser request timed out after {timeout_seconds}s: {url}",
            "timeout": True,
            "url": url,
        }
    finally:
        if session and session_id:
            # Only close sessions WE created (an explicit session_id is
            # the caller's to manage).
            await client.close_session(session_id)


async def _read_via_playwright_fallback(
    url: str,
    *,
    allowed_hosts: List[str],
    wait_until: str,
    selector: str,
    timeout_seconds: float,
    max_links: int,
    max_content_length: int,
) -> Dict[str, Any]:
    """Documented local fallback: the playwright engine module (see
    ``victor.tools.browser_playwright_fallback`` for its policy/cleanup
    guarantees). Never raises."""
    from victor.tools.browser_playwright_fallback import (
        BrowserDependencyError,
        _get_session_factory,
    )

    try:
        factory = _get_session_factory()
        session = factory()
        async with session:
            return await session.read(
                url,
                allowed_hosts=allowed_hosts,
                byte_cap=_PAGE_DOWNLOAD_CAP_BYTES,
                wait_until=wait_until,
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
    engine: str = "agentbrowser",
    session_id: str = "",
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Render a JavaScript-required page in a browser and extract content.

    Purpose:
    - Read pages that need JS to produce content (SPAs, dynamic pages).
    - Return title, final URL, main text, and outbound links as structured data.
    - Companion to web_fetch: use web_fetch first; use this when extraction
      comes back empty or the page explicitly requires JavaScript.

    The primary substrate is the **agentbrowser service** (default
    ``http://127.0.0.1:5709``; ``AGENTBROWSER_BASE_URL`` /
    ``AGENTBROWSER_API_KEY`` override). Victor enforces the destination
    allowlist (``tools.web_fetch_allowed_hosts``, the web_fetch policy) on
    the initial URL before any call and re-checks the final URL; agentbrowser's
    Cedar capability stack enforces its own navigation admission (a blocked
    navigation returns a structured ``navigation_blocked`` result with the
    service's reason). Sessions run headless and are closed after the read;
    pass ``session_id`` to reuse a live session you manage. If the service is
    unreachable the tool returns a structured error naming the startup command
    — or deploy the documented local fallback with ``engine="playwright"``
    (optional ``browser-playwright`` extra).

    Args:
        url: URL to render and read.
        wait_until: When navigation is considered done: 'networkidle'
            (default), 'load', or 'domcontentloaded' (agentbrowser enum;
            the playwright fallback also accepts 'commit').
        selector: Optional CSS selector to wait for before extraction
            (playwright fallback engine only).
        timeout_seconds: Per-request/navigation timeout (default: 30).
        max_links: Maximum number of outbound links to return (default: 50).
        engine: 'agentbrowser' (default; the service) or 'playwright'
            (the documented local fallback).
        session_id: Reuse this live agentbrowser session instead of
            creating and closing one per call.
        _exec_ctx: Tool execution context containing ToolConfig.

    Returns:
        Dictionary containing:
        - success: Whether the page was rendered and read
        - title: Page title
        - url: Final URL after any redirects
        - content: Extracted main text
        - links: List of {text, href} dicts (absolute hrefs, deduplicated)
        - link_count: Number of links returned
        - engine / session_id / redirect_chain / truncated: read accounting
        - error / destination_policy_denied / navigation_blocked / timeout /
          agentbrowser_unreachable / agentbrowser_error / playwright_missing /
          browser_binary_missing: structured failure fields
    """
    if not url:
        return {"success": False, "error": "Missing required parameter: url"}
    wait_mode = (wait_until or "networkidle").lower().strip()
    engine_mode = (engine or "agentbrowser").lower().strip()
    if engine_mode not in ("agentbrowser", "playwright"):
        return {
            "success": False,
            "error": "Invalid engine. Use 'agentbrowser' or 'playwright'.",
        }
    if wait_mode not in _AGENTBROWSER_WAIT_UNTIL + _FALLBACK_ONLY_WAIT_UNTIL:
        return {
            "success": False,
            "error": (
                f"Invalid wait_until '{wait_until}'. "
                f"Use one of: {', '.join(_AGENTBROWSER_WAIT_UNTIL + _FALLBACK_ONLY_WAIT_UNTIL)}."
            ),
        }
    if engine_mode == "agentbrowser" and wait_mode == "commit":
        return {
            "success": False,
            "error": "wait_until='commit' is only supported by the playwright fallback engine.",
        }

    config = _get_web_config(_exec_ctx)
    allowed_hosts = list(config.get("web_fetch_allowed_hosts", []) or [])
    max_content_length = int(config.get("max_content_length", 5000) or 5000)

    # Destination policy BEFORE anything else — a denied URL never reaches
    # the service or a local browser (same entry-point rule as web_fetch).
    try:
        _check_destination(url, allowed_hosts)
    except DestinationPolicyDenied as denied:
        return _destination_policy_error(denied)

    if engine_mode == "agentbrowser":
        client = _AgentbrowserClient(
            base_url=_service_base_url(),
            timeout_seconds=timeout_seconds,
        )
        return await _read_via_agentbrowser(
            url,
            client=client,
            wait_until=wait_mode,
            timeout_seconds=timeout_seconds,
            max_links=max_links,
            max_content_length=max_content_length,
            allowed_hosts=allowed_hosts,
            existing_session_id=session_id.strip(),
        )
    return await _read_via_playwright_fallback(
        url,
        allowed_hosts=allowed_hosts,
        wait_until=wait_mode,
        selector=selector,
        timeout_seconds=timeout_seconds,
        max_links=max_links,
        max_content_length=max_content_length,
    )
