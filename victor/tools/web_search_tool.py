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

"""Web search tool for accessing online information.

This tool provides:
1. Web search using DuckDuckGo (no API key required)
2. Result parsing and summarization
3. Content extraction from URLs
4. Source citations
5. Context injection for LLM
"""

import asyncio
import logging
import re
import time
from threading import RLock
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from victor.tools.base import AccessMode, CostTier, DangerLevel, Priority, ToolConfig
from victor.tools.decorators import tool
from victor.storage.cache.generic_result_cache import GenericResultCache, ResultType

# Constants
# Rotating User-Agent pool. A single static UA is trivially fingerprinted and
# blocked by search endpoints / anti-bot layers. Requests pick one at random.
_USER_AGENTS = (
    "Mozilla/5.0 (compatible; Victor/1.0; +https://github.com/vijaykumar/victor)",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
)
_USER_AGENT = _USER_AGENTS[0]  # Backwards-compat alias for static imports.
_DEFAULT_MAX_CONTENT_LENGTH = 5000

# DuckDuckGo endpoints tried in order. The HTML endpoint is the primary path
# but rate-limits aggressively and blocks headless UAs; `lite` is a lighter
# fallback; the rendered main site is the last resort (browser-based).
_DDG_HTML_URL = "https://html.duckduckgo.com/html/"
_DDG_LITE_URL = "https://lite.duckduckgo.com/lite/"
_DDG_RENDER_URL_TMPL = "https://duckduckgo.com/?q={query}"


def _pick_user_agent() -> str:
    """Return a random User-Agent from the rotating pool."""
    import random

    return random.choice(_USER_AGENTS)


_GENERIC_WEB_CACHE: Optional[GenericResultCache] = None
_GENERIC_WEB_CACHE_LOCK = RLock()


class RateLimiter:
    """Rate limiter with exponential backoff for HTTP requests.

    Tracks request timestamps per host and enforces rate limits.
    Implements exponential backoff retry on 429 (Too Many Requests) errors.

    Configuration:
        - requests_per_minute: Maximum requests per minute per host (default: 10)
        - max_retries: Maximum retry attempts (default: 5)
        - initial_delay: Initial backoff delay in seconds (default: 1.0)
        - max_delay: Maximum backoff delay in seconds (default: 60.0)
        - backoff_multiplier: Multiplier for exponential backoff (default: 2.0)
    """

    def __init__(
        self,
        requests_per_minute: int = 10,
        max_retries: int = 5,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_multiplier: float = 2.0,
    ):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute per host
            max_retries: Maximum retry attempts on rate limit errors
            initial_delay: Initial backoff delay in seconds
            max_delay: Maximum backoff delay in seconds
            backoff_multiplier: Multiplier for exponential backoff
        """
        self.requests_per_minute = requests_per_minute
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.min_interval = 60.0 / requests_per_minute

        # Track request timestamps per host
        self._request_times: Dict[str, List[float]] = {}
        self._lock = RLock()

        # Track retry counts per request
        self._retry_counts: Dict[str, int] = {}

        # Logger instance
        self._logger = logging.getLogger(__name__)

    def _get_host(self, url: str) -> str:
        """Extract host from URL.

        Args:
            url: URL to extract host from

        Returns:
            Host name (e.g., "html.duckduckgo.com")
        """
        parsed = urlparse(url)
        return parsed.netloc or url

    async def acquire(self, url: str) -> None:
        """Acquire permission to make a request.

        Blocks if rate limit would be exceeded. Implements token bucket algorithm.

        Args:
            url: Target URL
        """
        host = self._get_host(url)
        now = time.time()

        with self._lock:
            # Clean old requests outside the 1-minute window
            if host in self._request_times:
                cutoff = now - 60.0
                self._request_times[host] = [ts for ts in self._request_times[host] if ts > cutoff]

            # Check if we can make a request now
            if self._request_times.get(host, []):
                last_request = self._request_times[host][-1]
                elapsed = now - last_request

                if elapsed < self.min_interval:
                    # Need to wait
                    wait_time = self.min_interval - elapsed
                    self._logger.debug(
                        f"[RateLimiter] Rate limit: waiting "
                        f"{wait_time:.2f}s before request to {host}"
                    )
                    await asyncio.sleep(wait_time)

            # Record this request
            self._request_times.setdefault(host, []).append(time.time())

    def on_rate_limit_error(self, url: str, attempt: int) -> float:
        """Calculate backoff delay for rate limit error.

        Args:
            url: Target URL
            attempt: Retry attempt number (0-indexed)

        Returns:
            Delay in seconds before next retry
        """
        # Exponential backoff: initial_delay * (backoff_multiplier ^ attempt)
        delay = min(
            self.initial_delay * (self.backoff_multiplier**attempt),
            self.max_delay,
        )

        self._logger.warning(
            f"[RateLimiter] Rate limit error on attempt {attempt + 1} "
            f"to {self._get_host(url)}, backing off {delay:.2f}s"
        )

        return delay

    def should_retry(self, url: str, attempt: int) -> bool:
        """Check if we should retry a failed request.

        Args:
            url: Target URL
            attempt: Current retry attempt

        Returns:
            True if should retry
        """
        return attempt < self.max_retries

    def reset(self, url: Optional[str] = None) -> None:
        """Reset rate limiter state.

        Args:
            url: If provided, only reset for this host. Otherwise reset all.
        """
        with self._lock:
            if url:
                host = self._get_host(url)
                self._request_times.pop(host, None)
                self._retry_counts.pop(host, None)
            else:
                self._request_times.clear()
                self._retry_counts.clear()


# Global rate limiter instance
_RATE_LIMITER: Optional[RateLimiter] = None
_RATE_LIMITER_LOCK = RLock()


def _get_rate_limiter(
    requests_per_minute: int = 10,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_multiplier: float = 2.0,
) -> RateLimiter:
    """Get or create global rate limiter instance.

    Args:
        requests_per_minute: Maximum requests per minute per host
        max_retries: Maximum retry attempts
        initial_delay: Initial backoff delay in seconds
        max_delay: Maximum backoff delay in seconds
        backoff_multiplier: Multiplier for exponential backoff

    Returns:
        RateLimiter instance
    """
    global _RATE_LIMITER

    if _RATE_LIMITER is None:
        with _RATE_LIMITER_LOCK:
            if _RATE_LIMITER is None:
                _RATE_LIMITER = RateLimiter(
                    requests_per_minute=requests_per_minute,
                    max_retries=max_retries,
                    initial_delay=initial_delay,
                    max_delay=max_delay,
                    backoff_multiplier=backoff_multiplier,
                )

    return _RATE_LIMITER


def _get_web_config(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get web tool configuration from ToolConfig in execution context.

    Args:
        context: Tool execution context containing ToolConfig

    Returns:
        Dictionary with web tool settings (fetch_top, fetch_pool, max_content_length)
    """
    config = ToolConfig.from_context(context) if context else None
    if config:
        return {
            "provider": config.provider,
            "model": config.model,
            "fetch_top": config.web_fetch_top,
            "fetch_pool": config.web_fetch_pool,
            "max_content_length": config.max_content_length or _DEFAULT_MAX_CONTENT_LENGTH,
            "generic_result_cache_enabled": getattr(config, "generic_result_cache_enabled", False),
            "generic_result_cache_ttl": getattr(config, "generic_result_cache_ttl", 300),
            "http_connection_pool_enabled": getattr(config, "http_connection_pool_enabled", False),
            "http_connection_pool_max_connections": getattr(
                config,
                "http_connection_pool_max_connections",
                100,
            ),
            "http_connection_pool_max_connections_per_host": getattr(
                config,
                "http_connection_pool_max_connections_per_host",
                10,
            ),
            "http_connection_pool_connection_timeout": getattr(
                config,
                "http_connection_pool_connection_timeout",
                30,
            ),
            "http_connection_pool_total_timeout": getattr(
                config,
                "http_connection_pool_total_timeout",
                60,
            ),
            "rate_limiter_enabled": getattr(config, "rate_limiter_enabled", True),
            "rate_limiter_requests_per_minute": getattr(
                config,
                "rate_limiter_requests_per_minute",
                10,
            ),
            "rate_limiter_max_retries": getattr(config, "rate_limiter_max_retries", 5),
            "rate_limiter_initial_delay": getattr(config, "rate_limiter_initial_delay", 1.0),
            "rate_limiter_max_delay": getattr(config, "rate_limiter_max_delay", 60.0),
            "rate_limiter_backoff_multiplier": getattr(
                config,
                "rate_limiter_backoff_multiplier",
                2.0,
            ),
        }
    return {
        "provider": None,
        "model": None,
        "fetch_top": None,
        "fetch_pool": None,
        "max_content_length": _DEFAULT_MAX_CONTENT_LENGTH,
        "generic_result_cache_enabled": False,
        "generic_result_cache_ttl": 300,
        "http_connection_pool_enabled": True,  # Enable HTTP connection pooling by default for 20-30% latency reduction
        "http_connection_pool_max_connections": 100,
        "http_connection_pool_max_connections_per_host": 10,
        "http_connection_pool_connection_timeout": 30,
        "http_connection_pool_total_timeout": 60,
        "rate_limiter_enabled": True,
        "rate_limiter_requests_per_minute": 10,
        "rate_limiter_max_retries": 2,  # Reduced from 5 to fail faster on timeout
        "rate_limiter_initial_delay": 0.5,  # Reduced from 1.0 for faster retries
        "rate_limiter_max_delay": 10.0,  # Reduced from 60.0 to fail faster
        "rate_limiter_backoff_multiplier": 2.0,
    }


def _get_generic_web_cache(default_ttl: int) -> GenericResultCache:
    """Return singleton GenericResultCache used by web tools."""
    global _GENERIC_WEB_CACHE
    with _GENERIC_WEB_CACHE_LOCK:
        if _GENERIC_WEB_CACHE is None:
            _GENERIC_WEB_CACHE = GenericResultCache(default_ttl=max(1, int(default_ttl)))
        return _GENERIC_WEB_CACHE


async def _request_text(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    timeout_seconds: float = 15.0,
    follow_redirects: bool = True,
    web_config: Optional[Dict[str, Any]] = None,
) -> tuple[int, str]:
    """Perform HTTP request with rate limiting and retry logic.

    Implements:
    1. Rate limiting using token bucket algorithm
    2. Exponential backoff on 429 (rate limit) errors
    3. Configurable retry attempts and delays
    4. HTTP connection pooling (if enabled)

    Args:
        method: HTTP method (GET, POST, etc.)
        url: Target URL
        headers: Request headers
        data: Request data (for POST)
        timeout_seconds: Request timeout
        follow_redirects: Whether to follow redirects
        web_config: Configuration dictionary from _get_web_config

    Returns:
        Tuple of (status_code, response_text)

    Raises:
        httpx.HTTPError: On non-retryable errors or max retries exceeded
    """
    config = web_config or {}
    use_http_pool = bool(config.get("http_connection_pool_enabled", False))
    use_rate_limiter = bool(config.get("rate_limiter_enabled", True))
    logger = logging.getLogger(__name__)

    # Get rate limiter if enabled
    rate_limiter = None
    if use_rate_limiter:
        rate_limiter = _get_rate_limiter(
            requests_per_minute=int(config.get("rate_limiter_requests_per_minute", 10)),
            max_retries=int(config.get("rate_limiter_max_retries", 5)),
            initial_delay=float(config.get("rate_limiter_initial_delay", 1.0)),
            max_delay=float(config.get("rate_limiter_max_delay", 60.0)),
            backoff_multiplier=float(config.get("rate_limiter_backoff_multiplier", 2.0)),
        )

    # Retry loop for rate limit errors
    attempt = 0

    while True:
        # Acquire rate limiter permission before making request
        if rate_limiter:
            await rate_limiter.acquire(url)

        # Make the request
        try:
            if use_http_pool:
                try:
                    from victor.tools.http_pool import (
                        AIOHTTP_AVAILABLE,
                        ConnectionPoolConfig,
                        get_http_pool,
                    )

                    if AIOHTTP_AVAILABLE:
                        pool = await get_http_pool(
                            ConnectionPoolConfig(
                                max_connections=int(
                                    config.get("http_connection_pool_max_connections", 100)
                                ),
                                max_connections_per_host=int(
                                    config.get(
                                        "http_connection_pool_max_connections_per_host",
                                        10,
                                    )
                                ),
                                connection_timeout=int(
                                    config.get("http_connection_pool_connection_timeout", 30)
                                ),
                                total_timeout=int(
                                    config.get("http_connection_pool_total_timeout", 60)
                                ),
                            )
                        )
                        response = await pool.request(
                            method,
                            url,
                            headers=headers,
                            data=data,
                            allow_redirects=follow_redirects,
                            timeout=timeout_seconds,
                        )
                        try:
                            payload = await response.text()
                            status = int(response.status)

                            # Check for rate limit error (429)
                            if (
                                status == 429
                                and rate_limiter
                                and rate_limiter.should_retry(url, attempt)
                            ):
                                delay = rate_limiter.on_rate_limit_error(url, attempt)
                                await asyncio.sleep(delay)
                                attempt += 1
                                continue

                            return status, payload
                        finally:
                            response.release()
                except Exception as e:
                    logger.debug("HTTP pool request failed, falling back to httpx: %s", e)

            # Fallback to httpx
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.request(
                    method,
                    url,
                    data=data,
                    headers=headers,
                    follow_redirects=follow_redirects,
                )
                status = int(response.status_code)
                payload = response.text

                # Check for rate limit error (429)
                if status == 429 and rate_limiter and rate_limiter.should_retry(url, attempt):
                    delay = rate_limiter.on_rate_limit_error(url, attempt)
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue

                return status, payload

        except httpx.TimeoutException as e:
            # Don't retry timeouts - they're transient but we don't want to wait forever
            logger.warning(f"[web_search] Timeout on {url}: {e}")
            raise

        except httpx.HTTPError as e:
            # Check if it's a rate limit error (some servers return 429 as HTTPError)
            if "429" in str(e) and rate_limiter and rate_limiter.should_retry(url, attempt):
                delay = rate_limiter.on_rate_limit_error(url, attempt)
                await asyncio.sleep(delay)
                attempt += 1
                continue

            # Don't retry other HTTP errors
            logger.error(f"[web_search] HTTP error on {url}: {e}")
            raise

        except Exception as e:
            # Unexpected error - don't retry
            logger.error(f"[web_search] Unexpected error on {url}: {e}")
            raise


def _parse_ddg_results(html: str, max_results: int) -> List[Dict[str, str]]:
    """Parse DuckDuckGo HTML results.

    Args:
        html: HTML response
        max_results: Maximum number of results

    Returns:
        List of result dictionaries
    """
    soup = BeautifulSoup(html, "lxml")
    results = []

    # Find result divs
    for result in soup.find_all("div", class_="result", limit=max_results):
        try:
            # Extract title
            title_elem = result.find("a", class_="result__a")
            if not title_elem:
                continue

            title = title_elem.get_text(strip=True)
            url = title_elem.get("href", "")

            # Extract snippet
            snippet_elem = result.find("a", class_="result__snippet")
            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

            if title and url:
                results.append({"title": title, "url": url, "snippet": snippet})

        except Exception:
            continue

    return results


def _parse_ddg_lite_results(html: str, max_results: int) -> List[Dict[str, str]]:
    """Parse DuckDuckGo Lite results (fallback endpoint).

    The lite endpoint uses table rows (``<tr class="result-link">`` for the
    link and the following row for the snippet) rather than divs.

    Args:
        html: HTML response from lite.duckduckgo.com
        max_results: Maximum number of results

    Returns:
        List of result dictionaries
    """
    soup = BeautifulSoup(html, "lxml")
    results: List[Dict[str, str]] = []

    for link_row in soup.find_all("tr", class_=re.compile(r"result-link"), limit=max_results):
        try:
            a = link_row.find("a", class_="result-link")
            if not a:
                a = link_row.find("a")
            if not a:
                continue
            title = a.get_text(strip=True)
            url = a.get("href", "")
            # Snippet sits in the sibling result-snippet row.
            snippet = ""
            sib = link_row.find_next_sibling("tr", class_=re.compile(r"result-snippet"))
            if sib:
                snippet = sib.get_text(strip=True)
            if title and url:
                results.append({"title": title, "url": url, "snippet": snippet})
        except Exception:
            continue

    return results


async def _ddg_render_results(
    query: str,
    region: str,
    max_results: int,
    *,
    timeout_seconds: float = 30.0,
) -> List[Dict[str, str]]:
    """Last-resort: render the JS-backed DuckDuckGo main site via Playwright.

    Only invoked when both HTTP endpoints (html + lite) return nothing. Uses
    the same Playwright backend as ``_render_page_text``.

    Args:
        query: Search query
        region: DDG region code
        max_results: Maximum number of results
        timeout_seconds: Per-page navigation timeout

    Returns:
        List of result dictionaries (may be empty)
    """
    try:
        from playwright.async_api import async_playwright
    except Exception:
        return []

    from urllib.parse import quote_plus

    url = _DDG_RENDER_URL_TMPL.format(query=quote_plus(query))
    if region and region != "wt-wt":
        url += f"&kl={region}"
    browser = None
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(user_agent=_pick_user_agent())
            await page.goto(url, wait_until="networkidle", timeout=int(timeout_seconds * 1000))
            # DDG result links are anchors with data-testid="result-title-a" or
            # class ``result__a`` once rendered.
            anchors = await page.query_selector_all('a[data-testid="result-title-a"], a.result__a')
            results: List[Dict[str, str]] = []
            for a in anchors[:max_results]:
                title = (await a.inner_text()).strip()
                href = await a.get_attribute("href")
                if title and href:
                    results.append({"title": title, "url": href, "snippet": ""})
            return results
    except Exception:
        return []
    finally:
        if browser is not None:
            try:
                await browser.close()
            except Exception:
                pass


async def _ddg_search(
    query: str,
    region: str,
    max_results: int,
    safe_value: str,
    *,
    web_config: Dict[str, Any],
    timeout_seconds: float = 30.0,
) -> List[Dict[str, str]]:
    """Run a DuckDuckGo search across multiple endpoints with a browser fallback.

    Strategy (returns the first non-empty result set):
      1. POST html.duckduckgo.com/html/ (primary; parsed via _parse_ddg_results)
      2. POST lite.duckduckgo.com/lite/ (lighter fallback)
      3. Render duckduckgo.com via Playwright (only if 1 and 2 yield nothing)

    This keeps the common path fast (plain HTTP) and only spins up a browser on
    the hard cases where the HTML endpoint is rate-limited or blocked.

    Args:
        query: Search query
        region: DDG region code (e.g. "wt-wt")
        max_results: Maximum number of results
        safe_value: Safe-search value ("-1", "1", or "p")
        web_config: Web tool configuration from _get_web_config
        timeout_seconds: Per-request timeout

    Returns:
        List of result dictionaries (may be empty)
    """
    logger = logging.getLogger(__name__)
    data = {"q": query, "kl": region, "p": safe_value}
    headers = {"User-Agent": _pick_user_agent()}

    # 1. Primary HTML endpoint.
    try:
        status, payload = await _request_text(
            "POST",
            _DDG_HTML_URL,
            data=data,
            headers=headers,
            follow_redirects=True,
            timeout_seconds=timeout_seconds,
            web_config=web_config,
        )
        if status == 200:
            results = _parse_ddg_results(payload, max_results)
            if results:
                return results
    except Exception as exc:
        logger.debug("DDG html endpoint failed: %s", exc)

    # 2. Lite fallback endpoint.
    try:
        status, payload = await _request_text(
            "POST",
            _DDG_LITE_URL,
            data=data,
            headers={"User-Agent": _pick_user_agent()},
            follow_redirects=True,
            timeout_seconds=timeout_seconds,
            web_config=web_config,
        )
        if status == 200:
            results = _parse_ddg_lite_results(payload, max_results)
            if results:
                return results
    except Exception as exc:
        logger.debug("DDG lite endpoint failed: %s", exc)

    # 3. Browser-rendered fallback (Playwright optional; empty if unavailable).
    try:
        results = await _ddg_render_results(
            query, region, max_results, timeout_seconds=timeout_seconds
        )
        if results:
            return results
    except Exception as exc:
        logger.debug("DDG browser fallback failed: %s", exc)

    return []


def _format_results(query: str, results: List[Dict[str, str]]) -> str:
    """Format search results as text.

    Args:
        query: Search query
        results: List of results

    Returns:
        Formatted results string
    """
    output = [f"Search results for: {query}"]
    output.append("=" * 70)

    for i, result in enumerate(results, 1):
        output.append(f"\n{i}. {result['title']}")
        output.append(f"   URL: {result['url']}")
        if result["snippet"]:
            output.append(f"   {result['snippet']}")

    output.append(f"\nFound {len(results)} result(s)")

    return "\n".join(output)


def _extract_content(html: str, max_length: int = 5000) -> str:
    """Extract main content from HTML.

    Prefers `trafilatura` when available — it strips boilerplate/navigation/ads
    far better than hand-rolled heuristics and handles JS-heavy markup. Falls
    back to a BeautifulSoup heuristic when the optional dependency is absent.

    Args:
        html: HTML content
        max_length: Maximum content length

    Returns:
        Extracted text content
    """
    # Try trafilatura first (optional dependency; graceful fallback).
    try:
        from trafilatura import extract as traf_extract

        text = traf_extract(html, include_comments=False, include_tables=False, favor_recall=True)
        if text and len(text.strip()) > 100:
            text = re.sub(r"\n\s*\n", "\n\n", text)
            text = re.sub(r" +", " ", text).strip()
            return text[:max_length]
    except Exception:
        pass  # Fall through to BeautifulSoup heuristic.

    soup = BeautifulSoup(html, "lxml")

    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()

    # Try to find main content area
    content_areas = [
        soup.find("main"),
        soup.find("article"),
        soup.find("div", class_=re.compile("content|main|article", re.I)),
        soup.find("body"),
    ]

    for area in content_areas:
        if area:
            text = area.get_text(separator="\n", strip=True)
            # Clean up whitespace
            text = re.sub(r"\n\s*\n", "\n\n", text)
            text = re.sub(r" +", " ", text)

            if len(text) > 100:  # Minimum content length
                return text[:max_length]

    return ""


async def _render_page_text(
    url: str,
    *,
    timeout_seconds: float = 30.0,
    max_length: int = _DEFAULT_MAX_CONTENT_LENGTH,
) -> Dict[str, Any]:
    """Render a JavaScript-backed page with Playwright and return plaintext.

    This is intentionally isolated behind an explicit web_fetch render mode so
    normal HTTP fetches stay cheap. The output is plaintext by default to reduce
    LLM token pressure; the rendered HTML length is reported for diagnostics but
    the full DOM is not returned.
    """
    try:
        from playwright.async_api import async_playwright
    except Exception as exc:
        return {
            "success": False,
            "error": f"Playwright is not available: {exc}",
        }

    browser = None
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(user_agent=_USER_AGENT)
            await page.goto(
                url,
                wait_until="networkidle",
                timeout=int(timeout_seconds * 1000),
            )
            text = await page.locator("body").inner_text(timeout=int(timeout_seconds * 1000))
            html = await page.content()
            text = re.sub(r"\n\s*\n", "\n\n", text or "")
            text = re.sub(r" +", " ", text).strip()
            return {
                "success": True,
                "content": text[:max_length],
                "html_length": len(html or ""),
                "rendered": True,
            }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Browser render failed: {exc}",
        }
    finally:
        if browser is not None:
            try:
                await browser.close()
            except Exception:
                pass


@tool(
    cost_tier=CostTier.MEDIUM,
    category="web",
    priority=Priority.CONTEXTUAL,  # Only used when web search is requested
    access_mode=AccessMode.NETWORK,  # Makes external HTTP requests
    danger_level=DangerLevel.SAFE,  # No local side effects
    # Registry-driven metadata for tool selection and loop detection
    signature_params=["query"],  # Different queries indicate progress, not loops
    stages=["planning", "initial"],  # Conversation stages where relevant
    task_types=[
        "research",
        "analysis",
    ],  # Task types for classification-aware selection
    execution_category="network",  # Can run in parallel with read-only ops
    keywords=[
        "search",
        "web",
        "internet",
        "lookup",
        "find online",
        "google",
        "duckduckgo",
        "summarize",
        "web search",
    ],
    mandatory_keywords=[
        "search web",
        "search online",
        "look up online",
        "web search",
    ],  # Force inclusion
    aliases=["web"],  # Backward compatibility alias
)
async def web_search(
    query: str,
    max_results: int = 5,
    region: str = "wt-wt",
    safe_search: str = "moderate",
    ai_summarize: bool = False,
    fetch_top: Optional[int] = None,
    fetch_pool: Optional[int] = None,
    max_content_length: int = 5000,
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Search the web using DuckDuckGo. Optionally summarize with AI.

    Purpose:
    - Find links, docs, references on the public web.
    - Return titles, URLs, and snippets for relevance checking.
    - Optionally use AI to summarize search results.
    - Ideal when the user says "search the web", "find online", "lookup", "docs", "articles".

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (default: 5).
        region: Region for search results, e.g., 'us-en', 'uk-en', 'wt-wt' for worldwide (default: 'wt-wt').
        safe_search: Safe search level - 'on', 'moderate', 'off' (default: 'moderate').
        ai_summarize: If True, use AI to summarize results (default: False).
        fetch_top: Number of URLs to fetch for deeper summary (only with ai_summarize=True).
        fetch_pool: Pool of URLs to try fetching from (only with ai_summarize=True).
        max_content_length: Max content length to extract per URL (only with ai_summarize=True).
        _exec_ctx: Tool execution context containing ToolConfig for provider/model access.

    Returns:
        Dictionary containing:
        - success: Whether search succeeded
        - results: Formatted search results text
        - result_count: Number of results found
        - summary: AI summary (only if ai_summarize=True)
        - error: Error message if failed
    """
    # Route to summarize implementation if ai_summarize is True
    if ai_summarize:
        return await _summarize_search(
            query,
            max_results,
            region,
            safe_search,
            fetch_top,
            fetch_pool,
            max_content_length,
            context=_exec_ctx,
        )
    if not query:
        return {"success": False, "error": "Missing required parameter: query"}

    config = _get_web_config(_exec_ctx)

    # Map safe search to DuckDuckGo values
    safe_map = {"on": "1", "moderate": "-1", "off": "-2"}
    safe_value = safe_map.get(safe_search, "-1")
    logger = logging.getLogger(__name__)

    cache: Optional[GenericResultCache] = None
    cache_params = {
        "max_results": max_results,
        "region": region,
        "safe_search": safe_search,
    }
    if config.get("generic_result_cache_enabled"):
        cache = _get_generic_web_cache(int(config.get("generic_result_cache_ttl", 300)))
        cached_result = cache.get(ResultType.SEARCH, query, cache_params)
        if isinstance(cached_result, dict):
            payload = dict(cached_result)
            payload["cached"] = True
            return payload

    try:
        # DuckDuckGo search across multiple endpoints with a browser fallback.
        results = await _ddg_search(
            query,
            region,
            max_results,
            safe_value,
            web_config=config,
            timeout_seconds=30.0,  # Increased from 15.0 for better reliability
        )
        logger.info(
            f"[web_search] query='{query}', max_results={max_results}, parsed_results={len(results)}"
        )
        if results:
            sample_urls = [r.get("url", "") for r in results[:5]]
            logger.info(f"[web_search] top URLs: {sample_urls}")

        if not results:
            result_payload = {
                "success": True,
                "results": "No results found",
                "result_count": 0,
            }
            if cache is not None:
                cache.set(ResultType.SEARCH, query, dict(result_payload), params=cache_params)
            return result_payload

        # Format results
        output = _format_results(query, results)
        result_payload = {
            "success": True,
            "results": output,
            "result_count": len(results),
        }
        if cache is not None:
            cache.set(ResultType.SEARCH, query, dict(result_payload), params=cache_params)
        return result_payload

    except httpx.TimeoutException:
        return {"success": False, "error": "Search request timed out"}
    except Exception as e:
        return {"success": False, "error": f"Search failed: {str(e)}"}


@tool(
    cost_tier=CostTier.MEDIUM,
    category="web",
    priority=Priority.MEDIUM,  # Used when fetching specific URLs
    access_mode=AccessMode.NETWORK,  # Makes external HTTP requests
    danger_level=DangerLevel.SAFE,  # No local side effects
    # Registry-driven metadata for tool selection and loop detection
    signature_params=["url"],  # Different URLs indicate progress, not loops
    stages=["planning", "initial"],  # Conversation stages where relevant
    task_types=[
        "research",
        "analysis",
    ],  # Task types for classification-aware selection
    execution_category="network",  # Can run in parallel with read-only ops
    keywords=["fetch", "url", "webpage", "download", "http", "content", "web fetch"],
    aliases=["fetch"],  # Backward compatibility alias
)
async def web_fetch(
    url: str,
    render: str = "auto",
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fetch and extract main text content from a URL.

    Args:
        url: URL to fetch content from.
        render: "auto" (HTTP first, browser fallback), "http", or "browser".
    """
    if not url:
        return {"success": False, "error": "Missing required parameter: url"}
    render_mode = (render or "auto").lower().strip()
    if render_mode not in {"auto", "http", "browser"}:
        return {
            "success": False,
            "error": "Invalid render mode. Use 'auto', 'http', or 'browser'.",
        }

    config = _get_web_config(_exec_ctx)
    cache: Optional[GenericResultCache] = None
    cache_params = {
        "max_content_length": int(config.get("max_content_length", _DEFAULT_MAX_CONTENT_LENGTH)),
        "render": render_mode,
    }
    if config.get("generic_result_cache_enabled"):
        cache = _get_generic_web_cache(int(config.get("generic_result_cache_ttl", 300)))
        cached_result = cache.get(ResultType.SEARCH, f"fetch:{url}", cache_params)
        if isinstance(cached_result, dict):
            payload = dict(cached_result)
            payload["cached"] = True
            return payload

    try:
        max_content_length = int(config.get("max_content_length", _DEFAULT_MAX_CONTENT_LENGTH))
        if render_mode == "browser":
            rendered = await _render_page_text(
                url,
                timeout_seconds=30.0,
                max_length=max_content_length,
            )
            if not rendered.get("success"):
                return rendered
            result_payload = {
                "success": True,
                "content": rendered.get("content", ""),
                "url": url,
                "render": "browser",
                "html_length": rendered.get("html_length", 0),
            }
            if cache is not None:
                cache.set(
                    ResultType.SEARCH,
                    f"fetch:{url}",
                    dict(result_payload),
                    params=cache_params,
                )
            return result_payload

        status_code, payload = await _request_text(
            "GET",
            url,
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
            timeout_seconds=30.0,  # Increased from 15.0 for better reliability
            web_config=config,
        )
        if status_code != 200:
            return {
                "success": False,
                "error": f"Failed to fetch URL (status {status_code})",
            }

        # Extract text content
        content = _extract_content(
            payload,
            max_length=max_content_length,
        )

        if render_mode == "auto" and len(content.strip()) < 200:
            rendered = await _render_page_text(
                url,
                timeout_seconds=30.0,
                max_length=max_content_length,
            )
            if rendered.get("success") and rendered.get("content"):
                result_payload = {
                    "success": True,
                    "content": rendered.get("content", ""),
                    "url": url,
                    "render": "browser",
                    "html_length": rendered.get("html_length", 0),
                    "http_status_code": status_code,
                }
                if cache is not None:
                    cache.set(
                        ResultType.SEARCH,
                        f"fetch:{url}",
                        dict(result_payload),
                        params=cache_params,
                    )
                return result_payload

        if not content:
            return {"success": False, "error": "No content could be extracted from URL"}

        result_payload = {
            "success": True,
            "content": content,
            "url": url,
            "render": "http",
        }
        if cache is not None:
            cache.set(
                ResultType.SEARCH,
                f"fetch:{url}",
                dict(result_payload),
                params=cache_params,
            )
        return result_payload

    except httpx.TimeoutException:
        return {"success": False, "error": "Request timed out"}
    except Exception as e:
        return {"success": False, "error": f"Failed to fetch URL: {str(e)}"}


async def _summarize_search(
    query: str,
    max_results: int = 5,
    region: str = "wt-wt",
    safe_search: str = "moderate",
    fetch_top: Optional[int] = None,
    fetch_pool: Optional[int] = None,
    max_content_length: int = 5000,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Internal implementation for web search with AI summarization."""
    # Get config from context
    config = _get_web_config(context)
    provider = config.get("provider")
    model = config.get("model")

    if not query:
        return {"success": False, "error": "Missing required parameter: query"}

    # If no provider available, fall back to regular search
    if not provider:
        logger = logging.getLogger(__name__)
        logger.debug(
            "No LLM provider configured for AI summarization. "
            "Falling back to regular web search (no summarization). "
            "Set provider/model in ToolConfig to enable AI summarization."
        )
        # Call the regular web_search without ai_summarize
        return await web_search(
            query=query,
            max_results=max_results,
            region=region,
            safe_search=safe_search,
            ai_summarize=False,  # Disable summarization
            context=context,
        )

    # Map safe search to DuckDuckGo values
    safe_map = {"on": "1", "moderate": "-1", "off": "-2"}
    safe_value = safe_map.get(safe_search, "-1")
    logger = logging.getLogger(__name__)

    default_fetch_top = config.get("fetch_top")
    default_fetch_pool = config.get("fetch_pool")
    default_max_len = config.get("max_content_length", _DEFAULT_MAX_CONTENT_LENGTH)

    fetch_top = (
        fetch_top
        if fetch_top is not None
        else (default_fetch_top if default_fetch_top is not None else max_results)
    )
    fetch_pool = (
        fetch_pool
        if fetch_pool is not None
        else (
            default_fetch_pool
            if default_fetch_pool is not None
            else max(fetch_top + 2, max_results)
        )
    )
    max_content_length = max_content_length if max_content_length is not None else default_max_len

    fetch_top = max(0, min(fetch_top, 10))
    fetch_pool = max(fetch_top, min(fetch_pool, 12))
    max_content_length = max(500, min(max_content_length, 20000))

    try:
        # First, perform search across multiple DDG endpoints with a browser
        # fallback. Keeps the common path fast (plain HTTP) and only spins up
        # a browser when the HTTP endpoints yield nothing.
        results = await _ddg_search(
            query,
            region,
            fetch_pool,
            safe_value,
            web_config=config,
            timeout_seconds=30.0,  # Increased from 15.0 for better reliability
        )
        logger.info(f"[web_summarize] search query='{query}', parsed_results={len(results)}")
        if results:
            logger.info(f"[web_summarize] top URLs: {[r.get('url','') for r in results[:5]]}")

        if not results:
            return {
                "success": True,
                "summary": "No results found to summarize",
                "original_results": "",
            }

        # Format results
        results_text = _format_results(query, results)

        # Fetch top content for deeper summary (best-effort)
        fetched_contents = []
        for result in results[:fetch_pool]:
            if len(fetched_contents) >= fetch_top:
                break
            url = result.get("url")
            if not url:
                continue
            fetch_res = await web_fetch(url=url, context=context)
            if fetch_res.get("success") and fetch_res.get("content"):
                content = fetch_res["content"][:max_content_length]
                fetched_contents.append({"url": url, "content": content})

        logger.info(
            f"[web_summarize] fetch_top={fetch_top}, attempted_pool={fetch_pool}, fetched_ok={len(fetched_contents)}"
        )

        # Prepare prompt for summarization
        fetch_section = ""
        if fetched_contents:
            fetch_section = "\n\nFetched content excerpts:\n"
            for i, item in enumerate(fetched_contents, 1):
                fetch_section += (
                    f"\n{i}. URL: {item['url']}\nContent (excerpt):\n{item['content']}\n"
                )

        prompt = f"""Analyze these web search results and provide a comprehensive summary.

Query: {query}

Search Results:
{results_text}

Use fetched page excerpts if provided to improve accuracy:
{fetch_section}

Provide:
1. A clear summary of the key information
2. Important findings or facts
3. Any conflicting information
4. Sources used (include URLs)

Format the response clearly with sections."""

        logger.info(
            f"[web_summarize] building summary prompt for query='{query}', "
            f"results_count={len(results)}, prompt_chars={len(prompt)}"
        )

        try:
            from victor.providers.base import Message

            response = await provider.complete(
                model=model or "default",
                messages=[Message(role="user", content=prompt)],
                temperature=0.5,
                max_tokens=1000,
            )

            summary = response.content.strip()
            logger.info(
                f"[web_summarize] summarization model='{model}', summary_len={len(summary)}"
            )

            return {
                "success": True,
                "summary": summary,
                "original_results": results_text,
            }

        except Exception as e:
            # Fallback to just search results
            return {
                "success": True,
                "summary": f"AI summarization failed: {e}",
                "original_results": results_text,
            }

    except httpx.TimeoutException:
        return {"success": False, "error": "Search request timed out"}
    except Exception as e:
        return {"success": False, "error": f"Search failed: {str(e)}"}
