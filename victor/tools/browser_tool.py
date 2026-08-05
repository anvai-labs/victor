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

"""Browser automation tool for AI-assisted testing and debugging.

This module provides browser automation capabilities using Playwright,
enabling Victor to:
- Navigate to URLs and interact with web pages
- Take screenshots for visual debugging
- Capture console logs and errors
- Test web applications end-to-end
- Extract content from web pages

Safety features:
- Domain allowlist/blocklist
- Action rate limiting
- Configurable timeouts
- Sandboxed browser context

Usage:
    from victor.tools.browser_tool import BrowserTool

    tool = BrowserTool()
    await tool.initialize()

    # Navigate and screenshot
    result = await tool.navigate("https://example.com")
    screenshot = await tool.screenshot()

    # Interact with elements
    await tool.click("#login-button")
    await tool.type_text("#username", "test@example.com")
"""

import asyncio
import logging
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class BrowserAction(Enum):
    """Browser action types."""

    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    SCREENSHOT = "screenshot"
    EXTRACT = "extract"
    WAIT = "wait"
    EVALUATE = "evaluate"


@dataclass
class BrowserConfig:
    """Configuration for browser automation."""

    # Browser settings
    headless: bool = True
    browser_type: str = "chromium"  # chromium, firefox, webkit
    viewport_width: int = 1280
    viewport_height: int = 720
    user_agent: Optional[str] = None

    # Timeouts (milliseconds)
    navigation_timeout: int = 30000
    action_timeout: int = 10000
    element_timeout: int = 5000

    # Safety settings
    allowed_domains: List[str] = field(default_factory=list)  # Empty = all allowed
    blocked_domains: List[str] = field(
        default_factory=lambda: [
            "*.gov",  # Government sites
            "*.mil",  # Military sites
            "*.bank*",  # Banking sites
            "*.edu",  # Educational institutions
        ]
    )
    max_actions_per_page: int = 50
    max_pages_per_session: int = 20

    # Screenshot settings
    screenshot_dir: Optional[Path] = None
    max_screenshots: int = 20

    # Security
    disable_javascript: bool = False
    block_popups: bool = True
    block_downloads: bool = True


@dataclass
class BrowserState:
    """Current browser state."""

    url: str = ""
    title: str = ""
    is_loading: bool = False
    action_count: int = 0
    page_count: int = 0
    console_logs: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)


@dataclass
class ActionResult:
    """Result of a browser action."""

    success: bool
    action: BrowserAction
    message: str = ""
    data: Optional[Any] = None
    screenshot_path: Optional[str] = None
    error: Optional[str] = None
    duration_ms: int = 0


class BrowserTool:
    """Browser automation tool with safety controls.

    Features:
    - Navigate, click, type, scroll operations
    - Screenshot capture with automatic naming
    - Console log capture
    - Domain allowlist/blocklist
    - Action rate limiting
    - Configurable timeouts

    Requires playwright to be installed:
        pip install playwright
        playwright install chromium
    """

    def __init__(self, config: Optional[BrowserConfig] = None):
        """Initialize browser tool.

        Args:
            config: Browser configuration
        """
        self.config = config or BrowserConfig()
        self._browser = None
        self._context = None
        self._page = None
        self._state = BrowserState()
        self._initialized = False
        self._playwright = None

        # Set up screenshot directory
        if self.config.screenshot_dir is None:
            self.config.screenshot_dir = Path(tempfile.gettempdir()) / "victor_screenshots"
        self.config.screenshot_dir.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> bool:
        """Initialize browser and create context.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        try:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()

            # Select browser type
            if self.config.browser_type == "firefox":
                browser_launcher = self._playwright.firefox
            elif self.config.browser_type == "webkit":
                browser_launcher = self._playwright.webkit
            else:
                browser_launcher = self._playwright.chromium

            # Launch browser
            self._browser = await browser_launcher.launch(
                headless=self.config.headless,
            )

            # Create context with settings
            context_options = {
                "viewport": {
                    "width": self.config.viewport_width,
                    "height": self.config.viewport_height,
                },
                "java_script_enabled": not self.config.disable_javascript,
            }
            if self.config.user_agent:
                context_options["user_agent"] = self.config.user_agent

            self._context = await self._browser.new_context(**context_options)

            # Block popups if configured
            if self.config.block_popups:
                self._context.on("page", lambda p: asyncio.create_task(p.close()))

            # Create initial page
            self._page = await self._context.new_page()

            # Set up console log capture
            self._page.on("console", self._handle_console_message)
            self._page.on("pageerror", self._handle_page_error)

            # Set timeouts
            self._page.set_default_navigation_timeout(self.config.navigation_timeout)
            self._page.set_default_timeout(self.config.action_timeout)

            self._initialized = True
            logger.info(f"Browser initialized: {self.config.browser_type}")
            return True

        except ImportError:
            logger.error(
                "Playwright not installed. Install with: pip install playwright && playwright install chromium"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            return False

    def _handle_console_message(self, msg) -> None:
        """Handle console messages from the page."""
        self._state.console_logs.append(
            {
                "type": msg.type,
                "text": msg.text,
                "timestamp": datetime.now().isoformat(),
            }
        )
        # Keep only last 100 messages
        if len(self._state.console_logs) > 100:
            self._state.console_logs = self._state.console_logs[-100:]

    def _handle_page_error(self, error) -> None:
        """Handle page errors."""
        self._state.errors.append(str(error))
        # Keep only last 20 errors
        if len(self._state.errors) > 20:
            self._state.errors = self._state.errors[-20:]

    def _is_domain_allowed(self, url: str) -> bool:
        """Check if domain is allowed.

        Args:
            url: URL to check

        Returns:
            True if domain is allowed
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Invalid URL check - must have a valid domain
            if not domain or not parsed.scheme:
                logger.warning(f"Invalid URL: {url}")
                return False

            # Check blocked domains
            for pattern in self.config.blocked_domains:
                if self._match_domain_pattern(domain, pattern):
                    logger.warning(f"Domain blocked: {domain}")
                    return False

            # Check allowed domains (empty = all allowed)
            if self.config.allowed_domains:
                for pattern in self.config.allowed_domains:
                    if self._match_domain_pattern(domain, pattern):
                        return True
                logger.warning(f"Domain not in allowlist: {domain}")
                return False

            return True

        except Exception:
            return False

    def _match_domain_pattern(self, domain: str, pattern: str) -> bool:
        """Match domain against pattern (supports * wildcards).

        Patterns:
            - "example.com" - exact match
            - "*.example.com" - any subdomain of example.com
            - "*bank*" - any domain containing "bank"
        """
        pattern = pattern.lower()
        domain = domain.lower()

        # Handle patterns like "*bank*" (contains)
        if pattern.startswith("*") and pattern.endswith("*") and len(pattern) > 2:
            search_term = pattern[1:-1].replace(".", r"\.")
            return bool(re.search(search_term, domain))

        # Convert glob-style to regex
        regex_pattern = pattern.replace(".", r"\.").replace("*", ".*")
        return bool(re.match(f"^{regex_pattern}$", domain))

    def _check_rate_limit(self) -> bool:
        """Check if action rate limit exceeded.

        Returns:
            True if action allowed
        """
        if self._state.action_count >= self.config.max_actions_per_page:
            logger.warning("Action rate limit exceeded")
            return False
        return True

    async def navigate(self, url: str, wait_for: str = "load") -> ActionResult:
        """Navigate to a URL.

        Args:
            url: URL to navigate to
            wait_for: Wait condition ("load", "domcontentloaded", "networkidle")

        Returns:
            ActionResult
        """
        start = datetime.now()

        if not self._initialized:
            if not await self.initialize():
                return ActionResult(
                    success=False,
                    action=BrowserAction.NAVIGATE,
                    error="Browser not initialized",
                )

        if not self._is_domain_allowed(url):
            return ActionResult(
                success=False,
                action=BrowserAction.NAVIGATE,
                error=f"Domain not allowed: {url}",
            )

        if self._state.page_count >= self.config.max_pages_per_session:
            return ActionResult(
                success=False,
                action=BrowserAction.NAVIGATE,
                error="Max pages per session exceeded",
            )

        try:
            response = await self._page.goto(url, wait_until=wait_for)

            self._state.url = self._page.url
            self._state.title = await self._page.title()
            self._state.page_count += 1
            self._state.action_count = 0  # Reset per-page counter

            duration = int((datetime.now() - start).total_seconds() * 1000)

            return ActionResult(
                success=True,
                action=BrowserAction.NAVIGATE,
                message=f"Navigated to {self._state.url}",
                data={
                    "url": self._state.url,
                    "title": self._state.title,
                    "status": response.status if response else None,
                },
                duration_ms=duration,
            )

        except Exception as e:
            return ActionResult(
                success=False,
                action=BrowserAction.NAVIGATE,
                error=str(e),
            )

    async def click(self, selector: str) -> ActionResult:
        """Click an element.

        Args:
            selector: CSS selector or text selector

        Returns:
            ActionResult
        """
        start = datetime.now()

        if not self._initialized or not self._page:
            return ActionResult(
                success=False,
                action=BrowserAction.CLICK,
                error="Browser not initialized",
            )

        if not self._check_rate_limit():
            return ActionResult(
                success=False,
                action=BrowserAction.CLICK,
                error="Action rate limit exceeded",
            )

        try:
            await self._page.click(selector, timeout=self.config.element_timeout)
            self._state.action_count += 1

            duration = int((datetime.now() - start).total_seconds() * 1000)

            return ActionResult(
                success=True,
                action=BrowserAction.CLICK,
                message=f"Clicked: {selector}",
                duration_ms=duration,
            )

        except Exception as e:
            return ActionResult(
                success=False,
                action=BrowserAction.CLICK,
                error=str(e),
            )

    async def type_text(self, selector: str, text: str, clear_first: bool = True) -> ActionResult:
        """Type text into an input element.

        Args:
            selector: CSS selector for input element
            text: Text to type
            clear_first: Whether to clear existing text first

        Returns:
            ActionResult
        """
        start = datetime.now()

        if not self._initialized or not self._page:
            return ActionResult(
                success=False,
                action=BrowserAction.TYPE,
                error="Browser not initialized",
            )

        if not self._check_rate_limit():
            return ActionResult(
                success=False,
                action=BrowserAction.TYPE,
                error="Action rate limit exceeded",
            )

        try:
            if clear_first:
                await self._page.fill(selector, text, timeout=self.config.element_timeout)
            else:
                await self._page.type(selector, text, timeout=self.config.element_timeout)

            self._state.action_count += 1

            duration = int((datetime.now() - start).total_seconds() * 1000)

            # Don't log actual text for security
            return ActionResult(
                success=True,
                action=BrowserAction.TYPE,
                message=f"Typed {len(text)} characters into: {selector}",
                duration_ms=duration,
            )

        except Exception as e:
            return ActionResult(
                success=False,
                action=BrowserAction.TYPE,
                error=str(e),
            )

    async def scroll(self, direction: str = "down", amount: int = 500) -> ActionResult:
        """Scroll the page.

        Args:
            direction: "up", "down", "left", "right"
            amount: Pixels to scroll

        Returns:
            ActionResult
        """
        start = datetime.now()

        if not self._initialized or not self._page:
            return ActionResult(
                success=False,
                action=BrowserAction.SCROLL,
                error="Browser not initialized",
            )

        if not self._check_rate_limit():
            return ActionResult(
                success=False,
                action=BrowserAction.SCROLL,
                error="Action rate limit exceeded",
            )

        try:
            scroll_map = {
                "down": (0, amount),
                "up": (0, -amount),
                "right": (amount, 0),
                "left": (-amount, 0),
            }
            dx, dy = scroll_map.get(direction.lower(), (0, amount))

            await self._page.mouse.wheel(dx, dy)
            self._state.action_count += 1

            duration = int((datetime.now() - start).total_seconds() * 1000)

            return ActionResult(
                success=True,
                action=BrowserAction.SCROLL,
                message=f"Scrolled {direction} by {amount}px",
                duration_ms=duration,
            )

        except Exception as e:
            return ActionResult(
                success=False,
                action=BrowserAction.SCROLL,
                error=str(e),
            )

    async def screenshot(
        self,
        selector: Optional[str] = None,
        full_page: bool = False,
        name: Optional[str] = None,
    ) -> ActionResult:
        """Take a screenshot.

        Args:
            selector: Optional element selector to screenshot
            full_page: Whether to capture full page
            name: Optional filename (auto-generated if not provided)

        Returns:
            ActionResult with screenshot path
        """
        start = datetime.now()

        if not self._initialized or not self._page:
            return ActionResult(
                success=False,
                action=BrowserAction.SCREENSHOT,
                error="Browser not initialized",
            )

        if len(self._state.screenshots) >= self.config.max_screenshots:
            return ActionResult(
                success=False,
                action=BrowserAction.SCREENSHOT,
                error="Max screenshots exceeded",
            )

        try:
            # Generate filename
            if name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name = f"screenshot_{timestamp}_{len(self._state.screenshots)}.png"

            filepath = self.config.screenshot_dir / name

            # Take screenshot
            if selector:
                element = await self._page.query_selector(selector)
                if element:
                    await element.screenshot(path=str(filepath))
                else:
                    return ActionResult(
                        success=False,
                        action=BrowserAction.SCREENSHOT,
                        error=f"Element not found: {selector}",
                    )
            else:
                await self._page.screenshot(path=str(filepath), full_page=full_page)

            self._state.screenshots.append(str(filepath))

            duration = int((datetime.now() - start).total_seconds() * 1000)

            return ActionResult(
                success=True,
                action=BrowserAction.SCREENSHOT,
                message=f"Screenshot saved: {filepath}",
                screenshot_path=str(filepath),
                duration_ms=duration,
            )

        except Exception as e:
            return ActionResult(
                success=False,
                action=BrowserAction.SCREENSHOT,
                error=str(e),
            )

    async def extract_text(self, selector: Optional[str] = None) -> ActionResult:
        """Extract text content from page or element.

        Args:
            selector: Optional element selector (None = entire page)

        Returns:
            ActionResult with extracted text
        """
        start = datetime.now()

        if not self._initialized or not self._page:
            return ActionResult(
                success=False,
                action=BrowserAction.EXTRACT,
                error="Browser not initialized",
            )

        try:
            if selector:
                element = await self._page.query_selector(selector)
                if element:
                    text = await element.text_content()
                else:
                    return ActionResult(
                        success=False,
                        action=BrowserAction.EXTRACT,
                        error=f"Element not found: {selector}",
                    )
            else:
                text = await self._page.text_content("body")

            duration = int((datetime.now() - start).total_seconds() * 1000)

            return ActionResult(
                success=True,
                action=BrowserAction.EXTRACT,
                message=f"Extracted {len(text or '')} characters",
                data=text,
                duration_ms=duration,
            )

        except Exception as e:
            return ActionResult(
                success=False,
                action=BrowserAction.EXTRACT,
                error=str(e),
            )

    async def wait_for(
        self,
        selector: Optional[str] = None,
        state: str = "visible",
        timeout: Optional[int] = None,
    ) -> ActionResult:
        """Wait for element or condition.

        Args:
            selector: CSS selector to wait for
            state: "visible", "hidden", "attached", "detached"
            timeout: Custom timeout in ms

        Returns:
            ActionResult
        """
        start = datetime.now()

        if not self._initialized or not self._page:
            return ActionResult(
                success=False,
                action=BrowserAction.WAIT,
                error="Browser not initialized",
            )

        try:
            if selector:
                await self._page.wait_for_selector(
                    selector,
                    state=state,
                    timeout=timeout or self.config.element_timeout,
                )
            else:
                await self._page.wait_for_load_state("networkidle")

            duration = int((datetime.now() - start).total_seconds() * 1000)

            return ActionResult(
                success=True,
                action=BrowserAction.WAIT,
                message=f"Wait completed: {selector or 'page load'}",
                duration_ms=duration,
            )

        except Exception as e:
            return ActionResult(
                success=False,
                action=BrowserAction.WAIT,
                error=str(e),
            )

    async def evaluate(self, script: str) -> ActionResult:
        """Evaluate JavaScript in page context.

        Args:
            script: JavaScript code to execute

        Returns:
            ActionResult with evaluation result
        """
        start = datetime.now()

        if not self._initialized or not self._page:
            return ActionResult(
                success=False,
                action=BrowserAction.EVALUATE,
                error="Browser not initialized",
            )

        if self.config.disable_javascript:
            return ActionResult(
                success=False,
                action=BrowserAction.EVALUATE,
                error="JavaScript is disabled",
            )

        # Safety: Block potentially dangerous operations
        dangerous_patterns = [
            r"fetch\s*\(",
            r"XMLHttpRequest",
            r"localStorage",
            r"sessionStorage",
            r"document\.cookie",
            r"window\.location\s*=",
            r"eval\s*\(",
            r"Function\s*\(",
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, script, re.IGNORECASE):
                return ActionResult(
                    success=False,
                    action=BrowserAction.EVALUATE,
                    error=f"Potentially dangerous operation blocked: {pattern}",
                )

        try:
            result = await self._page.evaluate(script)

            duration = int((datetime.now() - start).total_seconds() * 1000)

            return ActionResult(
                success=True,
                action=BrowserAction.EVALUATE,
                message="Script executed",
                data=result,
                duration_ms=duration,
            )

        except Exception as e:
            return ActionResult(
                success=False,
                action=BrowserAction.EVALUATE,
                error=str(e),
            )

    def get_state(self) -> BrowserState:
        """Get current browser state.

        Returns:
            Current BrowserState
        """
        return self._state

    def get_console_logs(self, log_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get captured console logs.

        Args:
            log_type: Filter by type ("log", "error", "warning", etc.)

        Returns:
            List of console log entries
        """
        if log_type:
            return [log for log in self._state.console_logs if log["type"] == log_type]
        return self._state.console_logs

    def get_errors(self) -> List[str]:
        """Get captured page errors.

        Returns:
            List of error messages
        """
        return self._state.errors

    async def close(self) -> None:
        """Close browser and cleanup."""
        if self._browser:
            await self._browser.close()
            self._browser = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

        self._page = None
        self._context = None
        self._initialized = False
        logger.info("Browser closed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Convenience functions for tool integration


async def browser_navigate(url: str, **kwargs) -> Dict[str, Any]:
    """Navigate to URL and return page info.

    Args:
        url: URL to navigate to
        **kwargs: Additional config options

    Returns:
        Dict with page info or error
    """
    config = BrowserConfig(**kwargs) if kwargs else None
    async with BrowserTool(config) as browser:
        result = await browser.navigate(url)
        if result.success:
            screenshot = await browser.screenshot()
            return {
                "success": True,
                "url": result.data["url"],
                "title": result.data["title"],
                "screenshot": screenshot.screenshot_path,
            }
        return {"success": False, "error": result.error}


async def browser_screenshot(url: str, full_page: bool = False) -> Dict[str, Any]:
    """Navigate to URL and take screenshot.

    Args:
        url: URL to screenshot
        full_page: Whether to capture full page

    Returns:
        Dict with screenshot path or error
    """
    async with BrowserTool() as browser:
        nav_result = await browser.navigate(url)
        if not nav_result.success:
            return {"success": False, "error": nav_result.error}

        screenshot_result = await browser.screenshot(full_page=full_page)
        return {
            "success": screenshot_result.success,
            "screenshot": screenshot_result.screenshot_path,
            "error": screenshot_result.error,
        }
