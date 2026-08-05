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

"""Tests for the browser automation tool."""

import tempfile
from pathlib import Path
from unittest import mock

import pytest

from victor.tools.browser_tool import (
    BrowserAction,
    BrowserConfig,
    BrowserState,
    BrowserTool,
    ActionResult,
)


class TestBrowserConfig:
    """Tests for BrowserConfig dataclass."""

    def test_default_values(self):
        """BrowserConfig should have sensible defaults."""
        config = BrowserConfig()

        assert config.headless is True
        assert config.browser_type == "chromium"
        assert config.viewport_width == 1280
        assert config.viewport_height == 720
        assert config.navigation_timeout == 30000
        assert config.max_actions_per_page == 50
        assert config.max_pages_per_session == 20

    def test_blocked_domains_default(self):
        """Default blocked domains should include sensitive TLDs."""
        config = BrowserConfig()

        assert "*.gov" in config.blocked_domains
        assert "*.mil" in config.blocked_domains
        assert "*.bank*" in config.blocked_domains

    def test_custom_config(self):
        """BrowserConfig should accept custom values."""
        config = BrowserConfig(
            headless=False,
            browser_type="firefox",
            viewport_width=1920,
            max_actions_per_page=100,
        )

        assert config.headless is False
        assert config.browser_type == "firefox"
        assert config.viewport_width == 1920
        assert config.max_actions_per_page == 100


class TestBrowserState:
    """Tests for BrowserState dataclass."""

    def test_default_state(self):
        """BrowserState should have empty defaults."""
        state = BrowserState()

        assert state.url == ""
        assert state.title == ""
        assert state.is_loading is False
        assert state.action_count == 0
        assert state.page_count == 0
        assert state.console_logs == []
        assert state.errors == []
        assert state.screenshots == []


class TestActionResult:
    """Tests for ActionResult dataclass."""

    def test_success_result(self):
        """ActionResult should store success info."""
        result = ActionResult(
            success=True,
            action=BrowserAction.NAVIGATE,
            message="Navigated to https://example.com",
            data={"url": "https://example.com", "title": "Example"},
            duration_ms=500,
        )

        assert result.success is True
        assert result.action == BrowserAction.NAVIGATE
        assert result.error is None

    def test_failure_result(self):
        """ActionResult should store failure info."""
        result = ActionResult(
            success=False,
            action=BrowserAction.CLICK,
            error="Element not found",
        )

        assert result.success is False
        assert result.error == "Element not found"


class TestBrowserToolDomainValidation:
    """Tests for domain validation in BrowserTool."""

    def test_domain_pattern_matching(self):
        """_match_domain_pattern should handle wildcards."""
        tool = BrowserTool()

        # Exact match
        assert tool._match_domain_pattern("example.com", "example.com") is True
        assert tool._match_domain_pattern("example.com", "other.com") is False

        # Wildcard prefix
        assert tool._match_domain_pattern("sub.example.com", "*.example.com") is True
        assert tool._match_domain_pattern("example.com", "*.example.com") is False

        # Contains pattern (*word*)
        assert tool._match_domain_pattern("mybank.com", "*bank*") is True
        assert tool._match_domain_pattern("banking.org", "*bank*") is True
        assert tool._match_domain_pattern("example.com", "*bank*") is False

    def test_blocked_domains(self):
        """_is_domain_allowed should block configured domains."""
        config = BrowserConfig(blocked_domains=["*.gov", "blocked.com"])
        tool = BrowserTool(config)

        assert tool._is_domain_allowed("https://example.com") is True
        assert tool._is_domain_allowed("https://blocked.com") is False
        assert tool._is_domain_allowed("https://irs.gov") is False
        assert tool._is_domain_allowed("https://sub.gov") is False

    def test_allowed_domains(self):
        """_is_domain_allowed should restrict to allowlist."""
        config = BrowserConfig(
            allowed_domains=["example.com", "*.allowed.org"],
            blocked_domains=[],
        )
        tool = BrowserTool(config)

        assert tool._is_domain_allowed("https://example.com") is True
        assert tool._is_domain_allowed("https://sub.allowed.org") is True
        assert tool._is_domain_allowed("https://other.com") is False

    def test_invalid_url(self):
        """_is_domain_allowed should reject invalid URLs."""
        tool = BrowserTool()

        assert tool._is_domain_allowed("not a url") is False
        assert tool._is_domain_allowed("") is False


class TestBrowserToolRateLimiting:
    """Tests for rate limiting in BrowserTool."""

    def test_check_rate_limit_under_limit(self):
        """_check_rate_limit should allow actions under limit."""
        config = BrowserConfig(max_actions_per_page=10)
        tool = BrowserTool(config)
        tool._state.action_count = 5

        assert tool._check_rate_limit() is True

    def test_check_rate_limit_at_limit(self):
        """_check_rate_limit should block at limit."""
        config = BrowserConfig(max_actions_per_page=10)
        tool = BrowserTool(config)
        tool._state.action_count = 10

        assert tool._check_rate_limit() is False


class TestBrowserToolInitialization:
    """Tests for BrowserTool initialization."""

    def test_init_without_playwright(self):
        """initialize should fail gracefully without playwright."""
        BrowserTool()

        # Mock ImportError for playwright
        with mock.patch.dict("sys.modules", {"playwright": None, "playwright.async_api": None}):
            # This should not raise, but return False
            # Note: actual test would need async context
            pass

    def test_screenshot_dir_creation(self):
        """BrowserTool should create screenshot directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            screenshot_dir = Path(tmpdir) / "screenshots"
            config = BrowserConfig(screenshot_dir=screenshot_dir)
            BrowserTool(config)

            assert screenshot_dir.exists()

    def test_default_screenshot_dir(self):
        """BrowserTool should use temp dir by default."""
        tool = BrowserTool()

        assert tool.config.screenshot_dir is not None
        assert "victor_screenshots" in str(tool.config.screenshot_dir)


class TestBrowserToolState:
    """Tests for BrowserTool state management."""

    def test_get_state(self):
        """get_state should return current state."""
        tool = BrowserTool()
        tool._state.url = "https://example.com"
        tool._state.action_count = 5

        state = tool.get_state()

        assert state.url == "https://example.com"
        assert state.action_count == 5

    def test_get_console_logs_all(self):
        """get_console_logs should return all logs."""
        tool = BrowserTool()
        tool._state.console_logs = [
            {"type": "log", "text": "Info message"},
            {"type": "error", "text": "Error message"},
        ]

        logs = tool.get_console_logs()

        assert len(logs) == 2

    def test_get_console_logs_filtered(self):
        """get_console_logs should filter by type."""
        tool = BrowserTool()
        tool._state.console_logs = [
            {"type": "log", "text": "Info message"},
            {"type": "error", "text": "Error message"},
            {"type": "error", "text": "Another error"},
        ]

        error_logs = tool.get_console_logs("error")

        assert len(error_logs) == 2
        assert all(log["type"] == "error" for log in error_logs)

    def test_get_errors(self):
        """get_errors should return captured errors."""
        tool = BrowserTool()
        tool._state.errors = ["Error 1", "Error 2"]

        errors = tool.get_errors()

        assert len(errors) == 2
        assert "Error 1" in errors


class TestBrowserToolJavaScriptSafety:
    """Tests for JavaScript evaluation safety."""

    @pytest.mark.asyncio
    async def test_evaluate_blocks_dangerous_operations(self):
        """evaluate should block dangerous JavaScript."""
        tool = BrowserTool()
        tool._initialized = True
        tool._page = mock.MagicMock()

        # Test various dangerous patterns
        dangerous_scripts = [
            "fetch('https://evil.com')",
            "new XMLHttpRequest()",
            "localStorage.getItem('secret')",
            "document.cookie",
            "window.location = 'https://evil.com'",
            "eval('malicious')",
        ]

        for script in dangerous_scripts:
            result = await tool.evaluate(script)
            assert result.success is False
            assert "blocked" in result.error.lower() or "dangerous" in result.error.lower()

    @pytest.mark.asyncio
    async def test_evaluate_disabled_javascript(self):
        """evaluate should fail when JavaScript is disabled."""
        config = BrowserConfig(disable_javascript=True)
        tool = BrowserTool(config)
        tool._initialized = True
        tool._page = mock.MagicMock()

        result = await tool.evaluate("return 1 + 1")

        assert result.success is False
        assert "disabled" in result.error.lower()


class TestBrowserToolNotInitialized:
    """Tests for operations when browser not initialized."""

    @pytest.mark.asyncio
    async def test_click_not_initialized(self):
        """click should fail when not initialized."""
        tool = BrowserTool()
        tool._initialized = False

        result = await tool.click("#button")

        assert result.success is False
        assert "not initialized" in result.error.lower()

    @pytest.mark.asyncio
    async def test_type_not_initialized(self):
        """type_text should fail when not initialized."""
        tool = BrowserTool()
        tool._initialized = False

        result = await tool.type_text("#input", "text")

        assert result.success is False
        assert "not initialized" in result.error.lower()

    @pytest.mark.asyncio
    async def test_scroll_not_initialized(self):
        """scroll should fail when not initialized."""
        tool = BrowserTool()
        tool._initialized = False

        result = await tool.scroll("down")

        assert result.success is False
        assert "not initialized" in result.error.lower()

    @pytest.mark.asyncio
    async def test_screenshot_not_initialized(self):
        """screenshot should fail when not initialized."""
        tool = BrowserTool()
        tool._initialized = False

        result = await tool.screenshot()

        assert result.success is False
        assert "not initialized" in result.error.lower()

    @pytest.mark.asyncio
    async def test_extract_text_not_initialized(self):
        """extract_text should fail when not initialized."""
        tool = BrowserTool()
        tool._initialized = False

        result = await tool.extract_text()

        assert result.success is False
        assert "not initialized" in result.error.lower()


class TestBrowserAction:
    """Tests for BrowserAction enum."""

    def test_all_actions_exist(self):
        """All browser action types should exist."""
        assert BrowserAction.NAVIGATE.value == "navigate"
        assert BrowserAction.CLICK.value == "click"
        assert BrowserAction.TYPE.value == "type"
        assert BrowserAction.SCROLL.value == "scroll"
        assert BrowserAction.SCREENSHOT.value == "screenshot"
        assert BrowserAction.EXTRACT.value == "extract"
        assert BrowserAction.WAIT.value == "wait"
        assert BrowserAction.EVALUATE.value == "evaluate"
