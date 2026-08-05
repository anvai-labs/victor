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

"""Tests for provider/profile switching — ensures parameter names stay aligned.

Regression tests for the bug where AgentOrchestrator.switch_provider()
async method used 'provider' as param name while all callers passed
'provider_name=' as keyword arg, causing TypeError at runtime.
"""

import inspect
import io
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from victor.ui.slash.protocol import CommandContext

# ---------------------------------------------------------------------------
# Signature alignment tests — catch param name drift at import time
# ---------------------------------------------------------------------------


class TestSwitchProviderSignatureAlignment:
    """Verify switch_provider param names are consistent across sync/async."""

    def test_async_switch_provider_accepts_provider_name_kwarg(self):
        """Regression: async switch_provider must accept 'provider_name' kwarg."""
        from victor.agent.orchestrator import AgentOrchestrator

        sig = inspect.signature(AgentOrchestrator.switch_provider)
        params = list(sig.parameters.keys())
        # First positional after self should be 'provider_name'
        assert "provider_name" in params, (
            f"async switch_provider() must accept 'provider_name' parameter. "
            f"Got params: {params}"
        )

    def test_sync_and_async_switch_provider_param_names_match(self):
        """Both sync and async switch_provider should use 'provider_name'."""
        from victor.agent.orchestrator import AgentOrchestrator

        # The class has both sync and async versions; async shadows sync.
        # The visible method must use 'provider_name' for all callers.
        sig = inspect.signature(AgentOrchestrator.switch_provider)
        params = list(sig.parameters.keys())
        assert (
            params[1] == "provider_name"
        ), f"First param after 'self' should be 'provider_name', got '{params[1]}'"

    def test_switch_provider_does_not_use_bare_provider_param(self):
        """Regression guard: param must NOT be bare 'provider' (ambiguous)."""
        from victor.agent.orchestrator import AgentOrchestrator

        sig = inspect.signature(AgentOrchestrator.switch_provider)
        params = list(sig.parameters.keys())
        # 'provider' alone (without _name suffix) caused the original bug
        assert params[1] != "provider", (
            "switch_provider() must use 'provider_name' not 'provider' "
            "to match all callers (ProfileCommand, ProviderCommand, SwitchCommand)"
        )


# ---------------------------------------------------------------------------
# ProfileCommand integration tests
# ---------------------------------------------------------------------------


class TestProfileCommandSwitchProvider:
    """Test ProfileCommand calls switch_provider with correct kwargs."""

    def _make_ctx(self, args=None, agent=None, profiles=None):
        """Create a CommandContext with mocked dependencies."""
        console = Console(file=io.StringIO())
        settings = MagicMock()

        if profiles is None:

            @dataclass
            class FakeProfile:
                provider: str = "ollama"
                model: str = "qwen3-coder-tools:30b-64K"

            profiles = {"default": FakeProfile()}

        settings.load_profiles.return_value = profiles

        return CommandContext(
            console=console,
            settings=settings,
            agent=agent,
            args=args or [],
        )

    @pytest.mark.asyncio
    async def test_profile_switch_calls_switch_provider_with_provider_name(self):
        """ProfileCommand must call switch_provider(provider_name=..., model=...)."""
        from victor.ui.slash.commands.model import ProfileCommand

        agent = MagicMock()
        agent.switch_provider = AsyncMock(return_value=True)
        agent.get_current_provider_info.return_value = {
            "provider": "ollama",
            "model": "qwen3-coder-tools:30b-64K",
            "native_tool_calls": False,
            "thinking_mode": False,
        }

        ctx = self._make_ctx(args=["default"], agent=agent)
        cmd = ProfileCommand()
        await cmd.execute(ctx)

        agent.switch_provider.assert_called_once_with(
            provider_name="ollama",
            model="qwen3-coder-tools:30b-64K",
        )

    @pytest.mark.asyncio
    async def test_profile_switch_unknown_profile(self):
        """ProfileCommand should report error for unknown profile."""
        from victor.ui.slash.commands.model import ProfileCommand

        agent = MagicMock()
        ctx = self._make_ctx(args=["nonexistent"], agent=agent)
        cmd = ProfileCommand()
        await cmd.execute(ctx)

        agent.switch_provider.assert_not_called()

    @pytest.mark.asyncio
    async def test_profile_switch_no_agent(self):
        """ProfileCommand should handle missing agent gracefully."""
        from victor.ui.slash.commands.model import ProfileCommand

        ctx = self._make_ctx(args=["default"], agent=None)
        cmd = ProfileCommand()
        await cmd.execute(ctx)
        # Should not raise — just print a message


class TestProviderCommandSwitchProvider:
    """Test ProviderCommand calls switch_provider with correct kwargs."""

    def _make_ctx(self, args=None, agent=None):
        console = Console(file=io.StringIO())
        settings = MagicMock()
        return CommandContext(
            console=console,
            settings=settings,
            agent=agent,
            args=args or [],
        )

    @pytest.mark.asyncio
    @patch("victor.providers.registry.ProviderRegistry.list_providers")
    async def test_provider_switch_calls_with_provider_name(self, mock_list):
        """ProviderCommand must call switch_provider(provider_name=..., model=...)."""
        from victor.ui.slash.commands.model import ProviderCommand

        mock_list.return_value = ["ollama", "anthropic", "openai"]

        agent = MagicMock()
        agent.switch_provider = AsyncMock(return_value=True)
        agent.get_current_provider_info.return_value = {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
            "native_tool_calls": True,
            "thinking_mode": False,
        }

        ctx = self._make_ctx(args=["anthropic", "claude-sonnet-4-5"], agent=agent)
        cmd = ProviderCommand()
        await cmd.execute(ctx)

        agent.switch_provider.assert_called_once_with(
            provider_name="anthropic",
            model="claude-sonnet-4-5",
        )

    @pytest.mark.asyncio
    @patch("victor.providers.registry.ProviderRegistry.list_providers")
    async def test_provider_switch_with_colon_syntax(self, mock_list):
        """ProviderCommand should parse provider:model syntax correctly."""
        from victor.ui.slash.commands.model import ProviderCommand

        mock_list.return_value = ["anthropic"]

        agent = MagicMock()
        agent.switch_provider = AsyncMock(return_value=True)
        agent.get_current_provider_info.return_value = {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
            "native_tool_calls": True,
            "thinking_mode": False,
        }

        ctx = self._make_ctx(args=["anthropic:claude-sonnet-4-5"], agent=agent)
        cmd = ProviderCommand()
        await cmd.execute(ctx)

        agent.switch_provider.assert_called_once_with(
            provider_name="anthropic",
            model="claude-sonnet-4-5",
        )


class TestSwitchCommandSwitchProvider:
    """Test SwitchCommand._switch_provider calls with correct kwargs."""

    def _make_ctx(self, args=None, agent=None):
        console = Console(file=io.StringIO())
        settings = MagicMock()
        return CommandContext(
            console=console,
            settings=settings,
            agent=agent,
            args=args or [],
        )

    @pytest.mark.asyncio
    async def test_switch_command_calls_with_provider_name(self):
        """ModelCommand._switch_provider must use provider_name= kwarg."""
        from victor.ui.slash.commands.model import ModelCommand

        agent = MagicMock()
        agent.switch_provider = AsyncMock(return_value=True)
        agent.get_current_provider_info.return_value = {
            "provider": "openai",
            "model": "gpt-4o",
            "native_tool_calls": True,
            "thinking_mode": False,
        }

        ctx = self._make_ctx(agent=agent)
        cmd = ModelCommand()
        await cmd._switch_provider(ctx, "openai", "gpt-4o")

        agent.switch_provider.assert_called_once_with(
            provider_name="openai",
            model="gpt-4o",
        )
