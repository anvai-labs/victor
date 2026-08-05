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

"""Tests for victor.providers.health module."""

import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from victor.providers.health import (
    ProviderHealthChecker,
    ProviderHealthResult,
    check_provider_health,
)


class TestProviderHealthResult:
    """Tests for ProviderHealthResult dataclass."""

    def test_healthy_result(self):
        """Test healthy provider result."""
        result = ProviderHealthResult(
            healthy=True,
            provider="deepseek",
            model="deepseek-chat",
            issues=[],
            warnings=[],
            info={"registered": True},
        )

        assert result.healthy is True
        assert result.error_message == "Provider is healthy"
        assert result.to_dict()["status"] == "HEALTHY"

    def test_unhealthy_result(self):
        """Test unhealthy provider result."""
        result = ProviderHealthResult(
            healthy=False,
            provider="deepseek",
            model="deepseek-chat",
            issues=["API key not found", "Provider not registered"],
            warnings=[],
            info={},
        )

        assert result.healthy is False
        assert "API key not found" in result.error_message
        assert "Provider not registered" in result.error_message

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ProviderHealthResult(
            healthy=False,
            provider="anthropic",
            model="claude-3-5-haiku",
            issues=["No API key"],
            warnings=["Using keychain"],
            info={"key_source": "keyring"},
        )

        d = result.to_dict()
        assert d["healthy"] is False
        assert d["provider"] == "anthropic"
        assert d["model"] == "claude-3-5-haiku"
        assert d["status"] == "UNHEALTHY"
        assert len(d["issues"]) == 1
        assert len(d["warnings"]) == 1


class TestProviderHealthChecker:
    """Tests for ProviderHealthChecker class."""

    def test_local_provider_healthy(self):
        """Test local providers are always healthy (no API key needed)."""
        checker = ProviderHealthChecker()

        async def run_check():
            return await checker.check_provider("ollama", "qwen2.5:14b")

        result = asyncio.run(run_check())
        assert result.healthy is True
        assert result.provider == "ollama"

    def test_unregistered_provider(self):
        """Test unregistered provider fails health check."""
        checker = ProviderHealthChecker()

        with patch(
            "victor.providers.health.ProviderRegistry.list_providers",
            return_value=["ollama"],
        ):
            with patch(
                "victor.providers.health.ProviderRegistry.get",
                side_effect=Exception("Not found"),
            ):

                async def run_check():
                    return await checker.check_provider("unknown", "model")

                result = asyncio.run(run_check())
                assert result.healthy is False
                assert len(result.issues) > 0
                assert "not registered" in result.issues[0].lower()

    def test_api_key_missing(self):
        """Test missing API key causes unhealthy status."""
        checker = ProviderHealthChecker()

        with patch.dict("os.environ", {}, clear=True):
            with patch("victor.providers.health.ProviderRegistry.get", return_value=MagicMock()):

                async def run_check():
                    return await checker.check_provider("deepseek", "deepseek-chat")

                result = asyncio.run(run_check())
                assert result.healthy is False
                assert any("API key" in issue or "not found" in issue for issue in result.issues)

    def test_api_key_from_environment(self):
        """Test API key from environment passes health check."""
        checker = ProviderHealthChecker()

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test123456789"}):
            with patch("victor.providers.health.ProviderRegistry.get", return_value=MagicMock()):

                async def run_check():
                    return await checker.check_provider("deepseek", "deepseek-chat")

                result = asyncio.run(run_check())
                assert result.healthy is True
                assert result.info.get("key_source") is not None
                assert "environment" in result.info["key_source"].lower()

    def test_key_format_validation_invalid(self):
        """Test invalid key format fails validation."""
        checker = ProviderHealthChecker()

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "invalid-format"}):
            with patch("victor.providers.health.ProviderRegistry.get", return_value=MagicMock()):

                async def run_check():
                    return await checker.check_provider("deepseek", "deepseek-chat")

                result = asyncio.run(run_check())
                # DeepSeek key must start with sk-
                assert result.healthy is False
                assert any("format" in issue.lower() for issue in result.issues)

    def test_key_format_valid_deepseek(self):
        """Test valid DeepSeek key format passes validation."""
        checker = ProviderHealthChecker()

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-1234567890abcdef"}):
            with patch("victor.providers.health.ProviderRegistry.get", return_value=MagicMock()):

                async def run_check():
                    return await checker.check_provider("deepseek", "deepseek-chat")

                result = asyncio.run(run_check())
                assert result.healthy is True
                assert result.info.get("key_format_valid") is True

    def test_key_format_valid_anthropic(self):
        """Test valid Anthropic key format passes validation."""
        checker = ProviderHealthChecker()

        # Anthropic keys: sk-ant- followed by 95+ characters
        valid_key = "sk-ant-" + "a" * 95
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": valid_key}):
            with patch("victor.providers.health.ProviderRegistry.get", return_value=MagicMock()):

                async def run_check():
                    return await checker.check_provider("anthropic", "claude-3-5-haiku")

                result = asyncio.run(run_check())
                assert result.healthy is True
                assert result.info.get("key_format_valid") is True

    def test_keyring_warning_in_non_interactive(self):
        """Test warning when using keychain in non-interactive mode."""
        from victor.providers.resolution import APIKeyResult, KeySource

        checker = ProviderHealthChecker(non_interactive=True)

        # Create a mock result that simulates getting a key from keyring in non-interactive mode
        # Use valid key format (sk- followed by 20+ characters for DeepSeek)
        mock_result = APIKeyResult(
            key="sk-test12345678901234567890",  # Valid DeepSeek format
            source="keyring",
            source_detail="System keyring (test)",
            sources_attempted=[
                KeySource(
                    source="keyring",
                    description="System keyring",
                    found=True,
                )
            ],
            non_interactive=True,  # This should trigger the warning
            confidence="medium",
        )

        # Patch the resolver's get_api_key to return our mock result
        with patch.object(checker.resolver, "get_api_key", return_value=mock_result):
            with patch("victor.providers.health.ProviderRegistry.get", return_value=MagicMock()):

                async def run_check():
                    return await checker.check_provider("deepseek", "deepseek-chat")

                result = asyncio.run(run_check())
                assert result.healthy is True
                assert len(result.warnings) > 0
                assert any("keychain" in w.lower() for w in result.warnings)

    def test_get_env_var(self):
        """Test _get_env_var method."""
        checker = ProviderHealthChecker()
        assert checker._get_env_var("deepseek") == "DEEPSEEK_API_KEY"
        assert checker._get_env_var("anthropic") == "ANTHROPIC_API_KEY"
        assert checker._get_env_var("unknown") is None

    def test_validate_key_format_no_pattern(self):
        """Test validation returns True when no pattern defined."""
        checker = ProviderHealthChecker()
        assert checker._validate_key_format("unknown_provider", "any-key") is True

    def test_get_format_description(self):
        """Test format descriptions for various providers."""
        checker = ProviderHealthChecker()
        assert "sk-ant-" in checker._get_format_description("anthropic")
        assert "sk-" in checker._get_format_description("openai")
        assert "sk-" in checker._get_format_description("deepseek")
        assert "xai-" in checker._get_format_description("xai")


class TestCheckProviderHealth:
    """Tests for check_provider_health convenience function."""

    def test_convenience_function(self):
        """Test the convenience function works correctly."""
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test123"}):
            with patch("victor.providers.health.ProviderRegistry.get", return_value=MagicMock()):

                async def run_check():
                    return await check_provider_health("deepseek", "deepseek-chat")

                result = asyncio.run(run_check())
                assert isinstance(result, ProviderHealthResult)
                assert result.provider == "deepseek"
                assert result.model == "deepseek-chat"

    def test_passes_kwargs(self):
        """Test that kwargs are passed through."""
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}):
            with patch("victor.providers.health.ProviderRegistry.get", return_value=MagicMock()):

                async def run_check():
                    return await check_provider_health(
                        "deepseek",
                        "deepseek-chat",
                        check_connectivity=False,
                        timeout=10.0,
                    )

                result = asyncio.run(run_check())
                assert result.provider == "deepseek"
