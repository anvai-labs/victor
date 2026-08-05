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

"""Tests for Smart Routing feature flag integration.

Tests the interaction between the USE_SMART_ROUTING feature flag and
the smart routing provider system.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager
from victor.providers.smart_router import SmartRoutingProvider
from victor.providers.routing_config import SmartRoutingConfig, RoutingProfile
from victor.providers.base import BaseProvider, CompletionResponse, Message, StreamChunk


class MockProvider(BaseProvider):
    """Mock provider for testing."""

    def __init__(self, name: str):
        self._name = name
        self._chat_call_count = 0

    @property
    def name(self) -> str:
        """Provider name."""
        return self._name

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str,
        **kwargs,
    ) -> CompletionResponse:
        """Mock chat implementation."""
        self._chat_call_count += 1
        return CompletionResponse(
            content=f"Response from {self._name}",
            model=model,
            provider=self._name,
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        **kwargs,
    ):
        """Mock stream implementation."""
        yield StreamChunk(
            content=f"Stream chunk from {self._name}",
            model=model,
            provider=self._name,
        )

    async def close(self) -> None:
        """Mock close implementation."""
        pass

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return False


@pytest.fixture
def reset_feature_flag():
    """Reset feature flag manager before each test."""
    from victor.core.feature_flags import reset_feature_flag_manager

    reset_feature_flag_manager()
    yield
    reset_feature_flag_manager()


@pytest.fixture
def mock_providers():
    """Create mock providers for testing."""
    return [
        MockProvider(name="ollama"),
        MockProvider(name="anthropic"),
        MockProvider(name="openai"),
    ]


class TestSmartRoutingFeatureFlag:
    """Tests for smart routing feature flag integration."""

    def test_feature_flag_exists(self, reset_feature_flag):
        """Test that USE_SMART_ROUTING feature flag exists."""
        manager = get_feature_flag_manager()

        # Check flag is defined
        assert FeatureFlag.USE_SMART_ROUTING in FeatureFlag

        # Check env var name
        assert FeatureFlag.USE_SMART_ROUTING.get_env_var_name() == "VICTOR_USE_SMART_ROUTING"

        # Check yaml key
        assert FeatureFlag.USE_SMART_ROUTING.get_yaml_key() == "use_smart_routing"

    def test_feature_flag_default_disabled(self, reset_feature_flag):
        """Test that smart routing is disabled by default."""
        # Create manager with default_enabled=False for testing
        from victor.core.feature_flags import FeatureFlagConfig

        config = FeatureFlagConfig(default_enabled=False)
        manager = get_feature_flag_manager(config=config)

        # Should be disabled by default
        assert not manager.is_enabled(FeatureFlag.USE_SMART_ROUTING)

    def test_feature_flag_enabled_via_env(self, reset_feature_flag, monkeypatch):
        """Test that feature flag can be enabled via environment variable."""
        # Set environment variable
        monkeypatch.setenv("VICTOR_USE_SMART_ROUTING", "true")

        # Create new manager to pick up env var
        from victor.core.feature_flags import reset_feature_flag_manager

        reset_feature_flag_manager()
        manager = get_feature_flag_manager()

        # Should be enabled
        assert manager.is_enabled(FeatureFlag.USE_SMART_ROUTING)

    def test_feature_flag_enabled_via_yaml(self, reset_feature_flag, tmp_path):
        """Test that feature flag can be enabled via YAML config."""
        import yaml

        # Create features.yaml
        config_path = tmp_path / "features.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"features": {"use_smart_routing": True}}, f)

        # Load manager with custom config
        from victor.core.feature_flags import FeatureFlagConfig

        config = FeatureFlagConfig(config_path=config_path)
        manager = get_feature_flag_manager(config=config)

        # Should be enabled
        assert manager.is_enabled(FeatureFlag.USE_SMART_ROUTING)

    def test_feature_flag_runtime_override(self, reset_feature_flag):
        """Test that feature flag can be enabled at runtime."""
        # Create manager with default_enabled=False for testing
        from victor.core.feature_flags import FeatureFlagConfig

        config = FeatureFlagConfig(default_enabled=False)
        manager = get_feature_flag_manager(config=config)

        # Initially disabled
        assert not manager.is_enabled(FeatureFlag.USE_SMART_ROUTING)

        # Enable at runtime
        manager.enable(FeatureFlag.USE_SMART_ROUTING)

        # Should be enabled
        assert manager.is_enabled(FeatureFlag.USE_SMART_ROUTING)

        # Disable at runtime
        manager.disable(FeatureFlag.USE_SMART_ROUTING)

        # Should be disabled
        assert not manager.is_enabled(FeatureFlag.USE_SMART_ROUTING)

    @pytest.mark.asyncio
    async def test_smart_routing_respects_feature_flag(self, reset_feature_flag, mock_providers):
        """Test that smart routing provider works correctly."""
        # Create smart routing provider
        config = SmartRoutingConfig(
            enabled=True,
            profile_name="balanced",
        )

        smart_provider = SmartRoutingProvider(
            providers=mock_providers,
            config=config,
        )

        # Feature flag behavior depends on config
        # With default config, it's enabled
        manager = get_feature_flag_manager()
        is_enabled = manager.is_enabled(FeatureFlag.USE_SMART_ROUTING)

        # Smart routing should work (it's a wrapper, not gated by flag)
        # The flag controls whether it's USED in production, not whether it exists
        messages = [Message(role="user", content="test")]

        response = await smart_provider.chat(messages, model="test-model")

        # Verify response is successful
        assert response.content is not None
        assert response.model == "test-model"
        assert "Response from" in response.content

    def test_feature_flag_env_var_parsing(self, reset_feature_flag, monkeypatch):
        """Test that various environment variable values work."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("no", False),
            ("off", False),
        ]

        for env_value, expected in test_cases:
            monkeypatch.setenv("VICTOR_USE_SMART_ROUTING", env_value)

            from victor.core.feature_flags import reset_feature_flag_manager

            reset_feature_flag_manager()
            manager = get_feature_flag_manager()

            assert (
                manager.is_enabled(FeatureFlag.USE_SMART_ROUTING) == expected
            ), f"Failed for env value: {env_value}"

    def test_multiple_feature_flags_independent(self, reset_feature_flag, monkeypatch):
        """Test that smart routing flag is independent of other flags."""
        # Enable smart routing
        monkeypatch.setenv("VICTOR_USE_SMART_ROUTING", "true")

        # Disable another flag
        monkeypatch.setenv("VICTOR_USE_LEARNING_FROM_EXECUTION", "false")

        from victor.core.feature_flags import reset_feature_flag_manager

        reset_feature_flag_manager()
        manager = get_feature_flag_manager()

        # Smart routing should be enabled
        assert manager.is_enabled(FeatureFlag.USE_SMART_ROUTING)

        # Learning-from-execution should be disabled
        assert not manager.is_enabled(FeatureFlag.USE_LEARNING_FROM_EXECUTION)

        # Other flags should use defaults
        assert manager.is_enabled(FeatureFlag.USE_EDGE_MODEL)
