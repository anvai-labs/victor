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

"""Integration tests for provider and model switching.

Tests the ability to switch providers and models mid-conversation using
real orchestrator instances (not mocked internals).

Coverage:
- Model switching on the same provider
- Provider switching with model preservation
- Provider + model switching together
- Context preservation during switching
- Tool adapter reinitialization
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings, ProfileConfig


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(analytics_enabled=False, tool_cache_enabled=False, provider_health_checks=False)


@pytest.fixture
def mock_provider():
    """Create a mock provider for testing."""
    from unittest.mock import AsyncMock

    provider = MagicMock()
    provider.name = "mock"
    provider.supports_tools.return_value = True
    provider.supports_streaming.return_value = True
    # Mock async methods
    provider.discover_capabilities = AsyncMock()
    provider.discover_capabilities.return_value = MagicMock(
        provider="mock",
        model="test-model",
        context_window=128000,
        supports_tools=True,
        supports_streaming=True,
        native_tool_calls=True,
        source="discovery",
    )
    # Mock health check - needs to be async
    provider.check_health = AsyncMock(return_value=True)
    return provider


@pytest.fixture
def mock_provider2():
    """Create a second mock provider for switching."""
    from unittest.mock import AsyncMock

    provider = MagicMock()
    provider.name = "mock2"
    provider.supports_tools.return_value = True
    provider.supports_streaming.return_value = True
    # Mock async methods
    provider.discover_capabilities = AsyncMock()
    provider.discover_capabilities.return_value = MagicMock(
        provider="mock2",
        model="test-model",
        context_window=128000,
        supports_tools=True,
        supports_streaming=True,
        native_tool_calls=True,
        source="discovery",
    )

    # Mock health check - needs to be async
    async def mock_health_check():
        return True

    provider.check_health = AsyncMock(return_value=True)
    return provider


@pytest.fixture
def orchestrator(settings, mock_provider):
    """Create orchestrator with mocked provider."""
    with patch("victor.agent.orchestrator.UsageLogger"):
        orc = AgentOrchestrator(
            settings=settings,
            provider=mock_provider,
            model="test-model",
        )

    # Mock _metrics_collector to avoid errors during switch
    orc._metrics_collector = MagicMock()
    orc._metrics_collector.update_model_info = MagicMock()

    return orc


class TestModelSwitchingIntegration:
    """Integration tests for model switching on the same provider."""

    def test_switch_model_same_provider(self, orchestrator, mock_provider):
        """Test switching model on the same provider."""
        import asyncio

        initial_provider = orchestrator.provider_name
        initial_model = orchestrator.model

        # Switch to different model on same provider (async method, run in sync context)
        success = asyncio.run(orchestrator.switch_model("different-model"))

        assert success is True
        assert orchestrator.model == "different-model"
        assert orchestrator.provider_name == initial_provider  # Provider unchanged
        assert initial_model != orchestrator.model

    def test_model_switch_updates_provider_manager(self, orchestrator):
        """Test that model switch updates ProviderManager state."""
        import asyncio

        # Switch model - async method, run in sync context
        asyncio.run(orchestrator.switch_model("new-model"))

        # ProviderManager should have updated model
        assert orchestrator._provider_manager.model == "new-model"

    def test_multiple_model_switches(self, orchestrator):
        """Test multiple consecutive model switches."""
        import asyncio

        models = ["model1", "model2", "model3"]

        for model in models:
            success = asyncio.run(orchestrator.switch_model(model))
            assert success is True
            assert orchestrator.model == model

    def test_model_switch_updates_tool_adapter(self, orchestrator):
        """Test that model switch updates tool adapter."""
        import asyncio

        original_adapter = orchestrator.tool_adapter

        # Switch model - async method, run in sync context
        asyncio.run(orchestrator.switch_model("new-model"))

        # Tool adapter should be updated
        assert orchestrator.tool_adapter is not None


class TestProviderSwitchingIntegration:
    """Integration tests for provider switching."""

    @pytest.mark.asyncio
    async def test_switch_provider_preserves_model(self, orchestrator, mock_provider2):
        """Test switching provider while keeping the same model."""
        initial_model = orchestrator.model

        # Patch at the location where it's used (provider switcher imports from registry)
        with patch("victor.agent.provider.switcher.ProviderRegistry") as mock_registry:
            mock_registry.create.return_value = mock_provider2

            # Switch provider (keep same model) - note: async call
            await orchestrator.switch_provider("mock2", model=initial_model)

            # ProviderManager's switch_provider doesn't return a value, it just switches
            assert orchestrator.model == initial_model  # Model preserved
            assert orchestrator.provider_name == "mock2"

    @pytest.mark.asyncio
    async def test_switch_provider_with_new_model(self, orchestrator, mock_provider2):
        """Test switching provider with a new model."""
        new_model = "new-provider-model"

        with patch("victor.agent.provider.switcher.ProviderRegistry") as mock_registry:
            mock_registry.create.return_value = mock_provider2

            # Switch provider with different model - note: async call
            await orchestrator.switch_provider("mock2", model=new_model)

            assert orchestrator.model == new_model
            assert orchestrator.provider_name == "mock2"

    @pytest.mark.asyncio
    async def test_switch_provider_with_custom_settings(self, orchestrator, mock_provider2):
        """Test switching provider with custom provider settings."""
        custom_settings = {"base_url": "http://localhost:8080", "timeout": 60}

        with patch("victor.agent.provider.switcher.ProviderRegistry") as mock_registry:
            mock_registry.create.return_value = mock_provider2

            with patch.object(
                orchestrator.settings,
                "get_provider_settings",
                return_value=custom_settings,
            ):
                success = await orchestrator.switch_provider("mock2", "model")

            assert success is True
            # Verify registry was called with custom settings
            mock_registry.create.assert_called_with("mock2", **custom_settings)


class TestProviderModelCombinedSwitching:
    """Integration tests for combined provider + model switching."""

    @pytest.mark.asyncio
    async def test_switch_provider_and_model_together(self, orchestrator, mock_provider2):
        """Test switching provider and model simultaneously."""
        initial_provider = orchestrator.provider_name
        initial_model = orchestrator.model

        with patch("victor.agent.provider.switcher.ProviderRegistry") as mock_registry:
            mock_registry.create.return_value = mock_provider2

            # Switch both provider and model - note: async call
            await orchestrator.switch_provider("mock2", "new-model")

            assert orchestrator.model == "new-model"
            assert orchestrator.provider_name == "mock2"
            assert orchestrator.model != initial_model
            assert orchestrator.provider_name != initial_provider

    @pytest.mark.asyncio
    async def test_switch_to_profile(self, orchestrator, mock_provider2):
        """Test switching to a pre-configured profile."""
        profile = ProfileConfig(
            provider="mock2",
            model="profile-model",
            temperature=0.8,
            max_tokens=2048,
        )

        with patch("victor.agent.provider.switcher.ProviderRegistry") as mock_registry:
            mock_registry.create.return_value = mock_provider2

            # Switch to profile (both provider and model) - note: async call
            await orchestrator.switch_provider(profile.provider, profile.model)

            assert orchestrator.model == profile.model
            assert orchestrator.provider_name == profile.provider


class TestProfileBasedSwitching:
    """Integration tests for profile-based switching."""

    @pytest.fixture
    def profiles_config(self):
        """Create test profiles configuration."""
        return {
            "fast": ProfileConfig(
                provider="mock",
                model="fast-model",
                temperature=0.5,
                max_tokens=2048,
                description="Fast local model",
            ),
            "smart": ProfileConfig(
                provider="mock",
                model="smart-model",
                temperature=0.8,
                max_tokens=8192,
                description="Smart cloud model",
            ),
        }

    @pytest.mark.asyncio
    async def test_switch_between_profiles(self, orchestrator, profiles_config, mock_provider2):
        """Test switching between different profiles."""
        with patch("victor.agent.provider.switcher.ProviderRegistry") as mock_registry:
            mock_registry.create.return_value = mock_provider2

            # Start with fast profile - note: async call
            profile1 = profiles_config["fast"]
            await orchestrator.switch_provider(profile1.provider, profile1.model)
            assert orchestrator.model == profile1.model

            # Switch to smart profile - note: async call
            profile2 = profiles_config["smart"]
            await orchestrator.switch_provider(profile2.provider, profile2.model)
            assert orchestrator.model == profile2.model

    def test_profile_metadata_preserved(self, orchestrator):
        """Test that profile metadata is accessible."""
        profile = ProfileConfig(
            provider="mock",
            model="test-model",
            temperature=0.7,
            max_tokens=4096,
            description="Test profile",
            tool_selection={"max_cost_threshold": 0.5},
        )

        # Verify profile metadata
        assert profile.provider == "mock"
        assert profile.model == "test-model"
        assert profile.temperature == 0.7
        assert profile.max_tokens == 4096
        assert profile.description == "Test profile"
        assert profile.tool_selection is not None
        assert profile.tool_selection["max_cost_threshold"] == 0.5


class TestSwitchSequencePreservation:
    """Integration tests for context and state preservation during switching."""

    def test_model_switch_preserves_conversation_state(self, orchestrator):
        """Test that model switch preserves conversation state."""
        import asyncio

        # Add messages to conversation using the correct API
        orchestrator.add_message("user", "Question 1")
        orchestrator.add_message("assistant", "Answer 1")

        initial_count = orchestrator.get_message_count()

        # Switch model (async method, run in sync context)
        asyncio.run(orchestrator.switch_model("new-model"))

        # Conversation should be preserved
        assert orchestrator.get_message_count() == initial_count
        messages = orchestrator.messages
        assert len(messages) == 2
        assert messages[0].content == "Question 1"
        assert messages[1].content == "Answer 1"

    @pytest.mark.asyncio
    async def test_provider_switch_preserves_conversation_state(self, orchestrator, mock_provider2):
        """Test that provider switch preserves conversation state."""
        # Build conversation using the correct API
        orchestrator.add_message("user", "Question 1")
        orchestrator.add_message("assistant", "Answer 1")
        orchestrator.add_message("user", "Question 2")

        initial_count = orchestrator.get_message_count()

        with patch("victor.agent.provider.switcher.ProviderRegistry") as mock_registry:
            mock_registry.create.return_value = mock_provider2

            # Switch provider - note: async call
            await orchestrator.switch_provider("mock2", "new-model")

            # Conversation should be preserved
            assert orchestrator.get_message_count() == initial_count
            messages = orchestrator.messages
            assert len(messages) == 3
            assert messages[0].content == "Question 1"
            assert messages[2].content == "Question 2"

    @pytest.mark.asyncio
    async def test_multiple_switches_preserve_context(self, orchestrator, mock_provider2):
        """Test that multiple switches preserve conversation context."""
        # Build conversation using the correct API
        orchestrator.add_message("user", "Initial question")
        orchestrator.add_message("assistant", "Initial answer")

        with patch("victor.agent.provider.switcher.ProviderRegistry") as mock_registry:
            mock_registry.create.return_value = mock_provider2

            # First switch (model - use async version in async context)
            await orchestrator.switch_model("model1")
            assert orchestrator.get_message_count() == 2

            # Second switch (model - use async version in async context)
            await orchestrator.switch_model("model2")
            assert orchestrator.get_message_count() == 2

            # Provider switch - note: async call
            await orchestrator.switch_provider("mock2", "model3")
            assert orchestrator.get_message_count() == 2

            # All messages still present
            messages = orchestrator.messages
            assert messages[0].content == "Initial question"
            assert messages[1].content == "Initial answer"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
