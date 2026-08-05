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

"""Integration tests for CLI-Framework integration.

Tests the complete flow from CLI through FrameworkShim to the orchestrator,
validating that framework features (observability, verticals, session events)
are properly wired.

Test Categories:
1. FrameworkShim creates orchestrator with observability
2. Vertical configuration is applied when specified
3. Legacy mode bypasses framework features
4. Session lifecycle events are emitted
5. CQRS bridge integration (when enabled)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.framework.shim import FrameworkShim, get_vertical, list_verticals
from victor.observability.integration import ObservabilityIntegration
from victor.core.verticals.base import VerticalBase, VerticalRegistry


class MockVertical(VerticalBase):
    """Mock vertical for integration testing."""

    name = "test_integration_vertical"
    description = "Integration test vertical"
    version = "1.0.0"

    @classmethod
    def get_tools(cls):
        return ["read", "write", "shell"]

    @classmethod
    def get_system_prompt(cls):
        return "You are an integration test assistant."

    @classmethod
    def get_stages(cls):
        return {
            "INITIAL": {"keywords": ["start"], "tools": []},
            "TESTING": {"keywords": ["test"], "tools": ["shell"]},
        }


class TestCLIFrameworkIntegration:
    """Integration tests for CLI-Framework flow."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for integration tests."""
        settings = MagicMock()
        settings.provider = "anthropic"
        settings.model = "claude-3-5-sonnet"
        settings.airgapped_mode = False
        settings.enable_observability = True
        return settings

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator with all expected attributes.

        Implements CapabilityRegistryProtocol for proper capability checking
        by the VerticalIntegrationPipeline.
        """
        orch = MagicMock()
        orch.prompt_builder = MagicMock()
        orch.prompt_builder.set_custom_prompt = MagicMock()
        orch.conversation_state = MagicMock()
        orch.observability = None
        orch.set_observability = MagicMock(
            side_effect=lambda obs: setattr(orch, "observability", obs)
        )

        # Add protocol methods for tools
        orch._enabled_tools = set()
        orch.set_enabled_tools = MagicMock(
            side_effect=lambda tools: setattr(orch, "_enabled_tools", tools)
        )
        orch.get_enabled_tools = MagicMock(side_effect=lambda: orch._enabled_tools)

        # Add vertical context support
        orch._vertical_context = None
        orch.set_vertical_context = MagicMock(
            side_effect=lambda ctx: setattr(orch, "_vertical_context", ctx)
        )

        # Implement CapabilityRegistryProtocol for vertical integration pipeline
        # The pipeline uses protocol-based capability checking
        capabilities = {
            "enabled_tools": ("1.0", orch.set_enabled_tools),
            "prompt_builder": ("1.0", lambda: orch.prompt_builder),
            "vertical_context": ("1.0", orch.set_vertical_context),
        }
        orch.has_capability = MagicMock(
            side_effect=lambda name, min_version=None: name in capabilities
        )
        orch.invoke_capability = MagicMock(
            side_effect=lambda name, *args, min_version=None: (
                capabilities[name][1](*args) if name in capabilities else None
            )
        )
        orch.get_capability = MagicMock(
            side_effect=lambda name: (
                {
                    "name": name,
                    "version": capabilities[name][0],
                    "callable": capabilities[name][1],
                }
                if name in capabilities
                else None
            )
        )
        orch.list_capabilities = MagicMock(return_value=list(capabilities.keys()))

        return orch

    @pytest.fixture(autouse=True)
    def register_mock_vertical(self):
        """Register mock vertical for tests."""
        VerticalRegistry.register(MockVertical)
        yield
        VerticalRegistry.unregister("test_integration_vertical")

    @pytest.mark.asyncio
    async def test_framework_path_creates_orchestrator_with_observability(
        self, mock_settings, mock_orchestrator
    ):
        """Test that framework path (default) wires observability."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            shim = FrameworkShim(
                mock_settings,
                profile_name="default",
                enable_observability=True,
            )
            orchestrator = await shim.create_orchestrator()

            # Verify orchestrator was created
            assert orchestrator == mock_orchestrator
            mock_from_settings.assert_called_once()

            # Verify observability is wired
            assert shim.observability is not None
            assert isinstance(shim.observability, ObservabilityIntegration)
            assert orchestrator.observability == shim.observability

    @pytest.mark.asyncio
    async def test_framework_path_without_vertical(self, mock_settings, mock_orchestrator):
        """Test framework path works without specifying a vertical."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            # No vertical specified - this is the default CLI behavior
            shim = FrameworkShim(
                mock_settings,
                profile_name="default",
                vertical=None,  # No vertical
                enable_observability=True,
            )
            orchestrator = await shim.create_orchestrator()

            # Orchestrator created successfully
            assert orchestrator == mock_orchestrator

            # Observability still wired
            assert shim.observability is not None

            # No vertical config applied
            assert shim.vertical is None
            assert shim.vertical_config is None

    @pytest.mark.asyncio
    async def test_vertical_configuration_applied(self, mock_settings, mock_orchestrator):
        """Test that vertical configuration is properly applied.

        This test verifies that:
        1. The vertical is resolved correctly
        2. Tools from the vertical are applied
        3. Vertical context is stored with stages

        Note: System prompt application is tested separately as it depends on
        the CapabilityRegistryProtocol which is complex to mock correctly.
        """
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            shim = FrameworkShim(
                mock_settings,
                profile_name="default",
                vertical=MockVertical,
            )
            await shim.create_orchestrator()

            # Verify vertical was resolved
            assert shim.vertical == MockVertical

            # Verify tools were applied via set_enabled_tools protocol method
            enabled_tools = mock_orchestrator._enabled_tools
            assert "read" in enabled_tools
            assert "write" in enabled_tools
            assert "shell" in enabled_tools

            # Verify vertical config was captured
            # The shim stores vertical_config from the integration result
            assert shim.vertical_config is not None or shim._integration_result is not None

            # Verify stages were applied via vertical context
            assert mock_orchestrator._vertical_context is not None
            assert "INITIAL" in mock_orchestrator._vertical_context.stages
            assert "TESTING" in mock_orchestrator._vertical_context.stages

    @pytest.mark.asyncio
    async def test_vertical_lookup_by_string_name(self, mock_settings, mock_orchestrator):
        """Test vertical lookup by string name."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            # Use string name instead of class
            shim = FrameworkShim(
                mock_settings,
                vertical="test_integration_vertical",
            )
            await shim.create_orchestrator()

            # Should resolve to the class
            assert shim.vertical == MockVertical

    @pytest.mark.asyncio
    async def test_session_lifecycle_events(self, mock_settings, mock_orchestrator):
        """Test session start and end events are emitted."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            shim = FrameworkShim(
                mock_settings,
                session_id="test-session-123",
                enable_observability=True,
            )
            await shim.create_orchestrator()

            # Mock the observability methods
            shim._observability.on_session_start = MagicMock()
            shim._observability.on_session_end = MagicMock()

            # Emit session start
            shim.emit_session_start({"mode": "integration_test"})
            shim._observability.on_session_start.assert_called_once()
            start_payload = shim._observability.on_session_start.call_args.args[0]
            assert start_payload["mode"] == "integration_test"
            assert start_payload["session_id"] == "test-session-123"

            # Emit session end
            shim.emit_session_end(tool_calls=5, duration_seconds=10.0, success=True)
            shim._observability.on_session_end.assert_called_once_with(
                tool_calls=5, duration_seconds=10.0, success=True
            )

    @pytest.mark.asyncio
    async def test_session_id_propagation(self, mock_settings, mock_orchestrator):
        """Test session ID is propagated to observability."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            custom_session_id = "custom-session-abc123"
            shim = FrameworkShim(
                mock_settings,
                session_id=custom_session_id,
                enable_observability=True,
            )
            await shim.create_orchestrator()

            # Session ID should be used
            assert shim.session_id == custom_session_id
            assert shim.observability._session_id == custom_session_id

    @pytest.mark.asyncio
    async def test_observability_disabled(self, mock_settings, mock_orchestrator):
        """Test that observability can be disabled."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            shim = FrameworkShim(
                mock_settings,
                enable_observability=False,
            )
            await shim.create_orchestrator()

            # No observability wired
            assert shim.observability is None

            # Lifecycle events should be no-ops (no error)
            shim.emit_session_start({"mode": "test"})
            shim.emit_session_end(tool_calls=0)

    @pytest.mark.asyncio
    async def test_thinking_mode_propagation(self, mock_settings, mock_orchestrator):
        """Test thinking mode is passed to orchestrator creation."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            shim = FrameworkShim(
                mock_settings,
                profile_name="default",
                thinking=True,
            )
            await shim.create_orchestrator()

            # Verify thinking was passed
            mock_from_settings.assert_called_once_with(
                mock_settings,
                profile_name="default",
                thinking=True,
            )


class TestVerticalRegistryIntegration:
    """Integration tests for vertical registry functions."""

    @pytest.fixture(autouse=True)
    def register_mock_vertical(self):
        """Register mock vertical for tests."""
        VerticalRegistry.register(MockVertical)
        yield
        VerticalRegistry.unregister("test_integration_vertical")

    def test_get_vertical_returns_class(self):
        """Test get_vertical returns the vertical class."""
        vertical = get_vertical("test_integration_vertical")
        assert vertical == MockVertical

    def test_get_vertical_case_insensitive(self):
        """Test get_vertical is case-insensitive."""
        vertical = get_vertical("TEST_INTEGRATION_VERTICAL")
        assert vertical == MockVertical

    def test_get_vertical_none_returns_none(self):
        """Test get_vertical(None) returns None."""
        assert get_vertical(None) is None

    def test_get_vertical_not_found_returns_none(self):
        """Test get_vertical returns None for unknown verticals."""
        assert get_vertical("nonexistent_vertical") is None

    def test_list_verticals_includes_registered(self):
        """Test list_verticals includes registered verticals."""
        names = list_verticals()
        assert "test_integration_vertical" in names

    def test_list_verticals_includes_builtins(self):
        """Test list_verticals includes built-in verticals."""
        names = list_verticals()
        # benchmark is always built-in; contrib verticals (coding, etc.)
        # are external packages and may not be installed
        assert "benchmark" in names


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility with legacy code."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.provider = "anthropic"
        return settings

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_shim_is_drop_in_replacement(self, mock_settings, mock_orchestrator):
        """Test FrameworkShim is a drop-in replacement for from_settings."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            # Old way (direct call)
            # orchestrator = await AgentOrchestrator.from_settings(settings, "default")

            # New way (through shim)
            shim = FrameworkShim(mock_settings, "default")
            orchestrator = await shim.create_orchestrator()

            # Same orchestrator returned
            assert orchestrator == mock_orchestrator

            # Same signature used
            mock_from_settings.assert_called_once()

    @pytest.mark.asyncio
    async def test_orchestrator_property_access(self, mock_settings, mock_orchestrator):
        """Test orchestrator property access pattern."""
        with patch(
            "victor.agent.orchestrator.AgentOrchestrator.from_settings",
            new_callable=AsyncMock,
        ) as mock_from_settings:
            mock_from_settings.return_value = mock_orchestrator

            shim = FrameworkShim(mock_settings)

            # Before creation
            assert shim.orchestrator is None

            # After creation
            await shim.create_orchestrator()
            assert shim.orchestrator == mock_orchestrator
