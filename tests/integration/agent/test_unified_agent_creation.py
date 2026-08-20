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

"""Comprehensive tests for unified agent creation system (Phase 4).

Tests cover:
1. UnifiedAgentConfig - Configuration consolidation
2. Migration helpers - Backward compatibility
3. Protocol compliance
"""

import pytest

from victor.agent.config import AgentLifecycleMode, UnifiedAgentConfig, normalize_agent_config
from victor.framework.config import AgentConfig

# =============================================================================
# Test UnifiedAgentConfig
# =============================================================================


class TestUnifiedAgentConfig:
    """Test UnifiedAgentConfig configuration class."""

    def test_create_default_config(self):
        """Test creating default configuration."""
        config = UnifiedAgentConfig()

        assert config.mode == "foreground"
        assert config.provider == "anthropic"
        assert config.tool_budget == 50
        assert config.max_iterations == 25

    def test_foreground_factory_method(self):
        """Test foreground() factory method."""
        config = UnifiedAgentConfig.foreground(
            provider="openai",
            model="gpt-4-turbo",
            tool_budget=100,
        )

        assert config.mode == "foreground"
        assert config.provider == "openai"
        assert config.model == "gpt-4-turbo"
        assert config.tool_budget == 100

    def test_background_factory_method(self):
        """Test background() factory method."""
        config = UnifiedAgentConfig.background(
            task="Implement feature X",
            mode_type="build",
            tool_budget=200,
        )

        assert config.mode == "background"
        assert config.task == "Implement feature X"
        assert config.mode_type == "build"
        assert config.tool_budget == 200

    def test_team_member_factory_method(self):
        """Test team_member() factory method."""
        config = UnifiedAgentConfig.team_member(
            role="researcher",
            capabilities=["search", "analyze"],
            description="Research specialist",
        )

        assert config.mode == "team_member"
        assert config.role == "researcher"
        assert config.capabilities == ["search", "analyze"]
        assert config.description == "Research specialist"

    def test_minimal_preset(self):
        """Test minimal() preset."""
        config = UnifiedAgentConfig.minimal()

        assert config.tool_budget == 15
        assert config.max_iterations == 10
        assert config.enable_semantic_search is False
        assert config.enable_analytics is False
        assert config.enable_tool_cache is False

    def test_high_budget_preset(self):
        """Test high_budget() preset."""
        config = UnifiedAgentConfig.high_budget()

        assert config.tool_budget == 200
        assert config.max_iterations == 100
        assert config.max_concurrent_tools == 10
        assert config.enable_semantic_search is True

    def test_airgapped_preset(self):
        """Test airgapped() preset."""
        config = UnifiedAgentConfig.airgapped()

        assert config.enable_semantic_search is False
        assert config.enable_analytics is False
        assert config.enable_tool_cache is False

    def test_from_agent_config(self):
        """Test migration from AgentConfig."""
        agent_config = AgentConfig(
            tool_budget=100,
            max_iterations=50,
            enable_semantic_search=True,
        )

        unified_config = UnifiedAgentConfig.from_agent_config(agent_config)

        assert unified_config.mode == "foreground"
        assert unified_config.tool_budget == 100
        assert unified_config.max_iterations == 50
        assert unified_config.enable_semantic_search is True

    def test_to_agent_config(self):
        """Test conversion to AgentConfig."""
        unified_config = UnifiedAgentConfig(
            tool_budget=75,
            max_iterations=30,
            enable_parallel_tools=False,
        )

        agent_config = unified_config.to_agent_config()

        assert agent_config.tool_budget == 75
        assert agent_config.max_iterations == 30
        assert agent_config.enable_parallel_tools is False

    def test_to_settings_dict(self):
        """Test conversion to Settings dictionary."""
        config = UnifiedAgentConfig(
            tool_budget=100,
            max_iterations=50,
            streaming_timeout=600.0,
        )

        settings_dict = config.to_settings_dict()

        assert settings_dict["tool_call_budget"] == 100
        assert settings_dict["max_iterations"] == 50
        assert settings_dict["streaming_timeout"] == 600.0
        assert settings_dict["parallel_tool_execution"] is True

    def test_mode_specific_fields_foreground(self):
        """Test foreground-specific configuration fields."""
        config = UnifiedAgentConfig(
            mode="foreground",
            enable_context_compaction=True,
            enable_code_correction=True,
        )

        assert config.enable_context_compaction is True
        assert config.enable_code_correction is True

    def test_mode_specific_fields_background(self):
        """Test background-specific configuration fields."""
        config = UnifiedAgentConfig(
            mode="background",
            task="Test task",
            mode_type="plan",
            websocket=True,
            timeout_seconds=600,
        )

        assert config.task == "Test task"
        assert config.mode_type == "plan"
        assert config.websocket is True
        assert config.timeout_seconds == 600

    def test_mode_specific_fields_team_member(self):
        """Test team member-specific configuration fields."""
        config = UnifiedAgentConfig(
            mode="team_member",
            role="executor",
            allowed_tools=["read", "write", "shell"],
            can_spawn_subagents=True,
            context_limit=256000,
        )

        assert config.role == "executor"
        assert config.allowed_tools == ["read", "write", "shell"]
        assert config.can_spawn_subagents is True
        assert config.context_limit == 256000


# =============================================================================
# Test Migration and Backward Compatibility
# =============================================================================


class TestMigrationAndCompatibility:
    """Test migration helpers and backward compatibility."""

    def test_normalize_agent_config_converts_legacy_config(self):
        """Framework boundaries should normalize deprecated AgentConfig inputs."""
        legacy_config = AgentConfig(tool_budget=150, max_iterations=75)

        normalized = normalize_agent_config(legacy_config)

        assert isinstance(normalized, UnifiedAgentConfig)
        assert normalized.tool_budget == legacy_config.tool_budget
        assert normalized.max_iterations == legacy_config.max_iterations

    def test_agent_config_to_unified_roundtrip(self):
        """Test roundtrip conversion AgentConfig → UnifiedAgentConfig → AgentConfig."""
        original_config = AgentConfig(
            tool_budget=150,
            max_iterations=75,
            enable_parallel_tools=False,
            enable_semantic_search=True,
            streaming_timeout=400.0,
        )

        # Convert to UnifiedAgentConfig
        unified_config = UnifiedAgentConfig.from_agent_config(original_config)

        # Convert back to AgentConfig
        converted_config = unified_config.to_agent_config()

        # Verify all fields match
        assert converted_config.tool_budget == original_config.tool_budget
        assert converted_config.max_iterations == original_config.max_iterations
        assert converted_config.enable_parallel_tools == original_config.enable_parallel_tools
        assert converted_config.enable_semantic_search == original_config.enable_semantic_search
        assert converted_config.streaming_timeout == original_config.streaming_timeout

    def test_unified_config_preserves_extra_fields(self):
        """Test that extra fields are preserved in conversions."""
        config = UnifiedAgentConfig(
            mode="foreground",
            tool_budget=100,
            extra={"custom_field": "custom_value", "another_field": 123},
        )

        settings_dict = config.to_settings_dict()

        # Extra fields should be in settings dict
        assert "custom_field" in settings_dict
        assert settings_dict["custom_field"] == "custom_value"
        assert "another_field" in settings_dict
        assert settings_dict["another_field"] == 123

    def test_all_three_modes_produce_valid_configs(self):
        """Test that all three modes produce valid configurations."""
        # Foreground
        fg_config = UnifiedAgentConfig.foreground()
        assert fg_config.mode == "foreground"
        assert fg_config.tool_budget > 0

        # Background
        bg_config = UnifiedAgentConfig.background(task="Test")
        assert bg_config.mode == "background"
        assert bg_config.task == "Test"

        # Team member (capabilities is required)
        tm_config = UnifiedAgentConfig.team_member(role="tester", capabilities=["test"])
        assert tm_config.mode == "team_member"
        assert tm_config.role == "tester"


# =============================================================================
# Test AgentLifecycleMode Enum
# =============================================================================


class TestAgentLifecycleMode:
    """Test AgentLifecycleMode enumeration."""

    def test_agent_mode_values(self):
        """Test AgentLifecycleMode enum has correct values."""
        assert AgentLifecycleMode.FOREGROUND.value == "foreground"
        assert AgentLifecycleMode.BACKGROUND.value == "background"
        assert AgentLifecycleMode.TEAM_MEMBER.value == "team_member"

    def test_agent_mode_comparison(self):
        """Test AgentLifecycleMode enum comparison."""
        assert AgentLifecycleMode.FOREGROUND == "foreground"
        assert AgentLifecycleMode.BACKGROUND == "background"
        assert AgentLifecycleMode.TEAM_MEMBER == "team_member"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
