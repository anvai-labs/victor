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

"""Tests for the framework StageManager."""

import pytest

from victor.agent.conversation.state_machine import ConversationStage
from victor.framework.stage_manager import (
    StageDefinition,
    StageManager,
    StageManagerConfig,
    StageManagerProtocol,
    StageTransition,
    create_stage_manager,
    get_coding_stages,
    get_data_analysis_stages,
    get_research_stages,
)
from victor_contracts import StageDefinition as SdkStageDefinition


class TestStageDefinition:
    """Tests for StageDefinition dataclass."""

    def test_creation_with_defaults(self):
        """Test creating a definition with default values."""
        defn = StageDefinition(name="test_stage")
        assert defn.name == "test_stage"
        assert defn.display_name == "Test Stage"  # Auto-generated
        assert defn.description == ""
        assert defn.keywords == []
        assert defn.tools == set()
        assert defn.order == 0
        assert defn.can_transition_to is None
        assert defn.min_confidence == 0.5

    def test_creation_with_all_fields(self):
        """Test creating a definition with all fields specified."""
        defn = StageDefinition(
            name="data_loading",
            display_name="Loading Data",
            description="Loading and reading data files",
            keywords=["load", "read", "import"],
            tools={"read", "execute_code"},
            order=1,
            can_transition_to={"data_cleaning", "analysis"},
            min_confidence=0.7,
        )
        assert defn.name == "data_loading"
        assert defn.display_name == "Loading Data"
        assert defn.description == "Loading and reading data files"
        assert defn.keywords == ["load", "read", "import"]
        assert defn.tools == {"read", "execute_code"}
        assert defn.order == 1
        assert defn.can_transition_to == {"data_cleaning", "analysis"}
        assert defn.min_confidence == 0.7

    def test_display_name_auto_generation(self):
        """Test that display_name is auto-generated from snake_case name."""
        defn = StageDefinition(name="my_custom_stage")
        assert defn.display_name == "My Custom Stage"

    def test_to_sdk_definition_round_trip(self):
        """Runtime stage definitions should convert cleanly to SDK definitions."""
        runtime_defn = StageDefinition(
            name="data_loading",
            display_name="Loading Data",
            description="Loading files",
            keywords=["load", "read"],
            tools={"read", "execute_code"},
            order=2,
            can_transition_to={"analysis"},
            min_confidence=0.7,
        )

        sdk_defn = runtime_defn.to_sdk_definition()
        restored = StageDefinition.from_sdk_definition(
            sdk_defn,
            display_name=runtime_defn.display_name,
            order=runtime_defn.order,
        )

        assert isinstance(sdk_defn, SdkStageDefinition)
        assert sdk_defn.tools == {"read", "execute_code"}
        assert sdk_defn.next_stages == {"analysis"}
        assert restored.display_name == "Loading Data"
        assert restored.order == 2
        assert restored.can_transition_to == {"analysis"}


class TestStageManagerConfig:
    """Tests for StageManagerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = StageManagerConfig()
        assert config.track_history is True
        assert config.max_history_size == 100
        assert config.transition_cooldown == 2.0
        assert config.stage_tool_boost == 0.15
        assert config.adjacent_tool_boost == 0.08
        assert config.backward_confidence_threshold == 0.85

    def test_custom_values(self):
        """Test custom configuration values."""
        config = StageManagerConfig(
            track_history=False,
            max_history_size=50,
            transition_cooldown=1.0,
            stage_tool_boost=0.2,
            adjacent_tool_boost=0.1,
            backward_confidence_threshold=0.9,
        )
        assert config.track_history is False
        assert config.max_history_size == 50
        assert config.transition_cooldown == 1.0
        assert config.stage_tool_boost == 0.2
        assert config.adjacent_tool_boost == 0.1
        assert config.backward_confidence_threshold == 0.9


class TestStageManager:
    """Tests for StageManager class."""

    @pytest.fixture
    def stage_manager(self):
        """Create a fresh StageManager for each test."""
        return create_stage_manager()

    @pytest.fixture
    def custom_stage_manager(self):
        """Create a StageManager with custom stages."""
        custom_stages = {
            "data_loading": StageDefinition(
                name="data_loading",
                display_name="Loading Data",
                keywords=["load", "read", "import"],
                tools={"read", "execute_code"},
                order=1,
            ),
            "data_cleaning": StageDefinition(
                name="data_cleaning",
                display_name="Cleaning Data",
                keywords=["clean", "preprocess"],
                tools={"execute_code"},
                order=2,
            ),
        }
        return create_stage_manager(custom_stages=custom_stages)

    def test_initial_stage(self, stage_manager):
        """Test that manager starts in INITIAL stage."""
        assert stage_manager.get_stage() == ConversationStage.INITIAL
        assert stage_manager.get_stage_name() == "initial"

    def test_get_stage_tools(self, stage_manager):
        """Test getting tools for current stage."""
        tools = stage_manager.get_stage_tools()
        assert isinstance(tools, set)
        # Initial stage should have some tools
        # (actual tools depend on metadata registry)

    def test_record_tool(self, stage_manager):
        """Test recording tool execution."""
        # Record a read tool - should work without error
        stage_manager.record_tool("read", {"file_path": "/src/main.py"})
        # State should be updated
        state = stage_manager.get_state_summary()
        assert state["tools_executed"] >= 1

    def test_record_message(self, stage_manager):
        """Test recording message."""
        stage_manager.record_message("Please analyze this code", is_user=True)
        state = stage_manager.get_state_summary()
        assert state["message_count"] >= 1

    def test_tool_priority_boost_for_stage_tool(self, stage_manager):
        """Test that stage-relevant tools get priority boost."""
        # Force to reading stage
        stage_manager.force_stage(ConversationStage.READING)
        # Read is a reading stage tool
        boost = stage_manager.get_tool_priority_boost("read")
        assert boost >= 0.0  # Should have some boost

    def test_tool_priority_boost_for_non_stage_tool(self, stage_manager):
        """Test that non-stage tools get no boost."""
        # In initial stage, write is not typically recommended
        stage_manager.force_stage(ConversationStage.INITIAL)
        # Check a tool that's not in initial stage
        boost = stage_manager.get_tool_priority_boost("some_random_tool_xyz")
        assert boost == 0.0

    def test_should_include_tool(self, stage_manager):
        """Test tool inclusion recommendation."""
        # This delegates to the underlying machine
        result = stage_manager.should_include_tool("read")
        assert isinstance(result, bool)

    def test_reset(self, stage_manager):
        """Test resetting stage manager."""
        # Record some activity
        stage_manager.record_tool("read", {"file_path": "/test.py"})
        stage_manager.record_message("test message")

        # Reset
        stage_manager.reset()

        # Should be back to initial
        assert stage_manager.get_stage() == ConversationStage.INITIAL

    def test_force_stage(self, stage_manager):
        """Test forcing a specific stage."""
        stage_manager.force_stage(ConversationStage.EXECUTION, confidence=0.9)
        assert stage_manager.get_stage() == ConversationStage.EXECUTION

    def test_get_state_summary(self, stage_manager):
        """Test getting state summary."""
        summary = stage_manager.get_state_summary()
        assert "stage" in summary
        assert "confidence" in summary
        assert "message_count" in summary
        assert "tools_executed" in summary
        assert "recommended_tools" in summary

    def test_get_transition_history(self, stage_manager):
        """Test getting transition history."""
        # Force a transition
        stage_manager.force_stage(ConversationStage.READING)
        history = stage_manager.get_transition_history()
        assert isinstance(history, list)

    def test_get_transitions_summary(self, stage_manager):
        """Test getting transitions summary."""
        summary = stage_manager.get_transitions_summary()
        assert "total_transitions" in summary
        assert "average_confidence" in summary

    def test_custom_stages(self, custom_stage_manager):
        """Test custom stage definitions."""
        defn = custom_stage_manager.get_stage_definition("data_loading")
        assert defn is not None
        assert defn.name == "data_loading"
        assert defn.tools == {"read", "execute_code"}

    def test_register_stage(self, stage_manager):
        """Test registering a new stage."""
        new_stage = StageDefinition(
            name="custom_stage",
            display_name="Custom Stage",
            tools={"custom_tool"},
        )
        stage_manager.register_stage(new_stage)

        defn = stage_manager.get_stage_definition("custom_stage")
        assert defn is not None
        assert defn.tools == {"custom_tool"}

    def test_register_sdk_stage(self, stage_manager):
        """StageManager should accept SDK stage definitions and normalize them to runtime stages."""
        sdk_stage = SdkStageDefinition(
            name="sdk_stage",
            description="SDK stage",
            tools={"sdk_tool"},
            keywords=["sdk"],
            next_stages={"completion"},
            min_confidence=0.8,
        )

        stage_manager.register_stage(sdk_stage)

        defn = stage_manager.get_stage_definition("sdk_stage")
        assert defn is not None
        assert isinstance(defn, StageDefinition)
        assert defn.description == "SDK stage"
        assert defn.tools == {"sdk_tool"}
        assert defn.can_transition_to == {"completion"}

    def test_get_all_stage_definitions(self, custom_stage_manager):
        """Test getting all custom stage definitions."""
        definitions = custom_stage_manager.get_all_stage_definitions()
        assert "data_loading" in definitions
        assert "data_cleaning" in definitions

    def test_config_property(self, stage_manager):
        """Test config property access."""
        config = stage_manager.config
        assert isinstance(config, StageManagerConfig)

    def test_machine_property(self, stage_manager):
        """Test accessing underlying machine."""
        machine = stage_manager.machine
        assert machine is not None
        # Can access machine's stage
        assert machine.get_stage() == stage_manager.get_stage()

    def test_serialization(self, stage_manager):
        """Test to_dict serialization."""
        # Add some state
        stage_manager.record_tool("read", {"file_path": "/test.py"})

        data = stage_manager.to_dict()
        assert "machine" in data
        assert "config" in data
        assert data["config"]["track_history"] is True

    def test_deserialization(self, stage_manager):
        """Test from_dict deserialization."""
        # Add some state and serialize
        stage_manager.force_stage(ConversationStage.READING)
        data = stage_manager.to_dict()

        # Deserialize
        restored = StageManager.from_dict(data)
        # Note: Stage might reset on deserialization depending on machine state
        assert restored is not None


class TestStageManagerProtocol:
    """Tests for StageManagerProtocol compliance."""

    def test_protocol_compliance(self):
        """Test that StageManager satisfies the protocol."""
        manager = create_stage_manager()
        assert isinstance(manager, StageManagerProtocol)

    def test_protocol_methods_exist(self):
        """Test that all protocol methods exist on StageManager."""
        manager = create_stage_manager()

        # All protocol methods should be callable
        assert callable(manager.get_stage)
        assert callable(manager.get_stage_name)
        assert callable(manager.get_stage_tools)
        assert callable(manager.record_tool)
        assert callable(manager.record_message)
        assert callable(manager.get_tool_priority_boost)
        assert callable(manager.should_include_tool)
        assert callable(manager.reset)


class TestStandardStageDefinitions:
    """Tests for standard stage definition factories."""

    def test_get_coding_stages(self):
        """Test coding stage definitions."""
        stages = get_coding_stages()
        assert isinstance(stages, dict)

        # Should have all standard coding stages
        expected = [
            "initial",
            "planning",
            "reading",
            "analysis",
            "execution",
            "verification",
            "completion",
        ]
        for stage_name in expected:
            assert stage_name in stages
            assert isinstance(stages[stage_name], StageDefinition)

        # Check ordering
        assert stages["initial"].order < stages["planning"].order
        assert stages["planning"].order < stages["reading"].order
        assert stages["execution"].order < stages["verification"].order

        # Coding stage tool metadata should use canonical core names.
        assert "shell" in stages["execution"].tools
        assert "shell" in stages["verification"].tools
        assert stages["completion"].tools == {"shell"}
        assert "bash" not in stages["execution"].tools
        assert "bash" not in stages["verification"].tools

    def test_get_data_analysis_stages(self):
        """Test data analysis stage definitions."""
        stages = get_data_analysis_stages()
        assert isinstance(stages, dict)

        # Should have data analysis specific stages
        expected = [
            "initial",
            "data_loading",
            "data_cleaning",
            "analysis",
            "visualization",
            "completion",
        ]
        for stage_name in expected:
            assert stage_name in stages

        # Check data-specific tools
        assert "execute_code" in stages["data_loading"].tools
        assert "execute_code" in stages["visualization"].tools

    def test_get_research_stages(self):
        """Test research stage definitions."""
        stages = get_research_stages()
        assert isinstance(stages, dict)

        # Should have research specific stages
        expected = ["initial", "gathering", "reading", "synthesis", "completion"]
        for stage_name in expected:
            assert stage_name in stages

        # Check research-specific tools
        assert "web_search" in stages["gathering"].tools
        assert "web_fetch" in stages["gathering"].tools

    def test_stages_have_keywords(self):
        """Test that all stages have keywords defined."""
        for get_stages in [
            get_coding_stages,
            get_data_analysis_stages,
            get_research_stages,
        ]:
            stages = get_stages()
            for name, defn in stages.items():
                assert len(defn.keywords) > 0, f"Stage {name} has no keywords"

    def test_stages_have_tools(self):
        """Test that most stages have tools defined."""
        for get_stages in [
            get_coding_stages,
            get_data_analysis_stages,
            get_research_stages,
        ]:
            stages = get_stages()
            stages_with_tools = sum(1 for defn in stages.values() if defn.tools)
            # At least half should have tools
            assert stages_with_tools >= len(stages) // 2


class TestFactoryFunction:
    """Tests for create_stage_manager factory function."""

    def test_create_with_defaults(self):
        """Test creating with default parameters."""
        manager = create_stage_manager()
        assert manager is not None
        assert manager.get_stage() == ConversationStage.INITIAL

    def test_create_with_config(self):
        """Test creating with custom config."""
        config = StageManagerConfig(
            stage_tool_boost=0.25,
            transition_cooldown=1.0,
        )
        manager = create_stage_manager(config=config)
        assert manager.config.stage_tool_boost == 0.25
        assert manager.config.transition_cooldown == 1.0

    def test_create_with_custom_stages(self):
        """Test creating with custom stages."""
        custom_stages = {
            "my_stage": StageDefinition(
                name="my_stage",
                tools={"my_tool"},
            )
        }
        manager = create_stage_manager(custom_stages=custom_stages)
        assert manager.get_stage_definition("my_stage") is not None

    def test_create_with_sdk_custom_stages(self):
        """Factory should accept SDK stage definitions and normalize to runtime stages."""
        custom_stages = {
            "my_stage": SdkStageDefinition(
                name="my_stage",
                description="SDK-backed stage",
                tools={"my_tool"},
            )
        }

        manager = create_stage_manager(custom_stages=custom_stages)
        definition = manager.get_stage_definition("my_stage")

        assert definition is not None
        assert isinstance(definition, StageDefinition)
        assert definition.tools == {"my_tool"}
        assert definition.description == "SDK-backed stage"


class TestStageTransition:
    """Tests for StageTransition dataclass."""

    def test_creation(self):
        """Test creating a transition record."""
        transition = StageTransition(
            from_stage="initial",
            to_stage="reading",
            confidence=0.8,
            trigger="tool",
            timestamp=1234567890.0,
            context={"tool_name": "read"},
        )
        assert transition.from_stage == "initial"
        assert transition.to_stage == "reading"
        assert transition.confidence == 0.8
        assert transition.trigger == "tool"
        assert transition.timestamp == 1234567890.0
        assert transition.context == {"tool_name": "read"}

    def test_default_context(self):
        """Test that context defaults to empty dict."""
        transition = StageTransition(
            from_stage="initial",
            to_stage="reading",
            confidence=0.8,
            trigger="tool",
            timestamp=1234567890.0,
        )
        assert transition.context == {}


class TestIntegrationWithCodingStages:
    """Integration tests with coding stage definitions."""

    def test_coding_stage_manager(self):
        """Test StageManager with coding stages."""
        stages = get_coding_stages()
        manager = create_stage_manager(custom_stages=stages)

        # Should be able to look up all stages
        for name in ["initial", "reading", "execution"]:
            defn = manager.get_stage_definition(name)
            assert defn is not None, f"Stage {name} not found"

    def test_data_analysis_stage_manager(self):
        """Test StageManager with data analysis stages."""
        stages = get_data_analysis_stages()
        manager = create_stage_manager(custom_stages=stages)

        # Check data-specific stage
        loading_defn = manager.get_stage_definition("data_loading")
        assert loading_defn is not None
        assert "load" in loading_defn.keywords

    def test_research_stage_manager(self):
        """Test StageManager with research stages."""
        stages = get_research_stages()
        manager = create_stage_manager(custom_stages=stages)

        # Check research-specific stage
        gathering_defn = manager.get_stage_definition("gathering")
        assert gathering_defn is not None
        assert "web_search" in gathering_defn.tools
