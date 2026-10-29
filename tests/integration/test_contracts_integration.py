"""Integration tests for victor-contracts and victor-ai compatibility.

This test suite validates that:
1. victor-ai properly inherits from victor-contracts protocols
2. External verticals can use victor-contracts without dependencies
3. Backward compatibility is maintained
"""

import pytest

from victor.core.verticals.base import VerticalBase
from victor_contracts.verticals.protocols.base import VerticalBase as SdkVerticalBase
from victor_contracts.verticals.protocols import ToolProvider
from victor_contracts.core.types import VerticalDefinition


class TestSdkIntegration:
    """Tests for victor-contracts and victor-ai integration."""

    def test_victor_ai_inherits_from_sdk(self):
        """victor-ai's VerticalBase should inherit from SDK's VerticalBase."""
        assert issubclass(VerticalBase, SdkVerticalBase)

    def test_sdk_protocol_methods_available(self):
        """SDK protocol methods should be available on victor-ai's VerticalBase."""

        class TestVertical(VerticalBase):
            name = "test"
            description = "Test vertical"

            @classmethod
            def get_tools(cls):
                return ["read", "write"]

            @classmethod
            def get_system_prompt(cls):
                return "Test prompt"

        # SDK methods should work
        assert TestVertical.get_name() == "test"
        assert TestVertical.get_description() == "Test vertical"
        assert TestVertical.get_tools() == ["read", "write"]
        assert TestVertical.get_system_prompt() == "Test prompt"

    def test_sdk_config_generation(self):
        """SDK config generation should work with victor-ai implementation."""

        class TestVertical(VerticalBase):
            name = "test"
            description = "Test vertical"

            @classmethod
            def get_tools(cls):
                return ["read", "write"]

            @classmethod
            def get_system_prompt(cls):
                return "Test prompt"

        config = TestVertical.get_config()

        # Should return victor-ai's VerticalConfig (not SDK's)
        # The victor-ai version has tools as ToolSet
        assert hasattr(config.tools, "tools")

    def test_sdk_definition_generation(self):
        """SDK definition generation should work with victor-ai implementation."""

        class TestVertical(VerticalBase):
            name = "test"
            description = "Test vertical"
            version = "2.0.0"

            @classmethod
            def get_tools(cls):
                return ["read", "write"]

            @classmethod
            def get_system_prompt(cls):
                return "Test prompt"

        definition = TestVertical.get_definition()

        assert isinstance(definition, VerticalDefinition)
        assert definition.name == "test"
        assert definition.version == "2.0.0"
        assert definition.tools == ["read", "write"]
        assert definition.tool_requirements[0].tool_name == "read"

    @pytest.mark.asyncio
    async def test_victor_ai_create_agent_raises_not_implemented(self):
        """victor-ai VerticalBase.create_agent should raise NotImplementedError."""

        class TestVertical(VerticalBase):
            name = "test"
            description = "Test vertical"

            @classmethod
            def get_tools(cls):
                return ["read"]

            @classmethod
            def get_system_prompt(cls):
                return "Test prompt"

        with pytest.raises(NotImplementedError, match="has been removed"):
            await TestVertical.create_agent(
                provider="openai",
                model="gpt-4.1",
                thinking=True,
            )

    def test_sdk_definition_captures_prompt_and_workflow_metadata(self):
        """victor-ai subclasses should inherit SDK prompt/workflow definition hooks."""

        class TestVertical(VerticalBase):
            name = "test"
            description = "Test vertical"
            version = "2.0.0"

            @classmethod
            def get_tools(cls):
                return ["read", "write"]

            @classmethod
            def get_system_prompt(cls):
                return "Test prompt"

            @classmethod
            def get_prompt_templates(cls):
                return {"analysis": "Analyze the repository."}

            @classmethod
            def get_task_type_hints(cls):
                return {
                    "analysis": {
                        "hint": "Start with reading relevant files.",
                        "tool_budget": 10,
                    }
                }

            @classmethod
            def get_provider_hints(cls):
                return {"preferred_providers": ["anthropic"]}

            @classmethod
            def get_evaluation_criteria(cls):
                return ["accuracy"]

        definition = TestVertical.get_definition()

        assert definition.prompt_metadata.templates[0].task_type == "analysis"
        assert definition.prompt_metadata.task_type_hints[0].tool_budget == 10
        assert definition.workflow_metadata.provider_hints["preferred_providers"] == ["anthropic"]
        assert definition.workflow_metadata.evaluation_criteria == ["accuracy"]

    def test_sdk_definition_captures_team_metadata(self):
        """victor-ai subclasses should inherit SDK team metadata hooks."""

        class TestVertical(VerticalBase):
            name = "team_test"
            description = "Team test vertical"

            @classmethod
            def get_tools(cls):
                return ["read"]

            @classmethod
            def get_system_prompt(cls):
                return "Test prompt"

            @classmethod
            def get_team_declarations(cls):
                return {
                    "review_team": {
                        "name": "Review Team",
                        "formation": "sequential",
                        "members": [
                            {
                                "role": "researcher",
                                "goal": "Inspect the target.",
                            }
                        ],
                    }
                }

            @classmethod
            def get_default_team(cls):
                return "review_team"

        definition = TestVertical.get_definition()

        assert definition.team_metadata.default_team == "review_team"
        assert definition.team_metadata.teams[0].team_id == "review_team"
        assert TestVertical.get_team_spec_provider().get_default_team() == "review_team"
        assert TestVertical.get_team_specs()["review_team"].members[0].role == "researcher"

    def test_backward_compatibility_class_attributes(self):
        """Existing class attributes should still work."""

        class TestVertical(VerticalBase):
            name = "test"
            description = "Test vertical"
            version = "1.0.0"

            @classmethod
            def get_tools(cls):
                return []

            @classmethod
            def get_system_prompt(cls):
                return "Test"

        # Class attributes should be accessible
        assert TestVertical.name == "test"
        assert TestVertical.description == "Test vertical"
        assert TestVertical.version == "1.0.0"

    def test_backward_compatibility_get_stages(self):
        """get_stages() should return victor-ai's stage definitions."""

        class TestVertical(VerticalBase):
            name = "test"
            description = "Test"

            @classmethod
            def get_tools(cls):
                return []

            @classmethod
            def get_system_prompt(cls):
                return "Test"

        stages = TestVertical.get_stages()

        # Should have the 7-stage workflow from victor-ai
        assert "INITIAL" in stages
        assert "PLANNING" in stages
        assert "EXECUTION" in stages

    def test_backward_compatibility_get_extensions(self):
        """get_extensions() should work with victor-ai's implementation."""

        class TestVertical(VerticalBase):
            name = "test"
            description = "Test"

            @classmethod
            def get_tools(cls):
                return []

            @classmethod
            def get_system_prompt(cls):
                return "Test"

        extensions = TestVertical.get_extensions(use_cache=False)

        # Should return VerticalExtensions dataclass
        assert extensions is not None
        assert hasattr(extensions, "middleware")
        assert hasattr(extensions, "safety_extensions")

    def test_backward_compatibility_compose(self):
        """compose() should work with victor-ai's implementation."""

        class TestVertical(VerticalBase):
            name = "test"
            description = "Test"

            @classmethod
            def get_tools(cls):
                return []

            @classmethod
            def get_system_prompt(cls):
                return "Test"

        composer = TestVertical.compose()

        # Should return CapabilityComposer
        assert composer is not None
        assert hasattr(composer, "with_metadata")

    def test_mro_order(self):
        """Method Resolution Order should be correct."""

        class TestVertical(VerticalBase):
            name = "test"
            description = "Test"

            @classmethod
            def get_tools(cls):
                return []

            @classmethod
            def get_system_prompt(cls):
                return "Test"

        mro_names = [cls.__name__ for cls in TestVertical.__mro__]

        # First class in MRO is the concrete class itself
        assert mro_names[0] == "TestVertical"
        # SDK's VerticalBase should be in the MRO (after TestVertical)
        assert "VerticalBase" in mro_names  # victor-ai's VerticalBase
        # SDK's VerticalBase should come before ABC
        abc_index = mro_names.index("ABC")
        # There should be a VerticalBase before ABC
        assert "VerticalBase" in mro_names[:abc_index]

    def test_protocol_compatibility(self):
        """Vertical should be compatible with SDK protocols."""

        class TestVertical(VerticalBase):
            name = "test"
            description = "Test"

            @classmethod
            def get_tools(cls):
                return ["read", "write"]

            @classmethod
            def get_system_prompt(cls):
                return "Test"

        # Should be compatible with ToolProvider protocol
        assert isinstance(TestVertical, ToolProvider)
        # The protocol check works with runtime_checkable

    def test_concrete_vertical_integration(self):
        """Test with a realistic concrete vertical."""

        class CodingVertical(VerticalBase):
            name = "coding"
            description = "Software development assistant"
            version = "2.0.0"

            @classmethod
            def get_tools(cls):
                return [
                    "read",
                    "write",
                    "search",
                    "overview",
                    "code_search",
                    "plan",
                    "git",
                    "shell",
                    "lsp",
                ]

            @classmethod
            def get_system_prompt(cls):
                return "You are an expert software developer..."

        # Test all SDK methods
        assert CodingVertical.get_name() == "coding"
        assert CodingVertical.get_description() == "Software development assistant"
        assert len(CodingVertical.get_tools()) == 9
        assert "expert software developer" in CodingVertical.get_system_prompt()

        # Test victor-ai methods
        config = CodingVertical.get_config()
        assert config.tools.tools == set(CodingVertical.get_tools())
        assert len(CodingVertical.get_stages()) == 7

        # Test extensions
        extensions = CodingVertical.get_extensions(use_cache=False)
        assert extensions is not None


class TestZeroDependencyVertical:
    """Tests for creating a zero-dependency vertical using only victor-contracts."""

    def test_sdk_only_vertical(self):
        """A vertical can be defined using only victor-contracts protocols."""

        # This simulates an external vertical that only depends on victor-contracts
        from victor_contracts.verticals.protocols.base import VerticalBase as SdkBase

        class SdkOnlyVertical(SdkBase):
            @classmethod
            def get_name(cls) -> str:
                return "sdk-only"

            @classmethod
            def get_description(cls) -> str:
                return "contract-only vertical"

            @classmethod
            def get_tools(cls) -> list[str]:
                return ["read", "write"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "You are a helpful assistant."

        # SDK methods should work
        assert SdkOnlyVertical.get_name() == "sdk-only"
        assert SdkOnlyVertical.get_tools() == ["read", "write"]
        assert SdkOnlyVertical.get_system_prompt() == "You are a helpful assistant."

        # Note: get_config() will use the SDK's simple implementation
        # that returns a SDK VerticalConfig (not the enhanced victor-ai one)

    def test_sdk_only_vertical_definition(self):
        """contract-only verticals should produce valid definitions."""

        from victor_contracts.verticals.protocols.base import VerticalBase as SdkBase

        class SdkOnlyVertical(SdkBase):
            name = "sdk-only"
            description = "contract-only vertical"

            @classmethod
            def get_name(cls) -> str:
                return cls.name

            @classmethod
            def get_description(cls) -> str:
                return cls.description

            @classmethod
            def get_tools(cls) -> list[str]:
                return ["read", "write"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "You are a helpful assistant."

        definition = SdkOnlyVertical.get_definition()
        assert definition.name == "sdk-only"
        assert definition.tools == ["read", "write"]
