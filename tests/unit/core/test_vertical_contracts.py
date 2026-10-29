"""Contract tests for VerticalBase subclasses.

Provides reusable test patterns that verify verticals conform to
the expected contracts: abstract methods, naming, API version,
protocol conformance, stage graph connectivity, and config fields.
"""

import pytest
from typing import List

from victor.core.verticals.base import VerticalBase, VerticalConfig
from victor.core.vertical_types import TieredToolConfig


@pytest.fixture
def concrete_vertical():
    """Create a minimal VerticalBase subclass for testing."""

    class MinimalVertical(VerticalBase):
        name = "test_minimal"
        description = "Minimal vertical for contract testing"

        @classmethod
        def get_tools(cls) -> List[str]:
            return ["read", "write"]

        @classmethod
        def get_system_prompt(cls) -> str:
            return "You are a test assistant."

    return MinimalVertical


class TestVerticalBaseContract:
    """Verify VerticalBase contract requirements."""

    def test_has_name(self, concrete_vertical):
        """Vertical must have a non-empty name."""
        assert concrete_vertical.name
        assert isinstance(concrete_vertical.name, str)

    def test_has_description(self, concrete_vertical):
        """Vertical must have a description."""
        assert concrete_vertical.description
        assert isinstance(concrete_vertical.description, str)

    def test_get_tools_returns_list(self, concrete_vertical):
        """get_tools() must return a list of strings."""
        tools = concrete_vertical.get_tools()
        assert isinstance(tools, list)
        assert all(isinstance(t, str) for t in tools)

    def test_get_system_prompt_returns_str(self, concrete_vertical):
        """get_system_prompt() must return a string."""
        prompt = concrete_vertical.get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_has_api_version(self, concrete_vertical):
        """Vertical must have VERTICAL_API_VERSION."""
        assert hasattr(concrete_vertical, "VERTICAL_API_VERSION")
        assert isinstance(concrete_vertical.VERTICAL_API_VERSION, int)
        assert concrete_vertical.VERTICAL_API_VERSION >= 1

    def test_register_tools_exists(self, concrete_vertical):
        """Vertical must have register_tools() method."""
        assert hasattr(concrete_vertical, "register_tools")
        assert callable(concrete_vertical.register_tools)


class TestVerticalProtocolConformance:
    """Verify protocol satisfaction checks."""

    def test_is_subclass_of_vertical_base(self, concrete_vertical):
        """Concrete vertical must be a VerticalBase subclass."""
        assert issubclass(concrete_vertical, VerticalBase)

    def test_tiered_tool_config_type_safety(self):
        """VerticalExtensions.tiered_tool_config should accept TieredToolConfig."""
        from victor.core.verticals.protocols import VerticalExtensions

        config = TieredToolConfig(mandatory={"read"})
        ext = VerticalExtensions(tiered_tool_config=config)
        assert ext.tiered_tool_config is config


class TestStageGraphConnectivity:
    """Verify stage graph contracts."""

    def test_stages_have_valid_transitions(self, concrete_vertical):
        """All next_stages references must point to valid stage names."""
        stages = concrete_vertical.get_stages()
        stage_names = set(stages.keys())
        for name, stage in stages.items():
            if stage.next_stages:
                invalid = stage.next_stages - stage_names
                assert not invalid, f"Stage '{name}' references invalid next stages: {invalid}"

    def test_stages_have_terminal(self, concrete_vertical):
        """At least one stage should have no next_stages (terminal)."""
        stages = concrete_vertical.get_stages()
        terminals = [name for name, stage in stages.items() if not stage.next_stages]
        assert len(terminals) >= 1, "No terminal stage found"

    def test_all_stages_reachable(self, concrete_vertical):
        """All stages should be reachable from some stage's next_stages or be initial."""
        stages = concrete_vertical.get_stages()
        if len(stages) <= 1:
            return

        # Collect all referenced stages
        referenced = set()
        for stage in stages.values():
            referenced.update(stage.next_stages or set())

        stage_names = set(stages.keys())
        # At least one stage is an entry point (not referenced by others)
        entry_points = stage_names - referenced
        assert len(entry_points) >= 1, "No entry point stage found"


class TestVerticalConfigContract:
    """Verify config contract."""

    def test_get_config_returns_vertical_config(self, concrete_vertical):
        """get_config() must return a VerticalConfig."""
        concrete_vertical.clear_config_cache(clear_all=True)
        config = concrete_vertical.get_config()
        assert isinstance(config, VerticalConfig)

    def test_config_has_tools(self, concrete_vertical):
        """Config must have tools populated."""
        concrete_vertical.clear_config_cache(clear_all=True)
        config = concrete_vertical.get_config()
        assert config.tools is not None

    def test_config_has_system_prompt(self, concrete_vertical):
        """Config must have a system prompt."""
        concrete_vertical.clear_config_cache(clear_all=True)
        config = concrete_vertical.get_config()
        assert config.system_prompt
        assert isinstance(config.system_prompt, str)

    def test_config_caching_works(self, concrete_vertical):
        """Repeated get_config() calls should return the same object."""
        concrete_vertical.clear_config_cache(clear_all=True)
        config1 = concrete_vertical.get_config()
        config2 = concrete_vertical.get_config()
        assert config1 is config2

    def test_config_cache_clear(self, concrete_vertical):
        """clear_config_cache() should force rebuild."""
        concrete_vertical.clear_config_cache(clear_all=True)
        config1 = concrete_vertical.get_config()
        concrete_vertical.clear_config_cache()
        config2 = concrete_vertical.get_config()
        assert config1 is not config2
