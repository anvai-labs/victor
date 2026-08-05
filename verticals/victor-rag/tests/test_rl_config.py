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

"""Tests for RAG RL configuration."""

from victor.framework.rl import LearnerType


class TestRAGRLConfig:
    """Tests for RAGRLConfig."""

    def test_has_active_learners(self):
        """RAGRLConfig should have active learners defined."""
        from victor_rag.rl import RAGRLConfig

        config = RAGRLConfig()
        assert hasattr(config, "active_learners")
        assert isinstance(config.active_learners, list)
        assert len(config.active_learners) >= 1

    def test_active_learners_are_valid_types(self):
        """Active learners should be valid LearnerType enum values."""
        from victor_rag.rl import RAGRLConfig

        config = RAGRLConfig()
        for learner in config.active_learners:
            assert isinstance(learner, LearnerType)

    def test_has_task_type_mappings(self):
        """RAGRLConfig should have task type mappings."""
        from victor_rag.rl import RAGRLConfig

        config = RAGRLConfig()
        assert hasattr(config, "task_type_mappings")
        assert isinstance(config.task_type_mappings, dict)
        # Should have RAG-specific task types
        assert len(config.task_type_mappings) >= 1

    def test_task_type_mappings_use_canonical_names(self):
        """Task type mappings should use canonical tool names."""
        from victor_rag.rl import RAGRLConfig
        from victor_contracts.constants import ToolNames

        config = RAGRLConfig()

        # Get all canonical tool names
        canonical_names = {
            getattr(ToolNames, attr)
            for attr in dir(ToolNames)
            if not attr.startswith("_") and isinstance(getattr(ToolNames, attr), str)
        }

        # Check that all tools in mappings are canonical or RAG-specific
        for task_type, tools in config.task_type_mappings.items():
            for tool in tools:
                # Allow RAG-specific tools and canonical names
                is_valid = tool in canonical_names or tool.startswith("rag_")
                assert is_valid, f"Tool '{tool}' in task type '{task_type}' should be canonical"

    def test_has_quality_thresholds(self):
        """RAGRLConfig should have quality thresholds."""
        from victor_rag.rl import RAGRLConfig

        config = RAGRLConfig()
        assert hasattr(config, "quality_thresholds")
        assert isinstance(config.quality_thresholds, dict)

    def test_quality_thresholds_are_valid(self):
        """Quality thresholds should be valid floats between 0 and 1."""
        from victor_rag.rl import RAGRLConfig

        config = RAGRLConfig()
        for task_type, threshold in config.quality_thresholds.items():
            assert isinstance(threshold, float)
            assert 0.0 <= threshold <= 1.0, f"Threshold for '{task_type}' should be between 0 and 1"

    def test_get_tools_for_task(self):
        """get_tools_for_task should return tools for a task type."""
        from victor_rag.rl import RAGRLConfig

        config = RAGRLConfig()

        # Get first task type
        if config.task_type_mappings:
            task_type = next(iter(config.task_type_mappings.keys()))
            tools = config.get_tools_for_task(task_type)
            assert isinstance(tools, list)

    def test_get_tools_for_unknown_task(self):
        """get_tools_for_task should return empty list for unknown task."""
        from victor_rag.rl import RAGRLConfig

        config = RAGRLConfig()
        tools = config.get_tools_for_task("unknown_task_type_xyz")
        assert isinstance(tools, list)
        assert len(tools) == 0

    def test_get_quality_threshold(self):
        """get_quality_threshold should return threshold for task type."""
        from victor_rag.rl import RAGRLConfig

        config = RAGRLConfig()

        # Get first task type with threshold
        if config.quality_thresholds:
            task_type = next(iter(config.quality_thresholds.keys()))
            threshold = config.get_quality_threshold(task_type)
            assert isinstance(threshold, float)

    def test_get_quality_threshold_default(self):
        """get_quality_threshold should return default for unknown task."""
        from victor_rag.rl import RAGRLConfig

        config = RAGRLConfig()
        threshold = config.get_quality_threshold("unknown_task_type_xyz")
        assert isinstance(threshold, float)
        # Should return some reasonable default
        assert 0.0 <= threshold <= 1.0

    def test_has_default_patience(self):
        """RAGRLConfig should have default patience settings."""
        from victor_rag.rl import RAGRLConfig

        config = RAGRLConfig()
        assert hasattr(config, "default_patience")
        assert isinstance(config.default_patience, dict)

    def test_get_patience(self):
        """get_patience should return patience for provider."""
        from victor_rag.rl import RAGRLConfig

        config = RAGRLConfig()

        # Test with known providers
        patience = config.get_patience("anthropic")
        assert isinstance(patience, int)
        assert patience > 0

    def test_is_learner_active(self):
        """is_learner_active should check if learner is active."""
        from victor_rag.rl import RAGRLConfig

        config = RAGRLConfig()

        # First learner should be active
        if config.active_learners:
            first_learner = config.active_learners[0]
            assert config.is_learner_active(first_learner) is True

    def test_repr(self):
        """RAGRLConfig should have a useful repr."""
        from victor_rag.rl import RAGRLConfig

        config = RAGRLConfig()
        repr_str = repr(config)

        assert "RAGRLConfig" in repr_str


class TestRAGRLHooks:
    """Tests for RAGRLHooks."""

    def test_has_config(self):
        """RAGRLHooks should have a config property."""
        from victor_rag.rl import RAGRLHooks, RAGRLConfig

        hooks = RAGRLHooks()
        assert hasattr(hooks, "config")
        assert isinstance(hooks.config, RAGRLConfig)

    def test_accepts_custom_config(self):
        """RAGRLHooks should accept a custom config."""
        from victor_rag.rl import RAGRLHooks, RAGRLConfig

        custom_config = RAGRLConfig()
        hooks = RAGRLHooks(config=custom_config)
        assert hooks.config is custom_config

    def test_get_tool_recommendation(self):
        """get_tool_recommendation should return tool recommendations."""
        from victor_rag.rl import RAGRLHooks

        hooks = RAGRLHooks()

        # Get first task type from config
        if hooks.config.task_type_mappings:
            task_type = next(iter(hooks.config.task_type_mappings.keys()))
            tools = hooks.get_tool_recommendation(task_type)
            assert isinstance(tools, list)

    def test_get_tool_recommendation_with_filter(self):
        """get_tool_recommendation should filter by available tools."""
        from victor_rag.rl import RAGRLHooks

        hooks = RAGRLHooks()

        # Get first task type from config
        if hooks.config.task_type_mappings:
            task_type = next(iter(hooks.config.task_type_mappings.keys()))
            expected_tools = hooks.config.task_type_mappings[task_type]
            if expected_tools:
                # Filter to just one tool
                available = [expected_tools[0]]
                tools = hooks.get_tool_recommendation(task_type, available_tools=available)
                assert all(t in available for t in tools)

    def test_get_patience_recommendation(self):
        """get_patience_recommendation should return patience value."""
        from victor_rag.rl import RAGRLHooks

        hooks = RAGRLHooks()
        patience = hooks.get_patience_recommendation("anthropic", "claude-3-opus")
        assert isinstance(patience, int)
        assert patience > 0

    def test_get_quality_threshold(self):
        """get_quality_threshold should return quality threshold."""
        from victor_rag.rl import RAGRLHooks

        hooks = RAGRLHooks()

        # Get first task type with threshold
        if hooks.config.quality_thresholds:
            task_type = next(iter(hooks.config.quality_thresholds.keys()))
            threshold = hooks.get_quality_threshold(task_type)
            assert isinstance(threshold, float)
            assert 0.0 <= threshold <= 1.0

    def test_repr(self):
        """RAGRLHooks should have a useful repr."""
        from victor_rag.rl import RAGRLHooks

        hooks = RAGRLHooks()
        repr_str = repr(hooks)

        assert "RAGRLHooks" in repr_str


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def test_get_default_config(self):
        """get_default_config should return singleton config."""
        from victor_rag.rl import get_default_config, RAGRLConfig

        config = get_default_config()
        assert isinstance(config, RAGRLConfig)

        # Should return same instance
        config2 = get_default_config()
        assert config is config2

    def test_get_rag_rl_hooks(self):
        """get_rag_rl_hooks should return singleton hooks."""
        from victor_rag.rl import get_rag_rl_hooks, RAGRLHooks

        hooks = get_rag_rl_hooks()
        assert isinstance(hooks, RAGRLHooks)

        # Should return same instance
        hooks2 = get_rag_rl_hooks()
        assert hooks is hooks2
