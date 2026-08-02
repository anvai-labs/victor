# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the research RL configuration and hooks."""

from victor_contracts.constants import ToolNames
from victor_contracts.rl import BaseRLConfig

from victor_research.rl import (
    ResearchRLConfig,
    ResearchRLHooks,
    get_default_config,
    get_research_rl_hooks,
)


class TestResearchRLConfig:
    def test_inherits_base_config(self):
        assert issubclass(ResearchRLConfig, BaseRLConfig)

    def test_task_type_mappings_use_canonical_names(self):
        config = ResearchRLConfig()

        for task_type, tools in config.task_type_mappings.items():
            assert tools, f"{task_type} maps to no tools"

        assert ToolNames.WEB_SEARCH in config.task_type_mappings["research"]
        assert ToolNames.WEB_FETCH in config.task_type_mappings["fact_check"]

    def test_fact_check_has_highest_quality_bar(self):
        config = ResearchRLConfig()

        thresholds = config.quality_thresholds
        assert thresholds["fact_check"] == max(thresholds.values())
        assert thresholds["fact_check"] == 0.90
        assert all(0.0 < t <= 1.0 for t in thresholds.values())

    def test_get_tools_for_task(self):
        config = ResearchRLConfig()

        tools = config.get_tools_for_task("synthesis")
        assert tools == config.task_type_mappings["synthesis"]

    def test_preferred_providers_by_task(self):
        config = ResearchRLConfig()

        assert config.get_preferred_providers("fact_check") == ["anthropic", "openai"]
        # unknown tasks fall back to the full provider list
        assert config.get_preferred_providers("unknown") == ["anthropic", "openai", "google"]


class TestResearchRLHooks:
    def test_default_config_used_when_none_given(self):
        hooks = ResearchRLHooks()

        assert isinstance(hooks.config, ResearchRLConfig)

    def test_tool_recommendation_unfiltered(self):
        hooks = ResearchRLHooks()

        tools = hooks.get_tool_recommendation("research")
        assert ToolNames.WEB_SEARCH in tools

    def test_tool_recommendation_filters_by_availability(self):
        hooks = ResearchRLHooks()
        available = [ToolNames.WEB_SEARCH, ToolNames.READ]

        tools = hooks.get_tool_recommendation("research", available_tools=available)

        assert set(tools) <= set(available)
        assert ToolNames.WEB_SEARCH in tools

    def test_quality_threshold_delegates_to_config(self):
        hooks = ResearchRLHooks()

        assert hooks.get_quality_threshold("fact_check") == 0.90

    def test_should_verify_sources_for_evidence_tasks(self):
        hooks = ResearchRLHooks()

        assert hooks.should_verify_sources("fact_check") is True
        assert hooks.should_verify_sources("research") is True
        assert hooks.should_verify_sources("competitive") is False

    def test_min_sources_by_task(self):
        hooks = ResearchRLHooks()

        assert hooks.get_min_sources("fact_check") == 3
        assert hooks.get_min_sources("literature") == 5  # academic bar is higher
        assert hooks.get_min_sources("anything_else") == 2


class TestSingletons:
    def test_get_default_config_is_cached(self):
        assert get_default_config() is get_default_config()

    def test_get_research_rl_hooks_is_cached(self):
        assert get_research_rl_hooks() is get_research_rl_hooks()
