# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the data analysis RL configuration and hooks."""

from victor_contracts.constants import ToolNames
from victor_contracts.rl import BaseRLConfig

from victor_dataanalysis.rl import (
    DataAnalysisRLConfig,
    DataAnalysisRLHooks,
    get_data_analysis_rl_hooks,
    get_default_config,
)


class TestDataAnalysisRLConfig:
    def test_inherits_base_config(self):
        assert issubclass(DataAnalysisRLConfig, BaseRLConfig)

    def test_task_type_mappings_use_canonical_names(self):
        config = DataAnalysisRLConfig()

        for task_type, tools in config.task_type_mappings.items():
            assert tools, f"{task_type} maps to no tools"

        assert ToolNames.SHELL in config.task_type_mappings["ml"]
        assert ToolNames.READ in config.task_type_mappings["eda"]

    def test_statistics_has_highest_quality_bar(self):
        config = DataAnalysisRLConfig()

        thresholds = config.quality_thresholds
        assert thresholds["statistics"] == max(thresholds.values())
        assert all(0.0 < t <= 1.0 for t in thresholds.values())

    def test_get_tools_for_task(self):
        config = DataAnalysisRLConfig()

        assert config.get_tools_for_task("cleaning") == config.task_type_mappings["cleaning"]

    def test_preferred_output_length(self):
        config = DataAnalysisRLConfig()

        assert config.get_preferred_output_length("ml") == "long"
        assert config.get_preferred_output_length("profiling") == "short"
        assert config.get_preferred_output_length("unknown_task") == "medium"


class TestDataAnalysisRLHooks:
    def test_default_config_used_when_none_given(self):
        hooks = DataAnalysisRLHooks()

        assert isinstance(hooks.config, DataAnalysisRLConfig)

    def test_tool_recommendation_unfiltered(self):
        hooks = DataAnalysisRLHooks()

        tools = hooks.get_tool_recommendation("eda")
        assert ToolNames.READ in tools

    def test_tool_recommendation_filters_by_availability(self):
        hooks = DataAnalysisRLHooks()
        available = [ToolNames.SHELL, ToolNames.READ]

        tools = hooks.get_tool_recommendation("eda", available_tools=available)

        assert set(tools) <= set(available)
        assert ToolNames.READ in tools

    def test_quality_threshold_delegates_to_config(self):
        hooks = DataAnalysisRLHooks()

        assert hooks.get_quality_threshold("statistics") == 0.90

    def test_analysis_always_includes_code(self):
        hooks = DataAnalysisRLHooks()

        assert hooks.should_include_code("eda") is True
        assert hooks.should_include_code("reporting") is True

    def test_preferred_libraries_per_task(self):
        hooks = DataAnalysisRLHooks()

        assert "scipy" in hooks.get_preferred_libraries("statistics")
        assert "scikit-learn" in hooks.get_preferred_libraries("ml")
        assert hooks.get_preferred_libraries("unknown") == ["pandas", "numpy"]


class TestSingletons:
    def test_get_default_config_is_cached(self):
        assert get_default_config() is get_default_config()

    def test_get_data_analysis_rl_hooks_is_cached(self):
        assert get_data_analysis_rl_hooks() is get_data_analysis_rl_hooks()
