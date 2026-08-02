# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the DevOps RL configuration and hooks."""

from victor_contracts.constants import ToolNames
from victor_contracts.rl import BaseRLConfig

from victor_devops.rl import (
    DevOpsRLConfig,
    DevOpsRLHooks,
    get_default_config,
    get_devops_rl_hooks,
)


class TestDevOpsRLConfig:
    def test_inherits_base_config(self):
        assert issubclass(DevOpsRLConfig, BaseRLConfig)

    def test_task_type_mappings_use_canonical_names(self):
        config = DevOpsRLConfig()

        for task_type, tools in config.task_type_mappings.items():
            assert tools, f"{task_type} maps to no tools"

        assert ToolNames.DOCKER in config.task_type_mappings["containerization"]
        assert ToolNames.SHELL in config.task_type_mappings["deployment"]

    def test_deployment_has_highest_quality_bar(self):
        config = DevOpsRLConfig()

        thresholds = config.quality_thresholds
        assert thresholds["deployment"] == max(thresholds.values())
        assert all(0.0 < t <= 1.0 for t in thresholds.values())

    def test_get_tools_for_task(self):
        config = DevOpsRLConfig()

        tools = config.get_tools_for_task("monitoring")
        assert tools == config.task_type_mappings["monitoring"]

    def test_get_quality_threshold_known_task(self):
        config = DevOpsRLConfig()

        assert config.get_quality_threshold("deployment") == 0.90


class TestDevOpsRLHooks:
    def test_default_config_used_when_none_given(self):
        hooks = DevOpsRLHooks()

        assert isinstance(hooks.config, DevOpsRLConfig)

    def test_tool_recommendation_unfiltered(self):
        hooks = DevOpsRLHooks()

        tools = hooks.get_tool_recommendation("containerization")
        assert ToolNames.DOCKER in tools

    def test_tool_recommendation_filters_by_availability(self):
        hooks = DevOpsRLHooks()
        available = [ToolNames.DOCKER, ToolNames.READ]

        tools = hooks.get_tool_recommendation("containerization", available_tools=available)

        assert set(tools) <= set(available)
        assert ToolNames.DOCKER in tools

    def test_quality_threshold_delegates_to_config(self):
        hooks = DevOpsRLHooks()

        assert hooks.get_quality_threshold("deployment") == 0.90


class TestSingletons:
    def test_get_default_config_is_cached(self):
        assert get_default_config() is get_default_config()

    def test_get_devops_rl_hooks_is_cached(self):
        assert get_devops_rl_hooks() is get_devops_rl_hooks()
