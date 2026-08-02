# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for DevOps tool dependency configuration (YAML-backed)."""

from pathlib import Path

import pytest

import victor_devops.tool_dependencies as td
from victor_devops.tool_dependencies import (
    DEVOPS_COMPOSED_PATTERNS,
    DevOpsToolDependencyProvider,
    get_composed_pattern,
    get_devops_tool_graph,
    get_provider,
    list_composed_patterns,
    reset_devops_tool_graph,
)


class TestYamlConfig:
    def test_yaml_file_ships_with_package(self):
        yaml_path = Path(td.__file__).parent / "tool_dependencies.yaml"

        assert yaml_path.is_file()

    def test_config_loads_transitions_and_clusters(self):
        config = td._get_config()

        assert config.transitions, "expected non-empty tool transitions"
        assert config.clusters, "expected non-empty tool clusters"


class TestProviderContract:
    def test_provider_class_is_deprecated_but_functional(self):
        with pytest.warns(DeprecationWarning, match="DevOpsToolDependencyProvider"):
            provider = DevOpsToolDependencyProvider()

        deps = provider.get_dependencies()
        assert isinstance(deps, list)

    def test_entry_point_factory_returns_provider(self):
        with pytest.warns(DeprecationWarning):
            provider = get_provider()

        assert isinstance(provider, DevOpsToolDependencyProvider)


class TestDeprecatedConstants:
    def test_legacy_constant_access_warns_and_resolves(self):
        with pytest.warns(DeprecationWarning, match="DEVOPS_TOOL_TRANSITIONS"):
            transitions = td.DEVOPS_TOOL_TRANSITIONS

        assert transitions == td._get_config().transitions

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            td.NOT_A_REAL_EXPORT


class TestComposedPatterns:
    def test_expected_patterns_registered(self):
        names = list_composed_patterns()

        for expected in (
            "dockerfile_pipeline",
            "ci_cd_config",
            "kubernetes_manifest",
            "terraform_workflow",
            "helm_deploy",
        ):
            assert expected in names

    def test_pattern_shape(self):
        for name, pattern in DEVOPS_COMPOSED_PATTERNS.items():
            assert pattern["sequence"], f"{name} has empty sequence"
            assert pattern["description"]
            assert 0 < pattern["weight"] <= 1.0

    def test_get_composed_pattern_lookup(self):
        pattern = get_composed_pattern("dockerfile_pipeline")

        assert pattern is not None
        assert "docker" in pattern["sequence"]

    def test_get_composed_pattern_unknown_returns_none(self):
        assert get_composed_pattern("nonexistent") is None


class TestToolGraph:
    def test_graph_is_cached_until_reset(self):
        reset_devops_tool_graph()
        first = get_devops_tool_graph()
        second = get_devops_tool_graph()

        assert first is second

        reset_devops_tool_graph()
        third = get_devops_tool_graph()
        assert third is not first

    def test_graph_suggests_next_tools_after_read(self):
        reset_devops_tool_graph()
        graph = get_devops_tool_graph()

        suggestions = graph.suggest_next_tools("read")
        assert isinstance(suggestions, list)
