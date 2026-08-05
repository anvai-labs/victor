# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Contract tests for the DataAnalysisAssistant vertical definition.

Covers the vertical definition surface: tools, stages, system prompt,
and capability provider wiring.
"""

from victor_contracts.core.types import StageDefinition
from victor_contracts.verticals.protocols.base import VerticalBase

from victor_dataanalysis.assistant import DataAnalysisAssistant


class TestAssistantIdentity:
    def test_name_and_description(self):
        assert DataAnalysisAssistant.name == "dataanalysis"
        assert "Data exploration" in DataAnalysisAssistant.description
        assert issubclass(DataAnalysisAssistant, VerticalBase)

    def test_version_metadata(self):
        assert DataAnalysisAssistant.version == "1.0.0"


class TestAssistantTools:
    def test_tools_are_canonical_names(self):
        from victor.tools.tool_names import ToolNames

        tools = DataAnalysisAssistant.get_tools()

        assert len(tools) == len(set(tools)), "duplicate tool names"
        # File operations come from the framework FileOperationsCapability
        assert ToolNames.READ in tools
        assert ToolNames.WRITE in tools
        # Data analysis-specific additions
        assert ToolNames.SHELL in tools
        assert ToolNames.LS in tools
        assert ToolNames.GRAPH in tools

    def test_web_tools_for_datasets_and_docs(self):
        from victor.tools.tool_names import ToolNames

        tools = DataAnalysisAssistant.get_tools()

        assert ToolNames.WEB_SEARCH in tools
        assert ToolNames.WEB_FETCH in tools


class TestAssistantStages:
    def test_stage_graph_is_closed(self):
        """Every next_stage must reference a defined stage."""
        stages = DataAnalysisAssistant.get_stages()

        defined = set(stages)
        for name, stage in stages.items():
            assert isinstance(stage, StageDefinition)
            assert stage.name == name
            missing = stage.next_stages - defined
            assert not missing, f"{name} references undefined stages: {missing}"

    def test_analysis_lifecycle_stages_present(self):
        stages = DataAnalysisAssistant.get_stages()

        for expected in (
            "INITIAL",
            "DATA_LOADING",
            "EXPLORATION",
            "CLEANING",
            "ANALYSIS",
            "VISUALIZATION",
            "REPORTING",
            "COMPLETION",
        ):
            assert expected in stages

    def test_completion_is_terminal(self):
        stages = DataAnalysisAssistant.get_stages()

        assert stages["COMPLETION"].next_stages == set()

    def test_every_stage_has_tools_and_keywords(self):
        for name, stage in DataAnalysisAssistant.get_stages().items():
            assert stage.tools, f"stage {name} has no tools"
            assert stage.keywords, f"stage {name} has no routing keywords"

    def test_stage_tools_subset_of_vertical_tools(self):
        tools = set(DataAnalysisAssistant.get_tools())

        for name, stage in DataAnalysisAssistant.get_stages().items():
            extra = stage.tools - tools
            assert not extra, f"stage {name} uses tools outside get_tools(): {extra}"


class TestAssistantPrompt:
    def test_system_prompt_covers_core_capabilities(self):
        prompt = DataAnalysisAssistant.get_system_prompt()

        assert "data analysis" in prompt
        for domain in ("pandas", "Visualization", "Cleaning"):
            assert domain in prompt, f"system prompt missing {domain}"

    def test_system_prompt_has_privacy_guidance(self):
        prompt = DataAnalysisAssistant.get_system_prompt()

        assert "PII" in prompt
        assert "Privacy and Ethics" in prompt


class TestAssistantExtensions:
    def test_capability_provider_is_dataanalysis_provider(self):
        provider = DataAnalysisAssistant.get_capability_provider()

        assert type(provider).__name__ == "DataAnalysisCapabilityProvider"
