# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Contract tests for the ResearchAssistant vertical definition.

Covers the register(context) payload surface: tools, stages, system prompt,
and capability config wiring.
"""

from victor_contracts import StageDefinition, ToolNames, VerticalBase

from victor_research.assistant import ResearchAssistant


class TestAssistantIdentity:
    def test_name_and_description(self):
        assert ResearchAssistant.get_name() == "research"
        assert "fact-checking" in ResearchAssistant.get_description()
        assert issubclass(ResearchAssistant, VerticalBase)

    def test_version_metadata(self):
        assert ResearchAssistant.version == "1.0.0"
        assert ResearchAssistant.VERTICAL_API_VERSION == 1


class TestAssistantTools:
    def test_web_tools_are_core(self):
        tools = ResearchAssistant.get_tools()

        assert ToolNames.WEB_SEARCH in tools
        assert ToolNames.WEB_FETCH in tools

    def test_file_operations_capability_included(self):
        """Phase 3: file ops come from the framework FileOperationsCapability."""
        tools = ResearchAssistant.get_tools()

        for tool in (ToolNames.READ, ToolNames.WRITE, ToolNames.EDIT, ToolNames.GREP):
            assert tool in tools, f"file op {tool} missing"

    def test_no_duplicate_tools(self):
        tools = ResearchAssistant.get_tools()

        assert len(tools) == len(set(tools))


class TestAssistantStages:
    def test_stage_graph_is_closed(self):
        """Every next_stage must reference a defined stage."""
        stages = ResearchAssistant.get_stages()

        defined = set(stages)
        for name, stage in stages.items():
            assert isinstance(stage, StageDefinition)
            assert stage.name == name
            missing = stage.next_stages - defined
            assert not missing, f"{name} references undefined stages: {missing}"

    def test_research_lifecycle_stages_present(self):
        stages = ResearchAssistant.get_stages()

        for expected in (
            "INITIAL",
            "SEARCHING",
            "READING",
            "SYNTHESIZING",
            "WRITING",
            "VERIFICATION",
            "COMPLETION",
        ):
            assert expected in stages

    def test_completion_is_terminal(self):
        stages = ResearchAssistant.get_stages()

        assert stages["COMPLETION"].next_stages == set()

    def test_searching_stage_uses_web_tools(self):
        stage = ResearchAssistant.get_stages()["SEARCHING"]

        assert ToolNames.WEB_SEARCH in stage.tools
        assert ToolNames.WEB_FETCH in stage.tools

    def test_stage_tools_subset_of_vertical_tools(self):
        tools = set(ResearchAssistant.get_tools())

        for name, stage in ResearchAssistant.get_stages().items():
            extra = stage.tools - tools
            assert not extra, f"stage {name} uses tools outside get_tools(): {extra}"

    def test_every_non_terminal_stage_has_keywords(self):
        for name, stage in ResearchAssistant.get_stages().items():
            assert stage.keywords, f"stage {name} has no routing keywords"


class TestAssistantPrompt:
    def test_system_prompt_positions_web_research(self):
        prompt = ResearchAssistant.get_system_prompt()

        assert "WEB RESEARCH" in prompt
        assert "web_search" in prompt
        assert "web_fetch" in prompt

    def test_system_prompt_demands_citations(self):
        prompt = ResearchAssistant.get_system_prompt()

        assert "Attribution" in prompt
        assert "Never fabricate sources" in prompt


class TestCapabilityConfigs:
    def test_capability_configs_delegate_to_capabilities_module(self):
        from victor_research.capabilities import get_capability_configs

        assert ResearchAssistant.get_capability_configs() == get_capability_configs()
