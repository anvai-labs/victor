# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Contract tests for the DevOpsAssistant vertical definition.

Covers the register(context) payload surface: tools, stages, system prompt,
middleware, and capability provider wiring.
"""

from victor_contracts import StageDefinition, ToolNames, VerticalBase

from victor_devops.assistant import DevOpsAssistant


class TestAssistantIdentity:
    def test_name_and_description(self):
        assert DevOpsAssistant.get_name() == "devops"
        assert "Infrastructure" in DevOpsAssistant.get_description()
        assert issubclass(DevOpsAssistant, VerticalBase)

    def test_version_metadata(self):
        assert DevOpsAssistant.version == "1.0.0"
        assert DevOpsAssistant.VERTICAL_API_VERSION == 1


class TestAssistantTools:
    def test_tools_are_canonical_names(self):
        tools = DevOpsAssistant.get_tools()

        assert len(tools) == len(set(tools)), "duplicate tool names"
        assert ToolNames.READ in tools
        assert ToolNames.WRITE in tools
        assert ToolNames.SHELL in tools
        assert ToolNames.DOCKER in tools
        assert ToolNames.GIT in tools

    def test_devops_needs_web_tools_for_docs(self):
        tools = DevOpsAssistant.get_tools()

        assert ToolNames.WEB_SEARCH in tools
        assert ToolNames.WEB_FETCH in tools


class TestAssistantStages:
    def test_stage_graph_is_closed(self):
        """Every next_stage must reference a defined stage."""
        stages = DevOpsAssistant.get_stages()

        defined = set(stages)
        for name, stage in stages.items():
            assert isinstance(stage, StageDefinition)
            assert stage.name == name
            missing = stage.next_stages - defined
            assert not missing, f"{name} references undefined stages: {missing}"

    def test_stage_tools_subset_of_vertical_tools(self):
        tools = set(DevOpsAssistant.get_tools())

        for name, stage in DevOpsAssistant.get_stages().items():
            assert stage.tools, f"stage {name} has no tools"
            extra = stage.tools - tools
            assert not extra, f"stage {name} uses tools outside get_tools(): {extra}"

    def test_lifecycle_stages_present(self):
        stages = DevOpsAssistant.get_stages()

        for expected in (
            "INITIAL",
            "ASSESSMENT",
            "PLANNING",
            "IMPLEMENTATION",
            "VALIDATION",
            "DEPLOYMENT",
            "MONITORING",
            "COMPLETION",
        ):
            assert expected in stages

    def test_completion_is_terminal(self):
        stages = DevOpsAssistant.get_stages()

        assert stages["COMPLETION"].next_stages == set()

    def test_every_stage_has_keywords(self):
        for name, stage in DevOpsAssistant.get_stages().items():
            assert stage.keywords, f"stage {name} has no routing keywords"


class TestAssistantPrompt:
    def test_system_prompt_covers_core_domains(self):
        prompt = DevOpsAssistant.get_system_prompt()

        assert "DevOps" in prompt
        for domain in ("Docker", "Kubernetes", "Terraform", "CI/CD"):
            assert domain in prompt, f"system prompt missing {domain}"

    def test_system_prompt_has_security_guidance(self):
        prompt = DevOpsAssistant.get_system_prompt()

        assert "Never commit secrets" in prompt
        assert "Least privilege" in prompt


class TestAssistantExtensions:
    def test_middleware_chain_composition(self):
        middleware = DevOpsAssistant.get_middleware()

        assert len(middleware) == 3
        type_names = [type(m).__name__ for m in middleware]
        assert any("GitSafety" in n for n in type_names)
        assert any("SecretMasking" in n for n in type_names)
        assert any("Logging" in n for n in type_names)

    def test_capability_provider_is_devops_provider(self):
        provider = DevOpsAssistant.get_capability_provider()

        assert type(provider).__name__ == "DevOpsCapabilityProvider"
