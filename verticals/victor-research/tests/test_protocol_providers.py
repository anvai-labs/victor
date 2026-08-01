# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for victor.extension.protocols implementations in victor-research."""

from victor_research.protocols import (
    ResearchPromptProvider,
    ResearchSafetyProvider,
    ResearchToolProvider,
    ResearchToolSelectionStrategy,
    ResearchWorkflowProvider,
)


class TestToolProvider:
    def test_tool_list_contract(self):
        tools = ResearchToolProvider().get_tools()

        assert len(tools) == len(set(tools)), "duplicate tools"
        for expected in ("read", "web_search", "web_fetch", "arxiv_search", "fact_check"):
            assert expected in tools

    def test_covers_all_research_domains(self):
        tools = set(ResearchToolProvider().get_tools())

        assert tools & {"web_search", "web_fetch", "web_scrape", "web_crawl"}, "no web tools"
        assert tools & {"arxiv_search", "scholar_search", "pubmed_search"}, "no academic tools"
        assert tools & {
            "citation_extract",
            "citation_format",
            "bibliography_generate",
        }, "no citation tools"
        assert tools & {"pdf_extract", "document_parse"}, "no document tools"


class TestToolSelectionStrategy:
    def test_stage_specific_tools(self):
        strategy = ResearchToolSelectionStrategy()

        assert "arxiv_search" in strategy.get_tools_for_stage("discover", "research")
        assert "pdf_extract" in strategy.get_tools_for_stage("collect", "research")
        assert "fact_check" in strategy.get_tools_for_stage("verify", "fact_check")

    def test_unknown_stage_falls_back_to_search_read_write(self):
        strategy = ResearchToolSelectionStrategy()

        assert strategy.get_tools_for_stage("unknown_stage", "any") == [
            "web_search",
            "read",
            "write",
        ]


class TestSafetyProvider:
    def test_dangerous_web_patterns_registered(self):
        provider = ResearchSafetyProvider()

        patterns = [p["pattern"] for p in provider.get_bash_patterns()]
        assert "web_scrape --infinite" in patterns
        assert "web_crawl --depth 100" in patterns

    def test_tool_restrictions_limit_scraping(self):
        restrictions = ResearchSafetyProvider().get_tool_restrictions()

        assert "--infinite" in restrictions["web_scrape"]
        assert "web_crawl" in restrictions

    def test_no_file_patterns(self):
        assert ResearchSafetyProvider().get_file_patterns() == []


class TestPromptProvider:
    def test_system_prompt_sections(self):
        sections = ResearchPromptProvider().get_system_prompt_sections()

        assert {"role", "expertise", "methodology", "citations", "verification"} <= set(sections)
        assert "Research" in sections["role"]

    def test_task_type_hints_have_budgets(self):
        hints = ResearchPromptProvider().get_task_type_hints()

        assert "fact_check" in hints
        for task_type, hint in hints.items():
            assert hint["hint"], f"{task_type} hint empty"
            assert hint["tool_budget"] > 0


class TestWorkflowProviderProtocol:
    def test_workflow_lookup_roundtrip(self):
        provider = ResearchWorkflowProvider()

        names = provider.list_workflows()
        assert "literature_review" in names
        workflow = provider.get_workflow("literature_review")
        assert workflow["stages"] == ["discover", "collect", "analyze", "synthesize"]

    def test_fact_check_ends_with_verify_stage(self):
        workflow = ResearchWorkflowProvider().get_workflow("fact_check")

        assert workflow["stages"][-1] == "verify"

    def test_unknown_workflow_returns_none(self):
        assert ResearchWorkflowProvider().get_workflow("nope") is None
