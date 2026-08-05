# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Workflow tests for research vertical repair loops."""

from victor.workflows.definition import WorkflowDefinition


class TestResearchWorkflowProvider:
    """Tests for ResearchWorkflowProvider."""

    def test_has_deep_research_workflow(self):
        from victor_research.workflows import ResearchWorkflowProvider

        provider = ResearchWorkflowProvider()
        workflow = provider.get_workflow("deep_research")

        assert workflow is not None
        assert isinstance(workflow, WorkflowDefinition)


class TestDeepResearchWorkflow:
    """Tests for deep research repair topology."""

    def test_deep_research_has_gap_repair_router(self):
        from victor_research.workflows import ResearchWorkflowProvider

        provider = ResearchWorkflowProvider()
        workflow = provider.get_workflow("deep_research")
        assert workflow is not None

        next_nodes = workflow.get_next_nodes("gap_analysis")
        assert [node.id for node in next_nodes] == ["repair_research_gaps"]

        repair_router = workflow.nodes["repair_research_gaps"]
        assert repair_router.branches["search_more"] == "targeted_followup_search_plan"
        assert repair_router.branches["proceed"] == "synthesize"
        assert repair_router.branches["hitl"] == "request_more_research"

    def test_deep_research_has_targeted_followup_search_plan(self):
        from victor_research.workflows import ResearchWorkflowProvider

        provider = ResearchWorkflowProvider()
        workflow = provider.get_workflow("deep_research")
        assert workflow is not None

        assert "targeted_followup_search_plan" in workflow.nodes
        next_nodes = workflow.get_next_nodes("targeted_followup_search_plan")
        assert [node.id for node in next_nodes] == ["research_memory_reuse_router"]

    def test_deep_research_captures_memory_trace_after_validation(self):
        from victor_research.workflows import ResearchWorkflowProvider

        provider = ResearchWorkflowProvider()
        workflow = provider.get_workflow("deep_research")
        assert workflow is not None

        next_nodes = workflow.get_next_nodes("validate_sources")
        assert [node.id for node in next_nodes] == ["capture_research_memory_trace"]

        next_nodes = workflow.get_next_nodes("capture_research_memory_trace")
        assert [node.id for node in next_nodes] == ["check_coverage"]

    def test_deep_research_has_memory_reuse_router(self):
        from victor_research.workflows import ResearchWorkflowProvider

        provider = ResearchWorkflowProvider()
        workflow = provider.get_workflow("deep_research")
        assert workflow is not None

        assert "research_memory_reuse_router" in workflow.nodes
        router = workflow.nodes["research_memory_reuse_router"]
        assert router.branches["reuse_memory"] == "derive_research_memory_hints"
        assert router.branches["refresh"] == "increment_search_iterations"
        assert router.branches["hitl"] == "request_more_research"

        next_nodes = workflow.get_next_nodes("derive_research_memory_hints")
        assert [node.id for node in next_nodes] == ["increment_search_iterations"]


class TestResearchEscapeHatches:
    """Tests for research workflow escape hatches."""

    def test_research_gap_repair_decision_prefers_search_more(self):
        from victor_research.escape_hatches import research_gap_repair_decision

        ctx = {
            "coverage_score": 0.45,
            "gaps": ["pricing", "benchmarks", "integration"],
            "validated_sources": [
                {"url": "a", "credibility": 0.82},
                {"url": "b", "credibility": 0.79},
            ],
            "search_iterations": 1,
            "max_iterations": 3,
        }

        assert research_gap_repair_decision(ctx) == "search_more"

    def test_research_gap_repair_decision_routes_to_hitl_after_budget(self):
        from victor_research.escape_hatches import research_gap_repair_decision

        ctx = {
            "coverage_score": 0.4,
            "gaps": ["pricing", "benchmarks", "integration"],
            "validated_sources": [{"url": "a", "credibility": 0.4}],
            "search_iterations": 3,
            "max_iterations": 3,
        }

        assert research_gap_repair_decision(ctx) == "hitl"

    def test_research_memory_reuse_decision_prefers_reuse_memory(self):
        from victor_research.escape_hatches import research_memory_reuse_decision

        ctx = {
            "gaps": ["pricing", "benchmarks"],
            "validated_sources": [
                {
                    "url": "https://example.com/a",
                    "credibility": 0.86,
                    "source_type": "web",
                },
                {
                    "url": "https://example.com/b",
                    "credibility": 0.82,
                    "source_type": "academic",
                },
            ],
            "research_memory_trace": [
                {
                    "gaps": ["pricing", "benchmarks", "integration"],
                    "high_credibility_urls": ["https://example.com/a"],
                    "source_types": ["web", "academic"],
                }
            ],
            "search_iterations": 1,
            "max_iterations": 3,
        }

        assert research_memory_reuse_decision(ctx) == "reuse_memory"

    def test_research_memory_reuse_decision_routes_to_hitl_when_budget_exhausted(self):
        from victor_research.escape_hatches import research_memory_reuse_decision

        ctx = {
            "gaps": ["pricing", "benchmarks"],
            "validated_sources": [
                {
                    "url": "https://example.com/a",
                    "credibility": 0.86,
                    "source_type": "web",
                },
                {
                    "url": "https://example.com/b",
                    "credibility": 0.82,
                    "source_type": "academic",
                },
            ],
            "research_memory_trace": [
                {
                    "gaps": ["pricing", "benchmarks"],
                    "high_credibility_urls": ["https://example.com/a"],
                    "source_types": ["web", "academic"],
                }
            ],
            "search_iterations": 3,
            "max_iterations": 3,
        }

        assert research_memory_reuse_decision(ctx) == "hitl"
