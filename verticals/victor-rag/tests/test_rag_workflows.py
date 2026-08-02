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

"""Tests for RAG workflow provider."""

from victor.core.verticals.protocols import WorkflowProviderProtocol
from victor_contracts.workflow_runtime import WorkflowDefinition


class TestRAGWorkflowProvider:
    """Tests for RAGWorkflowProvider."""

    def test_implements_protocol(self):
        """RAGWorkflowProvider should implement WorkflowProviderProtocol."""
        from victor_rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        assert isinstance(provider, WorkflowProviderProtocol)

    def test_get_workflows_returns_dict(self):
        """get_workflows should return a dict of workflow definitions."""
        from victor_rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflows = provider.get_workflows()

        assert isinstance(workflows, dict)
        assert len(workflows) >= 3  # ingest, query, maintenance

    def test_has_ingest_workflow(self):
        """Provider should have an ingest workflow."""
        from victor_rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflows = provider.get_workflows()

        assert (
            "ingest" in workflows
            or "document_ingest" in workflows
            or "ingest_workflow" in workflows
        )
        # Get the ingest workflow (check different possible names)
        ingest_key = next(k for k in workflows if "ingest" in k.lower())
        workflow = workflows[ingest_key]
        assert isinstance(workflow, WorkflowDefinition)

    def test_has_query_workflow(self):
        """Provider should have a query workflow."""
        from victor_rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflows = provider.get_workflows()

        # Check for query workflow
        query_key = next((k for k in workflows if "query" in k.lower()), None)
        assert query_key is not None, "Should have a query workflow"
        workflow = workflows[query_key]
        assert isinstance(workflow, WorkflowDefinition)

    def test_has_maintenance_workflow(self):
        """Provider should have a maintenance workflow."""
        from victor_rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflows = provider.get_workflows()

        # Check for maintenance workflow
        maint_key = next(
            (k for k in workflows if "maintenance" in k.lower() or "optimize" in k.lower()),
            None,
        )
        assert maint_key is not None, "Should have a maintenance workflow"
        workflow = workflows[maint_key]
        assert isinstance(workflow, WorkflowDefinition)

    def test_get_workflow_by_name(self):
        """get_workflow should return a specific workflow by name."""
        from victor_rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflows = provider.get_workflows()
        first_name = next(iter(workflows.keys()))

        workflow = provider.get_workflow(first_name)
        assert workflow is not None
        assert isinstance(workflow, WorkflowDefinition)

    def test_get_workflow_names(self):
        """get_workflow_names should return list of workflow names."""
        from victor_rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        names = provider.get_workflow_names()

        assert isinstance(names, list)
        assert len(names) >= 3

    def test_get_auto_workflows(self):
        """get_auto_workflows should return pattern-workflow tuples."""
        from victor_rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        auto_workflows = provider.get_auto_workflows()

        assert isinstance(auto_workflows, list)
        # Should have at least some auto-trigger patterns
        assert len(auto_workflows) >= 1
        # Each entry should be (pattern, workflow_name) tuple
        for pattern, workflow_name in auto_workflows:
            assert isinstance(pattern, str)
            assert isinstance(workflow_name, str)

    def test_workflow_validation(self):
        """All workflows should pass validation."""
        from victor_rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflows = provider.get_workflows()

        for name, workflow in workflows.items():
            errors = workflow.validate()
            assert not errors, f"Workflow '{name}' has validation errors: {errors}"

    def test_workflow_has_agents(self):
        """Each workflow should have at least one agent node (except utility workflows)."""
        from victor_rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflows = provider.get_workflows()

        # Some workflows are utility/compute-only (no LLM agents)
        utility_workflows = {"incremental_update", "maintenance"}

        for name, workflow in workflows.items():
            if name in utility_workflows:
                # Utility workflows may not have agent nodes
                continue
            agent_count = workflow.get_agent_count()
            assert agent_count >= 1, f"Workflow '{name}' should have at least one agent"

    def test_repr(self):
        """Provider should have a useful repr."""
        from victor_rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        repr_str = repr(provider)

        assert "RAGWorkflowProvider" in repr_str


class TestIngestWorkflow:
    """Tests for the document ingest workflow."""

    def test_ingest_workflow_structure(self):
        """Ingest workflow should have proper structure."""
        from victor_rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflow = provider.get_workflow("document_ingest")
        assert workflow is not None
        assert isinstance(workflow, WorkflowDefinition)
        assert "ingest" in workflow.name.lower()

    def test_ingest_has_parse_step(self):
        """Ingest workflow should have a parsing/processing step."""
        from victor_rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflow = provider.get_workflow("document_ingest")
        assert workflow is not None
        nodes = workflow.nodes

        # Should have a node related to parsing/processing
        has_parse = any(
            "parse" in node_id.lower()
            or "process" in node_id.lower()
            or "extract" in node_id.lower()
            or "discover" in node_id.lower()
            for node_id in nodes
        )
        assert has_parse, "Ingest workflow should have a parse/process step"


class TestQueryWorkflow:
    """Tests for the query workflow."""

    def test_query_workflow_structure(self):
        """Query workflow should have proper structure."""
        from victor_rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflow = provider.get_workflow("rag_query")
        assert workflow is not None
        assert isinstance(workflow, WorkflowDefinition)
        assert "query" in workflow.name.lower()

    def test_query_has_search_step(self):
        """Query workflow should have a search/retrieve step."""
        from victor_rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflow = provider.get_workflow("rag_query")
        assert workflow is not None
        nodes = workflow.nodes

        # Should have a node related to search/retrieve
        has_search = any(
            "search" in node_id.lower() or "retrieve" in node_id.lower() for node_id in nodes
        )
        assert has_search, "Query workflow should have a search/retrieve step"

    def test_query_has_synthesis_step(self):
        """Query workflow should have a synthesis/answer step."""
        from victor_rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflow = provider.get_workflow("rag_query")
        assert workflow is not None
        nodes = workflow.nodes

        # Should have a node related to synthesis
        has_synthesis = any(
            "synth" in node_id.lower()
            or "answer" in node_id.lower()
            or "generate" in node_id.lower()
            for node_id in nodes
        )
        assert has_synthesis, "Query workflow should have a synthesis step"

    def test_query_has_retrieval_utility_scoring_step(self):
        """Query workflow should score retrieval utility before coverage checks."""
        from victor_rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflow = provider.get_workflow("rag_query")
        assert workflow is not None

        assert "score_retrieval_utility" in workflow.nodes
        next_nodes = workflow.get_next_nodes("rerank")
        assert [node.id for node in next_nodes] == ["score_retrieval_utility"]

    def test_query_uses_named_retrieval_utility_transform(self):
        """Utility scoring should resolve to the registered escape hatch."""
        from victor_rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflow = provider.get_workflow("rag_query")
        assert workflow is not None

        utility_node = workflow.nodes["score_retrieval_utility"]
        assert utility_node.transform.__name__ == "score_retrieval_utility"

    def test_query_has_retrieval_repair_branch(self):
        """Low-support retrieval should go through diagnosis and repair."""
        from victor_rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflow = provider.get_workflow("rag_query")
        assert workflow is not None

        coverage_decision = workflow.nodes["coverage_decision"]
        assert coverage_decision.branches["false"] == "diagnose_retrieval_gap"

        repair_decision = workflow.nodes["check_retrieval_repair"]
        assert repair_decision.branches["repair"] == "increment_retrieval_repair_attempt"
        assert repair_decision.branches["revise"] == "revise_answer"
        assert repair_decision.branches["clarify"] == "request_clarification"

        assert "repair_retrieval" in workflow.nodes
        next_nodes = workflow.get_next_nodes("increment_retrieval_repair_attempt")
        assert [node.id for node in next_nodes] == ["repair_retrieval"]

    def test_query_diagnosis_node_requests_explicit_gap_types(self):
        """Diagnosis prompt should request explicit retrieval gap classes."""
        from victor_rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflow = provider.get_workflow("rag_query")
        assert workflow is not None

        diagnosis = workflow.nodes["diagnose_retrieval_gap"]
        assert "gap_type" in diagnosis.goal
        assert "missing_support" in diagnosis.goal
        assert "weak_authority" in diagnosis.goal
        assert "contradictory_evidence" in diagnosis.goal
        assert "query_ambiguity" in diagnosis.goal


class TestRAGEscapeHatches:
    """Tests for RAG workflow escape hatches."""

    def test_score_retrieval_utility_reorders_for_authority_and_diversity(self):
        """Authority and source diversity should improve result ordering."""
        from victor_rag.escape_hatches import score_retrieval_utility

        ctx = {
            "ranked_results": [
                {
                    "id": "weak-snippet",
                    "score": 0.96,
                    "text": "A short unattributed answer fragment.",
                },
                {
                    "id": "official-doc",
                    "score": 0.92,
                    "source_title": "Product Documentation",
                    "source_url": "https://docs.example.com/guide",
                    "text": "Official product guidance with concrete steps.",
                },
                {
                    "id": "duplicate-doc",
                    "score": 0.91,
                    "source_title": "Product Documentation",
                    "source_url": "https://docs.example.com/reference",
                    "text": "Reference page from the same documentation site.",
                },
                {
                    "id": "independent-analysis",
                    "score": 0.90,
                    "source_title": "Independent Analysis",
                    "source_url": "https://analysis.example.org/report",
                    "text": "Independent write-up that corroborates the docs.",
                },
            ]
        }

        result = score_retrieval_utility(ctx)
        ranked_ids = [chunk["id"] for chunk in result["ranked_results"]]

        assert ranked_ids[0] == "official-doc"
        assert ranked_ids.index("independent-analysis") < ranked_ids.index("duplicate-doc")
        assert result["retrieval_utility"]["authority_hits"] == 3
        assert result["retrieval_utility"]["source_diversity"] == 2

    def test_score_retrieval_utility_reports_sparse_low_utility_context(self):
        """Sparse unattributed context should stay below the repair threshold."""
        from victor_rag.escape_hatches import score_retrieval_utility

        result = score_retrieval_utility(
            {
                "ranked_results": [
                    {
                        "id": "weak-snippet",
                        "score": 0.42,
                        "text": "Unattributed summary with limited support.",
                    }
                ]
            }
        )

        utility = result["retrieval_utility"]
        assert utility["candidate_count"] == 1
        assert utility["authority_hits"] == 0
        assert utility["utility_score"] < 0.55

    def test_classify_retrieval_gap_prefers_explicit_gap_type(self):
        """Structured diagnosis should override heuristic fallback."""
        from victor_rag.escape_hatches import classify_retrieval_gap

        ctx = {
            "retrieval_gap_diagnosis": {
                "gap_type": "weak_authority",
                "missing_information": ["authoritative sources"],
            },
            "coverage_assessment": {"has_answer": False, "confidence": "low"},
        }

        assert classify_retrieval_gap(ctx) == "weak_authority"

    def test_classify_retrieval_gap_detects_contradictory_evidence(self):
        """Verification issues about conflicts should classify as contradiction."""
        from victor_rag.escape_hatches import classify_retrieval_gap

        ctx = {
            "coverage_assessment": {"has_answer": True, "confidence": "medium"},
            "verification": {
                "passed": False,
                "issues": ["Sources conflict on the timeline"],
            },
            "retrieval_utility": {"utility_score": 0.8, "authority_hits": 3},
        }

        assert classify_retrieval_gap(ctx) == "contradictory_evidence"

    def test_retrieval_repair_decision_prefers_repair_for_missing_support(self):
        """Coverage gaps with remaining repair budget should retry retrieval."""
        from victor_rag.escape_hatches import retrieval_repair_decision

        ctx = {
            "coverage_assessment": {"has_answer": False, "confidence": "low"},
            "verification": {
                "passed": False,
                "issues": ["Missing supporting evidence"],
            },
            "repair_attempt_count": 0,
            "max_repair_attempts": 2,
        }

        assert retrieval_repair_decision(ctx) == "repair"

    def test_retrieval_repair_decision_clarifies_for_query_ambiguity(self):
        """Ambiguous or underspecified asks should prompt clarification."""
        from victor_rag.escape_hatches import retrieval_repair_decision

        ctx = {
            "retrieval_gap_diagnosis": {"gap_type": "query_ambiguity"},
            "coverage_assessment": {"has_answer": False, "confidence": "low"},
            "verification": {"passed": False, "issues": ["Question is ambiguous"]},
            "repair_attempt_count": 0,
            "max_repair_attempts": 2,
        }

        assert retrieval_repair_decision(ctx) == "clarify"

    def test_retrieval_repair_decision_revises_when_evidence_exists(self):
        """If evidence is adequate but the draft is weak, revise instead of repairing retrieval."""
        from victor_rag.escape_hatches import retrieval_repair_decision

        ctx = {
            "coverage_assessment": {"has_answer": True, "confidence": "high"},
            "verification": {
                "passed": False,
                "issues": ["Answer overstates one claim"],
            },
            "retrieval_utility": {"utility_score": 0.82, "authority_hits": 3},
            "repair_attempt_count": 0,
            "max_repair_attempts": 2,
        }

        assert retrieval_repair_decision(ctx) == "revise"

    def test_retrieval_repair_decision_escalates_after_budget_exhausted(self):
        """Repair budget exhaustion should stop automatic retries."""
        from victor_rag.escape_hatches import retrieval_repair_decision

        ctx = {
            "coverage_assessment": {"has_answer": False, "confidence": "low"},
            "verification": {
                "passed": False,
                "issues": ["Need external clarification"],
            },
            "repair_attempt_count": 2,
            "max_repair_attempts": 2,
        }

        assert retrieval_repair_decision(ctx) == "clarify"


class TestMaintenanceWorkflow:
    """Tests for the maintenance workflow."""

    def test_maintenance_workflow_structure(self):
        """Maintenance workflow should have proper structure."""
        from victor_rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflow = provider.get_workflow("maintenance")
        assert workflow is not None
        assert isinstance(workflow, WorkflowDefinition)
        assert "maintenance" in workflow.name.lower() or "optimize" in workflow.name.lower()
