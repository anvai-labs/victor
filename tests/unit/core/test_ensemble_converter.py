# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Ensemble to WorkflowDefinition converter."""

import pytest

from victor.agent.specs import (
    AgentSpec,
    AgentCapabilities,
    AgentConstraints,
    ModelPreference,
    Pipeline,
    Parallel,
    Hierarchical,
    EnsembleConverter,
    ensemble_to_workflow,
    workflow_to_ensemble,
)
from victor.agent.specs.ensemble import EnsembleType
from victor.workflows.definition import (
    AgentNode,
    ParallelNode,
    WorkflowDefinition,
    WorkflowNodeType,
)


@pytest.fixture
def researcher_agent():
    """Create researcher agent spec."""
    return AgentSpec(
        name="researcher",
        description="Research and gather information",
        capabilities=AgentCapabilities(
            tools={"web_search", "read_file", "code_search"},
        ),
        constraints=AgentConstraints(max_tool_calls=20),
        model_preference=ModelPreference.REASONING,
        system_prompt="You are a research specialist.",
    )


@pytest.fixture
def coder_agent():
    """Create coder agent spec."""
    return AgentSpec(
        name="coder",
        description="Write and modify code",
        capabilities=AgentCapabilities(
            tools={"edit_file", "write_file", "execute_bash"},
            can_execute_code=True,
        ),
        constraints=AgentConstraints(max_tool_calls=30),
        model_preference=ModelPreference.CODING,
        system_prompt="You are a coding specialist.",
    )


@pytest.fixture
def reviewer_agent():
    """Create reviewer agent spec."""
    return AgentSpec(
        name="reviewer",
        description="Review and validate work",
        capabilities=AgentCapabilities(
            tools={"read_file", "grep"},
        ),
        constraints=AgentConstraints(max_tool_calls=15),
        model_preference=ModelPreference.REASONING,
    )


class TestEnsembleConverter:
    """Tests for EnsembleConverter class."""

    def test_converter_initialization(self):
        """Test converter initialization."""
        converter = EnsembleConverter()
        assert converter.add_start_hitl is False
        assert converter.add_end_hitl is False
        assert converter.default_tool_budget == 15

    def test_converter_with_hitl_options(self):
        """Test converter with HITL options."""
        converter = EnsembleConverter(
            add_start_hitl=True,
            add_end_hitl=True,
            default_tool_budget=25,
        )
        assert converter.add_start_hitl is True
        assert converter.add_end_hitl is True
        assert converter.default_tool_budget == 25


class TestPipelineConversion:
    """Tests for converting Pipeline ensembles."""

    def test_pipeline_to_workflow(self, researcher_agent, coder_agent, reviewer_agent):
        """Test converting pipeline to workflow."""
        pipeline = Pipeline(
            agents=[researcher_agent, coder_agent, reviewer_agent],
            name="dev_pipeline",
            continue_on_error=False,
        )

        converter = EnsembleConverter()
        workflow = converter.to_workflow(pipeline)

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.name == "dev_pipeline"
        assert "ensemble_type" in workflow.metadata
        assert workflow.metadata["ensemble_type"] == EnsembleType.PIPELINE.value

    def test_pipeline_agents_converted(self, researcher_agent, coder_agent):
        """Test all pipeline agents are converted to nodes."""
        pipeline = Pipeline([researcher_agent, coder_agent])

        workflow = ensemble_to_workflow(pipeline)

        # Should have agent nodes for each agent
        agent_nodes = [n for n in workflow.nodes.values() if isinstance(n, AgentNode)]
        assert len(agent_nodes) >= 2

    def test_pipeline_with_hitl(self, researcher_agent, coder_agent):
        """Test pipeline conversion with HITL nodes."""
        pipeline = Pipeline([researcher_agent, coder_agent])

        converter = EnsembleConverter(
            add_start_hitl=True,
            add_end_hitl=True,
        )
        workflow = converter.to_workflow(pipeline)

        # Should have more nodes with HITL
        assert len(workflow.nodes) >= 4  # 2 agents + 2 HITL

    def test_pipeline_tool_budget_preserved(self, researcher_agent):
        """Test agent tool budgets are preserved."""
        pipeline = Pipeline([researcher_agent])

        workflow = ensemble_to_workflow(pipeline)

        # Find the agent node
        agent_nodes = [n for n in workflow.nodes.values() if isinstance(n, AgentNode)]
        assert len(agent_nodes) >= 1

        # Check tool budget
        node = agent_nodes[0]
        assert node.tool_budget == researcher_agent.constraints.max_tool_calls


class TestParallelConversion:
    """Tests for converting Parallel ensembles."""

    def test_parallel_to_workflow(self, researcher_agent, coder_agent, reviewer_agent):
        """Test converting parallel ensemble to workflow."""
        parallel = Parallel(
            agents=[researcher_agent, coder_agent, reviewer_agent],
            name="analysis_parallel",
            require_all=True,
        )

        converter = EnsembleConverter()
        workflow = converter.to_workflow(parallel)

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.name == "analysis_parallel"
        assert workflow.metadata["ensemble_type"] == EnsembleType.PARALLEL.value

    def test_parallel_has_parallel_node(self, researcher_agent, reviewer_agent):
        """Test parallel ensemble creates ParallelNode."""
        parallel = Parallel([researcher_agent, reviewer_agent])

        workflow = ensemble_to_workflow(parallel)

        # Should have a parallel execution node
        parallel_nodes = [n for n in workflow.nodes.values() if isinstance(n, ParallelNode)]
        assert len(parallel_nodes) == 1

        # Parallel node should reference agent nodes
        pnode = parallel_nodes[0]
        assert len(pnode.parallel_nodes) == 2

    def test_parallel_require_all(self, researcher_agent, reviewer_agent):
        """Test require_all setting is preserved."""
        parallel = Parallel(
            [researcher_agent, reviewer_agent],
            require_all=False,
        )

        workflow = ensemble_to_workflow(parallel)

        assert workflow.metadata["require_all"] is False


class TestHierarchicalConversion:
    """Tests for converting Hierarchical ensembles."""

    def test_hierarchical_to_workflow(self, researcher_agent, coder_agent, reviewer_agent):
        """Test converting hierarchical ensemble to workflow."""
        hierarchical = Hierarchical(
            manager=researcher_agent,
            workers=[coder_agent, reviewer_agent],
            name="managed_team",
            max_delegations=5,
        )

        converter = EnsembleConverter()
        workflow = converter.to_workflow(hierarchical)

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.name == "managed_team"
        assert workflow.metadata["ensemble_type"] == EnsembleType.HIERARCHICAL.value
        assert workflow.metadata["max_delegations"] == 5

    def test_hierarchical_has_manager_node(self, researcher_agent, coder_agent):
        """Test hierarchical has manager agent node."""
        hierarchical = Hierarchical(
            manager=researcher_agent,
            workers=[coder_agent],
        )

        workflow = ensemble_to_workflow(hierarchical)

        # Should have manager node
        assert "manager" in workflow.nodes
        manager_node = workflow.nodes["manager"]
        assert isinstance(manager_node, AgentNode)

    def test_hierarchical_workers_parallel(self, researcher_agent, coder_agent, reviewer_agent):
        """Test hierarchical workers execute in parallel."""
        hierarchical = Hierarchical(
            manager=researcher_agent,
            workers=[coder_agent, reviewer_agent],
        )

        workflow = ensemble_to_workflow(hierarchical)

        # Should have parallel worker execution node
        parallel_nodes = [n for n in workflow.nodes.values() if isinstance(n, ParallelNode)]
        assert len(parallel_nodes) == 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_ensemble_to_workflow_function(self, researcher_agent, coder_agent):
        """Test ensemble_to_workflow convenience function."""
        pipeline = Pipeline([researcher_agent, coder_agent])

        workflow = ensemble_to_workflow(pipeline)

        assert isinstance(workflow, WorkflowDefinition)

    def test_ensemble_to_workflow_with_hitl(self, researcher_agent):
        """Test ensemble_to_workflow with HITL option."""
        pipeline = Pipeline([researcher_agent])

        workflow = ensemble_to_workflow(pipeline, add_hitl=True)

        # Should have HITL nodes
        assert len(workflow.nodes) >= 2  # 1 agent + HITL nodes


class TestWorkflowToEnsemble:
    """Tests for converting workflows back to ensembles."""

    def test_pipeline_roundtrip(self, researcher_agent, coder_agent):
        """Test pipeline -> workflow -> pipeline roundtrip."""
        original = Pipeline(
            [researcher_agent, coder_agent],
            name="test_pipeline",
        )

        workflow = ensemble_to_workflow(original)
        recovered = workflow_to_ensemble(workflow)

        assert isinstance(recovered, Pipeline)
        assert recovered.name == "test_pipeline"

    def test_parallel_roundtrip(self, researcher_agent, reviewer_agent):
        """Test parallel -> workflow -> parallel roundtrip."""
        original = Parallel(
            [researcher_agent, reviewer_agent],
            name="test_parallel",
            require_all=False,
        )

        workflow = ensemble_to_workflow(original)
        recovered = workflow_to_ensemble(workflow)

        assert isinstance(recovered, Parallel)
        assert recovered.name == "test_parallel"
        assert recovered.require_all is False

    def test_hierarchical_roundtrip(self, researcher_agent, coder_agent):
        """Test hierarchical -> workflow -> hierarchical roundtrip."""
        original = Hierarchical(
            manager=researcher_agent,
            workers=[coder_agent],
            name="test_hierarchical",
            max_delegations=7,
        )

        workflow = ensemble_to_workflow(original)
        recovered = workflow_to_ensemble(workflow)

        assert isinstance(recovered, Hierarchical)
        assert recovered.name == "test_hierarchical"
        assert recovered.max_delegations == 7

    def test_workflow_without_metadata_fails(self):
        """Test workflow without ensemble metadata cannot be converted."""
        from victor.workflows.definition import WorkflowBuilder

        workflow = (
            WorkflowBuilder("plain_workflow")
            .add_agent("agent1", "executor", "Do something")
            .build()
        )

        with pytest.raises(ValueError, match="ensemble_type"):
            workflow_to_ensemble(workflow)


class TestWorkflowValidation:
    """Tests for workflow validation after conversion."""

    def test_converted_workflow_validates(self, researcher_agent, coder_agent):
        """Test converted workflows pass validation."""
        pipeline = Pipeline([researcher_agent, coder_agent])

        workflow = ensemble_to_workflow(pipeline)

        errors = workflow.validate()
        assert len(errors) == 0, f"Validation errors: {errors}"

    def test_converted_workflow_has_start_node(self, researcher_agent):
        """Test converted workflow has start node."""
        pipeline = Pipeline([researcher_agent])

        workflow = ensemble_to_workflow(pipeline)

        assert workflow.start_node is not None
        assert workflow.start_node in workflow.nodes


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_pipeline(self):
        """Test empty pipeline conversion raises validation error."""
        pipeline = Pipeline(agents=[], name="empty")

        # Empty pipeline should raise validation error
        with pytest.raises(ValueError, match="at least one node"):
            ensemble_to_workflow(pipeline)

    def test_single_agent_pipeline(self, researcher_agent):
        """Test single agent pipeline."""
        pipeline = Pipeline([researcher_agent])

        workflow = ensemble_to_workflow(pipeline)

        agent_nodes = [n for n in workflow.nodes.values() if isinstance(n, AgentNode)]
        assert len(agent_nodes) == 1

    def test_agent_with_no_tools(self):
        """Test agent with no tools specified."""
        agent = AgentSpec(
            name="basic",
            description="Basic agent",
        )
        pipeline = Pipeline([agent])

        workflow = ensemble_to_workflow(pipeline)

        assert isinstance(workflow, WorkflowDefinition)

    def test_metadata_preserved(self, researcher_agent):
        """Test metadata is preserved in conversion."""
        pipeline = Pipeline(
            [researcher_agent],
            name="test",
            continue_on_error=True,
        )

        workflow = ensemble_to_workflow(pipeline)

        assert workflow.metadata["source"] == "ensemble_converter"
        assert workflow.metadata["continue_on_error"] is True
