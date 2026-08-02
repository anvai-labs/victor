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

"""Ensemble to WorkflowDefinition converter.

Converts agent ensembles (Pipeline, Parallel, Hierarchical) to
Victor's WorkflowDefinition format for execution by the workflow engine.

This bridge enables:
- Declarative agent ensembles to use Victor's workflow execution
- Integration with HITL workflow nodes
- Checkpoint/state persistence
- Visualization and debugging
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.agent.specs.ensemble import (
    Ensemble,
    EnsembleType,
    Pipeline,
    Parallel,
    Hierarchical,
)
from victor.agent.specs.models import AgentSpec, ModelPreference
from victor.workflows.definition import (
    AgentNode,
    ParallelNode,
    WorkflowBuilder,
    WorkflowDefinition,
)

if TYPE_CHECKING:
    from victor.workflows.hitl import HITLNode

logger = logging.getLogger(__name__)


class EnsembleConverter:
    """Converts ensembles to workflow definitions.

    Provides a bridge between the declarative Agent Ensemble DSL
    and Victor's workflow execution engine.

    Example:
        # Create an ensemble
        pipeline = Pipeline([researcher, coder, reviewer])

        # Convert to workflow
        converter = EnsembleConverter()
        workflow = converter.to_workflow(pipeline)

        # Execute using workflow engine
        executor = WorkflowExecutor(workflow)
        result = await executor.execute({"task": "Fix bug #123"})
    """

    def __init__(
        self,
        add_start_hitl: bool = False,
        add_end_hitl: bool = False,
        default_tool_budget: int = 15,
    ):
        """Initialize converter.

        Args:
            add_start_hitl: Add approval node before workflow starts
            add_end_hitl: Add review node after workflow completes
            default_tool_budget: Default tool budget for agents
        """
        self.add_start_hitl = add_start_hitl
        self.add_end_hitl = add_end_hitl
        self.default_tool_budget = default_tool_budget

    def to_workflow(
        self,
        ensemble: Ensemble,
        add_checkpoints: bool = False,
    ) -> WorkflowDefinition:
        """Convert ensemble to workflow definition.

        Args:
            ensemble: Ensemble to convert
            add_checkpoints: Add checkpoint nodes between agents

        Returns:
            WorkflowDefinition ready for execution
        """
        if isinstance(ensemble, Pipeline):
            return self._convert_pipeline(ensemble, add_checkpoints)
        elif isinstance(ensemble, Parallel):
            return self._convert_parallel(ensemble)
        elif isinstance(ensemble, Hierarchical):
            return self._convert_hierarchical(ensemble, add_checkpoints)
        else:
            raise ValueError(f"Unknown ensemble type: {type(ensemble)}")

    def _convert_pipeline(
        self,
        pipeline: Pipeline,
        add_checkpoints: bool = False,
    ) -> WorkflowDefinition:
        """Convert a pipeline to sequential workflow nodes."""
        builder = WorkflowBuilder(
            name=pipeline.name or "pipeline_workflow",
            description=pipeline.description or "Converted from Pipeline ensemble",
        )

        # Add metadata
        builder.set_metadata("source", "ensemble_converter")
        builder.set_metadata("ensemble_type", EnsembleType.PIPELINE.value)
        builder.set_metadata("continue_on_error", pipeline.continue_on_error)

        # Add optional start HITL
        if self.add_start_hitl:
            builder.add_hitl_approval(
                "start_approval",
                prompt="Start pipeline execution?",
                fallback="abort",
            )

        # Add agent nodes in sequence
        for i, agent in enumerate(pipeline.agents):
            node_id = f"agent_{agent.name}_{i}"

            builder.add_agent(
                node_id=node_id,
                role=self._get_role_for_agent(agent),
                goal=agent.system_prompt or agent.description,
                tool_budget=agent.constraints.max_tool_calls or self.default_tool_budget,
                allowed_tools=(
                    list(agent.capabilities.tools) if agent.capabilities.tools else None
                ),
                output_key=f"{agent.name}_output",
            )

        # Add optional end HITL
        if self.add_end_hitl:
            builder.add_hitl_review(
                "end_review",
                prompt="Review pipeline results",
                fallback="continue",
            )

        return builder.build()

    def _convert_parallel(self, parallel: Parallel) -> WorkflowDefinition:
        """Convert a parallel ensemble to workflow with parallel node."""
        builder = WorkflowBuilder(
            name=parallel.name or "parallel_workflow",
            description=parallel.description or "Converted from Parallel ensemble",
        )

        builder.set_metadata("source", "ensemble_converter")
        builder.set_metadata("ensemble_type", EnsembleType.PARALLEL.value)
        builder.set_metadata("require_all", parallel.require_all)

        # Create agent nodes for each parallel agent
        agent_node_ids = []
        for i, agent in enumerate(parallel.agents):
            node_id = f"parallel_{agent.name}_{i}"
            agent_node_ids.append(node_id)

            # Add as standalone node (will be referenced by parallel node)
            node = AgentNode(
                id=node_id,
                name=agent.name,
                role=self._get_role_for_agent(agent),
                goal=agent.system_prompt or agent.description,
                tool_budget=agent.constraints.max_tool_calls or self.default_tool_budget,
                allowed_tools=(
                    list(agent.capabilities.tools) if agent.capabilities.tools else None
                ),
                output_key=f"{agent.name}_output",
                next_nodes=[],  # No explicit next nodes, parallel node handles flow
            )
            builder._nodes[node_id] = node

        # Add parallel execution node
        join_strategy = "all" if parallel.require_all else "any"
        builder.add_parallel(
            node_id="parallel_execution",
            parallel_nodes=agent_node_ids,
            name="Parallel Agent Execution",
            join_strategy=join_strategy,
        )

        # Set start node to the parallel execution
        if builder._first_node is None:
            builder._first_node = "parallel_execution"

        return builder.build()

    def _convert_hierarchical(
        self,
        hierarchical: Hierarchical,
        add_checkpoints: bool = False,
    ) -> WorkflowDefinition:
        """Convert hierarchical ensemble to manager-worker workflow."""
        builder = WorkflowBuilder(
            name=hierarchical.name or "hierarchical_workflow",
            description=hierarchical.description or "Converted from Hierarchical ensemble",
        )

        builder.set_metadata("source", "ensemble_converter")
        builder.set_metadata("ensemble_type", EnsembleType.HIERARCHICAL.value)
        builder.set_metadata("max_delegations", hierarchical.max_delegations)

        # Add manager agent node
        manager = hierarchical.manager
        builder.add_agent(
            node_id="manager",
            role=self._get_role_for_agent(manager),
            goal=manager.system_prompt or manager.description,
            name=f"Manager: {manager.name}",
            tool_budget=manager.constraints.max_tool_calls or self.default_tool_budget * 2,
            allowed_tools=(
                list(manager.capabilities.tools) if manager.capabilities.tools else None
            ),
            output_key="manager_plan",
        )

        # Add worker nodes in parallel
        worker_node_ids = []
        for i, worker in enumerate(hierarchical.workers):
            node_id = f"worker_{worker.name}_{i}"
            worker_node_ids.append(node_id)

            node = AgentNode(
                id=node_id,
                name=f"Worker: {worker.name}",
                role=self._get_role_for_agent(worker),
                goal=worker.system_prompt or worker.description,
                tool_budget=worker.constraints.max_tool_calls or self.default_tool_budget,
                allowed_tools=(
                    list(worker.capabilities.tools) if worker.capabilities.tools else None
                ),
                input_mapping={"task": "manager_plan"},
                output_key=f"{worker.name}_output",
                next_nodes=[],
            )
            builder._nodes[node_id] = node

        # Add parallel execution of workers
        builder.add_parallel(
            node_id="worker_execution",
            parallel_nodes=worker_node_ids,
            name="Worker Execution",
            join_strategy="all",
        )

        # Chain manager -> workers
        builder.chain("manager", "worker_execution")

        return builder.build()

    def _get_role_for_agent(self, agent: AgentSpec) -> str:
        """Map agent model preference to workflow role."""
        role_mapping = {
            ModelPreference.REASONING: "planner",
            ModelPreference.CODING: "executor",
            ModelPreference.FAST: "executor",
            ModelPreference.BALANCED: "executor",
            ModelPreference.BUDGET: "executor",
            ModelPreference.PREMIUM: "planner",
            ModelPreference.CREATIVE: "planner",
            ModelPreference.TOOL_USE: "executor",
            ModelPreference.LONG_CONTEXT: "planner",
            ModelPreference.DEFAULT: "executor",
        }
        return role_mapping.get(agent.model_preference, "executor")


def ensemble_to_workflow(
    ensemble: Ensemble,
    add_checkpoints: bool = False,
    add_hitl: bool = False,
) -> WorkflowDefinition:
    """Convenience function to convert an ensemble to workflow.

    Args:
        ensemble: Ensemble to convert
        add_checkpoints: Add checkpoint nodes between agents
        add_hitl: Add HITL approval/review nodes

    Returns:
        WorkflowDefinition
    """
    converter = EnsembleConverter(
        add_start_hitl=add_hitl,
        add_end_hitl=add_hitl,
    )
    return converter.to_workflow(ensemble, add_checkpoints)


def workflow_to_ensemble(
    workflow: WorkflowDefinition,
) -> Ensemble:
    """Convert a workflow definition back to an ensemble.

    Only works for workflows that were originally created from ensembles
    (checks for ensemble_type metadata).

    Args:
        workflow: WorkflowDefinition to convert

    Returns:
        Appropriate Ensemble subclass

    Raises:
        ValueError: If workflow cannot be converted
    """
    ensemble_type = workflow.metadata.get("ensemble_type")

    if not ensemble_type:
        raise ValueError(
            "Workflow does not have ensemble_type metadata. "
            "Only workflows created from ensembles can be converted back."
        )

    # Extract agents from AgentNodes
    agents = []
    for node in workflow.nodes.values():
        if isinstance(node, AgentNode):
            agent = _agent_node_to_spec(node)
            agents.append(agent)

    if ensemble_type == EnsembleType.PIPELINE.value:
        return Pipeline(
            agents=agents,
            name=workflow.name,
            continue_on_error=workflow.metadata.get("continue_on_error", False),
        )
    elif ensemble_type == EnsembleType.PARALLEL.value:
        return Parallel(
            agents=agents,
            name=workflow.name,
            require_all=workflow.metadata.get("require_all", True),
        )
    elif ensemble_type == EnsembleType.HIERARCHICAL.value:
        # First agent is manager, rest are workers
        if not agents:
            raise ValueError("No agents found in workflow")
        return Hierarchical(
            manager=agents[0],
            workers=agents[1:] if len(agents) > 1 else [],
            name=workflow.name,
            max_delegations=workflow.metadata.get("max_delegations", 10),
        )
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")


def _agent_node_to_spec(node: AgentNode) -> AgentSpec:
    """Convert an AgentNode back to AgentSpec."""
    from victor.agent.specs.models import AgentCapabilities, AgentConstraints

    # Map role back to model preference
    role_to_preference = {
        "planner": ModelPreference.REASONING,
        "researcher": ModelPreference.REASONING,
        "executor": ModelPreference.CODING,
        "reviewer": ModelPreference.REASONING,
    }

    return AgentSpec(
        name=node.name,
        description=node.goal,
        capabilities=AgentCapabilities(
            tools=set(node.allowed_tools) if node.allowed_tools else set(),
        ),
        constraints=AgentConstraints(
            max_tool_calls=node.tool_budget,
        ),
        model_preference=role_to_preference.get(node.role, ModelPreference.BALANCED),
        system_prompt=node.goal,
    )
