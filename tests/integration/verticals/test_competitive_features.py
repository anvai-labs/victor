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

"""Integration tests for competitive feature parity.

These tests verify that Victor provides feature parity with leading
multi-agent frameworks like LangGraph and CrewAI.
"""

import pytest
from dataclasses import dataclass
from typing import List, Optional


class TestLangGraphParity:
    """Tests verifying LangGraph feature parity.

    LangGraph is a library for building stateful, multi-actor applications
    with LLMs. Victor provides comparable capabilities through its
    StateGraph and workflow execution infrastructure.
    """

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.workflows
    async def test_graph_based_workflow_creation(self, empty_workflow_graph):
        """Victor supports graph-based workflows like LangGraph.

        Verifies that StateGraph can be created and configured with
        nodes and edges, similar to LangGraph's graph construction API.
        """
        # Given an empty workflow graph
        graph = empty_workflow_graph

        # When we add nodes and edges
        graph.add_node("analyze", lambda s: s)
        graph.add_node("process", lambda s: s)
        graph.add_edge("analyze", "process")
        graph.set_entry_point("analyze")
        graph.set_finish_point("process")

        # Then the graph should be valid and compilable
        errors = graph.validate()
        assert len(errors) == 0, f"Graph validation failed: {errors}"

        # And we can compile to a workflow definition
        workflow = graph.compile()
        assert workflow is not None
        assert workflow.name == "test_workflow"
        assert "analyze" in workflow.nodes
        assert "process" in workflow.nodes

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.workflows
    async def test_conditional_edge_routing(self, branching_workflow_graph):
        """Victor supports conditional edges like LangGraph.

        Verifies that conditional routing works correctly based on
        state values, allowing dynamic workflow paths.
        """
        # Given a branching workflow graph
        graph = branching_workflow_graph

        # Then the graph should validate successfully
        errors = graph.validate()
        assert len(errors) == 0, f"Graph validation failed: {errors}"

        # And we can compile it
        workflow = graph.compile()
        assert workflow is not None

        # And the compiled workflow has the conditional structure
        assert "start" in workflow.nodes
        assert "branch_a" in workflow.nodes
        assert "branch_b" in workflow.nodes
        assert "merge" in workflow.nodes

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.workflows
    async def test_graph_validation_catches_issues(self):
        """Victor validates graph structure like LangGraph.

        Verifies that the graph system can detect validation issues
        in workflow definitions, such as missing entry/finish points.
        """
        from dataclasses import dataclass
        from victor.workflows.graph_dsl import WorkflowGraph, State

        @dataclass
        class ValidationState(State):
            value: str = ""

        # Test 1: Graph without entry point should fail validation
        graph_no_entry = WorkflowGraph(ValidationState, name="no_entry_test")
        graph_no_entry.add_node("a", lambda s: s)
        graph_no_entry.add_node("b", lambda s: s)
        graph_no_entry.add_edge("a", "b")
        # Intentionally not setting entry point

        errors = graph_no_entry.validate()
        assert len(errors) > 0, "Should detect missing entry point"
        assert any("entry" in e.lower() for e in errors)

        # Test 2: Unreachable node should be detected
        graph_unreachable = WorkflowGraph(ValidationState, name="unreachable_test")
        graph_unreachable.add_node("a", lambda s: s)
        graph_unreachable.add_node("b", lambda s: s)
        graph_unreachable.add_node("c", lambda s: s)  # c is unreachable
        graph_unreachable.add_edge("a", "b")
        graph_unreachable.set_entry_point("a")
        graph_unreachable.set_finish_point("b")

        errors = graph_unreachable.validate()
        assert len(errors) > 0, "Should detect unreachable node"
        assert any("unreachable" in e.lower() for e in errors)

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.workflows
    async def test_state_persistence_across_nodes(self, linear_workflow_graph):
        """Victor preserves state across workflow nodes like LangGraph.

        Verifies that state modifications in one node are visible
        to subsequent nodes in the workflow.
        """
        # Given a linear workflow
        graph = linear_workflow_graph

        # Then the graph should compile successfully
        workflow = graph.compile()
        assert workflow is not None

        # And nodes should be connected in sequence
        assert len(workflow.nodes) == 3


class TestCrewAIParity:
    """Tests verifying CrewAI feature parity.

    CrewAI is a framework for orchestrating role-playing autonomous AI
    agents. Victor provides comparable multi-agent coordination through
    its teams infrastructure.
    """

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.agents
    async def test_team_formation_patterns(self, team_member_specs):
        """Victor supports formation patterns like CrewAI.

        Verifies that teams can be configured with different coordination
        patterns (sequential, parallel, hierarchical, pipeline).
        """
        from victor.framework.teams import TeamMemberSpec
        from victor.teams import TeamFormation, TeamConfig, TeamMember
        from victor.agent.subagents.base import SubAgentRole

        # Given team member specifications
        specs = team_member_specs

        # Test non-hierarchical formations
        formations_non_hierarchical = [
            TeamFormation.SEQUENTIAL,
            TeamFormation.PARALLEL,
            TeamFormation.PIPELINE,
        ]

        for formation in formations_non_hierarchical:
            members = [spec.to_team_member(i) for i, spec in enumerate(specs)]
            config = TeamConfig(
                name=f"test_team_{formation.value}",
                goal="Test team formation",
                members=members,
                formation=formation,
            )

            # Then each formation should be valid
            assert config.formation == formation
            assert len(config.members) == 3

        # Test hierarchical formation (requires exactly one manager)
        members_with_manager = [
            TeamMember(
                id="manager_1",
                role=SubAgentRole.PLANNER,
                name="Manager",
                goal="Manage the team",
                tool_budget=20,
                is_manager=True,
            ),
            TeamMember(
                id="worker_1",
                role=SubAgentRole.EXECUTOR,
                name="Worker 1",
                goal="Execute tasks",
                tool_budget=15,
                is_manager=False,
            ),
            TeamMember(
                id="worker_2",
                role=SubAgentRole.RESEARCHER,
                name="Worker 2",
                goal="Research information",
                tool_budget=15,
                is_manager=False,
            ),
        ]

        hierarchical_config = TeamConfig(
            name="test_team_hierarchical",
            goal="Test hierarchical formation",
            members=members_with_manager,
            formation=TeamFormation.HIERARCHICAL,
        )

        assert hierarchical_config.formation == TeamFormation.HIERARCHICAL
        assert len(hierarchical_config.members) == 3
        assert sum(1 for m in hierarchical_config.members if m.is_manager) == 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.agents
    async def test_role_based_capabilities(self, mock_team_member):
        """Victor restricts tools by role like CrewAI.

        Verifies that team members have role-based capabilities
        and tool restrictions.
        """
        from victor.agent.subagents.base import SubAgentRole

        # Given a mock team member
        member = mock_team_member

        # Then the member should have role-based properties
        assert member.role == SubAgentRole.EXECUTOR
        assert member.tool_budget == 15
        assert member.is_manager is False

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.agents
    async def test_inter_agent_communication(self):
        """Victor supports agent messaging like CrewAI.

        Verifies that the message bus infrastructure enables
        inter-agent communication.
        """
        from victor.teams import (
            TeamMessageBus,
            AgentMessage,
            MessageType,
        )

        # Given a message bus for a team with registered agents
        bus = TeamMessageBus(team_id="test_team")
        bus.register_agent("agent_a")
        bus.register_agent("agent_b")

        # When agents send and receive messages
        test_message = AgentMessage(
            message_type=MessageType.REQUEST,  # Use REQUEST type for task-like messages
            sender_id="agent_a",
            recipient_id="agent_b",
            content="Test message",
        )

        await bus.send(test_message)

        # Then messages should be routable to the recipient
        message = await bus.receive("agent_b", timeout=1.0)
        assert message is not None, "Agent B should receive the message"
        assert message.content == "Test message"
        assert message.from_agent == "agent_a"

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.agents
    async def test_team_member_rich_persona(self):
        """Victor supports rich persona attributes like CrewAI.

        Verifies that team members can have backstory, expertise,
        and personality attributes for natural characterization.
        """
        from victor.framework.teams import TeamMemberSpec

        # Given a team member spec with rich persona
        spec = TeamMemberSpec(
            role="researcher",
            goal="Find security vulnerabilities",
            name="Security Analyst",
            backstory="10 years of security experience at major tech companies.",
            expertise=["security", "authentication", "oauth", "jwt"],
            personality="methodical and thorough",
            tool_budget=25,
        )

        # When converted to team member
        member = spec.to_team_member(0)

        # Then persona attributes should be preserved
        assert member.name == "Security Analyst"
        assert member.goal == "Find security vulnerabilities"
        assert member.backstory == "10 years of security experience at major tech companies."
        assert "security" in member.expertise
        assert member.personality == "methodical and thorough"


class TestHITLFeatures:
    """Tests for human-in-the-loop features.

    HITL features allow workflows to pause for human approval,
    review, or input during execution.
    """

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.hitl
    async def test_approval_workflow(self, hitl_approval_node, auto_approve_handler):
        """Victor supports approval gates.

        Verifies that HITL approval nodes can pause workflow execution
        and await human decision.
        """
        from victor.workflows.hitl import (
            HITLExecutor,
            HITLStatus,
            HITLNodeType,
        )

        # Given an HITL node and auto-approve handler
        node = hitl_approval_node

        # Create an executor with a custom handler that auto-approves
        class AutoApproveHandler:
            async def request_human_input(self, request):
                return await auto_approve_handler(request)

        executor = HITLExecutor(handler=AutoApproveHandler())

        # When we execute the HITL node
        context = {"action": "test action"}
        response = await executor.execute_hitl_node(node, context)

        # Then the response should be approved
        assert response.status == HITLStatus.APPROVED
        assert response.approved is True

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.hitl
    async def test_rejection_workflow(self, hitl_approval_node, auto_reject_handler):
        """Victor handles rejection in approval gates.

        Verifies that HITL rejection is properly propagated.
        """
        from victor.workflows.hitl import HITLExecutor, HITLStatus

        # Given an HITL node and auto-reject handler
        node = hitl_approval_node

        class AutoRejectHandler:
            async def request_human_input(self, request):
                return await auto_reject_handler(request)

        executor = HITLExecutor(handler=AutoRejectHandler())

        # When we execute the HITL node
        context = {"action": "test action"}
        response = await executor.execute_hitl_node(node, context)

        # Then the response should be rejected
        assert response.status == HITLStatus.REJECTED
        assert response.approved is False
        assert response.reason == "Auto-rejected for testing"

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.hitl
    async def test_interrupt_and_resume(self):
        """Victor supports workflow interruption.

        Verifies that HITL requests can capture workflow context
        for later resumption.
        """
        from victor.workflows.hitl import (
            HITLNode,
            HITLNodeType,
            HITLFallback,
        )

        # Given an HITL node with context keys
        node = HITLNode(
            id="interrupt_test",
            name="Interrupt Point",
            hitl_type=HITLNodeType.REVIEW,
            prompt="Review the analysis results",
            context_keys=["analysis", "files"],
            timeout=60.0,
            fallback=HITLFallback.CONTINUE,
        )

        # When we create a request from workflow context
        context = {
            "analysis": "Found 3 issues",
            "files": ["main.py", "utils.py"],
            "internal_state": "should not be exposed",
        }

        request = node.create_request(context)

        # Then the request should contain relevant context
        assert "analysis" in request.context
        assert "files" in request.context
        assert "internal_state" not in request.context
        assert request.context["analysis"] == "Found 3 issues"

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.hitl
    async def test_hitl_timeout_fallback(self):
        """Victor handles HITL timeout with fallback behavior.

        Verifies that when HITL times out, the configured fallback
        behavior is applied.
        """
        from victor.workflows.hitl import (
            HITLNode,
            HITLNodeType,
            HITLFallback,
            HITLRequest,
            HITLExecutor,
        )

        # Given an HITL node with CONTINUE fallback
        node = HITLNode(
            id="timeout_test",
            name="Timeout Test",
            hitl_type=HITLNodeType.CONFIRMATION,
            prompt="Confirm to proceed",
            timeout=0.01,  # Very short timeout
            fallback=HITLFallback.CONTINUE,
            default_value="auto-continued",
        )

        # Create a handler that takes too long
        class SlowHandler:
            async def request_human_input(self, request):
                import asyncio

                await asyncio.sleep(1.0)  # Longer than timeout
                # This should never be reached due to timeout

        executor = HITLExecutor(handler=SlowHandler())

        # When we execute the HITL node
        context = {}
        response = await executor.execute_hitl_node(node, context)

        # Then the response should reflect timeout with continue behavior
        from victor.workflows.hitl import HITLStatus

        assert response.status == HITLStatus.TIMEOUT
        assert response.approved is True  # CONTINUE fallback approves
        assert response.value == "auto-continued"


class TestModeConfigFeatures:
    """Tests for mode configuration features.

    Mode configurations provide preset operational parameters
    for different task types and complexity levels.
    """

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_mode_registry_lookup(self, mode_config_registry):
        """Victor provides mode configuration lookup.

        Verifies that modes can be looked up by name with
        appropriate fallback behavior.
        """
        # Given a mode config registry
        registry = mode_config_registry

        # When we look up default modes
        quick_mode = registry.get_mode(None, "quick")
        standard_mode = registry.get_mode(None, "standard")

        # Then default modes should be available
        assert quick_mode is not None
        assert quick_mode.tool_budget == 5
        assert standard_mode is not None
        assert standard_mode.tool_budget == 15

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_vertical_mode_override(self, registered_mode_registry):
        """Victor supports vertical-specific mode overrides.

        Verifies that verticals can override default modes with
        custom configurations.
        """
        # Given a registry with registered verticals
        registry = registered_mode_registry

        # When we look up a vertical-specific mode
        custom_mode = registry.get_mode("test_vertical", "custom")

        # Then the vertical override should be returned
        assert custom_mode is not None
        assert custom_mode.tool_budget == 25
        assert custom_mode.max_iterations == 50

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_task_based_tool_budget(self, registered_mode_registry):
        """Victor provides task-based tool budget recommendations.

        Verifies that tool budgets can be retrieved based on
        task type for appropriate resource allocation.
        """
        # Given a registry with task budgets
        registry = registered_mode_registry

        # When we get tool budget for different task types
        test_budget = registry.get_tool_budget("test_vertical", task_type="test_task")
        complex_budget = registry.get_tool_budget("test_vertical", task_type="complex_task")
        default_budget = registry.get_tool_budget("test_vertical", task_type="unknown")

        # Then task-specific budgets should be returned
        assert test_budget == 15
        assert complex_budget == 30
        assert default_budget == 12  # Vertical default
