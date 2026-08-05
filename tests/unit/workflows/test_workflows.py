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

"""Unit tests for the workflows module."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.workflows import (
    WorkflowNodeType,
    WorkflowNode,
    AgentNode,
    ConditionNode,
    ParallelNode,
    TransformNode,
    WorkflowDefinition,
    WorkflowBuilder,
    workflow,
    get_registered_workflows,
    WorkflowMetadata,
    WorkflowRegistry,
    get_global_registry,
    ExecutorNodeStatus,
    NodeResult,
    WorkflowContext,
    WorkflowResult,
    WorkflowExecutor,
)


class TestWorkflowNodeType:
    """Test WorkflowNodeType enum."""

    def test_all_types_defined(self):
        """All expected node types are defined."""
        assert WorkflowNodeType.AGENT.value == "agent"
        assert WorkflowNodeType.CONDITION.value == "condition"
        assert WorkflowNodeType.PARALLEL.value == "parallel"
        assert WorkflowNodeType.TRANSFORM.value == "transform"
        assert WorkflowNodeType.START.value == "start"
        assert WorkflowNodeType.END.value == "end"


class TestAgentNode:
    """Test AgentNode class."""

    def test_minimal_node(self):
        """Create node with minimal fields."""
        node = AgentNode(id="test", name="Test Node")
        assert node.id == "test"
        assert node.name == "Test Node"
        assert node.node_type == WorkflowNodeType.AGENT
        assert node.role == "executor"
        assert node.tool_budget == 15

    def test_full_node(self):
        """Create node with all fields."""
        node = AgentNode(
            id="research",
            name="Research Node",
            role="researcher",
            goal="Find all API endpoints",
            tool_budget=20,
            allowed_tools=["read", "search"],
            input_mapping={"files": "target_files"},
            output_key="research_results",
            next_nodes=["review"],
        )
        assert node.role == "researcher"
        assert node.goal == "Find all API endpoints"
        assert node.tool_budget == 20
        assert node.allowed_tools == ["read", "search"]
        assert "review" in node.next_nodes

    def test_to_dict(self):
        """to_dict serializes correctly."""
        node = AgentNode(
            id="test",
            name="Test",
            role="executor",
            goal="Do something",
        )
        d = node.to_dict()
        assert d["id"] == "test"
        assert d["type"] == "agent"
        assert d["role"] == "executor"


class TestConditionNode:
    """Test ConditionNode class."""

    def test_minimal_node(self):
        """Create node with minimal fields."""
        node = ConditionNode(id="decide", name="Decision")
        assert node.id == "decide"
        assert node.node_type == WorkflowNodeType.CONDITION

    def test_evaluate_condition(self):
        """Condition evaluation works."""
        node = ConditionNode(
            id="decide",
            name="Decision",
            condition=lambda ctx: "fix" if ctx.get("issues", 0) > 0 else "done",
            branches={"fix": "fixer", "done": "report"},
        )

        # With issues
        result = node.evaluate({"issues": 5})
        assert result == "fixer"

        # Without issues
        result = node.evaluate({"issues": 0})
        assert result == "report"


class TestParallelNode:
    """Test ParallelNode class."""

    def test_minimal_node(self):
        """Create node with minimal fields."""
        node = ParallelNode(id="parallel", name="Parallel")
        assert node.id == "parallel"
        assert node.node_type == WorkflowNodeType.PARALLEL
        assert node.join_strategy == "all"

    def test_with_parallel_nodes(self):
        """Create node with parallel nodes."""
        node = ParallelNode(
            id="parallel",
            name="Parallel Research",
            parallel_nodes=["research_a", "research_b", "research_c"],
            join_strategy="merge",
        )
        assert len(node.parallel_nodes) == 3
        assert node.join_strategy == "merge"


class TestTransformNode:
    """Test TransformNode class."""

    def test_minimal_node(self):
        """Create node with minimal fields."""
        node = TransformNode(id="transform", name="Transform")
        assert node.id == "transform"
        assert node.node_type == WorkflowNodeType.TRANSFORM

    def test_with_transform(self):
        """Transform function works."""
        node = TransformNode(
            id="transform",
            name="Transform",
            transform=lambda ctx: {**ctx, "processed": True},
        )
        result = node.transform({"data": "value"})
        assert result["processed"] is True
        assert result["data"] == "value"


class TestWorkflowDefinition:
    """Test WorkflowDefinition class."""

    def test_minimal_definition(self):
        """Create definition with minimal fields."""
        defn = WorkflowDefinition(
            name="test",
            nodes={
                "start": AgentNode(id="start", name="Start"),
            },
        )
        assert defn.name == "test"
        assert defn.start_node == "start"

    def test_get_node(self):
        """get_node returns correct node."""
        node = AgentNode(id="test", name="Test")
        defn = WorkflowDefinition(
            name="test",
            nodes={"test": node},
        )
        assert defn.get_node("test") == node
        assert defn.get_node("nonexistent") is None

    def test_validate_empty(self):
        """Validation fails for empty workflow."""
        defn = WorkflowDefinition(name="test")
        errors = defn.validate()
        assert len(errors) > 0
        assert any("at least one node" in e for e in errors)

    def test_validate_missing_start(self):
        """Validation fails for missing start node."""
        defn = WorkflowDefinition(
            name="test",
            nodes={"a": AgentNode(id="a", name="A")},
            start_node="nonexistent",
        )
        errors = defn.validate()
        assert any("not found" in e for e in errors)

    def test_validate_broken_reference(self):
        """Validation fails for broken next_node reference."""
        defn = WorkflowDefinition(
            name="test",
            nodes={
                "a": AgentNode(id="a", name="A", next_nodes=["nonexistent"]),
            },
        )
        errors = defn.validate()
        assert any("non-existent node" in e for e in errors)

    def test_get_agent_count(self):
        """get_agent_count returns correct count."""
        defn = WorkflowDefinition(
            name="test",
            nodes={
                "a": AgentNode(id="a", name="A"),
                "b": AgentNode(id="b", name="B"),
                "c": ConditionNode(id="c", name="C"),
            },
        )
        assert defn.get_agent_count() == 2

    def test_get_total_budget(self):
        """get_total_budget sums agent budgets."""
        defn = WorkflowDefinition(
            name="test",
            nodes={
                "a": AgentNode(id="a", name="A", tool_budget=10),
                "b": AgentNode(id="b", name="B", tool_budget=20),
            },
        )
        assert defn.get_total_budget() == 30

    def test_to_dict(self):
        """to_dict serializes correctly."""
        defn = WorkflowDefinition(
            name="test",
            description="Test workflow",
            nodes={
                "a": AgentNode(id="a", name="A"),
            },
        )
        d = defn.to_dict()
        assert d["name"] == "test"
        assert d["description"] == "Test workflow"
        assert "a" in d["nodes"]


class TestWorkflowBuilder:
    """Test WorkflowBuilder class."""

    def test_simple_chain(self):
        """Build simple sequential workflow."""
        workflow = (
            WorkflowBuilder("test")
            .add_agent("analyze", "researcher", "Analyze code")
            .add_agent("review", "reviewer", "Review findings")
            .add_agent("report", "planner", "Create report")
            .build()
        )

        assert workflow.name == "test"
        assert len(workflow.nodes) == 3
        assert workflow.start_node == "analyze"

        # Check auto-chaining
        assert "review" in workflow.nodes["analyze"].next_nodes
        assert "report" in workflow.nodes["review"].next_nodes

    def test_with_condition(self):
        """Build workflow with condition node."""
        workflow = (
            WorkflowBuilder("test")
            .add_agent("analyze", "researcher", "Analyze")
            .add_condition(
                "decide",
                lambda ctx: "fix" if ctx.get("issues") else "done",
                {"fix": "fixer", "done": "report"},
            )
            .add_agent("fixer", "executor", "Fix issues", next_nodes=["report"])
            .add_agent("report", "planner", "Report")
            .build()
        )

        assert len(workflow.nodes) == 4
        assert "decide" in workflow.nodes

    def test_with_parallel(self):
        """Build workflow with parallel node."""
        workflow = (
            WorkflowBuilder("test")
            .add_agent("research_a", "researcher", "Research A")
            .add_agent("research_b", "researcher", "Research B")
            .add_parallel(
                "parallel",
                ["research_a", "research_b"],
                next_nodes=["merge"],
            )
            .add_agent("merge", "planner", "Merge results")
            .build()
        )

        parallel = workflow.nodes["parallel"]
        assert parallel.parallel_nodes == ["research_a", "research_b"]

    def test_explicit_chain(self):
        """chain() creates explicit connection."""
        builder = WorkflowBuilder("test")
        builder.add_agent("a", "executor", "A")
        builder.add_agent("b", "executor", "B")
        builder.chain("a", "b")

        workflow = builder.build()
        assert "b" in workflow.nodes["a"].next_nodes

    def test_metadata(self):
        """set_metadata adds metadata."""
        workflow = (
            WorkflowBuilder("test")
            .set_metadata("version", "1.0.0")
            .set_metadata("author", "test")
            .add_agent("a", "executor", "A")
            .build()
        )

        assert workflow.metadata["version"] == "1.0.0"
        assert workflow.metadata["author"] == "test"

    def test_duplicate_node_raises(self):
        """Adding duplicate node ID raises ValueError."""
        builder = WorkflowBuilder("test")
        builder.add_agent("a", "executor", "A")

        with pytest.raises(ValueError, match="already exists"):
            builder.add_agent("a", "executor", "A again")

    def test_chain_unknown_node_raises(self):
        """Chaining unknown node raises ValueError."""
        builder = WorkflowBuilder("test")
        builder.add_agent("a", "executor", "A")

        with pytest.raises(ValueError, match="not found"):
            builder.chain("a", "unknown")


class TestWorkflowDecorator:
    """Test @workflow decorator."""

    def test_decorator_registers_workflow(self):
        """Decorator registers workflow factory."""

        @workflow("test_decorated", "A test workflow")
        def test_workflow():
            return WorkflowBuilder("test_decorated").add_agent("a", "executor", "A").build()

        # Should be in registered workflows
        registered = get_registered_workflows()
        assert "test_decorated" in registered

        # Calling factory should return workflow
        wf = registered["test_decorated"]()
        assert wf.name == "test_decorated"
        assert wf.description == "A test workflow"


class TestWorkflowMetadata:
    """Test WorkflowMetadata class."""

    def test_from_definition(self):
        """from_definition creates correct metadata."""
        defn = WorkflowDefinition(
            name="test",
            description="Test workflow",
            nodes={
                "a": AgentNode(id="a", name="A", tool_budget=10),
                "b": AgentNode(id="b", name="B", tool_budget=20),
            },
            metadata={"tags": ["code", "review"], "version": "2.0.0"},
        )

        meta = WorkflowMetadata.from_definition(defn)
        assert meta.name == "test"
        assert meta.description == "Test workflow"
        assert meta.agent_count == 2
        assert meta.total_budget == 30
        assert "code" in meta.tags
        assert meta.version == "2.0.0"

    def test_to_dict(self):
        """to_dict serializes correctly."""
        meta = WorkflowMetadata(
            name="test",
            description="Test",
            agent_count=2,
            total_budget=30,
        )
        d = meta.to_dict()
        assert d["name"] == "test"
        assert d["agent_count"] == 2


class TestWorkflowRegistry:
    """Test WorkflowRegistry class."""

    def test_register_and_get(self):
        """Register and retrieve workflow."""
        registry = WorkflowRegistry()
        workflow = WorkflowDefinition(
            name="test",
            nodes={"a": AgentNode(id="a", name="A")},
        )

        registry.register(workflow)
        retrieved = registry.get("test")

        assert retrieved == workflow

    def test_register_duplicate_raises(self):
        """Registering duplicate raises ValueError."""
        registry = WorkflowRegistry()
        workflow = WorkflowDefinition(
            name="test",
            nodes={"a": AgentNode(id="a", name="A")},
        )

        registry.register(workflow)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(workflow)

    def test_register_with_replace(self):
        """Registering with replace=True overwrites."""
        registry = WorkflowRegistry()
        workflow1 = WorkflowDefinition(
            name="test",
            description="First",
            nodes={"a": AgentNode(id="a", name="A")},
        )
        workflow2 = WorkflowDefinition(
            name="test",
            description="Second",
            nodes={"a": AgentNode(id="a", name="A")},
        )

        registry.register(workflow1)
        registry.register(workflow2, replace=True)

        retrieved = registry.get("test")
        assert retrieved.description == "Second"

    def test_register_factory(self):
        """Register and use factory."""
        registry = WorkflowRegistry()

        def create_workflow():
            return WorkflowDefinition(
                name="lazy",
                nodes={"a": AgentNode(id="a", name="A")},
            )

        registry.register_factory("lazy", create_workflow)

        # Not instantiated yet
        assert "lazy" not in registry._definitions

        # Get triggers instantiation
        workflow = registry.get("lazy")
        assert workflow is not None
        assert workflow.name == "lazy"

        # Now cached
        assert "lazy" in registry._definitions

    def test_list_workflows(self):
        """list_workflows returns all names."""
        registry = WorkflowRegistry()
        registry.register(
            WorkflowDefinition(
                name="a",
                nodes={"x": AgentNode(id="x", name="X")},
            )
        )
        registry.register_factory(
            "b",
            lambda: WorkflowDefinition(
                name="b",
                nodes={"y": AgentNode(id="y", name="Y")},
            ),
        )

        names = registry.list_workflows()
        assert "a" in names
        assert "b" in names

    def test_unregister(self):
        """unregister removes workflow."""
        registry = WorkflowRegistry()
        registry.register(
            WorkflowDefinition(
                name="test",
                nodes={"a": AgentNode(id="a", name="A")},
            )
        )

        assert registry.unregister("test") is True
        assert registry.get("test") is None
        assert registry.unregister("test") is False

    def test_search_by_tags(self):
        """search filters by tags."""
        registry = WorkflowRegistry()
        registry.register(
            WorkflowDefinition(
                name="a",
                nodes={"x": AgentNode(id="x", name="X")},
                metadata={"tags": ["code", "review"]},
            )
        )
        registry.register(
            WorkflowDefinition(
                name="b",
                nodes={"y": AgentNode(id="y", name="Y")},
                metadata={"tags": ["test"]},
            )
        )

        results = registry.search(tags={"code"})
        assert len(results) == 1
        assert results[0].name == "a"

    def test_list_metadata_default_lazy(self):
        """list_metadata with default args doesn't materialize factories."""
        registry = WorkflowRegistry()
        factory_called = []

        def create_workflow():
            factory_called.append(True)
            return WorkflowDefinition(
                name="lazy",
                nodes={"a": AgentNode(id="a", name="A")},
            )

        registry.register_factory("lazy", create_workflow)

        # list_metadata with default (materialize=False) should not call factory
        metadata = registry.list_metadata()
        assert len(factory_called) == 0
        assert len(metadata) == 0  # Factory not materialized, so no metadata

    def test_list_metadata_materialize_true(self):
        """list_metadata with materialize=True instantiates all factories."""
        registry = WorkflowRegistry()
        factory_called = []

        def create_workflow():
            factory_called.append(True)
            return WorkflowDefinition(
                name="lazy",
                description="Lazy workflow",
                nodes={"a": AgentNode(id="a", name="A")},
            )

        registry.register_factory("lazy", create_workflow)

        # list_metadata with materialize=True should call factory
        metadata = registry.list_metadata(materialize=True)
        assert len(factory_called) == 1
        assert len(metadata) == 1
        assert metadata[0].name == "lazy"
        assert metadata[0].description == "Lazy workflow"

    def test_list_metadata_includes_eagerly_registered(self):
        """list_metadata returns metadata for eagerly registered workflows."""
        registry = WorkflowRegistry()
        registry.register(
            WorkflowDefinition(
                name="eager",
                description="Eager workflow",
                nodes={"a": AgentNode(id="a", name="A")},
            )
        )

        # Even with materialize=False, eager registrations have metadata
        metadata = registry.list_metadata()
        assert len(metadata) == 1
        assert metadata[0].name == "eager"

    def test_list_metadata_mixed_registration(self):
        """list_metadata handles mix of eager and lazy registrations."""
        registry = WorkflowRegistry()

        # Eager registration
        registry.register(
            WorkflowDefinition(
                name="eager",
                nodes={"a": AgentNode(id="a", name="A")},
            )
        )

        # Lazy registration
        factory_called = []

        def create_workflow():
            factory_called.append(True)
            return WorkflowDefinition(
                name="lazy",
                nodes={"b": AgentNode(id="b", name="B")},
            )

        registry.register_factory("lazy", create_workflow)

        # Without materialize, only eager has metadata
        metadata = registry.list_metadata(materialize=False)
        assert len(metadata) == 1
        assert metadata[0].name == "eager"
        assert len(factory_called) == 0

        # With materialize, both have metadata
        metadata = registry.list_metadata(materialize=True)
        assert len(metadata) == 2
        assert len(factory_called) == 1
        names = {m.name for m in metadata}
        assert names == {"eager", "lazy"}


class TestNodeResult:
    """Test NodeResult class."""

    def test_success_result(self):
        """Create successful result."""
        result = NodeResult(
            node_id="test",
            status=ExecutorNodeStatus.COMPLETED,
            output="Found 5 items",
            tool_calls_used=10,
        )
        assert result.success is True
        assert result.output == "Found 5 items"

    def test_failure_result(self):
        """Create failed result."""
        result = NodeResult(
            node_id="test",
            status=ExecutorNodeStatus.FAILED,
            error="Something went wrong",
        )
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_to_dict(self):
        """to_dict serializes correctly."""
        result = NodeResult(
            node_id="test",
            status=ExecutorNodeStatus.COMPLETED,
            output="Done",
        )
        d = result.to_dict()
        assert d["node_id"] == "test"
        assert d["status"] == "completed"


class TestWorkflowContext:
    """Test WorkflowContext class."""

    def test_get_set(self):
        """get and set work correctly."""
        ctx = WorkflowContext()
        ctx.set("key", "value")
        assert ctx.get("key") == "value"
        assert ctx.get("missing", "default") == "default"

    def test_update(self):
        """update adds multiple values."""
        ctx = WorkflowContext()
        ctx.update({"a": 1, "b": 2})
        assert ctx.get("a") == 1
        assert ctx.get("b") == 2

    def test_add_result(self):
        """add_result stores node result."""
        ctx = WorkflowContext()
        result = NodeResult(
            node_id="test",
            status=ExecutorNodeStatus.COMPLETED,
            output="Done",
        )
        ctx.add_result(result)
        assert ctx.get_result("test") == result

    def test_has_failures(self):
        """has_failures detects failures."""
        ctx = WorkflowContext()
        ctx.add_result(NodeResult(node_id="a", status=ExecutorNodeStatus.COMPLETED))
        assert ctx.has_failures() is False

        ctx.add_result(NodeResult(node_id="b", status=ExecutorNodeStatus.FAILED))
        assert ctx.has_failures() is True

    def test_get_outputs(self):
        """get_outputs returns successful outputs."""
        ctx = WorkflowContext()
        ctx.add_result(
            NodeResult(node_id="a", status=ExecutorNodeStatus.COMPLETED, output="A output")
        )
        ctx.add_result(NodeResult(node_id="b", status=ExecutorNodeStatus.FAILED))
        ctx.add_result(
            NodeResult(node_id="c", status=ExecutorNodeStatus.COMPLETED, output="C output")
        )

        outputs = ctx.get_outputs()
        assert outputs == {"a": "A output", "c": "C output"}


class TestWorkflowResult:
    """Test WorkflowResult class."""

    def test_success_result(self):
        """Create successful result."""
        ctx = WorkflowContext()
        ctx.add_result(NodeResult(node_id="a", status=ExecutorNodeStatus.COMPLETED, output="Done"))

        result = WorkflowResult(
            workflow_name="test",
            success=True,
            context=ctx,
            total_duration=5.0,
            total_tool_calls=10,
        )

        assert result.success is True
        assert result.get_output("a") == "Done"

    def test_to_dict(self):
        """to_dict serializes correctly."""
        ctx = WorkflowContext()
        result = WorkflowResult(
            workflow_name="test",
            success=True,
            context=ctx,
        )

        d = result.to_dict()
        assert d["workflow_name"] == "test"
        assert d["success"] is True


class TestWorkflowExecutor:
    """Test WorkflowExecutor class."""

    def test_initialization(self):
        """Executor can be initialized."""
        mock_orchestrator = MagicMock()
        executor = WorkflowExecutor(mock_orchestrator)

        assert executor.orchestrator == mock_orchestrator
        assert executor.max_parallel == 4

    @pytest.mark.asyncio
    async def test_execute_simple_workflow(self):
        """Execute simple sequential workflow."""
        mock_orchestrator = MagicMock()

        # Mock SubAgentOrchestrator result
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.summary = "Task completed"
        mock_result.error = None
        mock_result.tool_calls_used = 5

        # Create executor and inject mock sub_agents
        executor = WorkflowExecutor(mock_orchestrator)
        mock_sub_agents = MagicMock()
        mock_sub_agents.spawn = AsyncMock(return_value=mock_result)
        executor._sub_agents = mock_sub_agents

        workflow = (
            WorkflowBuilder("test").add_agent("analyze", "researcher", "Analyze code").build()
        )

        result = await executor.execute(workflow, {"files": ["main.py"]})

        assert result.success is True
        assert result.total_tool_calls == 5
        mock_sub_agents.spawn.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self):
        """Execute with timeout."""
        mock_orchestrator = MagicMock()

        async def slow_spawn(*args, **kwargs):
            await asyncio.sleep(10)
            return MagicMock(success=True)

        # Create executor and inject mock sub_agents
        executor = WorkflowExecutor(mock_orchestrator)
        mock_sub_agents = MagicMock()
        mock_sub_agents.spawn = slow_spawn
        executor._sub_agents = mock_sub_agents

        workflow = WorkflowBuilder("test").add_agent("slow", "executor", "Slow task").build()

        result = await executor.execute(workflow, timeout=0.1)

        assert result.success is False
        assert "timed out" in result.error.lower()


class TestWorkflowExecutorExtended:
    """Extended tests for WorkflowExecutor class."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        return MagicMock()

    @pytest.fixture
    def mock_sub_agent_result(self):
        """Create a mock SubAgent result."""
        result = MagicMock()
        result.success = True
        result.summary = "Task completed successfully"
        result.error = None
        result.tool_calls_used = 5
        return result

    def test_sub_agents_property_lazy_init(self, mock_orchestrator):
        """sub_agents property creates SubAgentOrchestrator on first access."""
        executor = WorkflowExecutor(mock_orchestrator)
        assert executor._sub_agents is None

        with patch("victor.agent.subagents.SubAgentOrchestrator") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result = executor.sub_agents

            mock_class.assert_called_once_with(mock_orchestrator)
            assert result == mock_instance

    def test_sub_agents_property_cached(self, mock_orchestrator):
        """sub_agents property returns cached instance."""
        executor = WorkflowExecutor(mock_orchestrator)
        mock_cached = MagicMock()
        executor._sub_agents = mock_cached

        result = executor.sub_agents

        assert result == mock_cached

    @pytest.mark.asyncio
    async def test_execute_empty_workflow_raises(self, mock_orchestrator):
        """Execute raises ValueError for workflow without start node."""
        executor = WorkflowExecutor(mock_orchestrator)
        workflow = WorkflowDefinition(
            name="empty",
            nodes={},
            start_node=None,
        )

        result = await executor.execute(workflow)

        assert result.success is False
        assert "no start node" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_missing_node_warning(self, mock_orchestrator, mock_sub_agent_result):
        """Execute logs warning for missing node and continues."""
        executor = WorkflowExecutor(mock_orchestrator)
        mock_sub_agents = MagicMock()
        mock_sub_agents.spawn = AsyncMock(return_value=mock_sub_agent_result)
        executor._sub_agents = mock_sub_agents

        workflow = WorkflowDefinition(
            name="test",
            nodes={
                "start": AgentNode(id="start", name="Start", next_nodes=["missing"]),
            },
            start_node="start",
        )

        result = await executor.execute(workflow)

        # Should complete despite missing node
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_continues_on_failure_when_configured(self, mock_orchestrator):
        """Execute continues after failure when continue_on_failure is set."""
        executor = WorkflowExecutor(mock_orchestrator)

        # First agent fails, second succeeds
        fail_result = MagicMock()
        fail_result.success = False
        fail_result.summary = None
        fail_result.error = "Failed"
        fail_result.tool_calls_used = 2

        success_result = MagicMock()
        success_result.success = True
        success_result.summary = "Done"
        success_result.error = None
        success_result.tool_calls_used = 3

        mock_sub_agents = MagicMock()
        mock_sub_agents.spawn = AsyncMock(side_effect=[fail_result, success_result])
        executor._sub_agents = mock_sub_agents

        workflow = WorkflowDefinition(
            name="test",
            nodes={
                "a": AgentNode(id="a", name="A", next_nodes=["b"]),
                "b": AgentNode(id="b", name="B"),
            },
            start_node="a",
            metadata={"continue_on_failure": True},
        )

        result = await executor.execute(workflow)

        # Both agents should have been called
        assert mock_sub_agents.spawn.call_count == 2
        # Overall result is failure because one node failed
        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_stops_on_failure_by_default(self, mock_orchestrator):
        """Execute stops after failure by default."""
        executor = WorkflowExecutor(mock_orchestrator)

        fail_result = MagicMock()
        fail_result.success = False
        fail_result.summary = None
        fail_result.error = "Failed"
        fail_result.tool_calls_used = 2

        mock_sub_agents = MagicMock()
        mock_sub_agents.spawn = AsyncMock(return_value=fail_result)
        executor._sub_agents = mock_sub_agents

        workflow = WorkflowDefinition(
            name="test",
            nodes={
                "a": AgentNode(id="a", name="A", next_nodes=["b"]),
                "b": AgentNode(id="b", name="B"),
            },
            start_node="a",
        )

        result = await executor.execute(workflow)

        # Only first agent should have been called
        assert mock_sub_agents.spawn.call_count == 1
        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_agent_node_with_all_roles(
        self, mock_orchestrator, mock_sub_agent_result
    ):
        """Execute agent node maps all role types correctly."""
        from victor.agent.subagents import SubAgentRole

        executor = WorkflowExecutor(mock_orchestrator)
        mock_sub_agents = MagicMock()
        mock_sub_agents.spawn = AsyncMock(return_value=mock_sub_agent_result)
        executor._sub_agents = mock_sub_agents

        roles = ["researcher", "planner", "executor", "reviewer", "tester", "unknown"]
        expected_roles = [
            SubAgentRole.RESEARCHER,
            SubAgentRole.PLANNER,
            SubAgentRole.EXECUTOR,
            SubAgentRole.REVIEWER,
            SubAgentRole.TESTER,
            SubAgentRole.EXECUTOR,  # unknown defaults to EXECUTOR
        ]

        for role, expected_role in zip(roles, expected_roles):
            mock_sub_agents.spawn.reset_mock()

            workflow = WorkflowDefinition(
                name="test",
                nodes={
                    "start": AgentNode(id="start", name="Start", role=role),
                },
                start_node="start",
            )

            await executor.execute(workflow)

            call_kwargs = mock_sub_agents.spawn.call_args[1]
            assert (
                call_kwargs["role"] == expected_role
            ), f"Role {role} should map to {expected_role}"

    @pytest.mark.asyncio
    async def test_execute_agent_node_with_output_key(
        self, mock_orchestrator, mock_sub_agent_result
    ):
        """Execute agent node stores output in context with output_key."""
        executor = WorkflowExecutor(mock_orchestrator)
        mock_sub_agents = MagicMock()
        mock_sub_agents.spawn = AsyncMock(return_value=mock_sub_agent_result)
        executor._sub_agents = mock_sub_agents

        workflow = WorkflowDefinition(
            name="test",
            nodes={
                "start": AgentNode(
                    id="start",
                    name="Start",
                    output_key="research_results",
                ),
            },
            start_node="start",
        )

        result = await executor.execute(workflow)

        assert result.context.get("research_results") == "Task completed successfully"

    @pytest.mark.asyncio
    async def test_execute_condition_node(self, mock_orchestrator, mock_sub_agent_result):
        """Execute condition node evaluates condition and picks branch."""
        executor = WorkflowExecutor(mock_orchestrator)
        mock_sub_agents = MagicMock()
        mock_sub_agents.spawn = AsyncMock(return_value=mock_sub_agent_result)
        executor._sub_agents = mock_sub_agents

        workflow = WorkflowDefinition(
            name="test",
            nodes={
                "start": AgentNode(id="start", name="Start", next_nodes=["decide"]),
                "decide": ConditionNode(
                    id="decide",
                    name="Decision",
                    condition=lambda ctx: "fix" if ctx.get("issues", 0) > 0 else "done",
                    branches={"fix": "fixer", "done": "reporter"},
                ),
                "fixer": AgentNode(id="fixer", name="Fixer"),
                "reporter": AgentNode(id="reporter", name="Reporter"),
            },
            start_node="start",
        )

        result = await executor.execute(workflow, {"issues": 5})

        # Should have executed start, decide, and fixer
        assert "start" in result.context.node_results
        assert "decide" in result.context.node_results
        assert "fixer" in result.context.node_results
        assert "reporter" not in result.context.node_results

    @pytest.mark.asyncio
    async def test_execute_condition_node_failure(self, mock_orchestrator):
        """Execute condition node handles exception in condition."""
        executor = WorkflowExecutor(mock_orchestrator)

        def bad_condition(ctx):
            raise ValueError("Bad condition")

        workflow = WorkflowDefinition(
            name="test",
            nodes={
                "decide": ConditionNode(
                    id="decide",
                    name="Decision",
                    condition=bad_condition,
                    branches={"a": "agent_a"},
                ),
            },
            start_node="decide",
        )

        result = await executor.execute(workflow)

        assert result.success is False
        condition_result = result.context.get_result("decide")
        assert condition_result.status == ExecutorNodeStatus.FAILED
        assert "Condition evaluation failed" in condition_result.error

    @pytest.mark.asyncio
    async def test_execute_transform_node(self, mock_orchestrator, mock_sub_agent_result):
        """Execute transform node updates context data."""
        executor = WorkflowExecutor(mock_orchestrator)
        mock_sub_agents = MagicMock()
        mock_sub_agents.spawn = AsyncMock(return_value=mock_sub_agent_result)
        executor._sub_agents = mock_sub_agents

        workflow = WorkflowDefinition(
            name="test",
            nodes={
                "transform": TransformNode(
                    id="transform",
                    name="Transform",
                    transform=lambda ctx: {
                        "processed": True,
                        "count": ctx.get("count", 0) * 2,
                    },
                    next_nodes=["agent"],
                ),
                "agent": AgentNode(id="agent", name="Agent"),
            },
            start_node="transform",
        )

        result = await executor.execute(workflow, {"count": 5})

        assert result.success is True
        assert result.context.get("processed") is True
        assert result.context.get("count") == 10

    @pytest.mark.asyncio
    async def test_execute_transform_node_failure(self, mock_orchestrator):
        """Execute transform node handles exception in transform."""
        executor = WorkflowExecutor(mock_orchestrator)

        def bad_transform(ctx):
            raise ValueError("Transform error")

        workflow = WorkflowDefinition(
            name="test",
            nodes={
                "transform": TransformNode(
                    id="transform",
                    name="Transform",
                    transform=bad_transform,
                ),
            },
            start_node="transform",
        )

        result = await executor.execute(workflow)

        assert result.success is False
        transform_result = result.context.get_result("transform")
        assert transform_result.status == ExecutorNodeStatus.FAILED
        assert "Transform failed" in transform_result.error

    @pytest.mark.asyncio
    async def test_execute_parallel_node_all_strategy(
        self, mock_orchestrator, mock_sub_agent_result
    ):
        """Execute parallel node with 'all' join strategy executes all nodes."""
        executor = WorkflowExecutor(mock_orchestrator)
        mock_sub_agents = MagicMock()
        mock_sub_agents.spawn = AsyncMock(return_value=mock_sub_agent_result)
        executor._sub_agents = mock_sub_agents

        workflow = WorkflowDefinition(
            name="test",
            nodes={
                "parallel": ParallelNode(
                    id="parallel",
                    name="Parallel",
                    parallel_nodes=["agent_a", "agent_b"],
                    join_strategy="all",
                ),
                "agent_a": AgentNode(id="agent_a", name="Agent A"),
                "agent_b": AgentNode(id="agent_b", name="Agent B"),
            },
            start_node="parallel",
        )

        result = await executor.execute(workflow)

        # Parallel node executes child nodes and returns COMPLETED
        # (workflow reference is now included in context.metadata)
        parallel_result = result.context.get_result("parallel")
        assert parallel_result.status == ExecutorNodeStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_parallel_node_any_strategy(
        self, mock_orchestrator, mock_sub_agent_result
    ):
        """Execute parallel node with 'any' join strategy executes all nodes."""
        executor = WorkflowExecutor(mock_orchestrator)

        mock_sub_agents = MagicMock()
        mock_sub_agents.spawn = AsyncMock(return_value=mock_sub_agent_result)
        executor._sub_agents = mock_sub_agents

        workflow = WorkflowDefinition(
            name="test",
            nodes={
                "parallel": ParallelNode(
                    id="parallel",
                    name="Parallel",
                    parallel_nodes=["agent_a", "agent_b"],
                    join_strategy="any",
                ),
                "agent_a": AgentNode(id="agent_a", name="Agent A"),
                "agent_b": AgentNode(id="agent_b", name="Agent B"),
            },
            start_node="parallel",
        )

        result = await executor.execute(workflow)

        # Parallel node executes child nodes and returns COMPLETED
        # (workflow reference is now included in context.metadata)
        parallel_result = result.context.get_result("parallel")
        assert parallel_result.status == ExecutorNodeStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_parallel_node_no_nodes(self, mock_orchestrator):
        """Execute parallel node with no nodes to execute."""
        executor = WorkflowExecutor(mock_orchestrator)

        workflow = WorkflowDefinition(
            name="test",
            nodes={
                "parallel": ParallelNode(
                    id="parallel",
                    name="Parallel",
                    parallel_nodes=["nonexistent_a", "nonexistent_b"],
                ),
            },
            start_node="parallel",
        )
        # No workflow metadata means nodes won't be found

        result = await executor.execute(workflow)

        parallel_result = result.context.get_result("parallel")
        assert parallel_result.status == ExecutorNodeStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_execute_unknown_node_type_skipped(self, mock_orchestrator):
        """Execute unknown node type returns SKIPPED status."""
        executor = WorkflowExecutor(mock_orchestrator)

        # Create a mock node that pretends to be an unknown type
        # WorkflowNode is abstract so we use a mock
        custom_node = MagicMock()
        custom_node.id = "custom"
        custom_node.name = "Custom"
        custom_node.node_type = MagicMock()
        custom_node.node_type.value = "unknown"
        custom_node.next_nodes = []

        workflow = WorkflowDefinition(
            name="test",
            nodes={"custom": custom_node},
            start_node="custom",
        )

        result = await executor.execute(workflow)

        custom_result = result.context.get_result("custom")
        assert custom_result.status == ExecutorNodeStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_execute_by_name_success(self, mock_orchestrator, mock_sub_agent_result):
        """execute_by_name retrieves workflow from registry."""
        executor = WorkflowExecutor(mock_orchestrator)
        mock_sub_agents = MagicMock()
        mock_sub_agents.spawn = AsyncMock(return_value=mock_sub_agent_result)
        executor._sub_agents = mock_sub_agents

        workflow = WorkflowDefinition(
            name="test_workflow",
            nodes={
                "start": AgentNode(id="start", name="Start"),
            },
            start_node="start",
        )

        with patch("victor.workflows.registry.get_global_registry") as mock_get_registry:
            mock_registry = MagicMock()
            mock_registry.get.return_value = workflow
            mock_get_registry.return_value = mock_registry

            result = await executor.execute_by_name("test_workflow", {"key": "value"})

            mock_registry.get.assert_called_once_with("test_workflow")
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_by_name_not_found(self, mock_orchestrator):
        """execute_by_name returns error for unknown workflow."""
        executor = WorkflowExecutor(mock_orchestrator)

        with patch("victor.workflows.registry.get_global_registry") as mock_get_registry:
            mock_registry = MagicMock()
            mock_registry.get.return_value = None
            mock_get_registry.return_value = mock_registry

            result = await executor.execute_by_name("nonexistent_workflow")

            assert result.success is False
            assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_build_agent_task_with_input_mapping(
        self, mock_orchestrator, mock_sub_agent_result
    ):
        """_build_agent_task substitutes placeholders in goal using input mapping."""
        executor = WorkflowExecutor(mock_orchestrator)
        mock_sub_agents = MagicMock()
        mock_sub_agents.spawn = AsyncMock(return_value=mock_sub_agent_result)
        executor._sub_agents = mock_sub_agents

        # Goal uses {placeholder} syntax for template substitution
        workflow = WorkflowDefinition(
            name="test",
            nodes={
                "start": AgentNode(
                    id="start",
                    name="Start",
                    goal="Analyze files {files} in {mode} mode",
                    input_mapping={"files": "target_files", "mode": "analysis_mode"},
                ),
            },
            start_node="start",
        )

        await executor.execute(
            workflow,
            {
                "target_files": ["main.py", "utils.py"],
                "analysis_mode": "full",
            },
        )

        # Check that task has placeholders substituted
        call_kwargs = mock_sub_agents.spawn.call_args[1]
        task = call_kwargs["task"]
        assert "Analyze files" in task
        # Placeholders are replaced with JSON-serialized values
        assert "main.py" in task
        assert "utils.py" in task
        assert "full" in task

    @pytest.mark.asyncio
    async def test_build_agent_task_with_previous_outputs(
        self, mock_orchestrator, mock_sub_agent_result
    ):
        """_build_agent_task uses goal with template substitution; previous outputs stored in context."""
        executor = WorkflowExecutor(mock_orchestrator)
        mock_sub_agents = MagicMock()
        mock_sub_agents.spawn = AsyncMock(return_value=mock_sub_agent_result)
        executor._sub_agents = mock_sub_agents

        workflow = WorkflowDefinition(
            name="test",
            nodes={
                "first": AgentNode(id="first", name="First", next_nodes=["second"]),
                "second": AgentNode(id="second", name="Second", goal="Continue work"),
            },
            start_node="first",
        )

        result = await executor.execute(workflow)

        # Both agents should be called
        assert mock_sub_agents.spawn.call_count == 2

        # Second agent's task is just the goal (template substitution only)
        second_call_kwargs = mock_sub_agents.spawn.call_args_list[1][1]
        task = second_call_kwargs["task"]
        assert task == "Continue work"

        # Previous outputs are stored in context, not appended to task
        assert result.context.get("first") == "Task completed successfully"

    @pytest.mark.asyncio
    async def test_build_agent_task_with_template_substitution(self, mock_orchestrator):
        """_build_agent_task substitutes template placeholders from context."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.summary = "Analysis complete"
        mock_result.error = None
        mock_result.tool_calls_used = 5

        executor = WorkflowExecutor(mock_orchestrator)
        mock_sub_agents = MagicMock()
        mock_sub_agents.spawn = AsyncMock(return_value=mock_result)
        executor._sub_agents = mock_sub_agents

        workflow = WorkflowDefinition(
            name="test",
            nodes={
                "start": AgentNode(
                    id="start",
                    name="Start",
                    goal="Process {item} with priority {level}",
                    input_mapping={"item": "target_item", "level": "priority_level"},
                ),
            },
            start_node="start",
        )

        result = await executor.execute(
            workflow,
            {"target_item": "config.yaml", "priority_level": "high"},
        )

        # Task should have placeholders substituted
        call_kwargs = mock_sub_agents.spawn.call_args[1]
        task = call_kwargs["task"]
        assert "Process config.yaml with priority high" == task
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_node_exception_handling(self, mock_orchestrator):
        """_execute_node handles exceptions gracefully."""
        executor = WorkflowExecutor(mock_orchestrator)
        mock_sub_agents = MagicMock()
        mock_sub_agents.spawn = AsyncMock(side_effect=Exception("Unexpected error"))
        executor._sub_agents = mock_sub_agents

        workflow = WorkflowDefinition(
            name="test",
            nodes={
                "start": AgentNode(id="start", name="Start"),
            },
            start_node="start",
        )

        result = await executor.execute(workflow)

        assert result.success is False
        node_result = result.context.get_result("start")
        assert node_result.status == ExecutorNodeStatus.FAILED
        assert "Unexpected error" in node_result.error

    @pytest.mark.asyncio
    async def test_execute_loop_prevention(self, mock_orchestrator, mock_sub_agent_result):
        """Execute prevents infinite loops from cyclic references."""
        executor = WorkflowExecutor(mock_orchestrator)
        mock_sub_agents = MagicMock()
        mock_sub_agents.spawn = AsyncMock(return_value=mock_sub_agent_result)
        executor._sub_agents = mock_sub_agents

        # Create a workflow with a cycle: a -> b -> a
        workflow = WorkflowDefinition(
            name="test",
            nodes={
                "a": AgentNode(id="a", name="A", next_nodes=["b"]),
                "b": AgentNode(id="b", name="B", next_nodes=["a"]),
            },
            start_node="a",
        )

        result = await executor.execute(workflow)

        # Should complete without infinite loop
        assert result.success is True
        # Each node should only be executed once
        assert mock_sub_agents.spawn.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_with_initial_context(self, mock_orchestrator, mock_sub_agent_result):
        """Execute accepts initial context data."""
        executor = WorkflowExecutor(mock_orchestrator)
        mock_sub_agents = MagicMock()
        mock_sub_agents.spawn = AsyncMock(return_value=mock_sub_agent_result)
        executor._sub_agents = mock_sub_agents

        workflow = WorkflowDefinition(
            name="test",
            nodes={
                "start": AgentNode(id="start", name="Start"),
            },
            start_node="start",
        )

        result = await executor.execute(workflow, {"custom_key": "custom_value"})

        assert result.context.get("custom_key") == "custom_value"

    @pytest.mark.asyncio
    async def test_execute_preserves_metadata(self, mock_orchestrator, mock_sub_agent_result):
        """Execute preserves workflow metadata in context."""
        executor = WorkflowExecutor(mock_orchestrator)
        mock_sub_agents = MagicMock()
        mock_sub_agents.spawn = AsyncMock(return_value=mock_sub_agent_result)
        executor._sub_agents = mock_sub_agents

        workflow = WorkflowDefinition(
            name="test_workflow",
            nodes={
                "start": AgentNode(id="start", name="Start"),
            },
            start_node="start",
        )

        result = await executor.execute(workflow)

        assert result.context.metadata["workflow_name"] == "test_workflow"
        assert "execution_id" in result.context.metadata

    def test_custom_max_parallel(self, mock_orchestrator):
        """Executor accepts custom max_parallel setting."""
        executor = WorkflowExecutor(mock_orchestrator, max_parallel=8)
        assert executor.max_parallel == 8

    def test_custom_default_timeout(self, mock_orchestrator):
        """Executor accepts custom default_timeout setting."""
        executor = WorkflowExecutor(mock_orchestrator, default_timeout=600.0)
        assert executor.default_timeout == 600.0


class TestNodeResultExtended:
    """Extended tests for NodeResult."""

    def test_skipped_status(self):
        """Skipped status is not successful."""
        result = NodeResult(
            node_id="test",
            status=ExecutorNodeStatus.SKIPPED,
        )
        assert result.success is False

    def test_pending_status(self):
        """Pending status is not successful."""
        result = NodeResult(
            node_id="test",
            status=ExecutorNodeStatus.PENDING,
        )
        assert result.success is False

    def test_running_status(self):
        """Running status is not successful."""
        result = NodeResult(
            node_id="test",
            status=ExecutorNodeStatus.RUNNING,
        )
        assert result.success is False


class TestWorkflowResultExtended:
    """Extended tests for WorkflowResult."""

    def test_get_output_missing_node(self):
        """get_output returns None for missing node."""
        result = WorkflowResult(
            workflow_name="test",
            success=True,
            context=WorkflowContext(),
        )
        assert result.get_output("nonexistent") is None

    def test_get_output_node_without_output(self):
        """get_output returns None for node without output."""
        ctx = WorkflowContext()
        ctx.add_result(NodeResult(node_id="test", status=ExecutorNodeStatus.COMPLETED))

        result = WorkflowResult(
            workflow_name="test",
            success=True,
            context=ctx,
        )
        assert result.get_output("test") is None

    def test_to_dict_includes_outputs(self):
        """to_dict includes node outputs."""
        ctx = WorkflowContext()
        ctx.add_result(
            NodeResult(node_id="a", status=ExecutorNodeStatus.COMPLETED, output="Output A")
        )

        result = WorkflowResult(
            workflow_name="test",
            success=True,
            context=ctx,
        )

        d = result.to_dict()
        assert d["outputs"]["a"] == "Output A"


class TestModuleExports:
    """Test module exports are correct."""

    def test_workflows_init_exports(self):
        """Workflows __init__ exports all expected symbols."""
        from victor.workflows import (
            WorkflowNodeType,
            WorkflowNode,
            AgentNode,
            ConditionNode,
            ParallelNode,
            WorkflowDefinition,
            WorkflowBuilder,
            workflow,
            WorkflowRegistry,
            ExecutorNodeStatus,
            NodeResult,
            WorkflowContext,
            WorkflowResult,
            WorkflowExecutor,
        )

        # If we get here without ImportError, all exports work
        assert True
