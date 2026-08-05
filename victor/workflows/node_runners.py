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

"""Node runner implementations for workflow execution.

This module provides NodeRunner protocol implementations for different
node types, following ISP (Interface Segregation) and DIP (Dependency
Inversion) principles.

Each runner is responsible for executing a specific node type:
- AgentNodeRunner: LLM-powered agent execution
- ComputeNodeRunner: Handler-based computation (no LLM)
- TransformNodeRunner: Data transformation
- HITLNodeRunner: Human-in-the-loop approval/review
- ConditionNodeRunner: Conditional branching
- ParallelNodeRunner: Parallel execution orchestration

Example:
    from victor.workflows.node_runners import (
        NodeRunnerRegistry,
        AgentNodeRunner,
        ComputeNodeRunner,
    )

    # Create registry with runners
    registry = NodeRunnerRegistry()
    registry.register(AgentNodeRunner(sub_agents=my_sub_agents))
    registry.register(ComputeNodeRunner(tool_registry=my_tools))

    # Execute a node
    runner = registry.get_runner("agent")
    context, result = await runner.execute(node_id, node_config, context)
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
)

from victor.workflows.protocols import NodeRunner, NodeRunnerResult, ProtocolNodeStatus
from victor.workflows.context import ExecutionContext

if TYPE_CHECKING:
    from victor.agent.subagents import SubAgentManager
    from victor.tools.registry import ToolRegistry
    from victor.workflows.hitl import HITLExecutor, HITLNode

logger = logging.getLogger(__name__)


# =============================================================================
# Base Node Runner
# =============================================================================


class BaseNodeRunner(ABC):
    """Abstract base class for node runners.

    Provides common functionality for timing, error handling, and result
    creation. Subclasses implement the specific execution logic.
    """

    def __init__(self, supported_types: List[str]):
        """Initialize the runner.

        Args:
            supported_types: List of node types this runner handles.
        """
        self._supported_types = set(supported_types)

    def supports_node_type(self, node_type: str) -> bool:
        """Check if this runner supports the given node type."""
        return node_type.lower() in self._supported_types

    async def execute(
        self,
        node_id: str,
        node_config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], NodeRunnerResult]:
        """Execute a node with timing and error handling.

        Args:
            node_id: Unique identifier of the node.
            node_config: Node configuration.
            context: Current execution context.

        Returns:
            Tuple of (updated_context, result).
        """
        start_time = time.time()

        try:
            # Update context with current node
            context["_current_node"] = node_id

            # Execute the node-specific logic
            updated_context, output = await self._execute_impl(node_id, node_config, context)

            duration = time.time() - start_time

            # Record result in context
            if "_node_results" not in updated_context:
                updated_context["_node_results"] = {}

            result = NodeRunnerResult(
                node_id=node_id,
                success=True,
                output=output,
                duration_seconds=duration,
            )

            updated_context["_node_results"][node_id] = {
                "success": True,
                "output": output,
                "duration_seconds": duration,
            }

            return updated_context, result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Node '{node_id}' failed: {e}", exc_info=True)

            # Record failure in context
            if "_node_results" not in context:
                context["_node_results"] = {}

            context["_node_results"][node_id] = {
                "success": False,
                "error": str(e),
                "duration_seconds": duration,
            }
            context["_error"] = str(e)

            result = NodeRunnerResult(
                node_id=node_id,
                success=False,
                error=str(e),
                duration_seconds=duration,
            )

            return context, result

    @abstractmethod
    async def _execute_impl(
        self,
        node_id: str,
        node_config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Any]:
        """Implement node-specific execution logic.

        Args:
            node_id: Unique identifier of the node.
            node_config: Node configuration.
            context: Current execution context.

        Returns:
            Tuple of (updated_context, output).

        Raises:
            Any exception - will be caught by execute() and converted to result.
        """
        ...


# =============================================================================
# Agent Node Runner
# =============================================================================


class AgentNodeRunner(BaseNodeRunner):
    """Runner for agent nodes that use LLM-powered sub-agents.

    Agent nodes spawn sub-agents with specific roles (researcher, planner,
    executor, reviewer, tester) to accomplish tasks using tools.

    Example:
        runner = AgentNodeRunner(sub_agents=my_sub_agent_manager)
        context, result = await runner.execute(
            "analyze_code",
            {"role": "researcher", "goal": "Analyze codebase", "tool_budget": 10},
            context,
        )
    """

    def __init__(self, sub_agents: Optional["SubAgentManager"] = None):
        """Initialize the agent runner.

        Args:
            sub_agents: SubAgentManager for spawning agents.
        """
        super().__init__(["agent"])
        self._sub_agents = sub_agents

    def set_sub_agents(self, sub_agents: "SubAgentManager") -> None:
        """Set the sub-agent manager (for deferred initialization)."""
        self._sub_agents = sub_agents

    async def _execute_impl(
        self,
        node_id: str,
        node_config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Any]:
        """Execute an agent node via sub-agent spawning."""
        if self._sub_agents is None:
            raise RuntimeError("SubAgentManager not configured for AgentNodeRunner")

        from victor.agent.subagents import SubAgentRole

        # Extract configuration
        role_str = node_config.get("role", "executor").lower()
        goal = node_config.get("goal", "")
        tool_budget = node_config.get("tool_budget", 10)
        allowed_tools = node_config.get("allowed_tools")
        output_key = node_config.get("output_key", node_id)

        # Map role string to enum
        role_map = {
            "researcher": SubAgentRole.RESEARCHER,
            "planner": SubAgentRole.PLANNER,
            "executor": SubAgentRole.EXECUTOR,
            "reviewer": SubAgentRole.REVIEWER,
            "tester": SubAgentRole.TESTER,
        }
        role = role_map.get(role_str, SubAgentRole.EXECUTOR)

        # Build task from goal and context data
        data = context.get("data", {})
        task = self._build_task(goal, data, node_config)

        # Spawn and execute agent
        result = await self._sub_agents.spawn(
            role=role,
            task=task,
            tool_budget=tool_budget,
            allowed_tools=allowed_tools,
        )

        # Store output in context data
        if result.success and result.summary:
            if "data" not in context:
                context["data"] = {}
            context["data"][output_key] = result.summary

        return context, result.summary if result.success else None

    def _build_task(
        self,
        goal: str,
        data: Dict[str, Any],
        config: Dict[str, Any],
    ) -> str:
        """Build task description from goal and context."""
        parts = [goal]

        # Add context data references
        context_keys = config.get("context_keys", [])
        for key in context_keys:
            if key in data:
                parts.append(f"\n{key}: {data[key]}")

        return "\n".join(parts)


# =============================================================================
# Compute Node Runner
# =============================================================================


class ComputeNodeRunner(BaseNodeRunner):
    """Runner for compute nodes that execute without LLM.

    Compute nodes use registered handlers or direct tool execution
    with constraint enforcement. They're ideal for deterministic
    operations like data processing, file I/O, and API calls.

    Example:
        runner = ComputeNodeRunner(tool_registry=my_tools)
        context, result = await runner.execute(
            "load_data",
            {"handler": "data_loader", "inputs": {"path": "/data/file.csv"}},
            context,
        )
    """

    def __init__(
        self,
        tool_registry: Optional["ToolRegistry"] = None,
        handler_registry: Optional[Dict[str, Callable]] = None,
    ):
        """Initialize the compute runner.

        Args:
            tool_registry: Registry for tool execution.
            handler_registry: Registry of compute handlers by name.
        """
        super().__init__(["compute"])
        self._tool_registry = tool_registry
        self._handler_registry = handler_registry or {}

    def set_tool_registry(self, registry: "ToolRegistry") -> None:
        """Set the tool registry (for deferred initialization)."""
        self._tool_registry = registry

    def register_handler(self, name: str, handler: Callable) -> None:
        """Register a compute handler.

        Args:
            name: Handler name to register.
            handler: Async callable that executes the computation.
        """
        self._handler_registry[name] = handler

    async def _execute_impl(
        self,
        node_id: str,
        node_config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Any]:
        """Execute a compute node via handler or tools."""
        handler_name = node_config.get("handler")
        output_key = node_config.get("output_key", node_id)

        # Try custom handler first
        if handler_name:
            handler = self._handler_registry.get(handler_name)
            if handler is None:
                # Try global handler registry
                from victor.workflows.compute_registry import get_compute_handler

                handler = get_compute_handler(handler_name)

            if handler:
                # Prepare inputs
                inputs = self._resolve_inputs(node_config.get("inputs", {}), context)
                result = await handler(inputs, context)

                # Store output
                if "data" not in context:
                    context["data"] = {}
                context["data"][output_key] = result

                return context, result
            else:
                raise ValueError(f"Handler '{handler_name}' not found")

        # Fall back to tool execution
        tools = node_config.get("tools", [])
        if tools and self._tool_registry:
            outputs = {}
            for tool_name in tools:
                tool = self._tool_registry.get_tool(tool_name)
                if tool:
                    params = self._resolve_inputs(
                        node_config.get("tool_params", {}).get(tool_name, {}),
                        context,
                    )
                    result = await tool.execute(**params)
                    outputs[tool_name] = result

            if "data" not in context:
                context["data"] = {}
            context["data"][output_key] = outputs

            return context, outputs

        raise ValueError(f"No handler or tools configured for compute node {node_id}")

    def _resolve_inputs(
        self,
        inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resolve input references to actual values.

        Supports $ctx.key and $data.key syntax for context references.
        """
        resolved = {}
        data = context.get("data", {})

        for key, value in inputs.items():
            if isinstance(value, str):
                if value.startswith("$ctx."):
                    ref_key = value[5:]
                    resolved[key] = context.get(ref_key)
                elif value.startswith("$data."):
                    ref_key = value[6:]
                    resolved[key] = data.get(ref_key)
                else:
                    resolved[key] = value
            else:
                resolved[key] = value

        return resolved


# =============================================================================
# Transform Node Runner
# =============================================================================


class TransformNodeRunner(BaseNodeRunner):
    """Runner for transform nodes that modify context data.

    Transform nodes apply a transformation function to context data,
    useful for data mapping, filtering, and aggregation.

    Example:
        runner = TransformNodeRunner()
        context, result = await runner.execute(
            "normalize",
            {"transform": my_transform_fn},
            context,
        )
    """

    def __init__(self, transform_registry: Optional[Dict[str, Callable]] = None):
        """Initialize the transform runner.

        Args:
            transform_registry: Registry of transform functions by name.
        """
        super().__init__(["transform"])
        self._transform_registry = transform_registry or {}

    def register_transform(self, name: str, transform: Callable) -> None:
        """Register a transform function."""
        self._transform_registry[name] = transform

    async def _execute_impl(
        self,
        node_id: str,
        node_config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Any]:
        """Execute a transform on context data."""
        transform = node_config.get("transform")
        transform_name = node_config.get("transform_name")

        # Get transform function
        if callable(transform):
            transform_fn = transform
        elif transform_name:
            transform_fn = self._transform_registry.get(transform_name)
            if transform_fn is None:
                raise ValueError(f"Transform '{transform_name}' not found")
        else:
            raise ValueError(f"No transform specified for node {node_id}")

        # Apply transform to data
        data = context.get("data", {})

        if asyncio.iscoroutinefunction(transform_fn):
            new_data = await transform_fn(data)
        else:
            new_data = transform_fn(data)

        # Update context data
        if isinstance(new_data, dict):
            context["data"] = {**data, **new_data}
        else:
            context["data"] = new_data

        return context, new_data


# =============================================================================
# HITL Node Runner
# =============================================================================


class HITLNodeRunner(BaseNodeRunner):
    """Runner for human-in-the-loop nodes.

    HITL nodes pause execution for human approval, review, or input.
    They support various interaction types and fallback behaviors.

    Example:
        runner = HITLNodeRunner(hitl_executor=my_executor)
        context, result = await runner.execute(
            "approve_deploy",
            {
                "hitl_type": "approval",
                "prompt": "Approve deployment to production?",
                "timeout": 300,
            },
            context,
        )
    """

    def __init__(self, hitl_executor: Optional["HITLExecutor"] = None):
        """Initialize the HITL runner.

        Args:
            hitl_executor: Executor for HITL interactions.
        """
        super().__init__(["hitl", "human", "approval", "review"])
        self._hitl_executor = hitl_executor

    def set_executor(self, executor: "HITLExecutor") -> None:
        """Set the HITL executor (for deferred initialization)."""
        self._hitl_executor = executor

    async def _execute_impl(
        self,
        node_id: str,
        node_config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Any]:
        """Execute a HITL node for human interaction."""
        if self._hitl_executor is None:
            raise RuntimeError("HITLExecutor not configured for HITLNodeRunner")

        from victor.workflows.hitl import HITLNode, HITLNodeType, HITLFallback

        # Build HITLNode from config
        hitl_type_str = node_config.get("hitl_type", "approval").upper()
        fallback_str = node_config.get("fallback", "abort").upper()

        node = HITLNode(
            id=node_id,
            name=node_config.get("name", node_id),
            hitl_type=HITLNodeType[hitl_type_str],
            prompt=node_config.get("prompt", ""),
            context_keys=node_config.get("context_keys", []),
            timeout=node_config.get("timeout", 300.0),
            fallback=HITLFallback[fallback_str],
            default_value=node_config.get("default_value"),
        )

        # Execute HITL interaction
        response = await self._hitl_executor.execute_hitl_node(node, context.get("data", {}))

        # Update context with response
        output_key = node_config.get("output_key", f"{node_id}_response")
        if "data" not in context:
            context["data"] = {}
        context["data"][output_key] = {
            "approved": response.approved,
            "value": response.value,
            "reason": response.reason,
            "status": response.status.value,
        }

        # Mark HITL state in context
        context["_hitl_pending"] = False
        context["_hitl_response"] = context["data"][output_key]

        if not response.approved and fallback_str == "ABORT":
            raise RuntimeError(f"HITL node '{node_id}' rejected: {response.reason}")

        return context, context["data"][output_key]


# =============================================================================
# Condition Node Runner
# =============================================================================


class ConditionNodeRunner(BaseNodeRunner):
    """Runner for condition nodes that evaluate routing logic.

    Condition nodes don't modify data but determine the next execution
    path based on evaluating context data.

    Example:
        runner = ConditionNodeRunner()
        context, result = await runner.execute(
            "check_quality",
            {"condition": lambda ctx: "pass" if ctx["score"] > 0.8 else "fail"},
            context,
        )
    """

    def __init__(self, condition_registry: Optional[Dict[str, Callable]] = None):
        """Initialize the condition runner.

        Args:
            condition_registry: Registry of condition functions by name.
        """
        super().__init__(["condition", "branch", "router"])
        self._condition_registry = condition_registry or {}

    def register_condition(self, name: str, condition: Callable) -> None:
        """Register a condition function."""
        self._condition_registry[name] = condition

    async def _execute_impl(
        self,
        node_id: str,
        node_config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Any]:
        """Evaluate condition and return branch name."""
        condition = node_config.get("condition")
        condition_name = node_config.get("condition_name")

        # Get condition function
        if callable(condition):
            condition_fn = condition
        elif condition_name:
            condition_fn = self._condition_registry.get(condition_name)
            if condition_fn is None:
                raise ValueError(f"Condition '{condition_name}' not found")
        else:
            raise ValueError(f"No condition specified for node {node_id}")

        # Evaluate condition
        data = context.get("data", {})

        if asyncio.iscoroutinefunction(condition_fn):
            branch = await condition_fn(data)
        else:
            branch = condition_fn(data)

        # Store branch decision
        if "_node_results" not in context:
            context["_node_results"] = {}
        context["_node_results"][node_id] = {
            "branch": branch,
            "success": True,
        }

        return context, branch


# =============================================================================
# Parallel Node Runner
# =============================================================================


class ParallelNodeRunner(BaseNodeRunner):
    """Runner for parallel node execution.

    Parallel nodes execute multiple child nodes concurrently,
    collecting their results when all complete.

    Example:
        runner = ParallelNodeRunner(child_runners={"agent": agent_runner})
        context, result = await runner.execute(
            "parallel_analysis",
            {
                "children": [
                    {"id": "analyze_a", "type": "agent", ...},
                    {"id": "analyze_b", "type": "agent", ...},
                ],
                "max_concurrency": 3,
            },
            context,
        )
    """

    def __init__(
        self,
        child_runners: Optional[Dict[str, "NodeRunner"]] = None,
        max_concurrency: int = 5,
    ):
        """Initialize the parallel runner.

        Args:
            child_runners: Runners for child node types.
            max_concurrency: Maximum concurrent child executions.
        """
        super().__init__(["parallel", "fan_out"])
        self._child_runners = child_runners or {}
        self._max_concurrency = max_concurrency

    def register_runner(self, node_type: str, runner: "NodeRunner") -> None:
        """Register a runner for child nodes."""
        self._child_runners[node_type] = runner

    async def _execute_impl(
        self,
        node_id: str,
        node_config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Any]:
        """Execute child nodes in parallel."""
        children = node_config.get("children", [])
        max_concurrency = node_config.get("max_concurrency", self._max_concurrency)

        if not children:
            return context, {}

        semaphore = asyncio.Semaphore(max_concurrency)
        results: Dict[str, Any] = {}

        async def execute_child(child_config: Dict[str, Any]) -> None:
            async with semaphore:
                child_id = child_config.get("id", "")
                child_type = child_config.get("type", "")

                runner = self._child_runners.get(child_type)
                if runner is None:
                    results[child_id] = {"error": f"No runner for type '{child_type}'"}
                    return

                _, result = await runner.execute(child_id, child_config, context.copy())
                results[child_id] = {
                    "success": result.success,
                    "output": result.output,
                    "error": result.error,
                }

        # Execute all children concurrently
        await asyncio.gather(*[execute_child(c) for c in children])

        # Store parallel results
        context["_parallel_results"] = results

        output_key = node_config.get("output_key", f"{node_id}_results")
        if "data" not in context:
            context["data"] = {}
        context["data"][output_key] = results

        return context, results


# =============================================================================
# Node Runner Registry
# =============================================================================


class NodeRunnerRegistry:
    """Registry for looking up node runners by type.

    Provides a centralized way to manage and access node runners,
    supporting the DIP by allowing executors to depend on the registry
    rather than concrete runner implementations.

    Example:
        registry = NodeRunnerRegistry()
        registry.register(AgentNodeRunner(sub_agents=mgr))
        registry.register(ComputeNodeRunner(tools=reg))

        runner = registry.get_runner("agent")
        if runner:
            context, result = await runner.execute(...)
    """

    def __init__(self):
        """Initialize an empty registry."""
        self._runners: List[NodeRunner] = []

    def register(self, runner: NodeRunner) -> "NodeRunnerRegistry":
        """Register a node runner.

        Args:
            runner: Runner implementing NodeRunner protocol.

        Returns:
            Self for fluent interface.
        """
        self._runners.append(runner)
        return self

    def get_runner(self, node_type: str) -> Optional[NodeRunner]:
        """Get a runner that supports the given node type.

        Args:
            node_type: Type of node to find runner for.

        Returns:
            NodeRunner if found, None otherwise.
        """
        for runner in self._runners:
            if runner.supports_node_type(node_type):
                return runner
        return None

    def get_all_runners(self) -> List[NodeRunner]:
        """Get all registered runners."""
        return list(self._runners)

    @classmethod
    def create_default(
        cls,
        sub_agents: Optional["SubAgentManager"] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        hitl_executor: Optional["HITLExecutor"] = None,
    ) -> "NodeRunnerRegistry":
        """Create a registry with default runners.

        Args:
            sub_agents: SubAgentManager for agent nodes.
            tool_registry: ToolRegistry for compute nodes.
            hitl_executor: HITLExecutor for HITL nodes.

        Returns:
            Registry configured with standard runners.
        """
        registry = cls()

        agent_runner = AgentNodeRunner(sub_agents)
        compute_runner = ComputeNodeRunner(tool_registry)
        transform_runner = TransformNodeRunner()
        hitl_runner = HITLNodeRunner(hitl_executor)
        condition_runner = ConditionNodeRunner()
        parallel_runner = ParallelNodeRunner()

        # Register child runners for parallel execution
        parallel_runner.register_runner("agent", agent_runner)
        parallel_runner.register_runner("compute", compute_runner)
        parallel_runner.register_runner("transform", transform_runner)

        registry.register(agent_runner)
        registry.register(compute_runner)
        registry.register(transform_runner)
        registry.register(hitl_runner)
        registry.register(condition_runner)
        registry.register(parallel_runner)

        return registry


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base
    "BaseNodeRunner",
    # Runners
    "AgentNodeRunner",
    "ComputeNodeRunner",
    "TransformNodeRunner",
    "HITLNodeRunner",
    "ConditionNodeRunner",
    "ParallelNodeRunner",
    # Registry
    "NodeRunnerRegistry",
]
