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

"""Workflow definition DSL.

Provides a LangGraph-like fluent API for defining multi-agent workflows
as directed acyclic graphs (DAGs).

Architecture Overview:
    See docs/archive/internal/workflow-architecture.md for diagrams and details.

    Two primary execution node types:
    - AgentNode: LLM-powered execution for complex reasoning tasks
    - ComputeNode: Direct tool execution for deterministic pipelines

    Key difference:
    - AgentNode = Brain + Hands (LLM decides, then executes)
    - ComputeNode = Just Hands (predefined tool sequence)

Example (Python builder):
    @workflow("code_review", "Comprehensive code review")
    def code_review_workflow():
        return (
            WorkflowBuilder("code_review")
            .add_agent("analyze", "researcher", "Find code patterns")
            .add_condition("decide", lambda ctx: "fix" if ctx.get("issues") else "done",
                          {"fix": "fixer", "done": "report"})
            .add_agent("fixer", "executor", "Fix issues", next_nodes=["report"])
            .add_agent("report", "planner", "Summarize findings")
            .build()
        )

Example (YAML):
    nodes:
      - id: fetch_data
        type: compute
        tools: [sec_filing, market_data]
        constraints:
          llm_allowed: false
          max_cost_tier: FREE

      - id: analyze
        type: agent
        role: researcher
        goal: "Analyze the data and provide insights"
        tool_budget: 20
"""

from __future__ import annotations

import functools
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.subagents import SubAgentRole
    from victor.workflows.protocols import RetryPolicy
    from victor.tools.write_path_policy import WritePathPolicy

logger = logging.getLogger(__name__)


class WorkflowNodeType(Enum):
    """Type of workflow definition node.

    Renamed from NodeType to be semantically distinct:
    - WorkflowNodeType (here): Workflow definition nodes (COMPUTE, HITL, START, END)
    - GraphNodeType (victor.workflows.graph_dsl): Graph DSL nodes (FUNCTION, CONDITIONAL, SUBGRAPH)
    - YAMLNodeType (victor.workflows.yaml_loader): YAML loader validation nodes
    """

    AGENT = "agent"  # Spawns an agent to execute
    COMPUTE = "compute"  # Direct tool execution without LLM (extensible)
    CONDITION = "condition"  # Branch based on condition
    PARALLEL = "parallel"  # Execute multiple nodes in parallel
    TRANSFORM = "transform"  # Transform context data
    HITL = "hitl"  # Human-in-the-loop interrupt
    TEAM = "team"  # Spawn ad-hoc multi-agent team (Phase 3.1)
    START = "start"  # Entry point
    END = "end"  # Terminal node


@dataclass
class WorkflowNode(ABC):
    """Base class for workflow nodes.

    Attributes:
        id: Unique identifier for this node
        name: Human-readable name
        next_nodes: IDs of nodes to execute after this one
        retry_policy: Optional retry policy for this node
        circuit_breaker_enabled: Whether to enable circuit breaker for this node
    """

    id: str
    name: str
    next_nodes: List[str] = field(default_factory=list)
    retry_policy: Optional["RetryPolicy"] = None
    circuit_breaker_enabled: bool = False

    @property
    @abstractmethod
    def node_type(self) -> WorkflowNodeType:
        """The type of this node."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary."""
        result = {
            "id": self.id,
            "name": self.name,
            "type": self.node_type.value,
            "next_nodes": self.next_nodes,
        }
        if self.retry_policy:
            result["retry_policy"] = {
                "max_retries": self.retry_policy.max_retries,
                "delay_seconds": self.retry_policy.delay_seconds,
                "exponential_backoff": self.retry_policy.exponential_backoff,
            }
        if self.circuit_breaker_enabled:
            result["circuit_breaker_enabled"] = True
        return result


@dataclass
class AgentNode(WorkflowNode):
    """Node that spawns an agent to execute a task.

    Attributes:
        role: Agent role (researcher, executor, etc.)
        goal: Task description for the agent
        tool_budget: Maximum tool calls allowed
        allowed_tools: Specific tools to allow (None = role defaults)
        input_mapping: Map context keys to agent inputs
        output_key: Key to store agent output in context
        llm_config: Optional LLM configuration (temperature, model_hint, etc.)
        timeout_seconds: Maximum execution time in seconds (None = no timeout)
        profile: Optional provider profile for this node (None = use workflow default)
        disable_embeddings: Disable codebase embeddings for this agent (default: False).
            When True, tools like code_search will use keyword search instead of semantic search.
            Useful for workflow agents that use shared services to avoid duplicate index loading.
    """

    role: str = "executor"
    goal: str = ""
    tool_budget: int = 15
    allowed_tools: Optional[List[str]] = None
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_key: Optional[str] = None
    llm_config: Optional[Dict[str, Any]] = None
    timeout_seconds: Optional[float] = None
    profile: Optional[str] = None
    disable_embeddings: bool = False

    @property
    def node_type(self) -> WorkflowNodeType:
        return WorkflowNodeType.AGENT

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "role": self.role,
                "goal": self.goal,
                "tool_budget": self.tool_budget,
                "allowed_tools": self.allowed_tools,
                "input_mapping": self.input_mapping,
                "output_key": self.output_key,
                "llm_config": self.llm_config,
                "timeout_seconds": self.timeout_seconds,
                "profile": self.profile,
                "disable_embeddings": self.disable_embeddings,
            }
        )
        return d


@dataclass
class ConditionNode(WorkflowNode):
    """Node that branches based on a condition.

    Attributes:
        condition: Function that takes context and returns branch name
        branches: Mapping from branch names to node IDs
    """

    condition: Callable[[Dict[str, Any]], str] = field(default=lambda ctx: "default")
    branches: Dict[str, str] = field(default_factory=dict)

    @property
    def node_type(self) -> WorkflowNodeType:
        return WorkflowNodeType.CONDITION

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "branches": self.branches,
                # Note: condition function is not serialized
            }
        )
        return d

    def evaluate(self, context: Dict[str, Any]) -> Optional[str]:
        """Evaluate condition and return next node ID."""
        branch = self.condition(context)
        return self.branches.get(branch)


@dataclass
class ParallelNode(WorkflowNode):
    """Node that executes multiple nodes in parallel.

    Attributes:
        parallel_nodes: IDs of nodes to execute in parallel
        join_strategy: How to combine results (all, any, merge)
    """

    parallel_nodes: List[str] = field(default_factory=list)
    join_strategy: str = "all"  # all, any, merge

    @property
    def node_type(self) -> WorkflowNodeType:
        return WorkflowNodeType.PARALLEL

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "parallel_nodes": self.parallel_nodes,
                "join_strategy": self.join_strategy,
            }
        )
        return d


@dataclass
class TransformNode(WorkflowNode):
    """Node that transforms context data.

    Attributes:
        transform: Function that transforms context
    """

    transform: Callable[[Dict[str, Any]], Dict[str, Any]] = field(default=lambda ctx: ctx)

    @property
    def node_type(self) -> WorkflowNodeType:
        return WorkflowNodeType.TRANSFORM


@dataclass
class TeamStepWorkflow(WorkflowNode):
    """Workflow-definition step that invokes an ad-hoc multi-agent team.

    This step type enables hybrid orchestration by invoking teams within
    workflow graphs. Teams use the ``victor.teams`` runtime infrastructure.

    Attributes:
        goal: Overall goal for the team
        team_formation: How to organize the team (sequential, parallel, etc.)
        members: List of team member configurations
        timeout_seconds: Maximum execution time (None = no limit)
        merge_strategy: How to merge team state ("dict", "list", "custom")
        merge_mode: Conflict resolution mode ("team_wins", "graph_wins", etc.)
        output_key: Key to store team result
        continue_on_error: Whether to continue if team fails
        shared_context: Initial context for team
        max_iterations: Maximum team iterations
        total_tool_budget: Total tool calls across all members
    """

    goal: str = ""
    team_formation: str = "sequential"
    members: List[Dict[str, Any]] = field(default_factory=list)
    timeout_seconds: Optional[float] = None
    merge_strategy: str = "dict"
    merge_mode: str = "team_wins"
    output_key: str = "team_result"
    continue_on_error: bool = True
    shared_context: Dict[str, Any] = field(default_factory=dict)
    max_iterations: int = 50
    total_tool_budget: int = 100

    @property
    def node_type(self) -> WorkflowNodeType:
        return WorkflowNodeType.TEAM

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary."""
        d = super().to_dict()
        d.update(
            {
                "goal": self.goal,
                "team_formation": self.team_formation,
                "members": self.members,
                "timeout_seconds": self.timeout_seconds,
                "merge_strategy": self.merge_strategy,
                "merge_mode": self.merge_mode,
                "output_key": self.output_key,
                "continue_on_error": self.continue_on_error,
                "shared_context": self.shared_context,
                "max_iterations": self.max_iterations,
                "total_tool_budget": self.total_tool_budget,
            }
        )
        return d


_DEPRECATED_ALIAS_MAP = {
    "TeamNodeWorkflow": TeamStepWorkflow,
}
_TEAM_NODE_REMOVAL_VERSION = "v0.9.0"
_TEAM_NODE_REMOVAL_DATE = "2027-03-31"


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_ALIAS_MAP:
        warnings.warn(
            f"{name} is deprecated; use "
            f"{_DEPRECATED_ALIAS_MAP[name].__name__} instead. "
            f"This compatibility alias remains supported through "
            f"{_TEAM_NODE_REMOVAL_VERSION} ({_TEAM_NODE_REMOVAL_DATE}) and "
            "will be removed after that milestone. "
            "See docs/architecture/migration.md for migration guidance.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEPRECATED_ALIAS_MAP[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_DEPRECATED_ALIAS_MAP))


class ConstraintsProtocol(ABC):
    """Protocol for task execution constraints.

    Extend this protocol to create domain-specific constraints.
    The framework provides TaskConstraints as a standard implementation.

    Example custom constraints:
        @dataclass
        class InvestmentConstraints(ConstraintsProtocol):
            max_api_calls_per_minute: int = 10
            require_sec_compliance: bool = True

            def allows_tool(self, tool_name: str, tool_cost_tier: str = "FREE") -> bool:
                # Custom validation logic
                ...
    """

    @abstractmethod
    def allows_tool(self, tool_name: str, tool_cost_tier: str = "FREE") -> bool:
        """Check if a tool is allowed under these constraints."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize constraints to dictionary."""
        pass

    @property
    @abstractmethod
    def timeout(self) -> float:
        """Maximum execution time in seconds."""
        pass

    @property
    @abstractmethod
    def max_tool_calls(self) -> int:
        """Maximum number of tool invocations."""
        pass


@dataclass
class TaskConstraints(ConstraintsProtocol):
    """Standard execution constraints for compute nodes.

    Defines what resources and capabilities a task can use.
    This enables airgapped execution, cost control, and security.

    Extend this class or implement ConstraintsProtocol for domain-specific needs.

    Attributes:
        llm_allowed: Whether LLM inference is permitted (default: False)
        network_allowed: Whether network calls are permitted (default: True)
        write_allowed: Whether disk writes are permitted (default: False)
        max_cost_tier: Maximum tool cost tier allowed (default: "FREE")
        max_tool_calls: Maximum number of tool invocations (default: 100)
        timeout: Maximum execution time in seconds (default: 300)
        allowed_tools: Explicit list of allowed tools (None = all)
        blocked_tools: Explicit list of blocked tools (None = none)
        write_path_policy: Granular path-tier write policy (takes precedence over write_allowed
            when set). Use WritePathPolicy.analysis_safe() to allow ./tmp/, ./victor/analysis/,
            and ./docs/*.md without enabling full source-code writes.

    Example YAML:
        constraints:
          llm_allowed: false
          network_allowed: true
          max_cost_tier: LOW
          timeout: 60

    Example extension:
        @dataclass
        class StrictConstraints(TaskConstraints):
            require_approval: bool = True
    """

    llm_allowed: bool = False
    network_allowed: bool = True
    write_allowed: bool = False
    max_cost_tier: str = "FREE"  # FREE, LOW, MEDIUM, HIGH
    _max_tool_calls: int = 100
    _timeout: float = 300.0
    allowed_tools: Optional[List[str]] = None
    blocked_tools: Optional[List[str]] = None
    write_path_policy: Optional["WritePathPolicy"] = None

    @property
    def timeout(self) -> float:
        return self._timeout

    @property
    def max_tool_calls(self) -> int:
        return self._max_tool_calls

    def allows_tool(self, tool_name: str, tool_cost_tier: str = "FREE") -> bool:
        """Check if a tool is allowed under these constraints."""
        # Check explicit blocklist
        if self.blocked_tools and tool_name in self.blocked_tools:
            return False

        # Check explicit allowlist
        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            return False

        # Check cost tier
        tier_order = {"FREE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
        max_tier = tier_order.get(self.max_cost_tier.upper(), 0)
        tool_tier = tier_order.get(tool_cost_tier.upper(), 0)
        if tool_tier > max_tier:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "llm_allowed": self.llm_allowed,
            "network_allowed": self.network_allowed,
            "write_allowed": self.write_allowed,
            "max_cost_tier": self.max_cost_tier,
            "max_tool_calls": self._max_tool_calls,
            "timeout": self._timeout,
            "allowed_tools": self.allowed_tools,
            "blocked_tools": self.blocked_tools,
        }
        if self.write_path_policy is not None:
            d["write_path_policy_tier"] = self.write_path_policy.max_tier.name
        return d


# Standard constraint presets
class AirgappedConstraints(TaskConstraints):
    """Constraints for airgapped/offline execution."""

    def __init__(self, **kwargs):
        super().__init__(
            llm_allowed=False,
            network_allowed=False,
            write_allowed=False,
            max_cost_tier="FREE",
            **kwargs,
        )


class ComputeOnlyConstraints(TaskConstraints):
    """Constraints for pure computation (no LLM, no writes)."""

    def __init__(self, **kwargs):
        super().__init__(
            llm_allowed=False,
            network_allowed=True,
            write_allowed=False,
            max_cost_tier="LOW",
            **kwargs,
        )


class FullAccessConstraints(TaskConstraints):
    """Constraints with full access (use carefully)."""

    def __init__(self, **kwargs):
        super().__init__(
            llm_allowed=True,
            network_allowed=True,
            write_allowed=True,
            max_cost_tier="HIGH",
            **kwargs,
        )


@dataclass
class ComputeNode(WorkflowNode):
    """Node that executes tools with configurable constraints.

    This node type enables LLM-free execution and fine-grained control
    over what resources a task can access. Ideal for:
    - Pure computation workflows (valuation, data processing)
    - Airgapped or security-sensitive environments
    - Cost-controlled execution

    Attributes:
        tools: List of tool names to execute in sequence
        input_mapping: Map context keys to tool parameters
        output_key: Key to store combined tool outputs in context
        constraints: Execution constraints (LLM, network, cost tier, etc.)
        handler: Optional custom handler name for domain-specific logic
        fail_fast: Stop on first tool failure (default: True)
        parallel: Execute tools in parallel (default: False)

    Example YAML:
        - id: run_valuation
          type: compute
          tools: [multi_model_valuation, sector_valuation]
          inputs:
            symbol: $ctx.symbol
            financials: $ctx.sec_data
          output: fair_values
          constraints:
            llm_allowed: false
            max_cost_tier: FREE
            timeout: 60
          next: [blend]

        # With custom handler for domain-specific logic
        - id: rl_weights
          type: compute
          handler: rl_decision  # Domain-specific handler
          inputs:
            features: $ctx.valuation_features
          output: model_weights
          constraints:
            llm_allowed: false
    """

    tools: List[str] = field(default_factory=list)
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_key: Optional[str] = None
    constraints: TaskConstraints = field(default_factory=TaskConstraints)
    handler: Optional[str] = None  # Custom handler name for extensibility
    fail_fast: bool = True
    parallel: bool = False
    # Execution target: "in-process", "subprocess", "docker"
    execution_target: str = "in-process"

    @property
    def node_type(self) -> WorkflowNodeType:
        return WorkflowNodeType.COMPUTE

    @property
    def timeout(self) -> float:
        """Convenience property for timeout from constraints."""
        return self.constraints.timeout

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "tools": self.tools,
                "input_mapping": self.input_mapping,
                "output_key": self.output_key,
                "constraints": self.constraints.to_dict(),
                "handler": self.handler,
                "fail_fast": self.fail_fast,
                "parallel": self.parallel,
                "execution_target": self.execution_target,
            }
        )
        return d


@dataclass
class WorkflowDefinition:
    """A complete workflow definition.

    Represents a DAG of nodes that define a multi-agent workflow.

    Attributes:
        name: Workflow name (used as identifier)
        description: Human-readable description
        nodes: All nodes in the workflow
        start_node: ID of the entry point node
        metadata: Additional workflow metadata
        max_execution_timeout_seconds: Overall workflow timeout (None = no limit)
        default_node_timeout_seconds: Default timeout for nodes (None = no default)
        max_iterations: Maximum workflow iterations/cycles (default: 25)
        max_retries: Maximum retries for the entire workflow (default: 0)
    """

    name: str
    description: str = ""
    nodes: Dict[str, WorkflowNode] = field(default_factory=dict)
    start_node: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_execution_timeout_seconds: Optional[float] = None
    default_node_timeout_seconds: Optional[float] = None
    max_iterations: int = 25
    max_retries: int = 0

    def __post_init__(self):
        """Validate workflow structure."""
        if self.nodes and not self.start_node:
            # Find first node added as start
            self.start_node = next(iter(self.nodes.keys()))

    def get_node(self, node_id: str) -> Optional[WorkflowNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def get_next_nodes(self, node_id: str) -> List[WorkflowNode]:
        """Get nodes that follow the given node."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[nid] for nid in node.next_nodes if nid in self.nodes]

    def validate(self) -> List[str]:
        """Validate workflow and return list of errors."""
        errors = []

        if not self.name:
            errors.append("Workflow must have a name")

        if not self.nodes:
            errors.append("Workflow must have at least one node")
            return errors

        if not self.start_node:
            errors.append("Workflow must have a start node")
        elif self.start_node not in self.nodes:
            errors.append(f"Start node '{self.start_node}' not found")

        # Check for unreachable nodes
        reachable = self._find_reachable_nodes()
        for node_id in self.nodes:
            if node_id not in reachable and node_id != self.start_node:
                errors.append(f"Node '{node_id}' is unreachable")

        # Check for broken references
        for node in self.nodes.values():
            for next_id in node.next_nodes:
                if next_id not in self.nodes:
                    errors.append(f"Node '{node.id}' references non-existent node '{next_id}'")

            if isinstance(node, ConditionNode):
                for branch, target in node.branches.items():
                    if target not in self.nodes:
                        errors.append(
                            f"Condition '{node.id}' branch '{branch}' "
                            f"references non-existent node '{target}'"
                        )

            if isinstance(node, ParallelNode):
                for parallel_id in node.parallel_nodes:
                    if parallel_id not in self.nodes:
                        errors.append(
                            f"Parallel '{node.id}' references non-existent " f"node '{parallel_id}'"
                        )

        return errors

    def _find_reachable_nodes(self) -> Set[str]:
        """Find all nodes reachable from start."""
        if not self.start_node:
            return set()

        reachable = set()
        to_visit = [self.start_node]

        while to_visit:
            node_id = to_visit.pop()
            if node_id in reachable:
                continue
            reachable.add(node_id)

            node = self.nodes.get(node_id)
            if not node:
                continue

            to_visit.extend(node.next_nodes)

            if isinstance(node, ConditionNode):
                to_visit.extend(node.branches.values())

            if isinstance(node, ParallelNode):
                to_visit.extend(node.parallel_nodes)

        return reachable

    def to_dict(self) -> Dict[str, Any]:
        """Serialize workflow to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "start_node": self.start_node,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "metadata": self.metadata,
            "max_execution_timeout_seconds": self.max_execution_timeout_seconds,
            "default_node_timeout_seconds": self.default_node_timeout_seconds,
            "max_iterations": self.max_iterations,
            "max_retries": self.max_retries,
        }

    def get_agent_count(self) -> int:
        """Count agent nodes in workflow."""
        return sum(1 for node in self.nodes.values() if isinstance(node, AgentNode))

    def get_total_budget(self) -> int:
        """Sum of all agent tool budgets."""
        return sum(node.tool_budget for node in self.nodes.values() if isinstance(node, AgentNode))


class WorkflowBuilder:
    """Fluent builder for creating workflow definitions.

    Provides a chainable API for constructing workflows:

    Example:
        workflow = (
            WorkflowBuilder("review")
            .add_agent("analyze", "researcher", "Find issues")
            .add_condition("decide", decide_func, {"fix": "fixer", "done": "end"})
            .add_agent("fixer", "executor", "Fix issues", next_nodes=["end"])
            .build()
        )
    """

    def __init__(self, name: str, description: str = ""):
        """Initialize builder.

        Args:
            name: Workflow name
            description: Optional description
        """
        self.name = name
        self.description = description
        self._nodes: Dict[str, WorkflowNode] = {}
        self._first_node: Optional[str] = None
        self._last_node: Optional[str] = None
        self._metadata: Dict[str, Any] = {}

    def add_agent(
        self,
        node_id: str,
        role: str,
        goal: str,
        *,
        name: Optional[str] = None,
        tool_budget: int = 15,
        allowed_tools: Optional[List[str]] = None,
        input_mapping: Optional[Dict[str, str]] = None,
        output_key: Optional[str] = None,
        next_nodes: Optional[List[str]] = None,
    ) -> "WorkflowBuilder":
        """Add an agent node.

        Args:
            node_id: Unique identifier
            role: Agent role (researcher, executor, etc.)
            goal: Task description
            name: Optional display name
            tool_budget: Max tool calls
            allowed_tools: Specific tools to allow
            input_mapping: Context key mappings
            output_key: Key for storing output
            next_nodes: Nodes to execute after

        Returns:
            Self for chaining
        """
        node = AgentNode(
            id=node_id,
            name=name or node_id,
            role=role,
            goal=goal,
            tool_budget=tool_budget,
            allowed_tools=allowed_tools,
            input_mapping=input_mapping or {},
            output_key=output_key or node_id,
            next_nodes=next_nodes or [],
        )
        return self._add_node(node)

    def add_condition(
        self,
        node_id: str,
        condition: Callable[[Dict[str, Any]], str],
        branches: Dict[str, str],
        *,
        name: Optional[str] = None,
    ) -> "WorkflowBuilder":
        """Add a condition node.

        Args:
            node_id: Unique identifier
            condition: Function taking context, returning branch name
            branches: Map from branch names to node IDs
            name: Optional display name

        Returns:
            Self for chaining
        """
        node = ConditionNode(
            id=node_id,
            name=name or node_id,
            condition=condition,
            branches=branches,
        )
        return self._add_node(node)

    def add_parallel(
        self,
        node_id: str,
        parallel_nodes: List[str],
        *,
        name: Optional[str] = None,
        join_strategy: str = "all",
        next_nodes: Optional[List[str]] = None,
    ) -> "WorkflowBuilder":
        """Add a parallel execution node.

        Args:
            node_id: Unique identifier
            parallel_nodes: Nodes to execute in parallel
            name: Optional display name
            join_strategy: How to combine results (all, any, merge)
            next_nodes: Nodes to execute after

        Returns:
            Self for chaining
        """
        node = ParallelNode(
            id=node_id,
            name=name or node_id,
            parallel_nodes=parallel_nodes,
            join_strategy=join_strategy,
            next_nodes=next_nodes or [],
        )
        return self._add_node(node)

    def add_transform(
        self,
        node_id: str,
        transform: Callable[[Dict[str, Any]], Dict[str, Any]],
        *,
        name: Optional[str] = None,
        next_nodes: Optional[List[str]] = None,
    ) -> "WorkflowBuilder":
        """Add a transform node.

        Args:
            node_id: Unique identifier
            transform: Function to transform context
            name: Optional display name
            next_nodes: Nodes to execute after

        Returns:
            Self for chaining
        """
        node = TransformNode(
            id=node_id,
            name=name or node_id,
            transform=transform,
            next_nodes=next_nodes or [],
        )
        return self._add_node(node)

    def add_compute(
        self,
        node_id: str,
        tools: Optional[List[str]] = None,
        *,
        name: Optional[str] = None,
        input_mapping: Optional[Dict[str, str]] = None,
        output_key: Optional[str] = None,
        handler: Optional[str] = None,
        constraints: Optional[TaskConstraints] = None,
        llm_allowed: bool = False,
        network_allowed: bool = True,
        max_cost_tier: str = "FREE",
        timeout: float = 60.0,
        fail_fast: bool = True,
        parallel: bool = False,
        next_nodes: Optional[List[str]] = None,
    ) -> "WorkflowBuilder":
        """Add a compute node for constrained task execution.

        Compute nodes execute tools with configurable constraints,
        enabling LLM-free execution, airgapped environments, and
        cost-controlled workflows.

        Args:
            node_id: Unique identifier
            tools: List of tool names to execute (optional if using handler)
            name: Optional display name
            input_mapping: Map context keys to tool parameters
            output_key: Key to store outputs in context
            handler: Custom handler name for domain-specific logic
            constraints: Full TaskConstraints object (overrides individual params)
            llm_allowed: Whether LLM inference is permitted
            network_allowed: Whether network calls are permitted
            max_cost_tier: Maximum tool cost tier (FREE, LOW, MEDIUM, HIGH)
            timeout: Timeout per tool in seconds
            fail_fast: Stop on first tool failure
            parallel: Execute tools in parallel
            next_nodes: Nodes to execute after

        Returns:
            Self for chaining

        Example:
            # Pure computation with tools
            workflow.add_compute(
                "valuation",
                tools=["multi_model_valuation", "sector_valuation"],
                input_mapping={"symbol": "ctx.symbol"},
                output_key="fair_values",
                llm_allowed=False,
                max_cost_tier="FREE",
            )

            # Custom handler for domain-specific logic
            workflow.add_compute(
                "rl_weights",
                handler="rl_decision",  # Domain-specific handler
                input_mapping={"features": "valuation_features"},
                output_key="model_weights",
            )
        """
        # Build constraints from individual params or use provided
        if constraints is None:
            constraints = TaskConstraints(
                llm_allowed=llm_allowed,
                network_allowed=network_allowed,
                max_cost_tier=max_cost_tier,
                _timeout=timeout,
            )

        node = ComputeNode(
            id=node_id,
            name=name or node_id,
            tools=tools or [],
            input_mapping=input_mapping or {},
            output_key=output_key or node_id,
            constraints=constraints,
            handler=handler,
            fail_fast=fail_fast,
            parallel=parallel,
            next_nodes=next_nodes or [],
        )
        return self._add_node(node)

    def add_hitl_approval(
        self,
        node_id: str,
        prompt: str,
        *,
        name: Optional[str] = None,
        context_keys: Optional[List[str]] = None,
        timeout: float = 300.0,
        fallback: str = "abort",
        next_nodes: Optional[List[str]] = None,
    ) -> "WorkflowBuilder":
        """Add a human approval gate.

        Pauses workflow execution until a human approves or rejects.

        Args:
            node_id: Unique identifier
            prompt: Message to display for approval
            name: Optional display name
            context_keys: Keys from context to include in display
            timeout: Timeout in seconds (default: 5 minutes)
            fallback: Behavior on timeout (abort, continue, skip)
            next_nodes: Nodes to execute after approval

        Returns:
            Self for chaining

        Example:
            workflow.add_hitl_approval(
                "approve_changes",
                prompt="Proceed with the following changes?",
                context_keys=["files_to_modify"],
                timeout=300.0,
            )
        """
        from victor.workflows.hitl import HITLNode, HITLNodeType, HITLFallback

        fallback_enum = HITLFallback(fallback) if fallback else HITLFallback.ABORT

        node = HITLNode(
            id=node_id,
            name=name or node_id,
            hitl_type=HITLNodeType.APPROVAL,
            prompt=prompt,
            context_keys=context_keys or [],
            timeout=timeout,
            fallback=fallback_enum,
            next_nodes=next_nodes or [],
        )
        return self._add_node(node)

    def add_hitl_choice(
        self,
        node_id: str,
        prompt: str,
        choices: List[str],
        *,
        name: Optional[str] = None,
        context_keys: Optional[List[str]] = None,
        default_value: Optional[str] = None,
        timeout: float = 300.0,
        fallback: str = "continue",
        next_nodes: Optional[List[str]] = None,
    ) -> "WorkflowBuilder":
        """Add a human choice selection node.

        Pauses workflow until human selects from options.

        Args:
            node_id: Unique identifier
            prompt: Message to display
            choices: Available options
            name: Optional display name
            context_keys: Keys from context to include in display
            default_value: Default choice if timeout
            timeout: Timeout in seconds
            fallback: Behavior on timeout
            next_nodes: Nodes to execute after

        Returns:
            Self for chaining
        """
        from victor.workflows.hitl import HITLNode, HITLNodeType, HITLFallback

        fallback_enum = HITLFallback(fallback) if fallback else HITLFallback.CONTINUE

        node = HITLNode(
            id=node_id,
            name=name or node_id,
            hitl_type=HITLNodeType.CHOICE,
            prompt=prompt,
            context_keys=context_keys or [],
            choices=choices,
            default_value=default_value,
            timeout=timeout,
            fallback=fallback_enum,
            next_nodes=next_nodes or [],
        )
        return self._add_node(node)

    def add_hitl_review(
        self,
        node_id: str,
        prompt: str,
        *,
        name: Optional[str] = None,
        context_keys: Optional[List[str]] = None,
        timeout: float = 600.0,
        fallback: str = "abort",
        next_nodes: Optional[List[str]] = None,
    ) -> "WorkflowBuilder":
        """Add a human review/modification node.

        Pauses workflow until human reviews and optionally modifies context.

        Args:
            node_id: Unique identifier
            prompt: Message to display
            name: Optional display name
            context_keys: Keys from context to include in review
            timeout: Timeout in seconds (default: 10 minutes)
            fallback: Behavior on timeout
            next_nodes: Nodes to execute after

        Returns:
            Self for chaining
        """
        from victor.workflows.hitl import HITLNode, HITLNodeType, HITLFallback

        fallback_enum = HITLFallback(fallback) if fallback else HITLFallback.ABORT

        node = HITLNode(
            id=node_id,
            name=name or node_id,
            hitl_type=HITLNodeType.REVIEW,
            prompt=prompt,
            context_keys=context_keys or [],
            timeout=timeout,
            fallback=fallback_enum,
            next_nodes=next_nodes or [],
        )
        return self._add_node(node)

    def add_hitl_confirmation(
        self,
        node_id: str,
        prompt: str = "Press Enter to continue",
        *,
        name: Optional[str] = None,
        timeout: float = 60.0,
        fallback: str = "continue",
        next_nodes: Optional[List[str]] = None,
    ) -> "WorkflowBuilder":
        """Add a simple confirmation gate.

        Pauses briefly for human to acknowledge before continuing.

        Args:
            node_id: Unique identifier
            prompt: Message to display
            name: Optional display name
            timeout: Timeout in seconds (default: 1 minute)
            fallback: Behavior on timeout (default: continue)
            next_nodes: Nodes to execute after

        Returns:
            Self for chaining
        """
        from victor.workflows.hitl import HITLNode, HITLNodeType, HITLFallback

        fallback_enum = HITLFallback(fallback) if fallback else HITLFallback.CONTINUE

        node = HITLNode(
            id=node_id,
            name=name or node_id,
            hitl_type=HITLNodeType.CONFIRMATION,
            prompt=prompt,
            timeout=timeout,
            fallback=fallback_enum,
            next_nodes=next_nodes or [],
        )
        return self._add_node(node)

    def set_metadata(self, key: str, value: Any) -> "WorkflowBuilder":
        """Set workflow metadata.

        Args:
            key: Metadata key
            value: Metadata value

        Returns:
            Self for chaining
        """
        self._metadata[key] = value
        return self

    def _add_node(self, node: WorkflowNode) -> "WorkflowBuilder":
        """Add a node to the workflow."""
        if node.id in self._nodes:
            raise ValueError(f"Node '{node.id}' already exists")

        # Auto-chain: connect previous node to this one
        if self._last_node and not self._nodes[self._last_node].next_nodes:
            # Only auto-chain if previous node has no explicit next_nodes
            # and is not a condition node (conditions have explicit branches)
            prev = self._nodes[self._last_node]
            if not isinstance(prev, ConditionNode):
                prev.next_nodes.append(node.id)

        self._nodes[node.id] = node
        if self._first_node is None:
            self._first_node = node.id
        self._last_node = node.id

        return self

    def chain(self, from_node: str, to_node: str) -> "WorkflowBuilder":
        """Explicitly chain one node to another.

        Args:
            from_node: Source node ID
            to_node: Target node ID

        Returns:
            Self for chaining
        """
        if from_node not in self._nodes:
            raise ValueError(f"Source node '{from_node}' not found")
        if to_node not in self._nodes:
            raise ValueError(f"Target node '{to_node}' not found")

        if to_node not in self._nodes[from_node].next_nodes:
            self._nodes[from_node].next_nodes.append(to_node)

        return self

    def build(self) -> WorkflowDefinition:
        """Build the workflow definition.

        Returns:
            Complete WorkflowDefinition

        Raises:
            ValueError: If workflow validation fails
        """
        workflow = WorkflowDefinition(
            name=self.name,
            description=self.description,
            nodes=self._nodes.copy(),
            start_node=self._first_node,
            metadata=self._metadata.copy(),
        )

        errors = workflow.validate()
        if errors:
            raise ValueError(f"Workflow validation failed: {'; '.join(errors)}")

        return workflow


# Registry for decorator-registered workflows
_workflow_registry: Dict[str, Callable[[], WorkflowDefinition]] = {}


def workflow(
    name: str,
    description: str = "",
) -> Callable[[Callable[[], WorkflowDefinition]], Callable[[], WorkflowDefinition]]:
    """Decorator to register a workflow factory function.

    The decorated function should return a WorkflowDefinition.

    Example:
        @workflow("code_review", "Review code quality")
        def code_review_workflow():
            return (
                WorkflowBuilder("code_review")
                .add_agent("analyze", "researcher", "Find issues")
                .build()
            )

    Args:
        name: Workflow name
        description: Optional description

    Returns:
        Decorator function
    """

    def decorator(
        func: Callable[[], WorkflowDefinition],
    ) -> Callable[[], WorkflowDefinition]:
        @functools.wraps(func)
        def wrapper() -> WorkflowDefinition:
            defn = func()
            # Override name/description from decorator
            defn.name = name
            if description:
                defn.description = description
            return defn

        _workflow_registry[name] = wrapper
        logger.debug(f"Registered workflow: {name}")
        return wrapper

    return decorator


def get_registered_workflows() -> Dict[str, Callable[[], WorkflowDefinition]]:
    """Get all registered workflow factories."""
    return _workflow_registry.copy()


if TYPE_CHECKING:
    TeamNodeWorkflow = TeamStepWorkflow


__all__ = [
    # Node types
    "WorkflowNodeType",
    "WorkflowNode",
    "AgentNode",
    "ComputeNode",
    "ConditionNode",
    "ParallelNode",
    "TransformNode",
    "TeamStepWorkflow",
    "TeamNodeWorkflow",
    # Constraints (extensible protocol + standard implementations)
    "ConstraintsProtocol",
    "TaskConstraints",
    "AirgappedConstraints",
    "ComputeOnlyConstraints",
    "FullAccessConstraints",
    # Workflow definition and builder
    "WorkflowDefinition",
    "WorkflowBuilder",
    "workflow",
    "get_registered_workflows",
]
