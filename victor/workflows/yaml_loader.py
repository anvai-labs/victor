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

"""YAML loader for declarative workflow definitions.

Enables defining workflows in YAML format for easier configuration and
version control without code changes.

Extended Schema Features:
- llm_config: Agent LLM settings (temperature, model_hint, max_tokens)
- $ref: External file references for node reuse
- batch_config: Workflow-level batch execution settings
- temporal_context: Point-in-time analysis for backtesting
- $env.VAR_NAME: Environment variable interpolation
- ${VAR:-default}: Shell-style env vars with defaults

Example YAML format:
    workflows:
      feature_implementation:
        description: "End-to-end feature development with review"

        metadata:
          version: "1.0"
          author: "team"
          vertical: dataanalysis

        # Batch execution settings
        batch_config:
          batch_size: 10
          max_concurrent: 5
          retry_strategy: end_of_batch

        # Point-in-time context for backtesting
        temporal_context:
          as_of_date: $ctx.analysis_date
          lookback_periods: 8
          period_type: quarters

        nodes:
          - id: research
            type: agent
            role: researcher
            goal: "Analyze codebase for relevant patterns"
            tool_budget: 20
            tools: [read, grep, code_search, overview]
            llm_config:
              temperature: 0.3
              model_hint: claude-3-sonnet
            output: research_findings
            next: [plan]

          - id: plan
            type: agent
            role: planner
            goal: "Create implementation plan"
            tool_budget: 10
            next: [decide]

          - id: decide
            type: condition
            condition: "has_tests"  # Simple key check
            branches:
              true: implement
              false: add_tests

          # External node reference
          - $ref: "./common_nodes.yaml#validation"

          - id: implement
            type: agent
            role: executor
            goal: "Implement the feature"
            tool_budget: 30

          - id: review
            type: hitl
            hitl_type: approval
            prompt: "Review the implementation?"
            timeout: 300
            fallback: continue
"""

from __future__ import annotations

import logging
import operator
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import yaml

from victor.core.yaml_utils import safe_load as yaml_safe_load

# =============================================================================
# Enums for controlled values (code maintenance)
# =============================================================================


class YAMLNodeType(str, Enum):
    """Valid node types for YAML workflow definitions.

    Renamed from NodeType to be semantically distinct:
    - YAMLNodeType (here): YAML loader validation nodes
    - WorkflowNodeType (victor.workflows.definition): Workflow definition nodes (COMPUTE, HITL, START, END)
    - GraphNodeType (victor.workflows.graph_dsl): Graph DSL nodes (FUNCTION, CONDITIONAL, SUBGRAPH)
    """

    AGENT = "agent"
    COMPUTE = "compute"
    CONDITION = "condition"
    PARALLEL = "parallel"
    TRANSFORM = "transform"
    HITL = "hitl"


class ConstraintType(str, Enum):
    """Valid constraint types for blocking resources.

    Used in list-format constraints: constraints: [llm, network, write]
    """

    LLM = "llm"
    NETWORK = "network"
    WRITE = "write"
    FILESYSTEM = "filesystem"  # Alias for strict sandboxing
    NONE = "none"  # Special: no constraints (everything allowed)


class JoinStrategy(str, Enum):
    """Join strategies for parallel nodes."""

    ALL = "all"
    ANY = "any"
    FIRST = "first"


class HITLType(str, Enum):
    """HITL node types."""

    APPROVAL = "approval"
    INPUT = "input"
    REVIEW = "review"
    CHOICE = "choice"


class HITLFallback(str, Enum):
    """HITL fallback behaviors on timeout."""

    ABORT = "abort"
    CONTINUE = "continue"
    DEFAULT = "default"


class ExecutionTarget(str, Enum):
    """Execution target for compute nodes.

    Specifies how a compute node should be executed:
    - IN_PROCESS: Direct Python execution (fastest, shared memory)
    - SUBPROCESS: Separate process (isolation, can timeout)
    - DOCKER: Container execution (full isolation, reproducible)
    """

    IN_PROCESS = "in-process"
    SUBPROCESS = "subprocess"
    DOCKER = "docker"


# =============================================================================
# Environment Variable Interpolation
# =============================================================================

# Pattern for $env.VAR_NAME syntax
ENV_VAR_PATTERN = re.compile(r"\$env\.([A-Za-z_][A-Za-z0-9_]*)")
# Pattern for ${VAR:-default} syntax
SHELL_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _interpolate_env_vars(value: Any) -> Any:
    """Recursively interpolate environment variables in YAML values.

    Supports two syntaxes:
    - $env.VAR_NAME: Simple env var reference
    - ${VAR_NAME:-default}: Shell-style with optional default

    Args:
        value: Any YAML value (str, dict, list, or primitive)

    Returns:
        Value with environment variables interpolated

    Example:
        input: "$env.DATABASE_URL"
        output: "postgresql://localhost:5432/db"

        input: "${API_KEY:-default_key}"
        output: value of API_KEY or "default_key" if not set
    """
    if isinstance(value, str):
        # Handle $env.VAR_NAME syntax
        def replace_env(match: re.Match) -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, f"$env.{var_name}")

        result = ENV_VAR_PATTERN.sub(replace_env, value)

        # Handle ${VAR:-default} syntax
        def replace_shell(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var_name, default)

        result = SHELL_VAR_PATTERN.sub(replace_shell, result)
        return result

    elif isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]

    return value


if TYPE_CHECKING:
    from victor.workflows.batch_executor import BatchConfig
    from victor.workflows.services.definition import ServiceConfig
    from victor.workflows.validation.tool_validator import ToolValidationResult

from victor.workflows.definition import (
    AgentNode,
    ComputeNode,
    ConditionNode,
    ParallelNode,
    TaskConstraints,
    TransformNode,
    WorkflowDefinition,
    WorkflowNode,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Extended Schema Dataclasses
# =============================================================================


@dataclass
class LLMConfig:
    """LLM configuration for agent nodes.

    Allows fine-tuning LLM behavior per agent node.

    Attributes:
        temperature: Sampling temperature (0.0-1.0)
        model_hint: Preferred model (e.g., claude-3-sonnet, claude-3-haiku)
        max_tokens: Maximum tokens for response
        top_p: Nucleus sampling parameter
        stop_sequences: Custom stop sequences

    Example YAML:
        llm_config:
          temperature: 0.3
          model_hint: claude-3-sonnet
          max_tokens: 4096
    """

    temperature: Optional[float] = None
    model_hint: Optional[str] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.model_hint is not None:
            result["model_hint"] = self.model_hint
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            result["top_p"] = self.top_p
        if self.stop_sequences is not None:
            result["stop_sequences"] = self.stop_sequences
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        return cls(
            temperature=data.get("temperature"),
            model_hint=data.get("model_hint"),
            max_tokens=data.get("max_tokens"),
            top_p=data.get("top_p"),
            stop_sequences=data.get("stop_sequences"),
        )


@dataclass
class TemporalContextConfig:
    """Point-in-time context configuration for backtesting.

    Integrates with TemporalContext from executor for historical analysis.

    Attributes:
        as_of_date: Reference date (YYYY-MM-DD or $ctx.key reference)
        lookback_periods: Number of periods to look back
        period_type: Type of period (days, weeks, months, quarters, years)
        include_end_date: Whether to include the as_of_date in range

    Example YAML:
        temporal_context:
          as_of_date: $ctx.analysis_date
          lookback_periods: 8
          period_type: quarters
    """

    as_of_date: Optional[str] = None
    lookback_periods: int = 1
    period_type: Literal["days", "weeks", "months", "quarters", "years"] = "quarters"
    include_end_date: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "as_of_date": self.as_of_date,
            "lookback_periods": self.lookback_periods,
            "period_type": self.period_type,
            "include_end_date": self.include_end_date,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalContextConfig":
        return cls(
            as_of_date=data.get("as_of_date"),
            lookback_periods=data.get("lookback_periods", 1),
            period_type=data.get("period_type", "quarters"),
            include_end_date=data.get("include_end_date", True),
        )


@dataclass
class ServiceConfigYAML:
    """Service configuration from YAML.

    Defines infrastructure services (databases, caches, message queues)
    that should be started before workflow execution.

    Attributes:
        name: Service name (used as key in workflow context)
        provider: Service provider (docker, kubernetes, local, external, aws_rds, etc.)
        preset: Use a preset (postgres, redis, kafka, elasticsearch, etc.)
        image: Docker image (for docker provider)
        command: Override container command
        ports: Port mappings (host:container or just container)
        environment: Environment variables
        volumes: Volume mounts
        health_check: Health check configuration
        depends_on: Services this service depends on
        lifecycle: Lifecycle configuration (startup_order, etc.)
        exports: Values to export to workflow context

    Example YAML:
        services:
          postgres:
            preset: postgres
            environment:
              POSTGRES_PASSWORD: $env.DB_PASSWORD
            exports:
              DATABASE_URL: postgresql://postgres:$env.DB_PASSWORD@localhost:5432/victor

          cache:
            provider: docker
            image: redis:7-alpine
            ports: [6379]
            health_check:
              type: tcp
              port: 6379
    """

    name: str
    provider: str = "docker"
    preset: Optional[str] = None
    image: Optional[str] = None
    command: Optional[str] = None
    ports: List[Union[int, str]] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    volumes: List[str] = field(default_factory=list)
    health_check: Optional[Dict[str, Any]] = None
    depends_on: List[str] = field(default_factory=list)
    lifecycle: Optional[Dict[str, Any]] = None
    exports: Dict[str, str] = field(default_factory=dict)
    # AWS-specific
    aws_config: Optional[Dict[str, Any]] = None
    # Kubernetes-specific
    k8s_config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "provider": self.provider,
        }
        if self.preset:
            result["preset"] = self.preset
        if self.image:
            result["image"] = self.image
        if self.command:
            result["command"] = self.command
        if self.ports:
            result["ports"] = self.ports
        if self.environment:
            result["environment"] = self.environment
        if self.volumes:
            result["volumes"] = self.volumes
        if self.health_check:
            result["health_check"] = self.health_check
        if self.depends_on:
            result["depends_on"] = self.depends_on
        if self.lifecycle:
            result["lifecycle"] = self.lifecycle
        if self.exports:
            result["exports"] = self.exports
        if self.aws_config:
            result["aws_config"] = self.aws_config
        if self.k8s_config:
            result["k8s_config"] = self.k8s_config
        return result

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "ServiceConfigYAML":
        return cls(
            name=name,
            provider=data.get("provider", "docker"),
            preset=data.get("preset"),
            image=data.get("image"),
            command=data.get("command"),
            ports=data.get("ports", []),
            environment=data.get("environment", {}),
            volumes=data.get("volumes", []),
            health_check=data.get("health_check"),
            depends_on=data.get("depends_on", []),
            lifecycle=data.get("lifecycle"),
            exports=data.get("exports", {}),
            aws_config=data.get("aws_config"),
            k8s_config=data.get("k8s_config"),
        )

    def to_service_config(self) -> "ServiceConfig":
        """Convert to ServiceConfig for the service registry.

        Returns:
            ServiceConfig instance for use with ServiceRegistry
        """
        from victor.workflows.services import (
            HealthCheckConfig,
            HealthCheckType,
            LifecycleConfig,
            PortMapping,
            ServiceConfig,
            ServicePresets,
            VolumeMount,
        )

        # Use preset if specified
        if self.preset:
            preset_factory = getattr(ServicePresets, self.preset, None)
            if preset_factory:
                # Get base config from preset
                base_config = preset_factory(name=self.name)
                # Override with YAML settings
                if self.environment:
                    base_config.environment.update(self.environment)
                if self.exports:
                    base_config.exports.update(self.exports)
                return base_config
            else:
                logger.warning(f"Unknown preset '{self.preset}', using manual config")

        # Parse port mappings
        port_mappings = []
        for port_spec in self.ports:
            if isinstance(port_spec, int):
                port_mappings.append(PortMapping(container_port=port_spec))
            elif isinstance(port_spec, str) and ":" in port_spec:
                host_port, container_port = port_spec.split(":", 1)
                port_mappings.append(
                    PortMapping(
                        container_port=int(container_port),
                        host_port=int(host_port),
                    )
                )
            else:
                port_mappings.append(PortMapping(container_port=int(port_spec)))

        # Parse volume mounts
        volume_mounts = []
        for vol_spec in self.volumes:
            if ":" in vol_spec:
                parts = vol_spec.split(":")
                volume_mounts.append(
                    VolumeMount(
                        host_path=parts[0],
                        container_path=parts[1],
                        read_only=len(parts) > 2 and parts[2] == "ro",
                    )
                )

        # Parse health check
        health_check = None
        if self.health_check:
            hc_type = self.health_check.get("type", "tcp")
            health_check = HealthCheckConfig(
                type=HealthCheckType(hc_type),
                port=self.health_check.get("port"),
                path=self.health_check.get("path"),
                interval=self.health_check.get("interval", 5.0),
                timeout=self.health_check.get("timeout", 30.0),
                retries=self.health_check.get("retries", 3),
            )

        # Parse lifecycle
        lifecycle = LifecycleConfig()
        if self.lifecycle:
            lifecycle = LifecycleConfig(
                startup_order=self.lifecycle.get("startup_order", 0),
                startup_timeout=self.lifecycle.get("startup_timeout", 60.0),
                shutdown_timeout=self.lifecycle.get("shutdown_timeout", 30.0),
                restart_policy=self.lifecycle.get("restart_policy", "no"),
            )

        return ServiceConfig(
            name=self.name,
            provider=self.provider,
            image=self.image,
            command=self.command,
            ports=port_mappings,
            environment=self.environment,
            volumes=volume_mounts,
            health_check=health_check,
            depends_on=self.depends_on,
            lifecycle=lifecycle,
            exports=self.exports,
        )


@dataclass
class BatchConfigYAML:
    """Batch execution configuration from YAML.

    Maps to BatchConfig from batch_executor for batch processing.

    Attributes:
        batch_size: Number of items per batch
        max_concurrent: Maximum parallel executions
        delay_seconds: Delay between batches
        retry_strategy: How to handle failures (none, immediate, end_of_batch)
        max_retries: Maximum retry attempts per item
        fail_fast: Stop on first failure

    Example YAML:
        batch_config:
          batch_size: 10
          max_concurrent: 5
          retry_strategy: end_of_batch
          max_retries: 2
    """

    batch_size: int = 5
    max_concurrent: int = 3
    delay_seconds: float = 1.0
    retry_strategy: str = "end_of_batch"
    max_retries: int = 2
    retry_delay_seconds: float = 5.0
    timeout_per_item: Optional[float] = None
    fail_fast: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "max_concurrent": self.max_concurrent,
            "delay_seconds": self.delay_seconds,
            "retry_strategy": self.retry_strategy,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "timeout_per_item": self.timeout_per_item,
            "fail_fast": self.fail_fast,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchConfigYAML":
        return cls(
            batch_size=data.get("batch_size", 5),
            max_concurrent=data.get("max_concurrent", 3),
            delay_seconds=data.get("delay_seconds", 1.0),
            retry_strategy=data.get("retry_strategy", "end_of_batch"),
            max_retries=data.get("max_retries", 2),
            retry_delay_seconds=data.get("retry_delay_seconds", 5.0),
            timeout_per_item=data.get("timeout_per_item"),
            fail_fast=data.get("fail_fast", False),
        )

    def to_batch_config(self) -> "BatchConfig":
        """Convert to BatchConfig for executor."""
        from victor.workflows.batch_executor import BatchConfig, BatchRetryStrategy

        strategy_map = {
            "none": BatchRetryStrategy.NONE,
            "immediate": BatchRetryStrategy.IMMEDIATE,
            "end_of_batch": BatchRetryStrategy.END_OF_BATCH,
            "exponential_backoff": BatchRetryStrategy.EXPONENTIAL_BACKOFF,
        }

        return BatchConfig(
            batch_size=self.batch_size,
            max_concurrent=self.max_concurrent,
            delay_seconds=self.delay_seconds,
            retry_strategy=strategy_map.get(self.retry_strategy, BatchRetryStrategy.END_OF_BATCH),
            max_retries=self.max_retries,
            retry_delay_seconds=self.retry_delay_seconds,
            timeout_per_item=self.timeout_per_item,
            fail_fast=self.fail_fast,
        )


# =============================================================================
# Error Types
# =============================================================================


class YAMLWorkflowError(Exception):
    """Error loading or parsing YAML workflow."""

    pass


# =============================================================================
# External Reference Resolution ($ref)
# =============================================================================


def _resolve_ref(
    ref: str,
    base_dir: Optional[Path],
    ref_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Resolve an external $ref reference to a node definition.

    Supports two formats:
    - "./path/to/file.yaml#node_id" - Reference specific node from file
    - "./path/to/file.yaml" - Load entire file (returns first node)

    Args:
        ref: The $ref value (e.g., "./common_nodes.yaml#validation")
        base_dir: Base directory for resolving relative paths
        ref_cache: Cache of previously loaded files

    Returns:
        Dict containing the referenced node definition

    Raises:
        YAMLWorkflowError: If reference cannot be resolved
    """
    if "#" in ref:
        file_path, node_id = ref.rsplit("#", 1)
    else:
        file_path = ref
        node_id = None

    # Resolve path relative to base_dir
    if base_dir and not Path(file_path).is_absolute():
        full_path = base_dir / file_path
    else:
        full_path = Path(file_path)

    # Validate resolved path stays within base_dir (prevent path traversal)
    resolved = full_path.resolve()
    if base_dir:
        base_resolved = base_dir.resolve()
        if not str(resolved).startswith(str(base_resolved)):
            raise YAMLWorkflowError(
                f"Path traversal detected: {ref} resolves outside base directory"
            )

    # Check cache
    cache_key = str(resolved)
    if cache_key not in ref_cache:
        if not full_path.exists():
            raise YAMLWorkflowError(f"Referenced file not found: {full_path}")

        try:
            content = full_path.read_text()
            ref_cache[cache_key] = yaml_safe_load(content)
        except yaml.YAMLError as e:
            raise YAMLWorkflowError(f"Invalid YAML in referenced file {full_path}: {e}")

    file_data = ref_cache[cache_key]

    # Extract specific node or return first node
    if node_id:
        # Look for node in "nodes" list
        nodes = file_data.get("nodes", [])
        for node in nodes:
            if node.get("id") == node_id:
                return node
        raise YAMLWorkflowError(f"Node '{node_id}' not found in {full_path}")
    else:
        # Return first node from file
        nodes = file_data.get("nodes", [])
        if not nodes:
            raise YAMLWorkflowError(f"No nodes found in {full_path}")
        return nodes[0]


def _expand_refs(
    node_list: List[Dict[str, Any]],
    base_dir: Optional[Path],
) -> List[Dict[str, Any]]:
    """Expand $ref references in node list.

    Args:
        node_list: List of node definitions (may contain $ref)
        base_dir: Base directory for resolving relative paths

    Returns:
        List with all $ref nodes replaced with actual definitions
    """
    ref_cache: Dict[str, Dict[str, Any]] = {}
    expanded = []

    for node_data in node_list:
        if "$ref" in node_data:
            # Resolve reference
            ref_value = node_data["$ref"]
            resolved = _resolve_ref(ref_value, base_dir, ref_cache)

            # Merge any overrides from the referencing node
            merged = {**resolved}
            for key, value in node_data.items():
                if key != "$ref":
                    merged[key] = value

            expanded.append(merged)
            logger.debug(f"Resolved $ref '{ref_value}' to node '{merged.get('id')}'")
        else:
            expanded.append(node_data)

    return expanded


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class YAMLWorkflowConfig:
    """Configuration for YAML workflow loading."""

    # Allow unsafe conditions (arbitrary Python expressions)
    allow_unsafe_conditions: bool = False
    # Base directory for relative imports
    base_dir: Optional[Path] = None
    # Custom condition functions
    condition_registry: Dict[str, Callable[[Dict[str, Any]], str]] = None
    # Custom transform functions
    transform_registry: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = None

    def __post_init__(self):
        if self.condition_registry is None:
            self.condition_registry = {}
        if self.transform_registry is None:
            self.transform_registry = {}


def _create_simple_condition(expr: str) -> Callable[[Dict[str, Any]], str]:
    """Create a condition function from a simple expression.

    Supported expressions:
    - "key" - check if key exists and is truthy
    - "key == value" - check equality
    - "key != value" - check inequality
    - "key > value" - numeric comparison
    - "key >= value" - numeric comparison
    - "key < value" - numeric comparison
    - "key <= value" - numeric comparison
    - "key in [a, b, c]" - check membership

    Returns:
        Function that evaluates the expression and returns "true" or "false"
    """
    expr = expr.strip()

    # Comparison operators
    comparisons = [
        (r"^(\w+)\s*==\s*(.+)$", operator.eq),
        (r"^(\w+)\s*!=\s*(.+)$", operator.ne),
        (r"^(\w+)\s*>=\s*(.+)$", operator.ge),
        (r"^(\w+)\s*<=\s*(.+)$", operator.le),
        (r"^(\w+)\s*>\s*(.+)$", operator.gt),
        (r"^(\w+)\s*<\s*(.+)$", operator.lt),
    ]

    for pattern, op in comparisons:
        match = re.match(pattern, expr)
        if match:
            key = match.group(1)
            value_str = match.group(2).strip()
            # Parse value
            value = _parse_value(value_str)

            def condition(ctx: Dict[str, Any], k=key, v=value, o=op) -> str:
                ctx_value = ctx.get(k)
                try:
                    return "true" if o(ctx_value, v) else "false"
                except TypeError:
                    return "false"

            return condition

    # Check for "in" operator
    in_match = re.match(r"^(\w+)\s+in\s+\[(.+)\]$", expr)
    if in_match:
        key = in_match.group(1)
        values_str = in_match.group(2)
        values = [_parse_value(v.strip()) for v in values_str.split(",")]

        def in_condition(ctx: Dict[str, Any], k=key, vs=values) -> str:
            return "true" if ctx.get(k) in vs else "false"

        return in_condition

    # Simple truthy check
    if re.match(r"^\w+$", expr):

        def truthy_condition(ctx: Dict[str, Any], k=expr) -> str:
            return "true" if ctx.get(k) else "false"

        return truthy_condition

    # Default: always return "default"
    return lambda ctx: "default"


def _parse_value(value_str: str) -> Any:
    """Parse a string value into appropriate Python type."""
    value_str = value_str.strip()

    # Handle quoted strings
    if (value_str.startswith('"') and value_str.endswith('"')) or (
        value_str.startswith("'") and value_str.endswith("'")
    ):
        return value_str[1:-1]

    # Handle booleans
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    if value_str.lower() == "none":
        return None

    # Handle numbers
    try:
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass

    # Return as string
    return value_str


def _create_transform(expr: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Create a transform function from a simple expression.

    Supported expressions:
    - "key = value" - set a key to a literal value
    - "key = ctx.other_key" - copy from another key
    - "key = merge(a, b)" - merge two dicts

    Returns:
        Function that transforms context
    """
    expr = expr.strip()

    # Assignment: key = value
    assign_match = re.match(r"^(\w+)\s*=\s*(.+)$", expr)
    if assign_match:
        key = assign_match.group(1)
        value_str = assign_match.group(2).strip()

        # Reference to context key
        if value_str.startswith("ctx."):
            ref_key = value_str[4:]

            def ref_transform(ctx: Dict[str, Any], k=key, rk=ref_key) -> Dict[str, Any]:
                result = ctx.copy()
                result[k] = ctx.get(rk)
                return result

            return ref_transform

        # Literal value
        value = _parse_value(value_str)

        def literal_transform(ctx: Dict[str, Any], k=key, v=value) -> Dict[str, Any]:
            result = ctx.copy()
            result[k] = v
            return result

        return literal_transform

    # Default: identity transform
    return lambda ctx: ctx


def _parse_agent_node(node_data: Dict[str, Any]) -> AgentNode:
    """Parse an agent node from YAML data.

    Supports llm_config for per-node LLM settings:
        - id: analyze
          type: agent
          role: researcher
          goal: "Analyze data patterns"
          profile: openai  # Optional: use specific provider profile
          llm_config:
            temperature: 0.3
            model_hint: claude-3-sonnet
            max_tokens: 4096
    """
    node_id = node_data["id"]

    # Parse llm_config if present
    llm_config_data = node_data.get("llm_config")
    llm_config = None
    if llm_config_data:
        llm_config = LLMConfig.from_dict(llm_config_data).to_dict()

    # Parse timeout (supports both 'timeout' and 'timeout_seconds' keys)
    timeout_seconds = node_data.get("timeout_seconds") or node_data.get("timeout")

    return AgentNode(
        id=node_id,
        name=node_data.get("name", node_id),
        role=node_data.get("role", "executor"),
        goal=node_data.get("goal", ""),
        tool_budget=node_data.get("tool_budget", 15),
        allowed_tools=node_data.get("tools"),
        input_mapping=node_data.get("input_mapping", {}),
        output_key=node_data.get("output", node_id),
        llm_config=llm_config,
        timeout_seconds=timeout_seconds,
        profile=node_data.get("profile"),  # Per-node provider profile
        next_nodes=node_data.get("next", []),
    )


def _parse_condition_node(
    node_data: Dict[str, Any],
    config: YAMLWorkflowConfig,
) -> ConditionNode:
    """Parse a condition node from YAML data."""
    node_id = node_data["id"]
    condition_expr = node_data.get("condition", "default")

    # Check for registered condition
    if condition_expr in config.condition_registry:
        condition_fn = config.condition_registry[condition_expr]
    else:
        condition_fn = _create_simple_condition(condition_expr)

    return ConditionNode(
        id=node_id,
        name=node_data.get("name", node_id),
        condition=condition_fn,
        branches=node_data.get("branches", {}),
        next_nodes=node_data.get("next", []),
    )


def _parse_parallel_node(node_data: Dict[str, Any]) -> ParallelNode:
    """Parse a parallel node from YAML data."""
    node_id = node_data["id"]
    return ParallelNode(
        id=node_id,
        name=node_data.get("name", node_id),
        parallel_nodes=node_data.get("parallel_nodes", []),
        join_strategy=node_data.get("join_strategy", "all"),
        next_nodes=node_data.get("next", []),
    )


def _parse_transform_node(
    node_data: Dict[str, Any],
    config: YAMLWorkflowConfig,
) -> TransformNode:
    """Parse a transform node from YAML data."""
    node_id = node_data["id"]
    transform_expr = node_data.get("transform", "")

    # Check for registered transform
    if transform_expr in config.transform_registry:
        transform_fn = config.transform_registry[transform_expr]
    else:
        transform_fn = _create_transform(transform_expr)

    return TransformNode(
        id=node_id,
        name=node_data.get("name", node_id),
        transform=transform_fn,
        next_nodes=node_data.get("next", []),
    )


def _parse_hitl_node(node_data: Dict[str, Any]) -> WorkflowNode:
    """Parse a HITL node from YAML data."""
    from victor.workflows.hitl import HITLFallback, HITLNode, HITLNodeType

    node_id = node_data["id"]
    hitl_type_str = node_data.get("hitl_type", "approval")
    hitl_type = HITLNodeType(hitl_type_str)

    fallback_str = node_data.get("fallback", "abort")
    fallback = HITLFallback(fallback_str)

    return HITLNode(
        id=node_id,
        name=node_data.get("name", node_id),
        hitl_type=hitl_type,
        prompt=node_data.get("prompt", ""),
        context_keys=node_data.get("context_keys", []),
        choices=node_data.get("choices", []),
        default_value=node_data.get("default_value"),
        timeout=node_data.get("timeout", 300.0),
        fallback=fallback,
        next_nodes=node_data.get("next", []),
    )


def _parse_constraints(constraints_data: Any, timeout: float = 60.0) -> TaskConstraints:
    """Parse constraints from YAML data.

    Supports two formats:
    1. List format (compact): constraints: [llm, network, write]
       - Items in list are BLOCKED
       - "none" = no constraints (everything allowed)
       - "none" cannot be combined with other values

    2. Dict format (legacy): constraints: {llm_allowed: false, ...}
       - Explicit key-value pairs

    Examples:
        constraints: none                    # No constraints at all
        constraints: [llm]                   # Only LLM blocked
        constraints: [llm, network, write]   # LLM, network, and write blocked
        constraints: [llm, filesystem]       # LLM and filesystem blocked
        constraints:                         # Legacy dict format
          llm_allowed: false
          network_allowed: true

    Valid constraint names for list format (from ConstraintType enum):
        - llm: Block LLM inference
        - network: Block network calls
        - write: Block disk writes
        - filesystem: Block filesystem access (alias for strict sandboxing)
        - none: Special value - no constraints (everything allowed)

    Args:
        constraints_data: Either a list, "none", or dict
        timeout: Default timeout value

    Returns:
        TaskConstraints instance

    Raises:
        YAMLWorkflowError: If "none" is combined with other values
    """
    # Handle None / missing constraints
    if constraints_data is None:
        return TaskConstraints(_timeout=timeout)

    # Handle string "none" - no constraints
    if isinstance(constraints_data, str):
        if constraints_data.lower() == ConstraintType.NONE.value:
            return TaskConstraints(
                llm_allowed=True,
                network_allowed=True,
                write_allowed=True,
                _timeout=timeout,
            )
        else:
            valid_values = [c.value for c in ConstraintType]
            raise YAMLWorkflowError(
                f"Invalid constraint string: '{constraints_data}'. " f"Valid values: {valid_values}"
            )

    # Handle list format (compact)
    if isinstance(constraints_data, list):
        # Normalize to lowercase
        normalized = [item.lower() for item in constraints_data]

        # Validate: "none" cannot be combined with other values
        if ConstraintType.NONE.value in normalized:
            if len(normalized) > 1:
                raise YAMLWorkflowError(
                    f"Constraint '{ConstraintType.NONE.value}' cannot be combined "
                    f"with other values. Got: {constraints_data}"
                )
            # Just "none" in list
            return TaskConstraints(
                llm_allowed=True,
                network_allowed=True,
                write_allowed=True,
                _timeout=timeout,
            )

        # Parse list of blocked items using enum values
        blocked = set(normalized)
        return TaskConstraints(
            llm_allowed=ConstraintType.LLM.value not in blocked,
            network_allowed=ConstraintType.NETWORK.value not in blocked,
            write_allowed=ConstraintType.WRITE.value not in blocked,
            # filesystem blocks both network and write for strict sandboxing
            _timeout=timeout,
        )

    # Handle dict format (legacy)
    if isinstance(constraints_data, dict):
        return TaskConstraints(
            llm_allowed=constraints_data.get("llm_allowed", False),
            network_allowed=constraints_data.get("network_allowed", True),
            write_allowed=constraints_data.get("write_allowed", False),
            max_cost_tier=constraints_data.get("max_cost_tier", "FREE"),
            _max_tool_calls=constraints_data.get("max_tool_calls", 100),
            _timeout=constraints_data.get("timeout", timeout),
            allowed_tools=constraints_data.get("allowed_tools"),
            blocked_tools=constraints_data.get("blocked_tools"),
        )

    raise YAMLWorkflowError(
        f"Invalid constraints format: {type(constraints_data)}. "
        f"Use '{ConstraintType.NONE.value}', a list [{ConstraintType.LLM.value}, "
        f"{ConstraintType.NETWORK.value}, {ConstraintType.WRITE.value}], or a dict."
    )


def _parse_compute_node(node_data: Dict[str, Any]) -> ComputeNode:
    """Parse a compute node from YAML data.

    Compute nodes execute tools with configurable constraints.
    Supports LLM-free execution and custom handlers.

    YAML format (new compact constraints):
        - id: run_valuation
          type: compute
          tools: [multi_model_valuation, sector_valuation]
          inputs:
            symbol: $ctx.symbol
            financials: $ctx.sec_data
          output: fair_values
          constraints: [llm, write]  # Blocks LLM and write, allows network
          timeout: 60
          next: [blend]

        # No constraints (everything allowed)
        - id: unrestricted_task
          type: compute
          handler: custom_handler
          constraints: none

    YAML format (legacy dict constraints):
        - id: run_valuation
          type: compute
          tools: [multi_model_valuation]
          constraints:
            llm_allowed: false
            network_allowed: true
            max_cost_tier: FREE
            timeout: 60

        # With custom handler
        - id: rl_weights
          type: compute
          handler: rl_decision
          inputs:
            features: $ctx.valuation_features
          output: model_weights
    """
    node_id = node_data["id"]

    # Parse input mapping - supports $ctx.key syntax
    input_mapping = {}
    inputs = node_data.get("inputs", node_data.get("input_mapping", {}))
    for key, value in inputs.items():
        if isinstance(value, str) and value.startswith("$ctx."):
            # Extract context key reference
            input_mapping[key] = value[5:]  # Remove $ctx. prefix
        else:
            input_mapping[key] = value

    # Parse constraints using new format-aware parser
    timeout = node_data.get("timeout", 60.0)
    constraints = _parse_constraints(node_data.get("constraints"), timeout)

    # Parse execution target with enum validation
    exec_target_str = node_data.get("execution_target", "in-process")
    try:
        exec_target = ExecutionTarget(exec_target_str.lower())
    except ValueError:
        valid_targets = [t.value for t in ExecutionTarget]
        raise YAMLWorkflowError(
            f"Invalid execution_target '{exec_target_str}'. " f"Valid values: {valid_targets}"
        )

    return ComputeNode(
        id=node_id,
        name=node_data.get("name", node_id),
        tools=node_data.get("tools", []),
        input_mapping=input_mapping,
        output_key=node_data.get("output", node_id),
        constraints=constraints,
        handler=node_data.get("handler"),
        fail_fast=node_data.get("fail_fast", True),
        parallel=node_data.get("parallel", False),
        execution_target=exec_target.value,
        next_nodes=node_data.get("next", []),
    )


def _parse_node(
    node_data: Dict[str, Any],
    config: YAMLWorkflowConfig,
) -> WorkflowNode:
    """Parse a workflow node from YAML data.

    Node type defaults to 'agent' if not specified, making the common case
    (LLM-powered agent nodes) more concise:

        nodes:
          # Agent node (type defaults to 'agent')
          - id: analyze
            role: researcher
            goal: "Analyze the codebase"
            tool_budget: 20

          # Compute node (must be explicit)
          - id: run_tests
            type: compute
            handler: test_runner
            constraints: [llm]

    This reduces YAML verbosity for the most common agentic workflows.
    """
    # Default to 'agent' if type not specified - makes common case concise
    node_type_str = node_data.get("type", YAMLNodeType.AGENT.value)

    # Use enum for matching to avoid string typos
    if node_type_str == YAMLNodeType.AGENT.value:
        return _parse_agent_node(node_data)
    elif node_type_str == YAMLNodeType.COMPUTE.value:
        return _parse_compute_node(node_data)
    elif node_type_str == YAMLNodeType.CONDITION.value:
        return _parse_condition_node(node_data, config)
    elif node_type_str == YAMLNodeType.PARALLEL.value:
        return _parse_parallel_node(node_data)
    elif node_type_str == YAMLNodeType.TRANSFORM.value:
        return _parse_transform_node(node_data, config)
    elif node_type_str == YAMLNodeType.HITL.value:
        return _parse_hitl_node(node_data)
    else:
        valid_types = [t.value for t in YAMLNodeType]
        raise YAMLWorkflowError(f"Unknown node type: '{node_type_str}'. Valid types: {valid_types}")


def load_workflow_from_dict(
    data: Dict[str, Any],
    name: str,
    config: Optional[YAMLWorkflowConfig] = None,
) -> WorkflowDefinition:
    """Load a workflow definition from a dictionary.

    Supports extended schema features:
    - $ref: External node references resolved from base_dir
    - batch_config: Batch execution settings stored in metadata
    - temporal_context: Point-in-time context stored in metadata

    Args:
        data: Dictionary containing workflow definition
        name: Name for the workflow
        config: Optional loader configuration

    Returns:
        WorkflowDefinition instance with extended metadata
    """
    config = config or YAMLWorkflowConfig()

    # Get node list and expand $ref references
    node_list = data.get("nodes", [])
    if config.base_dir:
        node_list = _expand_refs(node_list, config.base_dir)

    # Parse nodes
    nodes: Dict[str, WorkflowNode] = {}
    for node_data in node_list:
        if "id" not in node_data:
            raise YAMLWorkflowError("Node missing required 'id' field")

        node = _parse_node(node_data, config)
        nodes[node.id] = node

    # Auto-chain sequential nodes (like WorkflowBuilder)
    # Connect nodes that don't have explicit "next" to the following node
    for i, node_data in enumerate(node_list[:-1]):
        node_id = node_data["id"]
        node = nodes[node_id]
        # Skip condition nodes (they use branches)
        if isinstance(node, ConditionNode):
            continue
        # Auto-chain if no explicit next_nodes
        if not node.next_nodes:
            next_node_id = node_list[i + 1]["id"]
            node.next_nodes.append(next_node_id)

    # Determine start node
    start_node = data.get("start_node")
    if not start_node and node_list:
        start_node = node_list[0]["id"]

    # Build metadata with extended schema features
    metadata = dict(data.get("metadata", {}))

    # Parse batch_config if present
    batch_config_data = data.get("batch_config")
    if batch_config_data:
        batch_config = BatchConfigYAML.from_dict(batch_config_data)
        metadata["batch_config"] = batch_config.to_dict()
        logger.debug(f"Parsed batch_config for workflow '{name}'")

    # Parse temporal_context if present
    temporal_context_data = data.get("temporal_context")
    if temporal_context_data:
        temporal_context = TemporalContextConfig.from_dict(temporal_context_data)
        metadata["temporal_context"] = temporal_context.to_dict()
        logger.debug(f"Parsed temporal_context for workflow '{name}'")

    # Parse services if present
    services_data = data.get("services", {})
    if services_data:
        services = []
        for svc_name, svc_config in services_data.items():
            service = ServiceConfigYAML.from_dict(svc_name, svc_config)
            services.append(service.to_dict())
        metadata["services"] = services
        logger.debug(f"Parsed {len(services)} services for workflow '{name}'")

    # Parse workflow-level execution settings
    # These can be specified at workflow level or in metadata
    execution_settings = data.get("execution", {}) or {}
    max_execution_timeout = (
        execution_settings.get("timeout")
        or execution_settings.get("max_execution_timeout_seconds")
        or data.get("timeout")
        or data.get("max_execution_timeout_seconds")
    )
    default_node_timeout = (
        execution_settings.get("default_node_timeout")
        or execution_settings.get("default_node_timeout_seconds")
        or data.get("default_node_timeout")
        or data.get("default_node_timeout_seconds")
    )
    max_iterations = execution_settings.get("max_iterations") or data.get("max_iterations") or 25
    max_retries = execution_settings.get("max_retries") or data.get("max_retries") or 0

    # Parse schedule configuration if present
    schedule_data = data.get("schedule")
    if schedule_data:
        schedule_config = {
            "cron": schedule_data.get("cron"),
            "timezone": schedule_data.get("timezone", "UTC"),
            "catchup": schedule_data.get("catchup", False),
            "start_date": schedule_data.get("start_date"),
            "end_date": schedule_data.get("end_date"),
            "max_active_runs": schedule_data.get("max_active_runs", 1),
            "enabled": schedule_data.get("enabled", True),
        }
        # Also support interval syntax
        if schedule_data.get("interval"):
            schedule_config["interval"] = schedule_data["interval"]
        metadata["schedule"] = schedule_config
        logger.debug(f"Parsed schedule for workflow '{name}': {schedule_config.get('cron')}")

    workflow = WorkflowDefinition(
        name=name,
        description=data.get("description", ""),
        nodes=nodes,
        start_node=start_node,
        metadata=metadata,
        max_execution_timeout_seconds=max_execution_timeout,
        default_node_timeout_seconds=default_node_timeout,
        max_iterations=max_iterations,
        max_retries=max_retries,
    )

    # Validate
    errors = workflow.validate()
    if errors:
        raise YAMLWorkflowError(f"Workflow validation failed: {'; '.join(errors)}")

    return workflow


def load_workflow_from_yaml(
    yaml_content: str,
    workflow_name: Optional[str] = None,
    config: Optional[YAMLWorkflowConfig] = None,
) -> Union[WorkflowDefinition, Dict[str, WorkflowDefinition]]:
    """Load workflow(s) from YAML content.

    Args:
        yaml_content: YAML string content
        workflow_name: Optional specific workflow to load
        config: Optional loader configuration

    Returns:
        Single WorkflowDefinition if workflow_name specified,
        otherwise Dict of all workflows
    """
    try:
        data = yaml_safe_load(yaml_content)
    except yaml.YAMLError as e:
        raise YAMLWorkflowError(f"Invalid YAML: {e}")

    if not isinstance(data, dict):
        raise YAMLWorkflowError("YAML must contain a dictionary")

    # Interpolate environment variables
    data = _interpolate_env_vars(data)

    # Check for workflows key
    workflows_data = data.get("workflows", data)

    if workflow_name:
        if workflow_name not in workflows_data:
            raise YAMLWorkflowError(f"Workflow '{workflow_name}' not found")
        return load_workflow_from_dict(workflows_data[workflow_name], workflow_name, config)

    # Load all workflows
    workflows = {}
    for name, wf_data in workflows_data.items():
        if isinstance(wf_data, dict) and "nodes" in wf_data:
            workflows[name] = load_workflow_from_dict(wf_data, name, config)

    return workflows


def load_workflow_from_file(
    file_path: Union[str, Path],
    workflow_name: Optional[str] = None,
    config: Optional[YAMLWorkflowConfig] = None,
) -> Union[WorkflowDefinition, Dict[str, WorkflowDefinition]]:
    """Load workflow(s) from a YAML file.

    Args:
        file_path: Path to YAML file
        workflow_name: Optional specific workflow to load
        config: Optional loader configuration

    Returns:
        Single WorkflowDefinition if workflow_name specified,
        otherwise Dict of all workflows
    """
    path = Path(file_path)
    if not path.exists():
        raise YAMLWorkflowError(f"File not found: {path}")

    # Set base_dir from file location if not set
    if config is None:
        config = YAMLWorkflowConfig(base_dir=path.parent)
    elif config.base_dir is None:
        config.base_dir = path.parent

    yaml_content = path.read_text()
    return load_workflow_from_yaml(yaml_content, workflow_name, config)


def load_and_validate(
    file_path: Union[str, Path],
    workflow_name: Optional[str] = None,
    config: Optional[YAMLWorkflowConfig] = None,
    tool_registry: Optional[Any] = None,
    strict: bool = True,
) -> Tuple[Union[WorkflowDefinition, Dict[str, WorkflowDefinition]], "ToolValidationResult"]:
    """Load workflow(s) from a YAML file and validate tool dependencies.

    This function combines workflow loading with tool dependency validation,
    ensuring all tools referenced in workflow nodes exist and are available
    before workflow execution begins.

    Args:
        file_path: Path to YAML file
        workflow_name: Optional specific workflow to load
        config: Optional loader configuration
        tool_registry: Tool registry for validation (if None, validation passes)
        strict: If True, missing tools cause errors; if False, they cause warnings

    Returns:
        Tuple of (workflow(s), ToolValidationResult)

    Raises:
        YAMLWorkflowError: If file not found or YAML is invalid
        ToolValidationError: If strict=True and tools are missing

    Example:
        from victor.workflows.yaml_loader import load_and_validate
        from victor.tools.registry import ToolRegistry

        registry = ToolRegistry()
        workflow, validation = load_and_validate(
            "my_workflow.yaml",
            tool_registry=registry,
            strict=True,
        )

        if not validation.valid:
            print(f"Missing tools: {validation.missing_tools}")
            for error in validation.errors:
                print(f"  - {error}")
    """
    from victor.workflows.validation import (
        ToolDependencyValidator,
        ToolValidationResult,
        validate_workflow_tools,
    )

    # Load the workflow(s)
    workflows = load_workflow_from_file(file_path, workflow_name, config)

    # Validate tool dependencies
    if tool_registry is not None:
        validator = ToolDependencyValidator(tool_registry, strict=strict)

        if isinstance(workflows, dict):
            # Multiple workflows - validate each
            combined_result = ToolValidationResult()
            for wf_name, workflow in workflows.items():
                result = validator.validate(workflow)
                combined_result.validated_tools.update(result.validated_tools)
                combined_result.missing_tools.update(result.missing_tools)
                for error in result.errors:
                    combined_result.add_error(error)
                for warning in result.warnings:
                    combined_result.add_error(warning)
            validation_result = combined_result
        else:
            # Single workflow
            validation_result = validator.validate(workflows)

        # Raise if strict and invalid
        if strict and not validation_result.valid:
            raise YAMLWorkflowError(
                f"Tool validation failed: {validation_result.summary()}\n"
                f"Missing tools: {validation_result.missing_tools}"
            )
    else:
        # No registry - return valid result with all tools marked as validated
        validation_result = ToolValidationResult()
        if isinstance(workflows, dict):
            for workflow in workflows.values():
                for node in workflow.nodes.values():
                    tools = getattr(node, "allowed_tools", None) or getattr(node, "tools", None)
                    if tools:
                        validation_result.validated_tools.update(tools)
        else:
            for node in workflows.nodes.values():
                tools = getattr(node, "allowed_tools", None) or getattr(node, "tools", None)
                if tools:
                    validation_result.validated_tools.update(tools)

    return workflows, validation_result


def load_workflows_from_directory(
    directory: Union[str, Path],
    pattern: str = "*.yaml",
    config: Optional[YAMLWorkflowConfig] = None,
) -> Dict[str, WorkflowDefinition]:
    """Load all workflows from YAML files in a directory.

    Args:
        directory: Directory to scan
        pattern: Glob pattern for files (default: *.yaml)
        config: Optional loader configuration

    Returns:
        Dict mapping workflow names to definitions
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise YAMLWorkflowError(f"Not a directory: {dir_path}")

    workflows = {}
    for yaml_file in dir_path.glob(pattern):
        try:
            file_workflows = load_workflow_from_file(yaml_file, config=config)
            if isinstance(file_workflows, dict):
                workflows.update(file_workflows)
            else:
                workflows[file_workflows.name] = file_workflows
        except YAMLWorkflowError as e:
            logger.warning(f"Failed to load {yaml_file}: {e}")

    return workflows


class YAMLWorkflowProvider:
    """Workflow provider that loads workflows from YAML files.

    Can be used as a workflow provider for verticals by loading
    workflow definitions from a YAML file or directory.

    Example:
        provider = YAMLWorkflowProvider.from_file("workflows.yaml")
        workflow = provider.get_workflow("code_review")
    """

    def __init__(
        self,
        workflows: Dict[str, WorkflowDefinition],
        auto_workflows: Optional[List[tuple]] = None,
    ):
        """Initialize provider with pre-loaded workflows.

        Args:
            workflows: Dict of workflow name to definition
            auto_workflows: List of (pattern, workflow_name) for auto-selection
        """
        self._workflows = workflows
        self._auto_workflows = auto_workflows or []

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        config: Optional[YAMLWorkflowConfig] = None,
    ) -> "YAMLWorkflowProvider":
        """Create provider from a YAML file.

        Args:
            file_path: Path to YAML file
            config: Optional loader configuration

        Returns:
            YAMLWorkflowProvider instance
        """
        workflows = load_workflow_from_file(file_path, config=config)
        if not isinstance(workflows, dict):
            workflows = {workflows.name: workflows}
        return cls(workflows)

    @classmethod
    def from_directory(
        cls,
        directory: Union[str, Path],
        pattern: str = "*.yaml",
        config: Optional[YAMLWorkflowConfig] = None,
    ) -> "YAMLWorkflowProvider":
        """Create provider from YAML files in a directory.

        Args:
            directory: Directory to scan
            pattern: Glob pattern for files
            config: Optional loader configuration

        Returns:
            YAMLWorkflowProvider instance
        """
        workflows = load_workflows_from_directory(directory, pattern, config)
        return cls(workflows)

    def get_workflows(self) -> Dict[str, type]:
        """Get all available workflow factories."""
        return {name: type(wf) for name, wf in self._workflows.items()}

    def get_workflow(self, name: str) -> Optional[WorkflowDefinition]:
        """Get a specific workflow by name."""
        return self._workflows.get(name)

    def get_auto_workflows(self) -> List[tuple]:
        """Get auto-selection workflow mappings."""
        return self._auto_workflows.copy()

    def list_workflows(self) -> List[str]:
        """List all available workflow names."""
        return list(self._workflows.keys())


class YAMLWorkflowLoader:
    """Loader for YAML workflow definitions.

    Provides a simple interface for loading workflows from file paths
    or YAML string content. Used by WorkflowCompilerImpl for DI integration.

    Attributes:
        _enable_cache: Whether to enable definition caching
        _cache_ttl: Cache time-to-live in seconds
        _config: Optional YAMLWorkflowConfig for parsing

    Example:
        loader = YAMLWorkflowLoader(enable_cache=True, cache_ttl=3600)
        workflow_def = loader.load("path/to/workflow.yaml")
        # or
        workflow_def = loader.load("workflows:\\n  my_workflow:...")
    """

    def __init__(
        self,
        enable_cache: bool = True,
        cache_ttl: int = 3600,
        config: Optional[YAMLWorkflowConfig] = None,
    ):
        """Initialize the loader.

        Args:
            enable_cache: Whether to enable definition caching
            cache_ttl: Cache time-to-live in seconds
            config: Optional YAMLWorkflowConfig for parsing
        """
        self._enable_cache = enable_cache
        self._cache_ttl = cache_ttl
        self._config = config or YAMLWorkflowConfig()
        self._cache: Dict[str, Any] = {}

    def load(self, source: str, workflow_name: Optional[str] = None) -> Any:
        """Load workflow definition from file path or YAML string.

        Args:
            source: File path (if file exists) or YAML string content
            workflow_name: Optional specific workflow to load from multi-workflow files

        Returns:
            Parsed workflow definition (WorkflowDefinition or dict)

        Raises:
            YAMLWorkflowError: If parsing fails
            FileNotFoundError: If file path doesn't exist
        """
        # Check if source is a file path
        source_path = (
            Path(source) if not source.strip().startswith(("workflows:", "-", "{")) else None
        )

        if source_path:
            if not source_path.exists():
                raise FileNotFoundError(f"Workflow file not found: {source}")
            # Load from file
            return load_workflow_from_file(
                source_path, workflow_name=workflow_name, config=self._config
            )
        else:
            # Parse as YAML string
            return load_workflow_from_yaml(source, workflow_name=workflow_name, config=self._config)


# =============================================================================
# CLI Arguments Support
# =============================================================================


@dataclass
class WorkflowArgument:
    """Definition of a workflow input argument.

    Used to define expected inputs for workflows, similar to argparse.

    Attributes:
        name: Argument name (used as context key)
        type: Python type (str, int, float, bool, list)
        required: Whether argument is required
        default: Default value if not provided
        help: Help text for documentation
        choices: Valid choices (for enum-like args)
        env_var: Environment variable fallback

    Example YAML:
        workflows:
          deploy:
            arguments:
              - name: target
                type: str
                required: true
                help: "Deployment target (staging, production)"
                choices: [staging, production]
              - name: version
                type: str
                default: latest
                env_var: DEPLOY_VERSION
              - name: dry_run
                type: bool
                default: false
    """

    name: str
    type: str = "str"
    required: bool = False
    default: Any = None
    help: str = ""
    choices: Optional[List[Any]] = None
    env_var: Optional[str] = None

    def parse_value(self, value: Any) -> Any:
        """Parse and validate a value for this argument."""
        if value is None:
            # Check env var fallback
            if self.env_var:
                value = os.environ.get(self.env_var)
            if value is None:
                return self.default

        # Type conversion
        type_map = {
            "str": str,
            "int": int,
            "float": float,
            "bool": lambda v: (
                v.lower() in ("true", "1", "yes") if isinstance(v, str) else bool(v)
            ),
            "list": lambda v: v.split(",") if isinstance(v, str) else list(v),
        }
        converter = type_map.get(self.type, str)
        try:
            converted = converter(value)
        except (ValueError, TypeError) as e:
            raise YAMLWorkflowError(f"Invalid value for argument '{self.name}': {e}")

        # Validate choices
        if self.choices and converted not in self.choices:
            raise YAMLWorkflowError(
                f"Invalid value for argument '{self.name}': " f"'{converted}' not in {self.choices}"
            )

        return converted

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowArgument":
        return cls(
            name=data["name"],
            type=data.get("type", "str"),
            required=data.get("required", False),
            default=data.get("default"),
            help=data.get("help", ""),
            choices=data.get("choices"),
            env_var=data.get("env_var"),
        )


def parse_workflow_args(
    workflow_def: WorkflowDefinition,
    cli_args: Optional[Dict[str, Any]] = None,
    env_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse CLI-style arguments into workflow initial context.

    Args:
        workflow_def: Workflow definition with optional arguments metadata
        cli_args: Provided arguments (from CLI, API, etc.)
        env_prefix: Prefix for environment variable fallbacks

    Returns:
        Dict suitable for initial_context in workflow execution

    Raises:
        YAMLWorkflowError: If required arguments are missing

    Example:
        # YAML workflow with arguments defined
        workflow = load_workflow_from_file("deploy.yaml")
        context = parse_workflow_args(
            workflow,
            cli_args={"target": "staging", "version": "1.2.3"},
        )
        result = await executor.execute(workflow, initial_context=context)
    """
    cli_args = cli_args or {}
    context: Dict[str, Any] = {}

    # Get argument definitions from workflow metadata
    args_defs = workflow_def.metadata.get("arguments", [])
    if not args_defs:
        # No argument schema, pass through cli_args directly
        return dict(cli_args)

    # Parse each defined argument
    for arg_data in args_defs:
        arg = WorkflowArgument.from_dict(arg_data) if isinstance(arg_data, dict) else arg_data

        # Try to get value from cli_args
        value = cli_args.get(arg.name)

        # Try env var with optional prefix
        if value is None and env_prefix:
            env_key = f"{env_prefix}_{arg.name.upper()}"
            value = os.environ.get(env_key)

        # Parse and validate
        parsed = arg.parse_value(value)

        # Check required
        if parsed is None and arg.required:
            raise YAMLWorkflowError(f"Missing required argument: {arg.name}")

        if parsed is not None:
            context[arg.name] = parsed

    # Include any extra args not in schema
    for key, value in cli_args.items():
        if key not in context:
            context[key] = value

    return context


__all__ = [
    # Enums for controlled values
    "YAMLNodeType",
    "ConstraintType",
    "JoinStrategy",
    "HITLType",
    "HITLFallback",
    # Error types
    "YAMLWorkflowError",
    # Configuration
    "YAMLWorkflowConfig",
    "YAMLWorkflowProvider",
    # Extended schema types
    "LLMConfig",
    "TemporalContextConfig",
    "BatchConfigYAML",
    "ServiceConfigYAML",
    "WorkflowArgument",
    # Loading functions
    "load_workflow_from_dict",
    "load_workflow_from_yaml",
    "load_workflow_from_file",
    "load_workflows_from_directory",
    # Argument parsing
    "parse_workflow_args",
    # Constraint parsing (for external use)
    "_parse_constraints",
]
