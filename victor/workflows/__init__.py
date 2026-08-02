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

"""Workflow definition and execution system.

This package provides a LangGraph-like workflow DSL for defining
reusable multi-agent workflows as Python code or YAML.

Features:
- ComputeNode: LLM-free execution with constraints
- AgentNode: LLM-powered execution for complex reasoning
- Handlers: Extensible compute handlers for domain logic
- Isolation: Per-vertical sandbox configuration
- YAML Support: Declarative workflow definitions

Example (Python):
    from victor.workflows import WorkflowBuilder, workflow, WorkflowRegistry

    @workflow("code_review", "Review code quality")
    def code_review_workflow():
        return (
            WorkflowBuilder("code_review")
            .add_agent("analyze", "researcher", "Find code patterns")
            .add_agent("review", "reviewer", "Review quality")
            .add_agent("report", "planner", "Summarize findings")
            .build()
        )

Example (YAML):
    workflows:
      eda_pipeline:
        nodes:
          - id: load
            type: compute
            tools: [read]
            constraints:
              llm_allowed: false
          - id: analyze
            type: agent
            role: analyst
            goal: "Analyze data patterns"
"""

import warnings
from typing import Any

from victor_contracts.workflows import ExecutorNodeStatus, NodeResult
from victor.workflows.base import BaseWorkflow
from victor.workflows.context import (
    TemporalContext,
    WorkflowContext,
    WorkflowResult,
)
from victor.workflows.definition import (
    WorkflowNodeType,
    WorkflowNode,
    AgentNode,
    ConditionNode,
    ParallelNode,
    TeamStepWorkflow,
    TransformNode,
    WorkflowDefinition,
    WorkflowBuilder,
    workflow,
    get_registered_workflows,
)
from victor.workflows.registry import (
    WorkflowMetadata,
    WorkflowRegistry,
    get_global_registry,
)
from victor.workflows.unified_executor import WorkflowExecutor
from victor.workflows.protocols import (
    RetryPolicy,
    IWorkflowNode,
    IWorkflowEdge,
    IWorkflowGraph,
    ICheckpointStore,
    IWorkflowExecutor,
    IStreamingWorkflowExecutor,
    # Re-export ProtocolNodeStatus and NodeResult from protocols for graph API users
    ProtocolNodeStatus as GraphNodeStatus,
    NodeResult as GraphNodeResult,
)
from victor.workflows.streaming import (
    WorkflowEventType,
    WorkflowStreamChunk,
    WorkflowStreamContext,
)
from victor.workflows.streaming_executor import (
    StreamingWorkflowExecutor,
)
from victor.workflows.yaml_loader import (
    YAMLWorkflowError,
    YAMLWorkflowConfig,
    YAMLWorkflowProvider,
    load_workflow_from_dict,
    load_workflow_from_yaml,
    load_workflow_from_file,
    load_workflows_from_directory,
    load_and_validate,
)
from victor.workflows.cache import (
    WorkflowCacheConfig,
    WorkflowNodeCacheEntry,
    WorkflowCache,
    WorkflowCacheManager,
    get_workflow_cache_manager,
    configure_workflow_cache,
)
from victor.workflows.graph_dsl import (
    State,
    WorkflowGraph,  # Typed workflow graph DSL (compiles to WorkflowDefinition)
    GraphNode,
    GraphNodeType,
    NodeFunc,
    RouterFunc,
    Compilable,
    create_graph,
    compile_graph,
)
from victor.workflows.graph import (
    WorkflowNode as GraphWorkflowNode,
    WorkflowEdge,
    ConditionalEdge,
    BasicWorkflowGraph,  # Basic graph container
    DuplicateNodeError,
    InvalidEdgeError,
    GraphValidationError,
)

# New modules for extended workflow capabilities
from victor.workflows.definition import (
    ComputeNode,
    TaskConstraints,
    ConstraintsProtocol,
    AirgappedConstraints,
    ComputeOnlyConstraints,
    FullAccessConstraints,
)
from victor.workflows.compute_registry import (
    ComputeHandler,
    get_compute_handler,
    list_compute_handlers,
    register_compute_handler,
)
from victor.workflows.isolation import (
    SandboxType,
    ResourceLimits,
    IsolationConfig,
    IsolationMapper,
)
from victor.workflows.cost_router import (
    CostTier,
    ModelConfig,
    RoutingDecision,
    CostAwareRouter,
    get_default_router,
    route_for_cost,
)
from victor.workflows.sandbox_executor import (
    SandboxExecutionResult,
    SandboxedExecutor,
    get_sandboxed_executor,
)
from victor.workflows.handlers import (
    ParallelToolsHandler,
    SequentialToolsHandler,
    RetryBackoffHandler,
    DataTransformHandler,
    ConditionalBranchHandler,
    FRAMEWORK_HANDLERS,
    register_framework_handlers,
)
from victor.workflows.batch_executor import (
    BatchRetryStrategy,
    BatchConfig,
    BatchItemResult,
    BatchProgress,
    BatchResult,
    BatchWorkflowExecutor,
)
from victor.workflows.yaml_to_graph_compiler import (
    CompilerConfig,
    YAMLToStateGraphCompiler,
    NodeExecutorFactory,
    ConditionEvaluator,
    compile_yaml_workflow,
    execute_yaml_workflow,
)
from victor.workflows.executors import (
    WorkflowNodeExecutorRegistration,
    WorkflowNodeExecutorRegistry,
    clear_registered_workflow_node_executors,
    get_workflow_node_executor_registry,
    register_workflow_node_executor,
)
from victor.workflows.runtime_types import (
    WorkflowState,
    GraphNodeResult as CompiledGraphNodeResult,
    create_initial_workflow_state,
)
from victor.workflows.unified_executor import (
    ExecutorConfig,
    ExecutorResult,
    StateGraphExecutor,
    get_executor,
    execute_workflow,
)
from victor.workflows.deployment import (
    DeploymentTarget,
    DockerConfig,
    DockerComposeConfig,
    KubernetesConfig,
    ECSConfig,
    CloudRunConfig,
    AzureContainerConfig,
    LambdaConfig,
    RemoteConfig,
    SandboxConfig,
    DeploymentConfig,
    DeploymentHandler,
    get_deployment_handler,
)
from victor.workflows.service_lifecycle import (
    ServiceConfig,
    ServiceLifecycle,
    ServiceManager,
    PostgresService,
    RedisService,
    RabbitMQService,
    KafkaService,
    HTTPClientService,
    S3Service,
)
from victor.workflows.runtime import (
    HITLServerConfig,
    RuntimeConfig,
    RuntimeResult,
    WorkflowRuntime,
    run_workflow,
)
from victor.workflows.hitl import (
    HITLNodeType,
    HITLFallback,
    HITLMode,
    HITLCategory,
    HITL_MODE_CATEGORIES,
    HITLStatus,
    HITLRequest,
    HITLResponse,
    HITLHandler,
    HITLNode,
    HITLExecutor,
    DefaultHITLHandler,
    DEPLOYMENT_HITL_COMPATIBILITY,
    get_supported_hitl_modes,
    validate_hitl_deployment_compatibility,
)
from victor.workflows.hitl_transports import (
    # Base classes
    BaseTransportConfig,
    BaseTransport,
    HITLTransportProtocol,
    # Config classes
    EmailConfig,
    SMSConfig,
    SlackConfig,
    TeamsConfig,
    GitHubConfig,
    GitLabConfig,
    JiraConfig,
    PagerDutyConfig,
    TerraformCloudConfig,
    CustomHookConfig,
    # Transport classes
    EmailTransport,
    SMSTransport,
    SlackTransport,
    GitHubPRTransport,
    GitHubCheckTransport,
    CustomHookTransport,
    # Registry functions
    register_transport,
    get_transport,
    list_available_transports,
)
from victor.workflows.hitl_api import (
    StoredRequest,
    HITLStore,
    get_global_store,
    SQLiteHITLStore,
    get_sqlite_store,
    APIHITLHandler,
    create_hitl_router,
    create_hitl_app,
    run_hitl_server,
)
from victor.workflows.context import (
    ExecutionContext,
    create_execution_context,
    ExecutionContextWrapper,
    from_workflow_context,
    to_workflow_context,
    from_compiler_workflow_state,
    to_compiler_workflow_state,
    from_adapter_workflow_state,
    to_adapter_workflow_state,
)
from victor.workflows.node_runners import (
    BaseNodeRunner,
    AgentNodeRunner,
    ComputeNodeRunner,
    TransformNodeRunner,
    HITLNodeRunner,
    ConditionNodeRunner,
    ParallelNodeRunner,
    NodeRunnerRegistry,
)
from victor.workflows.protocols import (
    NodeRunner,
    NodeRunnerResult,
)
from victor.workflows.graph_compiler import (
    CompilerConfig,
    NodeRunnerWrapper,
    WorkflowGraphCompiler,
    WorkflowDefinitionCompiler,
    compile_workflow_graph,
    compile_workflow_definition,
)
from victor.workflows.observability import (
    StreamingObserver,
    AsyncStreamingObserver,
    FunctionObserver,
    ObservabilityEmitter,
    create_emitter,
    create_logging_observer,
)
from victor.workflows.unified_compiler import (
    UnifiedWorkflowCompiler,
    create_unified_compiler,
)

# Register framework handlers on module load
# Domain-specific handlers are registered by each vertical when loaded
register_framework_handlers()

__all__ = [
    # Base
    "BaseWorkflow",
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
    # Constraints
    "TaskConstraints",
    "ConstraintsProtocol",
    "AirgappedConstraints",
    "ComputeOnlyConstraints",
    "FullAccessConstraints",
    # Definition
    "WorkflowDefinition",
    "WorkflowBuilder",
    "workflow",
    "get_registered_workflows",
    # Registry
    "WorkflowMetadata",
    "WorkflowRegistry",
    "get_global_registry",
    # Executor
    "ExecutorNodeStatus",
    "NodeResult",
    "WorkflowContext",
    "WorkflowResult",
    "WorkflowExecutor",
    "TemporalContext",
    # Compute Handlers
    "ComputeHandler",
    "register_compute_handler",
    "get_compute_handler",
    "list_compute_handlers",
    # Framework handlers (domain handlers are provided by each vertical)
    "ParallelToolsHandler",
    "SequentialToolsHandler",
    "RetryBackoffHandler",
    "DataTransformHandler",
    "ConditionalBranchHandler",
    "FRAMEWORK_HANDLERS",
    "register_framework_handlers",
    # Isolation
    "SandboxType",
    "ResourceLimits",
    "IsolationConfig",
    "IsolationMapper",
    # Cost Routing
    "CostTier",
    "ModelConfig",
    "RoutingDecision",
    "CostAwareRouter",
    "get_default_router",
    "route_for_cost",
    # Sandbox Execution
    "SandboxExecutionResult",
    "SandboxedExecutor",
    "get_sandboxed_executor",
    # Batch Execution
    "BatchRetryStrategy",
    "BatchConfig",
    "BatchItemResult",
    "BatchProgress",
    "BatchResult",
    "BatchWorkflowExecutor",
    # Protocols (Graph API)
    "RetryPolicy",
    "IWorkflowNode",
    "IWorkflowEdge",
    "IWorkflowGraph",
    "ICheckpointStore",
    "IWorkflowExecutor",
    "IStreamingWorkflowExecutor",
    "GraphNodeStatus",
    "GraphNodeResult",
    # Streaming
    "WorkflowEventType",
    "WorkflowStreamChunk",
    "WorkflowStreamContext",
    "StreamingWorkflowExecutor",
    # YAML Loader
    "YAMLWorkflowError",
    "YAMLWorkflowConfig",
    "YAMLWorkflowProvider",
    "load_workflow_from_dict",
    "load_workflow_from_yaml",
    "load_workflow_from_file",
    "load_workflows_from_directory",
    "load_and_validate",
    # Cache
    "WorkflowCacheConfig",
    "WorkflowNodeCacheEntry",
    "WorkflowCache",
    "WorkflowCacheManager",
    "get_workflow_cache_manager",
    "configure_workflow_cache",
    # WorkflowGraph DSL (compiles to WorkflowDefinition)
    "State",
    "WorkflowGraph",  # Typed workflow graph DSL
    "StateGraph",  # Deprecated alias for WorkflowGraph
    "GraphNode",
    "GraphNodeType",
    "NodeFunc",
    "RouterFunc",
    "Compilable",
    "create_graph",
    "compile_graph",
    # Graph implementation (basic graph container)
    "GraphWorkflowNode",
    "WorkflowEdge",
    "ConditionalEdge",
    "BasicWorkflowGraph",  # Basic graph container (renamed from WorkflowGraph)
    "DuplicateNodeError",
    "InvalidEdgeError",
    "GraphValidationError",
    # YAML to StateGraph Compiler
    "WorkflowState",
    "CompiledGraphNodeResult",
    "create_initial_workflow_state",
    "CompilerConfig",
    "YAMLToStateGraphCompiler",
    "NodeExecutorFactory",
    "WorkflowNodeExecutorRegistration",
    "WorkflowNodeExecutorRegistry",
    "clear_registered_workflow_node_executors",
    "get_workflow_node_executor_registry",
    "register_workflow_node_executor",
    "ConditionEvaluator",
    "compile_yaml_workflow",
    "execute_yaml_workflow",
    # StateGraph Executor
    "ExecutorConfig",
    "ExecutorResult",
    "StateGraphExecutor",
    "get_executor",
    "execute_workflow",
    # Deployment
    "DeploymentTarget",
    "DockerConfig",
    "DockerComposeConfig",
    "KubernetesConfig",
    "ECSConfig",
    "CloudRunConfig",
    "AzureContainerConfig",
    "LambdaConfig",
    "RemoteConfig",
    "SandboxConfig",
    "DeploymentConfig",
    "DeploymentHandler",
    "get_deployment_handler",
    # Services
    "ServiceConfig",
    "ServiceLifecycle",
    "ServiceManager",
    "PostgresService",
    "RedisService",
    "RabbitMQService",
    "KafkaService",
    "HTTPClientService",
    "S3Service",
    # Runtime (serverless-like)
    "HITLServerConfig",
    "RuntimeConfig",
    "RuntimeResult",
    "WorkflowRuntime",
    "run_workflow",
    # HITL (Human-in-the-Loop)
    "HITLNodeType",
    "HITLFallback",
    "HITLMode",
    "HITLCategory",
    "HITL_MODE_CATEGORIES",
    "HITLStatus",
    "HITLRequest",
    "HITLResponse",
    "HITLHandler",
    "HITLNode",
    "HITLExecutor",
    "DefaultHITLHandler",
    "DEPLOYMENT_HITL_COMPATIBILITY",
    "get_supported_hitl_modes",
    "validate_hitl_deployment_compatibility",
    # HITL API (remote deployments)
    "StoredRequest",
    "HITLStore",
    "get_global_store",
    "SQLiteHITLStore",
    "get_sqlite_store",
    "APIHITLHandler",
    "create_hitl_router",
    "create_hitl_app",
    "run_hitl_server",
    # HITL Transports (external integrations)
    "BaseTransportConfig",
    "BaseTransport",
    "HITLTransportProtocol",
    "EmailConfig",
    "SMSConfig",
    "SlackConfig",
    "TeamsConfig",
    "GitHubConfig",
    "GitLabConfig",
    "JiraConfig",
    "PagerDutyConfig",
    "TerraformCloudConfig",
    "CustomHookConfig",
    "EmailTransport",
    "SMSTransport",
    "SlackTransport",
    "GitHubPRTransport",
    "GitHubCheckTransport",
    "CustomHookTransport",
    "register_transport",
    "get_transport",
    "list_available_transports",
    # Unified Execution Context
    "ExecutionContext",
    "create_execution_context",
    "ExecutionContextWrapper",
    "from_workflow_context",
    "to_workflow_context",
    "from_compiler_workflow_state",
    "to_compiler_workflow_state",
    "from_adapter_workflow_state",
    "to_adapter_workflow_state",
    # NodeRunner Protocol and Implementations (ISP + DIP)
    "NodeRunner",
    "NodeRunnerResult",
    "BaseNodeRunner",
    "AgentNodeRunner",
    "ComputeNodeRunner",
    "TransformNodeRunner",
    "HITLNodeRunner",
    "ConditionNodeRunner",
    "ParallelNodeRunner",
    "NodeRunnerRegistry",
    # Graph Compilers (Single Execution Engine - Phase 4)
    "CompilerConfig",
    "NodeRunnerWrapper",
    "WorkflowGraphCompiler",
    "WorkflowDefinitionCompiler",
    "compile_workflow_graph",
    "compile_workflow_definition",
    # Observability (Unified Streaming - Phase 5)
    "StreamingObserver",
    "AsyncStreamingObserver",
    "FunctionObserver",
    "ObservabilityEmitter",
    "create_emitter",
    "create_logging_observer",
    # Unified Compiler (Consistent Compilation and Caching)
    "UnifiedWorkflowCompiler",
    "create_unified_compiler",
]

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


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_DEPRECATED_ALIAS_MAP))
