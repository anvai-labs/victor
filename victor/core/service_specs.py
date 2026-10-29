"""Shared service specification metadata used by DI providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from victor.core.container import ServiceLifetime

# Agent protocols
from victor.agent.protocols import (
    ActionAuthorizerProtocol,
    ArgumentNormalizerProtocol,
    AutoCommitterProtocol,
    IBudgetManager,
    CodeExecutionManagerProtocol,
    CompactionSummarizerProtocol,
    ComplexityClassifierProtocol,
    ConversationEmbeddingStoreProtocol,
    ContextCompactorProtocol,
    DebugLoggerProtocol,
    HierarchicalCompactionProtocol,
    IToolAccessController,
    MCPBridgeProtocol,
    MetricsCollectorProtocol,
    ModeControllerProtocol,
    ParallelExecutorProtocol,
    ProjectContextProtocol,
    ProviderRegistryProtocol,
    ResponseCompleterProtocol,
    SafetyCheckerProtocol,
    SearchRouterProtocol,
    SemanticToolSelectorProtocol,
    SessionContextLinkerProtocol,
    SystemPromptBuilderProtocol,
    TaskTypeHinterProtocol,
    ToolCacheProtocol,
    ToolDeduplicationTrackerProtocol,
    ToolExecutorProtocol,
    ToolOutputFormatterProtocol,
    ToolSelectorProtocol,
    ToolSequenceTrackerProtocol,
    ToolDependencyGraphProtocol,
    ToolPluginRegistryProtocol,
    ContextTemperatureClassifierProtocol,
    UsageAnalyticsProtocol,
    UsageLoggerProtocol,
    WorkflowRegistryProtocol,
)
from victor.agent.services.protocols import (
    ChunkRuntimeProtocol,
    CoordinationAdvisorRuntimeProtocol,
    IntentClassifierProtocol,
    PromptRuntimeProtocol,
    RLLearningRuntimeProtocol,
    ReminderManagerProtocol,
    ResponseSanitizerProtocol,
    StateRuntimeProtocol,
    StreamingConfidenceMonitorProtocol,
    StreamingHandlerProtocol,
    StreamingMetricsCollectorProtocol,
    StreamingRecoveryRuntimeProtocol,
    TaskRuntimeProtocol,
    ToolPlanningRuntimeProtocol,
)

# Backward-compatible aliases for renamed protocols
ToolAccessControllerProtocol = IToolAccessController
BudgetManagerProtocol = IBudgetManager

# Workflow protocols / implementations
from victor.workflows.compiler_protocols import (
    ExecutionContextProtocol,
    NodeExecutorFactoryProtocol,
    WorkflowCompilerProtocol,
)
from victor.workflows.compiler.workflow_compiler_impl import WorkflowCompilerImpl
from victor.workflows.compiled_executor import WorkflowExecutor
from victor.workflows.orchestrator_pool import OrchestratorPool
from victor.workflows.validator import WorkflowValidator


@dataclass(frozen=True)
class ServiceSpec:
    protocol: type
    factory_attr: str
    lifetime: ServiceLifetime = ServiceLifetime.SINGLETON
    pass_container: bool = False
    depends_on: tuple = ()
    phase: str = "orchestrator"
    critical: bool = True


AGENT_SINGLETON_SPECS: List[ServiceSpec] = [
    ServiceSpec(ComplexityClassifierProtocol, "_create_complexity_classifier"),
    ServiceSpec(ActionAuthorizerProtocol, "_create_action_authorizer"),
    ServiceSpec(SearchRouterProtocol, "_create_search_router"),
    ServiceSpec(ResponseSanitizerProtocol, "_create_response_sanitizer"),
    ServiceSpec(ArgumentNormalizerProtocol, "_create_argument_normalizer"),
    ServiceSpec(ProjectContextProtocol, "_create_project_context"),
    ServiceSpec(CodeExecutionManagerProtocol, "_create_code_execution_manager"),
    ServiceSpec(WorkflowRegistryProtocol, "_create_workflow_registry"),
    ServiceSpec(UsageAnalyticsProtocol, "_create_usage_analytics"),
    ServiceSpec(ToolSequenceTrackerProtocol, "_create_tool_sequence_tracker"),
    ServiceSpec(ContextCompactorProtocol, "_create_context_compactor"),
    ServiceSpec(ModeControllerProtocol, "_create_mode_controller"),
    ServiceSpec(ToolDeduplicationTrackerProtocol, "_create_tool_deduplication_tracker"),
    ServiceSpec(DebugLoggerProtocol, "_create_debug_logger"),
    ServiceSpec(TaskTypeHinterProtocol, "_create_task_type_hinter"),
    ServiceSpec(
        ReminderManagerProtocol,
        "_create_reminder_manager",
        ServiceLifetime.SCOPED,
    ),
    ServiceSpec(RLLearningRuntimeProtocol, "_create_rl_coordinator"),
    ServiceSpec(SafetyCheckerProtocol, "_create_safety_checker"),
    ServiceSpec(AutoCommitterProtocol, "_create_auto_committer"),
    ServiceSpec(MCPBridgeProtocol, "_create_mcp_bridge"),
    ServiceSpec(ToolDependencyGraphProtocol, "_create_tool_dependency_graph"),
    ServiceSpec(ToolPluginRegistryProtocol, "_create_tool_plugin_registry"),
    ServiceSpec(
        ContextTemperatureClassifierProtocol,
        "_create_context_temperature_classifier",
        critical=False,
    ),
    ServiceSpec(SemanticToolSelectorProtocol, "_create_semantic_tool_selector"),
    ServiceSpec(ProviderRegistryProtocol, "_create_provider_registry"),
    ServiceSpec(ConversationEmbeddingStoreProtocol, "_create_conversation_embedding_store"),
    ServiceSpec(MetricsCollectorProtocol, "_create_metrics_collector"),
    ServiceSpec(ToolCacheProtocol, "_create_tool_cache"),
    ServiceSpec(UsageLoggerProtocol, "_create_usage_logger"),
    ServiceSpec(
        StreamingMetricsCollectorProtocol,
        "_create_streaming_metrics_collector",
        ServiceLifetime.SCOPED,
    ),
    ServiceSpec(IntentClassifierProtocol, "_create_intent_classifier"),
    ServiceSpec(SystemPromptBuilderProtocol, "_create_system_prompt_builder"),
    ServiceSpec(ToolSelectorProtocol, "_create_tool_selector"),
    ServiceSpec(ToolExecutorProtocol, "_create_tool_executor"),
    ServiceSpec(ToolOutputFormatterProtocol, "_create_tool_output_formatter"),
    ServiceSpec(ParallelExecutorProtocol, "_create_parallel_executor"),
    ServiceSpec(ResponseCompleterProtocol, "_create_response_completer"),
    ServiceSpec(
        StreamingHandlerProtocol,
        "_create_streaming_handler",
        ServiceLifetime.SCOPED,
    ),
    ServiceSpec(
        StreamingConfidenceMonitorProtocol,
        "_create_confidence_monitor",
        ServiceLifetime.SCOPED,
        critical=False,
    ),
    ServiceSpec(StreamingRecoveryRuntimeProtocol, "_create_recovery_coordinator"),
    ServiceSpec(ChunkRuntimeProtocol, "_create_chunk_generator"),
    ServiceSpec(ToolPlanningRuntimeProtocol, "_create_tool_planner"),
    ServiceSpec(TaskRuntimeProtocol, "_create_task_coordinator"),
    ServiceSpec(CoordinationAdvisorRuntimeProtocol, "_create_coordination_advisor_runtime"),
    ServiceSpec(CompactionSummarizerProtocol, "_create_compaction_summarizer"),
    ServiceSpec(HierarchicalCompactionProtocol, "_create_hierarchical_compaction_manager"),
    ServiceSpec(SessionContextLinkerProtocol, "_create_session_context_linker"),
    ServiceSpec(
        StateRuntimeProtocol,
        "_create_state_runtime",
        ServiceLifetime.SCOPED,
        pass_container=True,
    ),
    ServiceSpec(
        PromptRuntimeProtocol,
        "_create_prompt_runtime",
        ServiceLifetime.SCOPED,
    ),
]

# Vertical extension specs — used by BaseVerticalServiceProvider
VERTICAL_EXTENSION_SPECS: List[ServiceSpec] = []

WORKFLOW_SINGLETON_SPECS: List[ServiceSpec] = [
    ServiceSpec(NodeExecutorFactoryProtocol, "_create_node_executor_factory"),
    ServiceSpec(WorkflowValidator, "_create_workflow_validator"),
    ServiceSpec(OrchestratorPool, "_create_orchestrator_pool", pass_container=True),
]

WORKFLOW_SCOPED_SPECS: List[ServiceSpec] = [
    ServiceSpec(
        ExecutionContextProtocol,
        "_create_execution_context",
        ServiceLifetime.SCOPED,
        pass_container=True,
    ),
]

WORKFLOW_TRANSIENT_SPECS: List[ServiceSpec] = [
    ServiceSpec(
        WorkflowCompilerImpl,
        "_create_workflow_compiler_impl",
        ServiceLifetime.TRANSIENT,
    ),
    ServiceSpec(
        WorkflowCompilerProtocol,
        "_create_workflow_compiler_impl",
        ServiceLifetime.TRANSIENT,
    ),
    ServiceSpec(WorkflowExecutor, "_create_workflow_executor", ServiceLifetime.TRANSIENT),
]
