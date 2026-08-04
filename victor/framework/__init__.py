from __future__ import annotations

"""Victor Framework - Simplified API for AI coding agents.

This module provides a "golden path" API that covers 90% of use cases
with 5 core concepts:

1. **Agent** - Single entry point for creating agents
2. **Task** - What the agent should accomplish (TaskResult)
3. **Tools** - Available capabilities (ToolSet with presets)
4. **State** - Observable conversation state (Stage enum)
5. **Event** - Stream of observations (EventType enum)

Quick Start:
    from victor.framework import Agent

    # Simple use case
    agent = await Agent.create(provider="anthropic")
    result = await agent.run("Write a hello world function")
    print(result.content)

Advanced Config:
    from victor.framework import Agent, UnifiedAgentConfig

    agent = await Agent.create(config=UnifiedAgentConfig.high_budget())

Streaming:
    async for event in agent.stream("Refactor this code"):
        if event.type == EventType.CONTENT:
            print(event.content, end="")
        elif event.type == EventType.TOOL_CALL:
            print(f"Using {event.tool_name}")

Tools:
    # Default tools (recommended)
    agent = await Agent.create(tools=ToolSet.default())

    # Minimal for simple tasks
    agent = await Agent.create(tools=ToolSet.minimal())

    # Full access to every registered tool in the selected ToolSet
    agent = await Agent.create(tools=ToolSet.full())

State Observation:
    print(f"Stage: {agent.state.stage}")
    print(f"Tools used: {agent.state.tool_calls_used}/{agent.state.tool_budget}")

    agent.on_state_change(lambda old, new: print(f"{old.stage} -> {new.stage}"))

Extensions:
    # Build integrations on documented framework APIs and extension contracts.
    # Runtime internals are intentionally not part of this public quick-start.
"""

import warnings

# === CORE: Always imported (the 5 concepts + errors + protocols) ===
from victor.framework._api import *  # noqa: F401,F403
from victor.framework._api import PUBLIC_API_NAMES

_CORE_NAMES = PUBLIC_API_NAMES + ["discover"]

# === LAZY: Feature groups loaded on demand via PEP 562 ===
_LAZY_IMPORTS: dict[str, list[str]] = {
    "victor.framework.event_registry": [
        "BaseEventConverter",
        "EventConverterProtocol",
        "EventRegistry",
        "EventTarget",
        "convert_from_cqrs",
        "convert_from_observability",
        "convert_to_cqrs",
        "convert_to_observability",
        "get_event_registry",
    ],
    "victor.framework.agent_components": [
        "AgentBridge",
        "AgentBuilder",
        "AgentBuildOptions",
        "AgentSession",
        "BridgeConfiguration",
        "BuilderPreset",
        "SessionContext",
        "SessionLifecycleHooks",
        "SessionMetrics",
        "SessionState",
        "create_bridge",
        "create_builder",
        "create_session",
    ],
    "victor.framework.tool_config": [
        "AirgappedFilter",
        "CostTierFilter",
        "SecurityFilter",
        "ToolConfig",
        "ToolConfigBuilder",
        "ToolConfigEntry",
        "ToolConfigMode",
        "ToolConfigResult",
        "ToolConfigurator",
        "ToolFilterProtocol",
        "configure_tools",
        "configure_tools_from_toolset",
        "get_tool_configurator",
    ],
    "victor.framework.service_provider": [
        "AgentBuilderService",
        "AgentSessionService",
        "EventRegistryService",
        "FrameworkScope",
        "FrameworkServiceProvider",
        "ToolConfiguratorService",
        "configure_framework_services",
        "create_framework_scope",
    ],
    "victor.framework.resilience": [
        "CircuitBreaker",
        "CircuitBreakerError",
        "CircuitBreakerRegistry",
        "CircuitState",
        "CircuitBreakerConfig",
        "CircuitBreakerState",
        "CircuitOpenError",
        "ProviderUnavailableError",
        "ResilientProvider",
        "ProviderRetryConfig",
        "RetryExhaustedError",
        "ProviderRetryStrategy",
        "ExponentialBackoffStrategy",
        "FixedDelayStrategy",
        "LinearBackoffStrategy",
        "NoRetryStrategy",
        "RetryContext",
        "RetryExecutor",
        "RetryOutcome",
        "RetryResult",
        "BaseRetryStrategy",
        "connection_retry_strategy",
        "provider_retry_strategy",
        "tool_retry_strategy",
        "with_retry",
        "with_retry_sync",
    ],
    "victor.framework.health": [
        "BaseHealthCheck",
        "CacheHealthCheck",
        "CallableHealthCheck",
        "ComponentHealth",
        "HealthCheckProtocol",
        "HealthChecker",
        "HealthReport",
        "HealthStatus",
        "MemoryHealthCheck",
        "ProviderHealthCheck",
        "ToolHealthCheck",
        "create_default_health_checker",
        "HealthCheckResult",
        "ProviderHealthResult",
        "ProviderHealthStatus",
        "ProviderHealthChecker",
        "ProviderHealthReport",
        "get_provider_health_checker",
        "reset_provider_health_checker",
    ],
    "victor.framework.metrics": [
        "Counter",
        "Gauge",
        "Histogram",
        "Metric",
        "MetricLabels",
        "MetricsCollector",
        "MetricsRegistry",
        "Timer",
        "TimerContext",
        "get_meter",
        "get_tracer",
        "is_telemetry_enabled",
        "setup_opentelemetry",
    ],
    "victor.framework.teams": [
        "AgentTeam",
        "TeamEvent",
        "TeamEventType",
        "TeamMemberSpec",
        "member_complete_event",
        "member_start_event",
        "team_complete_event",
        "team_start_event",
    ],
    "victor.teams": [
        "MemberResult",
        "TeamFormation",
        "TeamResult",
    ],
    "victor.teams.types": [
        "TeamConfig",
        "TeamMember",
    ],
    "victor.framework.rl": [
        "BaseLearner",
        "LearnerStats",
        "LearnerType",
        "RLCoordinator",
        "RLManager",
        "RLOutcome",
        "RLRecommendation",
        "RLStats",
        "analyze_prompt_rollout_experiment",
        "analyze_prompt_rollout_experiment_async",
        "apply_prompt_rollout_recommendation",
        "apply_prompt_rollout_recommendation_async",
        "process_prompt_candidate_evaluation_suite",
        "process_prompt_candidate_evaluation_suite_async",
        "PromptCandidateSuiteWorkflowResult",
        "create_prompt_rollout_experiment",
        "create_prompt_rollout_experiment_async",
        "create_outcome",
        "get_rl_coordinator",
        "get_rl_coordinator_async",
        "record_tool_success",
    ],
    "victor.framework.team_registry": [
        "TeamSpecRegistry",
        "TeamSpecEntry",
        "get_team_registry",
        "register_team_spec",
        "get_team_spec",
    ],
    "victor.framework.graph": [
        "StateGraph",
        "CompiledGraph",
        "Node",
        "Edge",
        "EdgeType",
        "FrameworkNodeStatus",
        "GraphExecutionResult",
        "GraphConfig",
        "WorkflowCheckpoint",
        "CheckpointerProtocol",
        "MemoryCheckpointer",
        "RLCheckpointerAdapter",
        "StateProtocol",
        "NodeFunctionProtocol",
        "ConditionFunctionProtocol",
        "END",
        "START",
        "create_graph",
    ],
    "victor.framework.execution_checkpoint": [
        "ApprovalState",
        "ExecutionCheckpoint",
        "execution_checkpoint_trace_metadata",
        "normalize_execution_checkpoint_context",
    ],
    "victor.framework.module_loader": [
        "CachedEntryPoints",
        "DebouncedReloadTimer",
        "DynamicModuleLoader",
        "EntryPointCache",
        "get_entry_point_cache",
    ],
    "victor.framework.entry_point_loader": [
        "load_safety_rules_from_entry_points",
        "load_tool_dependency_provider_from_entry_points",
        "load_rl_config_from_entry_points",
        "register_escape_hatches_from_entry_points",
        "register_commands_from_entry_points",
        "list_installed_verticals",
    ],
    "victor.framework.capability_loader": [
        "CapabilityLoader",
        "CapabilityEntry",
        "CapabilityLoadError",
        "capability",
        "create_capability_loader",
        "get_default_capability_loader",
    ],
    "victor.framework.tool_naming": [
        "CANONICAL_TO_ALIASES",
        "TOOL_ALIASES",
        "ToolNameEntry",
        "ToolNames",
        "canonicalize_dependencies",
        "canonicalize_tool_dict",
        "canonicalize_tool_list",
        "canonicalize_tool_set",
        "canonicalize_transitions",
        "get_aliases",
        "get_all_canonical_names",
        "get_canonical_name",
        "get_legacy_names_report",
        "get_name_mapping",
        "is_valid_tool_name",
        "validate_tool_names",
    ],
    "victor.framework.middleware": [
        "GitSafetyMiddleware",
        "LoggingMiddleware",
        "MetricsMiddleware",
        "SecretMaskingMiddleware",
        "ToolMetrics",
    ],
    "victor.framework.task_types": [
        "TaskCategory",
        "TaskTypeDefinition",
        "TaskTypeRegistry",
        "get_task_budget",
        "get_task_hint",
        "get_task_type_registry",
        "register_vertical_task_type",
        "register_coding_task_types",
        "register_data_analysis_task_types",
        "register_devops_task_types",
        "register_research_task_types",
        "setup_vertical_task_types",
    ],
    "victor.framework.vertical_integration": [
        "IntegrationResult",
        "OrchestratorVerticalProtocol",
        "VerticalIntegrationPipeline",
        "apply_vertical",
        "create_integration_pipeline",
        "create_integration_pipeline_with_handlers",
    ],
    "victor.framework.vertical_cache_policy": [
        "InMemoryLRUVerticalIntegrationCachePolicy",
        "VerticalIntegrationCachePolicy",
    ],
    "victor.framework.step_handlers": [
        "BaseStepHandler",
        "ConfigStepHandler",
        "ContextStepHandler",
        "ExtensionsStepHandler",
        "FrameworkStepHandler",
        "MiddlewareStepHandler",
        "PromptStepHandler",
        "SafetyStepHandler",
        "StepHandlerProtocol",
        "StepHandlerRegistry",
        "ToolStepHandler",
    ],
    "victor.framework.hitl": [
        "ApprovalHandler",
        "ApprovalRequest",
        "ApprovalStatus",
        "HITLCheckpoint",
        "HITLController",
    ],
    "victor.framework.personas": [
        "Persona",
        "PERSONA_REGISTRY",
        "get_persona",
        "register_persona",
        "list_personas",
    ],
    "victor.framework.agent_protocols": [
        "AgentCapability",
        "AgentMessage",
        "IAgentPersona",
        "IAgentRole",
        "ITeamCoordinator",
        "ITeamMember",
        "MessageType",
    ],
    "victor.framework.agent_roles": [
        "ExecutorRole",
        "ManagerRole",
        "ResearcherRole",
        "ReviewerRole",
        "ROLE_REGISTRY",
        "SupervisorRole",
        "get_role",
    ],
    "victor.framework.chain_registry": [
        "ChainMetadata",
        "ChainRegistry",
        "get_chain",
        "get_chain_registry",
        "register_chain",
    ],
    "victor.framework.persona_registry": [
        "PersonaRegistry",
        "PersonaSpec",
        "get_persona_registry",
        "get_persona_spec",
        "register_persona_spec",
    ],
    "victor.framework.prompt_builder": [
        "PromptBuilder",
        "PromptSection",
        "ToolHint",
        "create_coding_prompt_builder",
        "create_data_analysis_prompt_builder",
        "create_devops_prompt_builder",
        "create_research_prompt_builder",
    ],
    "victor.framework.prompt_document": [
        "PromptBlock",
        "PromptDocument",
        "PromptDocumentProcessor",
        "PromptDeduplicationProcessor",
        "PromptPriorityTrimProcessor",
    ],
    "victor.framework.prompt_sections": [
        "GROUNDING_RULES_EXTENDED",
        "GROUNDING_RULES_MINIMAL",
        "PARALLEL_READ_GUIDANCE",
        "CODING_GUIDELINES",
        "CODING_IDENTITY",
        "CODING_TOOL_USAGE",
        "DEVOPS_COMMON_PITFALLS",
        "DEVOPS_GROUNDING",
        "DEVOPS_IDENTITY",
        "DEVOPS_SECURITY_CHECKLIST",
        "RESEARCH_GROUNDING",
        "RESEARCH_IDENTITY",
        "RESEARCH_QUALITY_CHECKLIST",
        "RESEARCH_SOURCE_HIERARCHY",
        "DATA_ANALYSIS_GROUNDING",
        "DATA_ANALYSIS_IDENTITY",
        "DATA_ANALYSIS_LIBRARIES",
        "DATA_ANALYSIS_OPERATIONS",
    ],
    "victor.framework.stage_manager": [
        "StageDefinition",
        "StageManager",
        "StageManagerConfig",
        "StageManagerProtocol",
        "StageTransition",
        "create_stage_manager",
        "get_coding_stages",
        "get_data_analysis_stages",
        "get_research_stages",
    ],
    "victor.framework.lsp_protocols": [
        "LSPCompletionItem",
        "LSPDiagnostic",
        "LSPHoverInfo",
        "LSPLocation",
        "LSPPoolProtocol",
        "LSPPosition",
        "LSPRange",
        "LSPServiceProtocol",
        "LSPSymbol",
    ],
    "victor.framework.workflow_engine": [
        "WorkflowErrorType",
        "WorkflowExecutionResult",
        "WorkflowEngine",
        "WorkflowEngineConfig",
        "WorkflowEngineProtocol",
        "WorkflowEvent",
        "create_workflow_engine",
        "run_graph_workflow",
        "run_yaml_workflow",
    ],
    "victor.framework.observability": [
        "ObservabilityManager",
        "ObservabilityConfig",
        "DashboardData",
        "MetricType",
        "MetricLabel",
        "CounterMetric",
        "GaugeMetric",
        "HistogramBucket",
        "HistogramMetric",
        "SummaryMetric",
        "MetricsSnapshot",
        "MetricsCollection",
        "MetricSource",
        "MetricNames",
    ],
    "victor.framework.contextual_errors": [
        "ContextualError",
        "ProviderConnectionError",
        "ToolExecutionError",
        "FileOperationError",
        "ResourceError",
        "create_provider_error",
        "create_tool_error",
        "create_file_error",
        "wrap_error",
        "format_exception_for_user",
    ],
    "victor.framework.perception_integration": [
        "Perception",
        "PerceptionIntegration",
        "Requirement",
        "RequirementType",
        "SimilarExperience",
        "perceive",
    ],
    "victor.framework.evaluation_nodes": [
        "EvaluationCheckpoint",
        "EvaluationDecision",
        "EvaluationNode",
        "EvaluationResult",
        "add_evaluation",
        "composite_evaluator",
        "convergence_evaluator",
        "create_agentic_loop_graph",
        "error_count_evaluator",
        "progress_tracking_evaluator",
        "simple_score_evaluator",
    ],
    "victor.framework.fulfillment": [
        "FulfillmentConfig",
        "FulfillmentDetector",
        "FulfillmentResult",
        "FulfillmentStatus",
        "FulfillmentStrategy",
        "TaskType",
    ],
    "victor.framework.agentic_loop": [
        "AgenticLoop",
        "LoopIteration",
        "LoopResult",
        "LoopStage",
    ],
}

# Phase 9 export counts for testing
_RESILIENCE_EXPORTS = [
    "CircuitBreaker",
    "CircuitBreakerError",
    "CircuitBreakerRegistry",
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "CircuitOpenError",
    "ProviderUnavailableError",
    "ResilientProvider",
    "ProviderRetryConfig",
    "RetryExhaustedError",
    "ProviderRetryStrategy",
    "ExponentialBackoffStrategy",
    "FixedDelayStrategy",
    "LinearBackoffStrategy",
    "NoRetryStrategy",
    "RetryContext",
    "RetryExecutor",
    "RetryOutcome",
    "RetryResult",
    "BaseRetryStrategy",
    "connection_retry_strategy",
    "provider_retry_strategy",
    "tool_retry_strategy",
    "with_retry",
    "with_retry_sync",
]

_HEALTH_EXPORTS = [
    "BaseHealthCheck",
    "CacheHealthCheck",
    "CallableHealthCheck",
    "ComponentHealth",
    "HealthCheckProtocol",
    "HealthChecker",
    "HealthReport",
    "HealthStatus",
    "MemoryHealthCheck",
    "ProviderHealthCheck",
    "ToolHealthCheck",
    "create_default_health_checker",
    "HealthCheckResult",
    "ProviderHealthResult",
    "ProviderHealthStatus",
    "ProviderHealthChecker",
    "ProviderHealthReport",
    "get_provider_health_checker",
    "reset_provider_health_checker",
]

_METRICS_EXPORTS = [
    "Counter",
    "Gauge",
    "Histogram",
    "Metric",
    "MetricLabels",
    "MetricsCollector",
    "MetricsRegistry",
    "Timer",
    "TimerContext",
    "get_meter",
    "get_tracer",
    "is_telemetry_enabled",
    "setup_opentelemetry",
]

# Aliased imports: name -> (module, real_name)
_ALIASED_IMPORTS: dict[str, tuple[str, str]] = {
    "AgentTeamFormation": ("victor.framework.agent_protocols", "TeamFormation"),
    "CoordinatorMemberResult": ("victor.teams", "MemberResult"),
    "CoordinatorTeamResult": ("victor.teams", "TeamResult"),
    # create_builder exists in both agent_components and service_provider;
    # keep the service_provider version as the canonical lazy target since
    # agent_components also defines it (first-wins in _NAME_TO_MODULE)
}

_DEPRECATED_COMPAT_EXPORTS: dict[str, tuple[str, str, str]] = {
    "FrameworkShim": (
        "victor.framework.shim",
        "FrameworkShim",
        "victor.framework.FrameworkShim is deprecated. "
        "Use Agent.create() from the Framework API instead. "
        "This warning-backed compatibility export remains supported through "
        "v1.0.0 (2027-06-30). "
        "See docs/architecture/migration.md for migration guidance.",
    ),
}

# Build reverse lookup: name -> module path
_NAME_TO_MODULE: dict[str, str] = {}
for _mod, _names in _LAZY_IMPORTS.items():
    for _name in _names:
        if _name not in _NAME_TO_MODULE:
            _NAME_TO_MODULE[_name] = _mod


def __getattr__(name: str):
    if name in _DEPRECATED_COMPAT_EXPORTS:
        import importlib

        mod_path, real_name, message = _DEPRECATED_COMPAT_EXPORTS[name]
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        mod = importlib.import_module(mod_path)
        value = getattr(mod, real_name)
        globals()[name] = value
        return value

    # Check aliased imports first
    if name in _ALIASED_IMPORTS:
        import importlib

        mod_path, real_name = _ALIASED_IMPORTS[name]
        mod = importlib.import_module(mod_path)
        value = getattr(mod, real_name)
        globals()[name] = value
        return value

    # Check lazy imports
    if name in _NAME_TO_MODULE:
        import importlib

        mod = importlib.import_module(_NAME_TO_MODULE[name])
        value = getattr(mod, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module 'victor.framework' has no attribute {name!r}")


def __dir__():
    return sorted(set(_CORE_NAMES) | set(_NAME_TO_MODULE.keys()) | set(_ALIASED_IMPORTS.keys()))


__all__ = list(__dir__())


def discover() -> dict:
    """Discover all Victor framework capabilities programmatically.

    Returns a dictionary containing all available tools, verticals, personas,
    teams, chains, handlers, task types, providers, and events.

    Example:
        from victor.framework import discover

        caps = discover()
        print(f"Available tools: {len(caps['tools'])}")
        print(f"Available verticals: {caps['verticals']}")

    Returns:
        dict: Capability manifest with keys:
            - tools: List of tool names
            - tool_categories: List of category names
            - verticals: List of vertical names
            - personas: List of persona names
            - teams: List of team names
            - chains: List of chain/workflow names
            - handlers: List of handler names
            - task_types: List of task type names
            - providers: List of provider names
            - events: List of event type names
            - summary: Aggregated counts
    """
    from victor.ui.commands.capabilities import get_capability_discovery

    discovery = get_capability_discovery()
    manifest = discovery.discover_all()
    return manifest.to_dict()


# Version sourced from pyproject.toml via importlib.metadata
try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("victor-ai")
except Exception:
    __version__ = "0.0.0"
