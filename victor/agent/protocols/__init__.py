"""Protocols for orchestrator services.

All protocols are organized in domain-grouped submodules but remain importable
from this package for full backward compatibility via lazy loading.

Submodules:
    agent_factory        — IAgentFactory, IAgent
    tool_protocols       — Tool registry, pipeline, executor, selector, access control
    conversation_protocols — Conversation controller, state machine, message history
    streaming_protocols  — Streaming controller, recovery, chunk generator
    provider_protocols   — Provider manager, registry, health monitor, switcher
    analysis_protocols   — Task analyzer, complexity classifier, intent classifier
    coordination_protocols — Tool planner, task coordinator, state coordinator
    infrastructure_protocols — Observability, metrics, recovery, sanitizer, etc.
    budget_protocols     — Mode controller, budget manager, budget tracking
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

# Map every exported name to its submodule
_SUBMODULE_MAP: dict[str, str] = {}

_MODULE_MEMBERS = {
    "agent_factory": [
        "IAgentFactory",
        "IAgent",
    ],
    "tool_protocols": [
        "AgentToolSelectionContext",
        "ToolSelectorFeatures",
        "ToolRegistryProtocol",
        "ToolSelectorProtocol",
        "ToolExecutorProtocol",
        "ToolCacheProtocol",
        "ToolRegistrarProtocol",
        "ToolSequenceTrackerProtocol",
        "ToolOutputFormatterProtocol",
        "ToolCallTrackerProtocol",
        "ToolDeduplicationTrackerProtocol",
        "ToolDependencyGraphProtocol",
        "ToolPluginRegistryProtocol",
        "SemanticToolSelectorProtocol",
        "IToolAdapterCoordinator",
        "AccessPrecedence",
        "ToolAccessDecision",
        "ToolAccessContext",
        "IToolAccessController",
    ],
    "conversation_protocols": [
        "ConversationControllerProtocol",
        "ConversationStateMachineProtocol",
        "MessageHistoryProtocol",
        "ConversationEmbeddingStoreProtocol",
        "IMessageStore",
        "IContextOverflowHandler",
        "ISessionManager",
        "IEmbeddingManager",
    ],
    "streaming_protocols": [
        "StreamingControllerProtocol",
        "StreamingCoordinatorProtocol",
        "StreamingRecoveryCoordinatorProtocol",
        "ChunkGeneratorProtocol",
        "StreamingHandlerProtocol",
        "StreamingMetricsCollectorProtocol",
        "StreamingConfidenceMonitorProtocol",
    ],
    "provider_protocols": [
        "ProviderRegistryProtocol",
        "IProviderHealthMonitor",
        "IProviderSwitcher",
        "IProviderEventEmitter",
        "IProviderClassificationStrategy",
    ],
    "analysis_protocols": [
        "TaskAnalyzerProtocol",
        "ComplexityClassifierProtocol",
        "ActionAuthorizerProtocol",
        "SearchRouterProtocol",
        "IntentClassifierProtocol",
        "TaskTypeHinterProtocol",
        "IToolCallClassifier",
    ],
    "coordination_protocols": [
        "CoordinationAdvisorRuntimeProtocol",
        "ToolPlannerProtocol",
        "TaskCoordinatorProtocol",
        "ToolCoordinatorProtocol",
        "StateCoordinatorProtocol",
        "PromptCoordinatorProtocol",
        "UnifiedMemoryCoordinatorProtocol",
    ],
    "infrastructure_protocols": [
        "ObservabilityProtocol",
        "MetricsCollectorProtocol",
        "FailureDetectorProtocol",
        "RecoveryExecutorProtocol",
        "RecoveryDiagnosticsProtocol",
        "RecoveryHandlerProtocol",
        "ResponseSanitizerProtocol",
        "ArgumentNormalizerProtocol",
        "ProjectContextProtocol",
        "CodeExecutionManagerProtocol",
        "WorkflowRegistryProtocol",
        "UsageAnalyticsProtocol",
        "ContextCompactorProtocol",
        "DebugLoggerProtocol",
        "ReminderManagerProtocol",
        "RLCoordinatorProtocol",
        "SafetyCheckerProtocol",
        "AutoCommitterProtocol",
        "MCPBridgeProtocol",
        "UsageLoggerProtocol",
        "SystemPromptBuilderProtocol",
        "ParallelExecutorProtocol",
        "ResponseCompleterProtocol",
        "VerticalMiddlewareStorageProtocol",
        "SafetyPatternStorageProtocol",
        "TeamSpecStorageProtocol",
        "VerticalStorageProtocol",
        "TaskTrackerProtocol",
        "CompactionSummarizerProtocol",
        "HierarchicalCompactionProtocol",
        "SessionContextLinkerProtocol",
    ],
    "budget_protocols": [
        "ModeControllerProtocol",
        "BudgetType",
        "BudgetStatus",
        "BudgetConfig",
        "IBudgetManager",
        "IBudgetTracker",
        "IMultiplierCalculator",
        "IModeCompletionChecker",
    ],
    "context_protocols": [
        "ContextTemperatureClassifierProtocol",
    ],
}

# Build the reverse map
for _mod, _names in _MODULE_MEMBERS.items():
    for _name in _names:
        _SUBMODULE_MAP[_name] = _mod

__all__ = list(_SUBMODULE_MAP.keys())

_DEPRECATED_EXPORTS = {
    "IntentClassifierProtocol": (
        "victor.agent.protocols.IntentClassifierProtocol is deprecated "
        "compatibility surface. Prefer "
        "victor.agent.services.protocols.IntentClassifierProtocol."
    ),
    "CoordinationAdvisorRuntimeProtocol": (
        "victor.agent.protocols.CoordinationAdvisorRuntimeProtocol is deprecated "
        "compatibility surface. Prefer "
        "victor.agent.services.protocols.CoordinationAdvisorRuntimeProtocol."
    ),
    "ReminderManagerProtocol": (
        "victor.agent.protocols.ReminderManagerProtocol is deprecated "
        "compatibility surface. Prefer "
        "victor.agent.services.protocols.ReminderManagerProtocol."
    ),
    "ResponseSanitizerProtocol": (
        "victor.agent.protocols.ResponseSanitizerProtocol is deprecated "
        "compatibility surface. Prefer "
        "victor.agent.services.protocols.ResponseSanitizerProtocol."
    ),
    "ToolCoordinatorProtocol": (
        "victor.agent.protocols.ToolCoordinatorProtocol is deprecated compatibility "
        "surface. Prefer ToolServiceProtocol."
    ),
    "ChunkGeneratorProtocol": (
        "victor.agent.protocols.ChunkGeneratorProtocol is deprecated compatibility "
        "surface. Prefer victor.agent.services.protocols.ChunkRuntimeProtocol."
    ),
    "ToolPlannerProtocol": (
        "victor.agent.protocols.ToolPlannerProtocol is deprecated compatibility "
        "surface. Prefer victor.agent.services.protocols.ToolPlanningRuntimeProtocol."
    ),
    "TaskCoordinatorProtocol": (
        "victor.agent.protocols.TaskCoordinatorProtocol is deprecated compatibility "
        "surface. Prefer victor.agent.services.protocols.TaskRuntimeProtocol."
    ),
    "StateCoordinatorProtocol": (
        "victor.agent.protocols.StateCoordinatorProtocol is deprecated compatibility "
        "surface. Prefer victor.agent.services.protocols.StateRuntimeProtocol."
    ),
    "PromptCoordinatorProtocol": (
        "victor.agent.protocols.PromptCoordinatorProtocol is deprecated compatibility "
        "surface. Prefer victor.agent.services.protocols.PromptRuntimeProtocol."
    ),
    "StreamingRecoveryCoordinatorProtocol": (
        "victor.agent.protocols.StreamingRecoveryCoordinatorProtocol is deprecated "
        "compatibility surface. Prefer "
        "victor.agent.services.protocols.StreamingRecoveryRuntimeProtocol."
    ),
    "StreamingHandlerProtocol": (
        "victor.agent.protocols.StreamingHandlerProtocol is deprecated "
        "compatibility surface. Prefer "
        "victor.agent.services.protocols.StreamingHandlerProtocol."
    ),
    "StreamingMetricsCollectorProtocol": (
        "victor.agent.protocols.StreamingMetricsCollectorProtocol is deprecated "
        "compatibility surface. Prefer "
        "victor.agent.services.protocols.StreamingMetricsCollectorProtocol."
    ),
    "StreamingConfidenceMonitorProtocol": (
        "victor.agent.protocols.StreamingConfidenceMonitorProtocol is deprecated "
        "compatibility surface. Prefer "
        "victor.agent.services.protocols.StreamingConfidenceMonitorProtocol."
    ),
    "RLCoordinatorProtocol": (
        "victor.agent.protocols.RLCoordinatorProtocol is deprecated compatibility "
        "surface. Prefer victor.agent.services.protocols.RLLearningRuntimeProtocol."
    ),
}


def __getattr__(name: str) -> Any:
    """Lazy import: resolve protocol names on first access."""
    if name in _SUBMODULE_MAP:
        mod = importlib.import_module(f"victor.agent.protocols.{_SUBMODULE_MAP[name]}")
        value = getattr(mod, name)
        if name in _DEPRECATED_EXPORTS:
            warnings.warn(
                _DEPRECATED_EXPORTS[name],
                DeprecationWarning,
                stacklevel=2,
            )
            return value

        # Cache on the module for subsequent access
        globals()[name] = value
        return value
    raise AttributeError(f"module 'victor.agent.protocols' has no attribute {name!r}")
