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

"""Agent module - orchestrator and supporting components."""

import os
import sys as _sys
import types as _types
from typing import Any

_LIGHT_IMPORT = str(os.getenv("VICTOR_LIGHT_IMPORT", "")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

if not _LIGHT_IMPORT:
    from victor.agent.argument_normalizer import (
        ArgumentNormalizer,
        NormalizationStrategy,
    )

    # OBSOLETE: ConfigLoader moved to victor/agent/archive/obsolete/config_loader.py
    # Configuration is now loaded via Settings objects and composed by OrchestratorFactory
    from victor.agent.config import UnifiedAgentConfig, AgentLifecycleMode
    from victor.agent.tool_selection import get_critical_tools
    from victor.agent.message_history import MessageHistory

    # OBSOLETE: observability moved to victor/agent/archive/obsolete/observability.py
    # Observability is now handled by victor.core.events (get_observability_bus)
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.stream_handler import (
        StreamHandler,
        StreamResult,
        StreamMetrics,
        StreamBuffer,
    )
    from victor.agent.tool_executor import ToolExecutor, ToolExecutionResult

    # New decomposed components
    from victor.agent.conversation.controller import (
        ConversationController,
        ConversationConfig,
        ContextMetrics,
    )
    from victor.agent.tool_pipeline import (
        ToolPipeline,
        ToolPipelineConfig,
        ToolCallResult,
        PipelineExecutionResult,
    )
    from victor.agent.streaming_controller import (
        StreamingController,
        StreamingControllerConfig,
        StreamingSession,
    )
    from victor.agent.task_analyzer import (
        TaskAnalyzer,
        TaskAnalysis,
        get_task_analyzer,
        reset_task_analyzer,
    )

    # Runtime-intelligence components (RL-based learning, quality scoring,
    # prompt optimization)
    from victor.agent.runtime_intelligence_pipeline import (
        RuntimeIntelligencePipeline,
        RequestContext,
        ResponseResult,
        PipelineStats,
        get_runtime_intelligence_pipeline,
        clear_runtime_intelligence_pipeline_cache,
    )
    from victor.agent.orchestrator_integration import (
        OrchestratorIntegration,
        IntegrationConfig,
        IntegrationMetrics,
        enhance_orchestrator,
    )

    # Backward-compatible aliases while external imports migrate.
    IntelligentAgentPipeline = RuntimeIntelligencePipeline
    get_intelligent_agent = get_runtime_intelligence_pipeline
    clear_intelligent_agent_cache = clear_runtime_intelligence_pipeline_cache

    __all__ = [
        "AgentOrchestrator",
        "ArgumentNormalizer",
        "NormalizationStrategy",
        # "ConfigLoader",  # OBSOLETE - moved to archive/obsolete/
        "UnifiedAgentConfig",
        "AgentLifecycleMode",
        "get_critical_tools",
        # Conversation
        "MessageHistory",
        "ConversationController",
        "ConversationConfig",
        "ContextMetrics",
        # Observability (OBSOLETE - use victor.core.events instead)
        # "TracingProvider",
        # "Span",
        # "SpanKind",
        # "SpanStatus",
        # "get_observability",
        # "set_observability",
        # "traced",
        # Streaming
        "StreamHandler",
        "StreamResult",
        "StreamMetrics",
        "StreamBuffer",
        "StreamingController",
        "StreamingControllerConfig",
        "StreamingSession",
        # Tool execution
        "ToolExecutor",
        "ToolExecutionResult",
        "ToolPipeline",
        "ToolPipelineConfig",
        "ToolCallResult",
        "PipelineExecutionResult",
        # Task analysis
        "TaskAnalyzer",
        "TaskAnalysis",
        "get_task_analyzer",
        "reset_task_analyzer",
        # Runtime Intelligence
        "RuntimeIntelligencePipeline",
        "IntelligentAgentPipeline",
        "RequestContext",
        "ResponseResult",
        "PipelineStats",
        "get_runtime_intelligence_pipeline",
        "clear_runtime_intelligence_pipeline_cache",
        "get_intelligent_agent",
        "clear_intelligent_agent_cache",
        # Orchestrator Integration
        "OrchestratorIntegration",
        "IntegrationConfig",
        "IntegrationMetrics",
        "enhance_orchestrator",
    ]
else:
    __all__ = []


class _CallableAgentModule(_types.ModuleType):
    """Make ``victor.agent`` callable so ``@victor.agent`` keeps working.

    ``victor.agent`` is both a real subpackage (mock.patch targets like
    ``victor.agent.orchestrator`` must resolve through it) and the public
    ``@victor.agent`` decorator. With the top-level lazy ``victor/__init__``
    (PEP 562), the import machinery binds this module object onto the
    ``victor`` package on first import — so the decorator has to live on the
    module itself rather than on a shadowing proxy object.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from victor.framework.decorators import agent as _agent_decorator

        return _agent_decorator(*args, **kwargs)


_sys.modules[__name__].__class__ = _CallableAgentModule
