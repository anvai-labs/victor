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

"""Property facade for AgentOrchestrator.

Extracts ~25 property definitions from AgentOrchestrator to reduce its LOC
while preserving the same access patterns.  Properties are installed onto
the AgentOrchestrator class via ``install_properties()``, keeping full
compatibility with ``unittest.mock.patch.object`` and introspection.

Groups:
  1. Simple component accessors -- return self._xxx directly
  2. Lazy coordinator constructors -- check-and-create on first access
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.services.recovery_service import RecoveryService
    from victor.agent.services.chunk_runtime import ChunkGenerator
    from victor.agent.services.tool_planning_runtime import ToolPlanner
    from victor.agent.services.task_runtime import TaskCoordinator
    from victor.agent.orchestrator_integration import OrchestratorIntegration
    from victor.agent.conversation.controller import ConversationController
    from victor.agent.tool_pipeline import ToolPipeline
    from victor.agent.streaming_controller import StreamingController
    from victor.agent.task_analyzer import TaskAnalyzer
    from victor.agent.provider_manager import ProviderManager
    from victor.agent.context_compactor import ContextCompactor
    from victor.agent.tool_output_formatter import ToolOutputFormatter
    from victor.agent.usage_analytics import UsageAnalytics
    from victor.agent.tool_sequence_tracker import ToolSequenceTracker
    from victor.agent.streaming.handler import StreamingChatHandler
    from victor.agent.vertical_context import VerticalContext
    from victor.agent.recovery import RecoveryHandler
    from victor.agent.orchestrator_recovery import OrchestratorRecoveryIntegration
    from victor.agent.session_state_manager import SessionStateManager

logger = logging.getLogger(__name__)


# =====================================================================
# Group 1: Simple component accessor properties
# =====================================================================


def _conversation_controller(self: "AgentOrchestrator") -> "ConversationController":
    """Get the canonical conversation controller component."""
    return self._conversation_controller


def _tool_pipeline(self: "AgentOrchestrator") -> "ToolPipeline":
    """Get the canonical tool pipeline component."""
    return self._tool_pipeline


def _streaming_controller(self: "AgentOrchestrator") -> "StreamingController":
    """Get the canonical streaming controller component."""
    return self._streaming_controller


def _streaming_handler(self: "AgentOrchestrator") -> "StreamingChatHandler":
    """Get the canonical streaming chat handler component."""
    return self._streaming_handler


def _task_analyzer(self: "AgentOrchestrator") -> "TaskAnalyzer":
    """Get the canonical task analyzer component."""
    return self._task_analyzer


def _provider_manager(self: "AgentOrchestrator") -> "ProviderManager":
    """Get the canonical provider manager component."""
    return self._provider_manager


def _context_compactor(self: "AgentOrchestrator") -> "ContextCompactor":
    """Get the canonical context compactor component."""
    return self._context_compactor


def _tool_output_formatter(self: "AgentOrchestrator") -> "ToolOutputFormatter":
    """Get the canonical tool output formatter component."""
    return self._tool_output_formatter


def _usage_analytics(self: "AgentOrchestrator") -> "UsageAnalytics":
    """Get the canonical usage analytics singleton."""
    return self._usage_analytics


def _sequence_tracker(self: "AgentOrchestrator") -> "ToolSequenceTracker":
    """Get the canonical tool sequence tracker."""
    return self._sequence_tracker


def _recovery_coordinator(self: "AgentOrchestrator") -> "RecoveryService":
    """Get the canonical RecoveryService for centralized recovery logic.

    Note: The property name 'recovery_coordinator' is retained for compatibility.
    This now returns RecoveryService with native streaming runtime enabled.
    """
    # Access the private attribute directly to avoid property recursion
    return object.__getattribute__(self, "_recovery_coordinator")


def _chunk_generator(self: "AgentOrchestrator") -> "ChunkGenerator":
    """Get the canonical chunk generator for streaming output."""
    return self._chunk_generator


def _tool_planner(self: "AgentOrchestrator") -> "ToolPlanner":
    """Get the tool planner for tool planning operations."""
    return self._tool_planner


def _task_coordinator(self: "AgentOrchestrator") -> "TaskCoordinator":
    """Get the task coordinator for task coordination operations."""
    return self._task_coordinator


def _skill_matcher(self: "AgentOrchestrator") -> Any:
    """Get the shared framework skill matcher when initialized."""
    return getattr(self, "_skill_matcher", None)


def _session_ledger_get(self: "AgentOrchestrator") -> Any:
    """Get the canonical session ledger for structured state tracking."""
    return self._session_ledger


def _session_ledger_set(self: "AgentOrchestrator", value: Any) -> None:
    self._session_ledger = value


def _code_correction_middleware(self: "AgentOrchestrator") -> Optional[Any]:
    """Get the canonical code correction middleware."""
    return self._code_correction_middleware


def _checkpoint_manager(self: "AgentOrchestrator") -> Optional[Any]:
    """Get the canonical checkpoint manager for time-travel debugging."""
    return self._checkpoint_manager


def _vertical_context(self: "AgentOrchestrator") -> "VerticalContext":
    """Get the canonical vertical context for unified vertical state access."""
    return self._vertical_context


def _resolve_lazy_surface(self: "AgentOrchestrator", attr_name: str) -> Any:
    """Resolve a lazily-proxied facade surface and cache the concrete instance."""
    surface = getattr(self, attr_name, None)
    if hasattr(surface, "get_instance"):
        resolved = surface.get_instance()
        setattr(self, attr_name, resolved)
        return resolved
    return surface


def _orchestration_facade(self: "AgentOrchestrator") -> Any:
    """Get the canonical orchestration facade surface."""
    return _resolve_lazy_surface(self, "_orchestration_facade")


# =====================================================================
# Group 2: Lazy coordinator constructor properties
# =====================================================================


def _protocol_adapter(self: "AgentOrchestrator") -> Any:
    """Get the protocol adapter for DIP compliance (lazy init)."""
    if getattr(self, "_protocol_adapter", None) is None:
        from victor.agent.services.orchestrator_protocol_adapter import (
            OrchestratorProtocolAdapter,
        )

        self._protocol_adapter = OrchestratorProtocolAdapter(self)
    return self._protocol_adapter


def _turn_executor(self: "AgentOrchestrator") -> Any:
    """Get the execution coordinator for agentic loop (lazy init)."""
    if self._turn_executor is None:
        from victor.agent.services.turn_execution_runtime import TurnExecutor

        # Governance message gate (REQUEST/RESPONSE phases) is built once at
        # component assembly and shared with the streaming executor. None unless
        # the policy engine is enabled with message-phase policies configured —
        # zero behaviour change otherwise.
        self._turn_executor = TurnExecutor(
            chat_context=self.protocol_adapter,
            tool_context=self.protocol_adapter,
            provider_context=self.protocol_adapter,
            execution_provider=self.protocol_adapter,
            message_policy_gate=getattr(self, "_message_policy_gate", None),
        )
    return self._turn_executor


def _temperature_resolver(self: "AgentOrchestrator") -> Any:
    """Unified sampling-temperature resolver (ADR-013), built once and cached.

    Composes profile/settings/task-hint sources + ratchet/bounds modifiers from
    ``settings.temperature``. The reactive recovery modifier is added later (PR-E); for now the
    resolver does base resolution + bounds, so behaviour matches the prior task-hint path.
    """
    cached = getattr(self, "_temperature_resolver_cached", None)
    if cached is None:
        from victor.framework.capabilities.task_hints import (
            TaskTypeHintCapabilityProvider,
        )
        from victor.framework.temperature import build_resolver_from_settings

        cached = build_resolver_from_settings(
            getattr(self.settings, "temperature_policy", None),
            hint_provider=TaskTypeHintCapabilityProvider(),
        )
        self._temperature_resolver_cached = cached
    return cached


def _profile_task_temperatures(self: "AgentOrchestrator") -> Any:
    """The active profile's per-task ``temperatures`` map (ADR-013), or ``{}``. Cached."""
    cached = getattr(self, "_profile_task_temps_cached", None)
    if cached is None:
        cached = {}
        try:
            name = getattr(self, "_profile_name", None)
            if name:
                profiles = type(self.settings).load_profiles()
                profile = profiles.get(name)
                cached = dict(getattr(profile, "temperatures", None) or {})
        except Exception:  # profile lookup is best-effort; fall back to no per-task overrides
            cached = {}
        self._profile_task_temps_cached = cached
    return cached


def _settings_task_temperatures(self: "AgentOrchestrator") -> Any:
    """The settings-level ``temperature.task_defaults`` ops table (ADR-013), or ``{}``."""
    temp_settings = getattr(self.settings, "temperature_policy", None)
    return dict(getattr(temp_settings, "task_defaults", None) or {})


def _temperature_ratchet_state(self: "AgentOrchestrator") -> Any:
    """Per-session spin-escape ratchet state (ADR-013, PR-E), shared across buffered/streaming turns.

    The agentic loop advances it once per turn (``RatchetState.record_turn``) from the SpinDetector
    signal; the temperature resolver reads its step count to apply the escape-velocity ratchet. Lives
    on the orchestrator so both the loop (writer) and the resolution seam (reader) share one instance.
    """
    cached = getattr(self, "_temperature_ratchet_state_obj", None)
    if cached is None:
        from victor.framework.temperature import RatchetState

        cached = RatchetState()
        self._temperature_ratchet_state_obj = cached
    return cached


def _runtime_intelligence_integration(
    self: "AgentOrchestrator",
) -> Optional["OrchestratorIntegration"]:
    """Get the runtime-intelligence integration (lazy init)."""
    if not self._runtime_intelligence_enabled:
        return None

    if self._runtime_intelligence_integration is None:
        try:
            from victor.agent.orchestrator_integration import OrchestratorIntegration
            from victor.agent.runtime_intelligence_pipeline import (
                RuntimeIntelligencePipeline,
            )
            from victor.config.settings import get_project_paths
            from victor.context.project_context import VICTOR_DIR_NAME

            if self.project_context.context_file:
                parent_dir = self.project_context.context_file.parent
                if parent_dir.name == VICTOR_DIR_NAME:
                    intelligent_project_root = str(parent_dir.parent)
                else:
                    intelligent_project_root = str(parent_dir)
            else:
                intelligent_project_root = str(get_project_paths().project_root)

            pipeline = RuntimeIntelligencePipeline(
                provider_name=self.provider_name,
                model=self.model,
                profile_name=f"{self.provider_name}:{self.model}",
                project_root=intelligent_project_root,
            )
            self._runtime_intelligence_integration = OrchestratorIntegration(
                orchestrator=self,
                pipeline=pipeline,
                config=self._runtime_intelligence_integration_config,
            )
            logger.info(
                f"Runtime-intelligence integration initialized for "
                f"{self.provider_name}:{self.model}"
            )
        except ImportError as e:
            logger.debug(f"RuntimeIntelligencePipeline dependencies not available: {e}")
            self._runtime_intelligence_enabled = False
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Failed to initialize RuntimeIntelligencePipeline (config error): {e}")
            self._runtime_intelligence_enabled = False

    return self._runtime_intelligence_integration


def _subagent_orchestrator(self: "AgentOrchestrator") -> Optional[Any]:
    """Get the sub-agent orchestrator (lazy init)."""
    if not self._subagent_orchestration_enabled:
        return None

    if self._subagent_orchestrator is None:
        try:
            from victor.agent.subagents import SubAgentOrchestrator

            self._subagent_orchestrator = SubAgentOrchestrator(parent=self)
            logger.info("SubAgentOrchestrator initialized")
        except ImportError as e:
            logger.debug(f"SubAgentOrchestrator module not available: {e}")
            self._subagent_orchestration_enabled = False
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Failed to initialize SubAgentOrchestrator (config error): {e}")
            self._subagent_orchestration_enabled = False

    return self._subagent_orchestrator


def _coordination(self: "AgentOrchestrator") -> Any:
    """Compatibility alias for the framework-facing coordination advisor."""
    return self.coordination_advisor


def _coordination_advisor(self: "AgentOrchestrator") -> Any:
    """Get the framework-facing coordination advisor (lazy init)."""
    if getattr(self, "_coordination_advisor", None) is None:
        self._coordination_advisor = self._factory.create_coordination_advisor(
            self._vertical_context
        )
        logger.debug("Coordination advisor initialized on first access")

    return self._coordination_advisor


# =====================================================================
# Group 3: Recovery properties (lazy resolution via get_instance)
# =====================================================================


def _recovery_handler(self: "AgentOrchestrator") -> Optional["RecoveryHandler"]:
    """Get the recovery handler for model failure recovery."""
    handler = getattr(self, "_recovery_handler", None)
    if hasattr(handler, "get_instance"):
        resolved = handler.get_instance()
        self._recovery_handler = resolved
        return resolved
    return handler


def _recovery_integration(
    self: "AgentOrchestrator",
) -> "OrchestratorRecoveryIntegration":
    """Get the recovery integration submodule."""
    integration = getattr(self, "_recovery_integration", None)
    if hasattr(integration, "get_instance"):
        resolved = integration.get_instance()
        self._recovery_integration = resolved
        return resolved
    return integration


# =====================================================================
# Group 4: Session state delegation properties (with setters)
# =====================================================================


def _session_state_get(self: "AgentOrchestrator") -> "SessionStateManager":
    """Get the session state manager."""
    return self._session_accessor.session_state


def _tool_calls_used_get(self: "AgentOrchestrator") -> int:
    """Get the number of tool calls used in this session."""
    return self._session_accessor.tool_calls_used


def _tool_calls_used_set(self: "AgentOrchestrator", value: int) -> None:
    self._session_accessor.tool_calls_used = value


def _observed_files_get(self: "AgentOrchestrator") -> set:
    """Get set of files observed/read during this session."""
    return self._session_accessor.observed_files


def _observed_files_set(self: "AgentOrchestrator", value: set) -> None:
    self._session_accessor.observed_files = value


def _executed_tools_get(self: "AgentOrchestrator") -> list:
    """Get list of executed tool names in order."""
    return self._session_accessor.executed_tools


def _executed_tools_set(self: "AgentOrchestrator", value: list) -> None:
    self._session_accessor.executed_tools = value


def _failed_tool_signatures_get(self: "AgentOrchestrator") -> set:
    """Get set of (tool_name, args_hash) tuples for failed calls."""
    return self._session_accessor.failed_tool_signatures


def _failed_tool_signatures_set(self: "AgentOrchestrator", value: set) -> None:
    self._session_accessor.failed_tool_signatures = value


def _tool_capability_warned_get(self: "AgentOrchestrator") -> bool:
    """Get whether we've warned about tool capability limitations."""
    return self._session_accessor.tool_capability_warned


def _tool_capability_warned_set(self: "AgentOrchestrator", value: bool) -> None:
    self._session_accessor.tool_capability_warned = value


def _read_files_session_get(self: "AgentOrchestrator") -> set:
    """Get files read during this session for task completion detection."""
    return self._session_accessor.read_files_session


def _required_files_get(self: "AgentOrchestrator") -> list:
    """Get required files extracted from user prompts."""
    return self._session_accessor.required_files


def _required_files_set(self: "AgentOrchestrator", value: list) -> None:
    self._session_accessor.required_files = value


def _required_outputs_get(self: "AgentOrchestrator") -> list:
    """Get required outputs extracted from user prompts."""
    return self._session_accessor.required_outputs


def _required_outputs_set(self: "AgentOrchestrator", value: list) -> None:
    self._session_accessor.required_outputs = value


def _all_files_read_nudge_sent_get(self: "AgentOrchestrator") -> bool:
    """Get whether we've sent a nudge that all required files are read."""
    return self._session_accessor.all_files_read_nudge_sent


def _all_files_read_nudge_sent_set(self: "AgentOrchestrator", value: bool) -> None:
    self._session_accessor.all_files_read_nudge_sent = value


def _cumulative_token_usage_get(self: "AgentOrchestrator") -> dict:
    """Get cumulative token usage for evaluation/benchmarking."""
    return self._session_accessor.cumulative_token_usage


def _cumulative_token_usage_set(self: "AgentOrchestrator", value: dict) -> None:
    self._session_accessor.cumulative_token_usage = value


# =====================================================================
# Group 5: Deprecated compatibility handles
# =====================================================================


def _get_provider_runtime_component(
    self: "AgentOrchestrator", component_name: str
) -> Optional[Any]:
    """Resolve a provider compatibility component from provider_runtime."""
    runtime = getattr(self, "_provider_runtime", None)
    if runtime is None:
        return None
    return getattr(runtime, component_name, None)


# =====================================================================
# Property installation registry
# =====================================================================

# =====================================================================
# Group 6: ToolService convenience methods (delegates to canonical service)
# =====================================================================


def _is_tool_enabled_get(self: "AgentOrchestrator") -> Any:
    """Check if a tool is enabled via ToolService."""
    if hasattr(self, "_tool_service") and self._tool_service:
        return self._tool_service.is_tool_enabled
    raise AttributeError("ToolService not available")


def _get_enabled_tools_get(self: "AgentOrchestrator") -> Any:
    """Get enabled tools via ToolService."""
    if hasattr(self, "_tool_service") and self._tool_service:
        return self._tool_service.get_enabled_tools
    raise AttributeError("ToolService not available")


def _set_enabled_tools_get(self: "AgentOrchestrator") -> Any:
    """Set enabled tools via ToolService."""
    if hasattr(self, "_tool_service") and self._tool_service:
        return self._tool_service.set_enabled_tools
    raise AttributeError("ToolService not available")


def _resolve_tool_alias_get(self: "AgentOrchestrator") -> Any:
    """Resolve tool alias via ToolService."""
    if hasattr(self, "_tool_service") and self._tool_service:
        return self._tool_service.resolve_tool_alias
    raise AttributeError("ToolService not available")


# Map of public property name -> (getter, setter_or_None)
_PROPERTY_REGISTRY: dict[str, Any] = {
    # Group 1: Simple accessors (getter only)
    "conversation_controller": (_conversation_controller, None),
    "tool_pipeline": (_tool_pipeline, None),
    "streaming_controller": (_streaming_controller, None),
    "streaming_handler": (_streaming_handler, None),
    "task_analyzer": (_task_analyzer, None),
    "provider_manager": (_provider_manager, None),
    "context_compactor": (_context_compactor, None),
    "tool_output_formatter": (_tool_output_formatter, None),
    "usage_analytics": (_usage_analytics, None),
    "sequence_tracker": (_sequence_tracker, None),
    "recovery_coordinator": (_recovery_coordinator, None),
    "chunk_generator": (_chunk_generator, None),
    "tool_planner": (_tool_planner, None),
    "task_coordinator": (_task_coordinator, None),
    "skill_matcher": (_skill_matcher, None),
    "session_ledger": (_session_ledger_get, _session_ledger_set),
    "code_correction_middleware": (_code_correction_middleware, None),
    "checkpoint_manager": (_checkpoint_manager, None),
    "vertical_context": (_vertical_context, None),
    "orchestration_facade": (_orchestration_facade, None),
    # Group 2: Lazy coordinators (getter only)
    "protocol_adapter": (_protocol_adapter, None),
    "turn_executor": (_turn_executor, None),
    "temperature_resolver": (_temperature_resolver, None),
    "profile_task_temperatures": (_profile_task_temperatures, None),
    "settings_task_temperatures": (_settings_task_temperatures, None),
    "temperature_ratchet_state": (_temperature_ratchet_state, None),
    "runtime_intelligence_integration": (_runtime_intelligence_integration, None),
    "subagent_orchestrator": (_subagent_orchestrator, None),
    "coordination_advisor": (_coordination_advisor, None),
    "coordination": (_coordination, None),
    # Group 3: Recovery (getter only, with lazy resolution)
    "recovery_handler": (_recovery_handler, None),
    "recovery_integration": (_recovery_integration, None),
    # Group 4: Session state delegation (getter + setter)
    "session_state": (_session_state_get, None),
    "tool_calls_used": (_tool_calls_used_get, _tool_calls_used_set),
    "observed_files": (_observed_files_get, _observed_files_set),
    "executed_tools": (_executed_tools_get, _executed_tools_set),
    "failed_tool_signatures": (
        _failed_tool_signatures_get,
        _failed_tool_signatures_set,
    ),
    "_tool_capability_warned": (
        _tool_capability_warned_get,
        _tool_capability_warned_set,
    ),
    "_read_files_session": (_read_files_session_get, None),
    "_required_files": (_required_files_get, _required_files_set),
    "_required_outputs": (_required_outputs_get, _required_outputs_set),
    "_all_files_read_nudge_sent": (
        _all_files_read_nudge_sent_get,
        _all_files_read_nudge_sent_set,
    ),
    "_cumulative_token_usage": (
        _cumulative_token_usage_get,
        _cumulative_token_usage_set,
    ),
    # Group 6: ToolService convenience methods (delegates to canonical service)
    "is_tool_enabled": (_is_tool_enabled_get, None),
    "get_enabled_tools": (_get_enabled_tools_get, None),
    "set_enabled_tools": (_set_enabled_tools_get, None),
    "resolve_tool_alias": (_resolve_tool_alias_get, None),
}


def install_properties(cls: type) -> type:
    """Install extracted property descriptors onto *cls*.

    Called once at module load time (after ``AgentOrchestrator`` class body
    is defined) to inject the property descriptors.  Because they live in
    the class ``__dict__``, ``unittest.mock.patch.object`` and normal
    attribute introspection continue to work.

    Supports both getter-only and getter+setter properties.

    Args:
        cls: The AgentOrchestrator class (or compatible subclass).

    Returns:
        The same class, with property descriptors installed.
    """
    for name, (getter, setter) in _PROPERTY_REGISTRY.items():
        prop = property(getter, setter, doc=getter.__doc__)
        setattr(cls, name, prop)
    return cls
