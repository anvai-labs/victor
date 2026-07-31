# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Factory for creating AgentOrchestrator components.

This module is a thin facade that delegates to focused builder modules in
``victor.agent.factory``:

- **tool_builders**: Tool registry, executor, pipeline, selector, cache, etc.
- **runtime_builders**: Streaming, conversation, provider, lifecycle, etc.
- **infrastructure_builders**: Observability, tracing, metrics, analytics, etc.
- **coordination_builders**: Recovery, workflow, team, safety, middleware, etc.

Part of CRITICAL-001: Monolithic Orchestrator decomposition.

Usage:
    factory = OrchestratorFactory(settings, provider, model)
    sanitizer = factory.create_sanitizer()
    prompt_builder = factory.create_prompt_builder(tool_adapter, capabilities)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Mode-aware mixin for consistent mode controller access
from victor.protocols.mode_aware import ModeAwareMixin

# Builder mixins (grouped by domain)
from victor.agent.factory.tool_builders import ToolBuildersMixin
from victor.agent.factory.runtime_builders import RuntimeBuildersMixin
from victor.agent.factory.infrastructure_builders import InfrastructureBuildersMixin
from victor.agent.factory.coordination_builders import CoordinationBuildersMixin

if TYPE_CHECKING:
    from victor.config.settings import Settings
    from victor.providers.base import BaseProvider
    from victor.agent.tool_calling import (
        BaseToolCallingAdapter,
        ToolCallingCapabilities,
    )
    from victor.agent.response_sanitizer import ResponseSanitizer
    from victor.agent.prompt_builder import SystemPromptBuilder
    from victor.agent.context_project import ProjectContext
    from victor.framework.task import TaskComplexityService as ComplexityClassifier
    from victor.agent.action_authorizer import ActionAuthorizer
    from victor.agent.search_router import SearchRouter
    from victor.agent.metrics_collector import MetricsCollector
    from victor.agent.conversation.controller import ConversationController
    from victor.agent.tool_pipeline import ToolPipeline
    from victor.agent.streaming_controller import StreamingController
    from victor.agent.context_compactor import ContextCompactor
    from victor.agent.usage_analytics import UsageAnalytics
    from victor.agent.tool_sequence_tracker import ToolSequenceTracker
    from victor.agent.tool_output_formatter import ToolOutputFormatter
    from victor.agent.recovery import RecoveryHandler
    from victor.observability.integration import ObservabilityIntegration

    # Additional types for method signatures
    from victor.tools.registry import ToolRegistry
    from victor.agent.tool_executor import ToolExecutor
    from victor.agent.tool_registrar import ToolRegistrar
    from victor.agent.conversation.store import ConversationStore
    from victor.observability.analytics.logger import UsageLogger  # noqa: F811
    from victor.observability.analytics.streaming_metrics import (
        StreamingMetricsCollector,
    )
    from victor.agent.middleware_chain import MiddlewareChain
    from victor.agent.parallel_executor import ParallelToolExecutor
    from victor.agent.response_completer import ResponseCompleter
    from victor.agent.unified_task_tracker import UnifiedTaskTracker
    from victor.agent.orchestrator_recovery import OrchestratorRecoveryIntegration
    from victor.agent.orchestrator_integration import IntegrationConfig
    from victor.agent.message_history import MessageHistory
    from victor.agent.tool_selection import ToolSelector
    from victor.storage.embeddings.intent_classifier import IntentClassifier
    from victor.agent.debug_logger import DebugLogger
    from victor.agent.tool_access_controller import ToolAccessController
    from victor.agent.session_ledger import SessionLedger
    from victor.agent.compaction_summarizer import LedgerAwareCompactionSummarizer
    from victor.agent.tool_result_deduplicator import ToolResultDeduplicator
    from victor.agent.conversation.assembler import TurnBoundaryContextAssembler
    from victor.agent.streaming.streaming_coordinator import StreamingCoordinator
    from victor.agent.streaming.handler import StreamingChatHandler
    from victor.agent.services.chat_stream_executor import StreamingChatExecutor
    from victor.agent.lifecycle_manager import LifecycleManager
    from victor.agent.tool_call_tracker import (
        ToolCallTracker as ToolDeduplicationTracker,
    )
    from victor.agent.provider_manager import ProviderManager
    from victor.agent.presentation.protocols import PresentationProtocol
    from victor.tools.plugin_registry import ToolPluginRegistry
    from victor.tools.semantic_selector import SemanticToolSelector
    from victor.storage.cache.tool_cache import ToolCache
    from victor.storage.checkpoints import ConversationCheckpointManager
    from victor.observability.tracing import ExecutionTracer, ToolCallTracer
    from victor.config.model_capabilities import ToolCallingMatrix
    from victor.agent.argument_normalizer import ArgumentNormalizer
    from victor.agent.conversation.state_machine import ConversationStateMachine
    from victor.agent.protocols.tool_protocols import ToolDependencyGraphProtocol
    from victor.agent.protocols.infrastructure_protocols import (
        SafetyCheckerProtocol,
        AutoCommitterProtocol,
        TaskTrackerProtocol,
        CodeExecutionManagerProtocol,
        WorkflowRegistryProtocol,
        ArgumentNormalizerProtocol,
        DebugLoggerProtocol,
    )
    from victor.agent.services.protocols import (
        ChunkRuntimeProtocol as ChunkGeneratorProto,
        ReminderManagerProtocol,
        RLLearningRuntimeProtocol as RLCoordinatorProtocol,
        ResponseSanitizerProtocol,
        StreamingMetricsCollectorProtocol,
        StreamingRecoveryRuntimeProtocol as StreamingRecoveryCoordinatorProto,
        TaskRuntimeProtocol as TaskCoordinatorProtocol,
        ToolPlanningRuntimeProtocol as ToolPlannerProtocol,
    )
    from victor.agent.protocols.budget_protocols import (
        IBudgetManager,
        IModeCompletionChecker,
    )
    from victor.agent.protocols.provider_protocols import (
        IProviderSwitcher,
        IProviderHealthMonitor,
    )

logger = logging.getLogger(__name__)


# =========================================================================
# OrchestratorFactory facade
# =========================================================================


class OrchestratorFactory(
    ToolBuildersMixin,
    RuntimeBuildersMixin,
    InfrastructureBuildersMixin,
    CoordinationBuildersMixin,
    ModeAwareMixin,
):
    """Factory for creating AgentOrchestrator components.

    This factory is a thin facade that delegates to focused builder mixins:

    - :class:`ToolBuildersMixin` -- tool registry, executor, pipeline, etc.
    - :class:`RuntimeBuildersMixin` -- streaming, conversation, provider, etc.
    - :class:`InfrastructureBuildersMixin` -- observability, metrics, analytics, etc.
    - :class:`CoordinationBuildersMixin` -- recovery, workflow, team, safety, etc.

    Uses ModeAwareMixin for consistent mode controller access when applying
    mode-specific configurations (e.g., exploration multipliers).

    Example:
        factory = OrchestratorFactory(settings, provider, model)

        # Create all components at once
        components = factory.create_all_components()

        # Or create individual components for testing
        sanitizer = factory.create_sanitizer()
    """

    def __init__(
        self,
        settings: "Settings",
        provider: "BaseProvider",
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        provider_name: Optional[str] = None,
        profile_name: Optional[str] = None,
        tool_selection: Optional[Dict[str, Any]] = None,
        thinking: bool = False,
    ):
        """Initialize the factory with core configuration.

        Args:
            settings: Application settings
            provider: LLM provider instance
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            provider_name: Provider label (for OpenAI-compatible disambiguation)
            profile_name: Profile name for session tracking
            tool_selection: Tool selection configuration
            thinking: Enable extended thinking mode
        """
        self.settings = settings
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider_name = provider_name
        self.profile_name = profile_name
        self.tool_selection = tool_selection or {}
        self.thinking = thinking

        # Lazy-initialized container
        self._container = None

    @property
    def container(self):
        """Get or create the DI container."""
        if self._container is None:
            from victor.core.bootstrap import ensure_bootstrapped

            self._container = ensure_bootstrapped(self.settings)
        return self._container

    # -----------------------------------------------------------------
    # Core service creation (kept on facade for backward compatibility)
    # -----------------------------------------------------------------

    def create_sanitizer(self) -> "ResponseSanitizer":
        """Create response sanitizer (from DI container with fallback)."""
        from victor.agent.services.protocols import ResponseSanitizerProtocol

        sanitizer = self.container.get_optional(ResponseSanitizerProtocol)
        if sanitizer is not None:
            return sanitizer
        # Fallback: construct directly when not registered (e.g. tests patching bootstrap)
        from victor.agent.response_sanitizer import ResponseSanitizer

        return ResponseSanitizer()

    def create_prompt_builder(
        self,
        tool_adapter: "BaseToolCallingAdapter",
        capabilities: "ToolCallingCapabilities",
    ) -> "SystemPromptBuilder":
        """Create system prompt builder.

        Args:
            tool_adapter: Tool calling adapter
            capabilities: Tool calling capabilities

        Returns:
            Configured SystemPromptBuilder
        """
        from victor.agent.prompt_builder import SystemPromptBuilder
        from victor.core.verticals.protocols import VerticalExtensions

        prompt_contributors = []
        try:
            extensions = self.container.get_optional(VerticalExtensions)
            if extensions and extensions.prompt_contributors:
                prompt_contributors = extensions.prompt_contributors
        except Exception as e:
            logger.debug(f"Could not load vertical prompt contributors: {e}")

        # Detect if provider supports prompt caching (e.g., Anthropic ephemeral cache)
        provider_caches = (
            hasattr(self.provider, "supports_prompt_caching")
            and self.provider.supports_prompt_caching()
        )
        provider_has_kv_cache = (
            hasattr(self.provider, "supports_kv_prefix_caching")
            and self.provider.supports_kv_prefix_caching()
        ) or provider_caches  # API caching providers also have KV caching

        return SystemPromptBuilder(
            provider_name=self.provider_name or self.provider.__class__.__name__.lower(),
            model=self.model,
            tool_adapter=tool_adapter,
            capabilities=capabilities,
            prompt_contributors=prompt_contributors,
            provider_caches=provider_caches,
            provider_has_kv_cache=provider_has_kv_cache,
            headless_mode=self.settings.headless.headless_mode,
        )

    def create_project_context(self) -> "ProjectContext":
        """Create project context loader (from DI container)."""
        from victor.agent.protocols import ProjectContextProtocol

        return self.container.get(ProjectContextProtocol)

    def create_complexity_classifier(self) -> "ComplexityClassifier":
        """Create complexity classifier (from DI container)."""
        from victor.agent.protocols import ComplexityClassifierProtocol

        return self.container.get(ComplexityClassifierProtocol)

    def create_action_authorizer(self) -> "ActionAuthorizer":
        """Create action authorizer (from DI container)."""
        from victor.agent.protocols import ActionAuthorizerProtocol

        return self.container.get(ActionAuthorizerProtocol)

    def create_search_router(self) -> "SearchRouter":
        """Create search router (from DI container)."""
        from victor.agent.protocols import SearchRouterProtocol

        return self.container.get(SearchRouterProtocol)

    def create_system_prompt_builder(
        self,
        provider_name: str,
        model: str,
        tool_adapter: "BaseToolCallingAdapter",
        tool_calling_caps: "ToolCallingCapabilities",
    ) -> "SystemPromptBuilder":
        """Create SystemPromptBuilder with vertical prompt contributors.

        Args:
            provider_name: Provider name string
            model: Model identifier
            tool_adapter: Tool calling adapter instance
            tool_calling_caps: Tool calling capabilities

        Returns:
            SystemPromptBuilder instance configured with prompt contributors
        """
        from victor.agent.prompt_builder import SystemPromptBuilder

        prompt_contributors = []
        try:
            from victor.core.verticals.protocols import VerticalExtensions

            extensions = self.container.get_optional(VerticalExtensions)
            if extensions and extensions.prompt_contributors:
                prompt_contributors = extensions.prompt_contributors
                logger.debug(
                    f"Loaded {len(prompt_contributors)} prompt contributors from verticals"
                )
        except Exception as e:
            logger.debug(f"Could not load vertical prompt contributors: {e}")

        # Detect if provider supports prompt caching
        provider_caches = (
            hasattr(self.provider, "supports_prompt_caching")
            and self.provider.supports_prompt_caching()
        )
        provider_has_kv_cache = (
            hasattr(self.provider, "supports_kv_prefix_caching")
            and self.provider.supports_kv_prefix_caching()
        ) or provider_caches

        prompt_builder = SystemPromptBuilder(
            provider_name=provider_name,
            model=model,
            tool_adapter=tool_adapter,
            capabilities=tool_calling_caps,
            prompt_contributors=prompt_contributors,
            provider_caches=provider_caches,
            provider_has_kv_cache=provider_has_kv_cache,
            headless_mode=self.settings.headless.headless_mode,
        )

        logger.debug(f"SystemPromptBuilder created for {provider_name}/{model}")
        return prompt_builder

    def create_prompt_pipeline(
        self,
        builder: Any,
        get_context_window: Optional[Any] = None,
    ) -> Any:
        """Create UnifiedPromptPipeline with all dependencies.

        Args:
            builder: SystemPromptBuilder instance
            get_context_window: Callable returning model context window size

        Returns:
            Configured UnifiedPromptPipeline
        """
        from victor.agent.content_registry import create_default_registry
        from victor.agent.optimization_injector import OptimizationInjector
        from victor.agent.prompt_pipeline import UnifiedPromptPipeline
        from victor.agent.services.runtime_intelligence import (
            RuntimeIntelligenceService,
        )
        from victor.evaluation.runtime_feedback import (
            runtime_evaluation_feedback_scope_from_context,
        )

        optimizer = OptimizationInjector()
        registry = create_default_registry()
        runtime_intelligence = RuntimeIntelligenceService(
            optimization_injector=optimizer,
            evaluation_feedback_scope=runtime_evaluation_feedback_scope_from_context(
                {
                    "provider_name": getattr(self.provider, "provider_name", None)
                    or getattr(self.provider, "name", None),
                    "model": getattr(builder, "model", None)
                    or getattr(self.provider, "model", None),
                }
            ),
        )

        # Fetch optional services for prompt pipeline
        from victor.agent.tool_pipeline import ToolPipeline
        from victor.framework.rl.credit_tracking_service import CreditTrackingService

        credit_tracking_service = self.container.get_optional(CreditTrackingService)
        tool_pipeline = self.container.get_optional(ToolPipeline)

        return UnifiedPromptPipeline(
            provider=self.provider,
            builder=builder,
            registry=registry,
            optimizer=optimizer,
            runtime_intelligence=runtime_intelligence,
            get_context_window=get_context_window,
            credit_tracking_service=credit_tracking_service,
            tool_pipeline=tool_pipeline,
        )

    # -----------------------------------------------------------------
    # Unified Agent Creation (Phase 4)
    # -----------------------------------------------------------------

    async def create_agent(
        self,
        mode: str = "foreground",
        config: Optional[Any] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Create ANY agent type using shared factory infrastructure.

        This is the ONLY method that should create agents. All other entrypoints
        (Agent.create, BackgroundAgentManager.start_agent, Vertical.create_agent)
        must delegate here to ensure consistent code maintenance and eliminate
        code proliferation (SOLID SRP, DIP).

        Args:
            mode: Agent creation mode
                - "foreground": Interactive Agent instance (default)
                - "background": BackgroundAgent for async task execution
                - "team_member": TeamMember/SubAgent for multi-agent teams
            config: Optional unified agent configuration (UnifiedAgentConfig)
            task: Optional task description (for background agents)
            **kwargs: Additional agent-specific parameters

        Returns:
            Agent instance based on mode

        Raises:
            ValueError: If mode is invalid or required parameters are missing
            ProviderError: If provider initialization fails
        """
        from victor.agent.orchestrator import AgentOrchestrator

        # Extract parameters from UnifiedAgentConfig if provided
        if config:
            from victor.agent.config import UnifiedAgentConfig

            if isinstance(config, UnifiedAgentConfig):
                mode = config.mode
                task = config.task or task

                merged_kwargs = {}

                # Common settings
                if config.provider != "anthropic":
                    merged_kwargs["provider"] = config.provider
                if config.model:
                    merged_kwargs["model"] = config.model
                if config.temperature != 0.7:
                    merged_kwargs["temperature"] = config.temperature
                if config.max_tokens != 4096:
                    merged_kwargs["max_tokens"] = config.max_tokens

                # Foreground-specific
                if mode == "foreground":
                    merged_kwargs["tool_budget"] = config.tool_budget
                    merged_kwargs["max_iterations"] = config.max_iterations
                    merged_kwargs["enable_parallel_tools"] = config.enable_parallel_tools
                    merged_kwargs["max_concurrent_tools"] = config.max_concurrent_tools
                    merged_kwargs["enable_context_compaction"] = config.enable_context_compaction
                    merged_kwargs["enable_semantic_search"] = config.enable_semantic_search
                    merged_kwargs["enable_analytics"] = config.enable_analytics

                # Background-specific
                elif mode == "background":
                    merged_kwargs["task"] = config.task or task
                    merged_kwargs["mode_type"] = config.mode_type
                    merged_kwargs["websocket"] = config.websocket
                    merged_kwargs["timeout_seconds"] = config.timeout_seconds
                    merged_kwargs["tool_budget"] = config.tool_budget

                # Team member-specific
                elif mode == "team_member":
                    merged_kwargs["role"] = config.role
                    merged_kwargs["capabilities"] = config.capabilities
                    merged_kwargs["description"] = config.description
                    merged_kwargs["allowed_tools"] = config.allowed_tools
                    merged_kwargs["can_spawn_subagents"] = config.can_spawn_subagents

                # Apply extra settings
                merged_kwargs.update(config.extra)

                # Merge with provided kwargs (kwargs take precedence)
                merged_kwargs.update(kwargs)
                kwargs = merged_kwargs

        # Create orchestrator using existing factory methods
        orchestrator = await AgentOrchestrator.from_settings(
            self.settings,
            profile_name=self.profile_name or "default",
            thinking=self.thinking,
        )

        # Override provider/model if different from defaults
        provider = kwargs.get("provider")
        model = kwargs.get("model")
        if provider or model:
            if hasattr(orchestrator, "_provider_manager") or hasattr(
                orchestrator, "provider_manager"
            ):
                pm = getattr(orchestrator, "_provider_manager", None) or getattr(
                    orchestrator, "provider_manager", None
                )
                if pm:
                    await pm.switch_provider(
                        provider or pm.provider_name,
                        model or self.model,
                    )

        # Create agent based on mode
        if mode == "foreground":
            from victor.framework.agent import Agent

            agent = Agent(orchestrator)

            tools = kwargs.get("tools")
            if tools:
                from victor.framework._internal import configure_tools

                configure_tools(orchestrator, tools, airgapped=kwargs.get("airgapped", False))

            vertical = kwargs.get("vertical")
            if vertical:
                vertical_config = vertical.get_config()
                if vertical_config.system_prompt:
                    orchestrator.system_prompt = vertical_config.system_prompt

            return agent

        elif mode == "background":
            from victor.agent.background_agent import BackgroundAgent

            agent = BackgroundAgent(
                orchestrator=orchestrator,
                task=task or kwargs.get("task", ""),
                mode_type=kwargs.get("mode_type", "build"),
                websocket=kwargs.get("websocket", False),
                enable_observability=kwargs.get("enable_observability", True),
            )
            return agent

        elif mode == "team_member":
            from victor.teams.types import TeamMember

            member = TeamMember(
                agent_ref=kwargs.get("agent_ref"),
                role=kwargs.get("role", "team_member"),
                capabilities=kwargs.get("capabilities", []),
                description=kwargs.get("description", ""),
            )
            return member

        else:
            raise ValueError(
                f"Invalid agent mode: {mode!r}. "
                "Must be 'foreground', 'background', or 'team_member'"
            )


# =========================================================================
# Convenience function
# =========================================================================


def create_orchestrator_factory(
    settings: "Settings",
    provider: "BaseProvider",
    model: str,
    **kwargs,
) -> OrchestratorFactory:
    """Create an OrchestratorFactory instance.

    Args:
        settings: Application settings
        provider: LLM provider instance
        model: Model identifier
        **kwargs: Additional factory configuration

    Returns:
        Configured OrchestratorFactory
    """
    return OrchestratorFactory(
        settings=settings,
        provider=provider,
        model=model,
        **kwargs,
    )
