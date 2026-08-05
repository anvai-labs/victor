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

"""Tool-related builder methods for OrchestratorFactory.

Provides creation methods for tool registry, executor, pipeline, selector,
cache, registrar, dependency graph, and related tool infrastructure.

Part of CRITICAL-001: Monolithic Orchestrator decomposition.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING
from unittest.mock import Mock

from victor.config.tool_selection_access import is_semantic_tool_selection_enabled

if TYPE_CHECKING:
    from victor.config.settings import Settings
    from victor.providers.base import BaseProvider
    from victor.agent.tool_calling import (
        BaseToolCallingAdapter,
        ToolCallingCapabilities,
    )
    from victor.tools.registry import ToolRegistry
    from victor.agent.tool_executor import ToolExecutor
    from victor.agent.tool_registrar import ToolRegistrar
    from victor.agent.tool_pipeline import ToolPipeline
    from victor.agent.middleware_chain import MiddlewareChain
    from victor.agent.tool_output_formatter import ToolOutputFormatter
    from victor.agent.tool_call_tracker import (
        ToolCallTracker as ToolDeduplicationTracker,
    )
    from victor.agent.tool_selection import ToolSelector
    from victor.agent.tool_access_controller import ToolAccessController
    from victor.agent.argument_normalizer import ArgumentNormalizer
    from victor.agent.context_compactor import ContextCompactor
    from victor.agent.search_router import SearchRouter
    from victor.agent.conversation.state_machine import ConversationStateMachine
    from victor.agent.unified_task_tracker import UnifiedTaskTracker
    from victor.agent.parallel_executor import ParallelToolExecutor
    from victor.agent.tool_result_deduplicator import ToolResultDeduplicator
    from victor.tools.plugin_registry import ToolPluginRegistry
    from victor.tools.semantic_selector import SemanticToolSelector
    from victor.storage.cache.tool_cache import ToolCache
    from victor.config.model_capabilities import ToolCallingMatrix
    from victor.agent.protocols.tool_protocols import ToolDependencyGraphProtocol
    from victor.agent.protocols.infrastructure_protocols import SafetyCheckerProtocol
    from victor.agent.services.protocols import (
        ToolPlanningRuntimeProtocol as ToolPlannerProtocol,
    )

logger = logging.getLogger(__name__)
_MISSING = object()


class ToolBuildersMixin:
    """Mixin providing tool-related factory methods.

    Requires the host class to provide:
        - self.settings: Settings
        - self.provider: BaseProvider
        - self.model: str
        - self.provider_name: Optional[str]
        - self.container: DI container
        - self.mode_controller: property from ModeAwareMixin
    """

    @staticmethod
    def _resolve_setting_value(source: Any, name: str, default: Any = _MISSING) -> Any:
        """Return a concrete setting value while ignoring auto-created mock attributes."""
        if source is None:
            return default

        value = getattr(source, name, default)
        if isinstance(value, Mock) and name not in getattr(source, "__dict__", {}):
            return default
        return value

    def _tool_setting(self, name: str, default: Any) -> Any:
        """Resolve a tool setting from nested ToolSettings or legacy flat settings."""
        tool_settings = self._resolve_setting_value(self.settings, "tools", None)
        nested_value = self._resolve_setting_value(tool_settings, name, _MISSING)
        if nested_value is not _MISSING:
            return nested_value
        return self._resolve_setting_value(self.settings, name, default)

    def _cache_setting(self, name: str, default: Any) -> Any:
        """Resolve a cache setting from nested CacheConfig or legacy flat settings."""
        cache_settings = self._resolve_setting_value(self.settings, "cache", None)
        nested_value = self._resolve_setting_value(cache_settings, name, _MISSING)
        if nested_value is not _MISSING:
            return nested_value
        return self._resolve_setting_value(self.settings, name, default)

    def create_tool_registry(self) -> "ToolRegistry":
        """Create tool registry for managing available tools.

        Returns:
            ToolRegistry instance for tool storage and management
        """
        from victor.tools.registry import ToolRegistry

        registry = ToolRegistry()
        logger.debug("ToolRegistry created")
        return registry

    def create_tool_executor(
        self,
        tools: "ToolRegistry",
        argument_normalizer: "ArgumentNormalizer",
        tool_cache: Optional["ToolCache"],
        safety_checker: "SafetyCheckerProtocol",
        code_correction_middleware: Optional[Any],
    ) -> "ToolExecutor":
        """Create tool executor for centralized tool execution.

        Handles retry logic, caching, validation, and metrics collection.

        Args:
            tools: Tool registry containing available tools
            argument_normalizer: ArgumentNormalizer for handling malformed arguments
            tool_cache: Optional cache for tool results
            safety_checker: SafetyChecker for validating tool execution
            code_correction_middleware: Optional middleware for code correction

        Returns:
            ToolExecutor instance configured with settings
        """
        from victor.agent.tool_executor import ToolExecutor, ValidationMode

        # Parse validation mode from settings
        validation_mode_str = getattr(self.settings, "tool_validation_mode", "lenient").lower()
        validation_mode_map = {
            "strict": ValidationMode.STRICT,
            "lenient": ValidationMode.LENIENT,
            "off": ValidationMode.OFF,
        }
        validation_mode = validation_mode_map.get(validation_mode_str, ValidationMode.LENIENT)
        per_tool_timeout_seconds = self._tool_setting("per_tool_timeout_seconds", 60.0)

        # Create executor with all settings
        executor = ToolExecutor(
            tool_registry=tools,
            argument_normalizer=argument_normalizer,
            tool_cache=tool_cache,
            max_retries=getattr(self.settings, "tool_retry_max_attempts", 3),
            retry_delay=getattr(self.settings, "tool_retry_base_delay", 1.0),
            validation_mode=validation_mode,
            safety_checker=safety_checker,
            code_correction_middleware=code_correction_middleware,
            enable_code_correction=getattr(self.settings, "code_correction_enabled", True),
            default_timeout_seconds=per_tool_timeout_seconds,
            tool_call_tracer=getattr(self, "_tool_call_tracer", None),
        )

        # Inject ToolConfig into executor context for DI-style tool configuration
        from victor.tools.base import ToolConfig

        tool_config = ToolConfig(
            provider=self.provider,
            model=getattr(self.settings, "model", None),
            max_complexity=getattr(self.settings, "max_complexity", 10),
            web_fetch_top=getattr(self.settings, "web_fetch_top", 5),
            web_fetch_pool=getattr(self.settings, "web_fetch_pool", 3),
            max_content_length=getattr(self.settings, "max_content_length", 5000),
            batch_concurrency=getattr(self.settings, "batch_concurrency", 5),
            batch_max_files=getattr(self.settings, "batch_max_files", 100),
            generic_result_cache_enabled=self._tool_setting(
                "generic_result_cache_enabled",
                False,
            ),
            generic_result_cache_ttl=self._tool_setting("generic_result_cache_ttl", 300),
            http_connection_pool_enabled=self._cache_setting(
                "http_connection_pool_enabled",
                False,
            ),
            http_connection_pool_max_connections=self._cache_setting(
                "http_connection_pool_max_connections",
                100,
            ),
            http_connection_pool_max_connections_per_host=self._cache_setting(
                "http_connection_pool_max_connections_per_host",
                10,
            ),
            http_connection_pool_connection_timeout=self._cache_setting(
                "http_connection_pool_connection_timeout",
                30,
            ),
            http_connection_pool_total_timeout=self._cache_setting(
                "http_connection_pool_total_timeout",
                60,
            ),
        )
        executor.update_context(tool_config=tool_config)

        logger.debug(f"ToolExecutor created with validation_mode={validation_mode}")
        return executor

    def create_tool_pipeline(
        self,
        tools: "ToolRegistry",
        tool_executor: "ToolExecutor",
        tool_budget: int,
        tool_cache: Optional["ToolCache"],
        argument_normalizer: "ArgumentNormalizer",
        on_tool_start: Callable,
        on_tool_complete: Callable,
        on_tool_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        deduplication_tracker: Optional["ToolDeduplicationTracker"] = None,
        middleware_chain: Optional["MiddlewareChain"] = None,
        semantic_cache: Optional[Any] = None,
        search_router: Optional["SearchRouter"] = None,
    ) -> "ToolPipeline":
        """Create tool pipeline for coordinating tool execution flow.

        Args:
            tools: Tool registry containing available tools
            tool_executor: ToolExecutor for executing tool calls
            tool_budget: Maximum tool calls allowed per session
            tool_cache: Optional cache for tool results
            argument_normalizer: Normalizer for tool arguments
            on_tool_start: Callback invoked when tool execution starts
            on_tool_complete: Callback invoked when tool execution completes
            on_tool_event: Callback invoked when tool events should be emitted
            deduplication_tracker: Optional tracker for preventing duplicate calls
            middleware_chain: Optional middleware chain for processing tool calls
            semantic_cache: Optional FAISS-based semantic cache for tool results
            search_router: Optional search router for enriching search tool calls

        Returns:
            ToolPipeline instance coordinating tool execution
        """
        from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig

        # Initialize permission policy from settings if available
        permission_policy = None
        try:
            from victor.security.permissions import PermissionPolicy, PermissionMode

            settings = getattr(self, "_settings", None)
            perm_settings = getattr(settings, "permissions", None) if settings else None
            if perm_settings and getattr(perm_settings, "permission_mode", None):
                mode = PermissionMode.from_string(perm_settings.permission_mode)
                permission_policy = PermissionPolicy(active_mode=mode)
                # Apply per-tool overrides from config
                for tool_name, perm_str in getattr(
                    perm_settings, "permission_tool_overrides", {}
                ).items():
                    permission_policy.register_tool_permission(
                        tool_name, PermissionMode.from_string(perm_str)
                    )
                # Sync from tool metadata if available
                permission_policy.sync_from_tool_metadata()
        except Exception:
            pass  # Permission system is optional

        # Read cross-turn dedup settings from ToolSettings
        cross_turn_enabled = self._tool_setting("cross_turn_dedup_enabled", True)
        cross_turn_ttl = float(self._tool_setting("cross_turn_dedup_ttl", 300))

        pipeline = ToolPipeline(
            tool_registry=tools,
            tool_executor=tool_executor,
            config=ToolPipelineConfig(
                tool_budget=tool_budget,
                enable_caching=tool_cache is not None,
                enable_analytics=True,
                enable_failed_signature_tracking=True,
                enable_semantic_caching=semantic_cache is not None,
                enable_cross_turn_dedup=cross_turn_enabled,
                cross_turn_dedup_ttl=cross_turn_ttl,
                per_tool_timeout_seconds=self._tool_setting("per_tool_timeout_seconds", 60.0),
            ),
            tool_cache=tool_cache,
            argument_normalizer=argument_normalizer,
            on_tool_start=on_tool_start,
            on_tool_complete=on_tool_complete,
            on_tool_event=on_tool_event,
            deduplication_tracker=deduplication_tracker,
            middleware_chain=middleware_chain,
            semantic_cache=semantic_cache,
            search_router=search_router,
            permission_policy=permission_policy,
        )
        logger.debug(
            "ToolPipeline created%s%s",
            " with middleware chain" if middleware_chain else "",
            " with semantic cache" if semantic_cache else "",
        )
        return pipeline

    def create_tool_selector(
        self,
        tools: "ToolRegistry",
        semantic_selector: Optional["SemanticToolSelector"],
        conversation_state: "ConversationStateMachine",
        unified_tracker: "UnifiedTaskTracker",
        model: str,
        provider_name: str,
        tool_selection: Dict[str, Any],
        on_selection_recorded: Optional[Callable],
    ) -> "ToolSelector":
        """Create unified tool selector for semantic and keyword-based selection.

        Args:
            tools: ToolRegistry instance
            semantic_selector: SemanticToolSelector instance
            conversation_state: ConversationStateMachine instance
            unified_tracker: UnifiedTaskTracker instance
            model: Model identifier
            provider_name: Provider name
            tool_selection: Tool selection configuration
            on_selection_recorded: Callback for recording tool selections

        Returns:
            ToolSelector instance configured with all dependencies
        """
        from victor.agent.services.runtime_intelligence import (
            RuntimeIntelligenceService,
        )
        from victor.agent.tool_selection import ToolSelector

        fallback_max_tools = getattr(self.settings, "fallback_max_tools", 8)
        runtime_intelligence = None
        if getattr(self, "container", None) is not None:
            runtime_intelligence = RuntimeIntelligenceService.from_container(self.container)

        # Merge ToolSettings into tool_selection_config so ToolSelector
        # has access to max_tool_schema_tokens, schema_promotion_threshold,
        # and max_mcp_tools_per_turn without a separate settings reference.
        merged_config = dict(tool_selection)
        tool_settings = self._resolve_setting_value(self.settings, "tools", None)
        if tool_settings is not None:
            if "max_tool_schema_tokens" not in merged_config:
                merged_config["max_tool_schema_tokens"] = self._resolve_setting_value(
                    tool_settings, "max_tool_schema_tokens", 0
                )
            if "schema_promotion_threshold" not in merged_config:
                merged_config["schema_promotion_threshold"] = self._resolve_setting_value(
                    tool_settings, "schema_promotion_threshold", 0.8
                )
            if "max_mcp_tools_per_turn" not in merged_config:
                merged_config["max_mcp_tools_per_turn"] = self._resolve_setting_value(
                    tool_settings, "max_mcp_tools_per_turn", 12
                )

        selector = ToolSelector(
            tools=tools,
            semantic_selector=semantic_selector,
            conversation_state=conversation_state,
            task_tracker=unified_tracker,
            model=model,
            provider_name=provider_name,
            tool_selection_config=merged_config,
            fallback_max_tools=fallback_max_tools,
            on_selection_recorded=on_selection_recorded,
            runtime_intelligence=runtime_intelligence,
        )

        logger.debug(f"ToolSelector created with fallback_max_tools={fallback_max_tools}")
        return selector

    def create_tool_cache(self) -> Optional["ToolCache"]:
        """Create tool cache if enabled.

        Returns:
            ToolCache instance or None if disabled
        """
        if not self._tool_setting("tool_cache_enabled", True):
            return None

        from victor.storage.cache.config import CacheConfig
        from victor.config.settings import get_project_paths
        from victor.storage.cache.tool_cache import ToolCache

        cache_dir = getattr(self.settings, "tool_cache_dir_override", None)
        if not cache_dir:
            cache_dir = getattr(self.settings, "tool_cache_dir", None)
        if cache_dir:
            cache_dir = Path(cache_dir).expanduser()
        else:
            cache_dir = Path(get_project_paths().global_cache_dir).expanduser()

        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache = ToolCache(
                ttl=self._tool_setting("tool_cache_ttl", 600),
                allowlist=self._tool_setting("tool_cache_allowlist", []),
                cache_config=CacheConfig(disk_path=cache_dir),
            )
        except Exception as exc:
            logger.warning("Tool cache disabled: failed to initialize %s: %s", cache_dir, exc)
            return None
        logger.debug(f"ToolCache created with TTL={cache.ttl}s")
        return cache

    def create_tool_registrar(
        self,
        tools: "ToolRegistry",
        tool_graph: "ToolDependencyGraphProtocol",
        provider: "BaseProvider",
        model: str,
    ) -> "ToolRegistrar":
        """Create tool registrar for dynamic tool discovery and registration.

        Args:
            tools: ToolRegistry instance for tool storage
            tool_graph: ToolGraph instance for dependency tracking
            provider: LLM provider instance
            model: Model identifier

        Returns:
            ToolRegistrar instance configured with settings
        """
        from victor.agent.tool_registrar import ToolRegistrar, ToolRegistrarConfig

        lazy_startup_env = os.getenv("VICTOR_TOOL_LAZY_STARTUP")
        lazy_startup_default = not (
            lazy_startup_env is not None
            and lazy_startup_env.strip().lower() in {"0", "false", "no", "off"}
        )
        lazy_startup = bool(getattr(self.settings, "tool_lazy_startup", lazy_startup_default))

        registrar = ToolRegistrar(
            tools=tools,
            settings=self.settings,
            provider=provider,
            model=model,
            tool_graph=tool_graph,
            config=ToolRegistrarConfig(
                enable_plugins=getattr(self.settings, "plugin_enabled", True),
                enable_mcp=getattr(self.settings, "use_mcp_tools", False),
                enable_tool_graph=True,
                airgapped_mode=getattr(self.settings, "airgapped_mode", False),
                plugin_dirs=getattr(self.settings, "plugin_dirs", []),
                disabled_plugins=set(getattr(self.settings, "disabled_plugins", [])),
                plugin_packages=getattr(self.settings, "plugin_packages", []),
                max_workers=4,
                max_complexity=10,
                lazy_startup=lazy_startup,
            ),
        )

        logger.debug(
            f"ToolRegistrar created with plugins={registrar.config.enable_plugins}, "
            f"mcp={registrar.config.enable_mcp}"
        )
        return registrar

    def create_tool_dependency_graph(self) -> "ToolDependencyGraphProtocol":
        """Create tool dependency graph from DI container.

        Returns:
            ToolDependencyGraph instance
        """
        from victor.agent.protocols import ToolDependencyGraphProtocol

        tool_graph = self.container.get(ToolDependencyGraphProtocol)

        logger.debug("ToolDependencyGraph created")
        return tool_graph

    def create_tool_output_formatter(
        self, context_compactor: "ContextCompactor"
    ) -> "ToolOutputFormatter":
        """Create tool output formatter for LLM-context-aware output formatting.

        Args:
            context_compactor: ContextCompactor instance for smart truncation

        Returns:
            ToolOutputFormatter instance configured with settings
        """
        from victor.agent.tool_output_formatter import (
            create_tool_output_formatter,
            ToolOutputFormatterConfig,
        )

        formatter = create_tool_output_formatter(
            config=ToolOutputFormatterConfig(
                max_output_chars=getattr(self.settings, "max_tool_output_chars", 15000),
                file_structure_threshold=getattr(self.settings, "file_structure_threshold", 50000),
            ),
            truncator=context_compactor,
        )
        logger.debug("ToolOutputFormatter created")
        return formatter

    def create_tool_deduplication_tracker(self) -> Optional["ToolDeduplicationTracker"]:
        """Create tool deduplication tracker for preventing redundant calls.

        Returns:
            ToolDeduplicationTracker instance if enabled, None otherwise
        """
        if not getattr(self.settings, "enable_tool_deduplication", False):
            logger.debug("Tool deduplication disabled")
            return None

        try:
            from victor.agent.tool_call_tracker import (
                ToolCallTracker as ToolDeduplicationTracker,
            )

            window_size = getattr(self.settings, "tool_deduplication_window_size", 10)
            tracker = ToolDeduplicationTracker(window_size=window_size)
            logger.info(f"ToolDeduplicationTracker initialized (window: {window_size})")
            return tracker
        except Exception as e:
            logger.warning(f"Failed to initialize ToolDeduplicationTracker: {e}")
            return None

    def create_tool_access_controller(
        self, registry: Optional["ToolRegistry"] = None
    ) -> "ToolAccessController":
        """Create ToolAccessController for unified tool access control.

        Args:
            registry: Optional tool registry for tool lookup

        Returns:
            ToolAccessController instance
        """
        from victor.agent.tool_access_controller import create_tool_access_controller

        controller = create_tool_access_controller(registry=registry)

        mc = self.mode_controller
        if mc is not None and hasattr(mc.config, "exploration_multiplier"):
            multiplier = mc.config.exploration_multiplier
            logger.debug(f"ToolAccessController created with mode multiplier: {multiplier}")

        return controller

    def create_semantic_selector(self) -> Optional["SemanticToolSelector"]:
        """Create semantic tool selector if enabled.

        Returns:
            SemanticToolSelector instance or None
        """
        use_semantic_selection = is_semantic_tool_selection_enabled(self.settings, default=False)

        if not use_semantic_selection:
            return None

        from victor.tools.semantic_selector import SemanticToolSelector

        semantic_selector = SemanticToolSelector(
            embedding_model=self.settings.tools.embedding_model,
            embedding_provider=self.settings.tools.embedding_provider,
            ollama_base_url=self.settings.provider.ollama_base_url,
            cache_embeddings=True,
        )
        logger.debug("SemanticToolSelector created")
        return semantic_selector

    def create_parallel_executor(self, tool_executor: "ToolExecutor") -> "ParallelToolExecutor":
        """Create parallel tool executor for concurrent independent tool calls.

        Args:
            tool_executor: The ToolExecutor instance to wrap

        Returns:
            ParallelExecutor instance wrapping the tool executor
        """
        from victor.agent.parallel_executor import create_parallel_executor

        parallel_enabled = getattr(self.settings, "parallel_tool_execution", True)
        max_concurrent = getattr(self.settings, "max_concurrent_tools", 5)

        parallel_executor = create_parallel_executor(
            tool_executor=tool_executor,
            max_concurrent=max_concurrent,
            enable=parallel_enabled,
        )
        logger.debug(
            f"ParallelExecutor created (enabled={parallel_enabled}, max_concurrent={max_concurrent})"
        )
        return parallel_executor

    def create_argument_normalizer(self, provider: "BaseProvider") -> "ArgumentNormalizer":
        """Create argument normalizer for handling malformed tool arguments.

        Args:
            provider: Provider instance for extracting provider name

        Returns:
            ArgumentNormalizer instance configured with provider name
        """
        from victor.agent.protocols import ArgumentNormalizerProtocol

        return self.container.get(ArgumentNormalizerProtocol)

    def create_tool_calling_matrix(self) -> tuple[Dict[str, Any], "ToolCallingMatrix"]:
        """Create ToolCallingMatrix for managing tool calling capabilities.

        Returns:
            Tuple of (tool_calling_models, tool_capabilities)
        """
        from victor.config.model_capabilities import ToolCallingMatrix

        tool_calling_models = getattr(self.settings, "tool_calling_models", {})

        tool_capabilities = ToolCallingMatrix(
            tool_calling_models,
            always_allow_providers=[
                "openai",
                "anthropic",
                "google",
                "xai",
                "cerebras",
                "groq",
                "deepseek",
                "mistral",
                "together",
                "fireworks",
                "openrouter",
                "cohere",
                "moonshot",
            ],
        )

        logger.debug(f"ToolCallingMatrix created with {len(tool_calling_models)} model configs")
        return (tool_calling_models, tool_capabilities)

    def initialize_plugin_system(
        self, tool_registrar: "ToolRegistrar"
    ) -> Optional["ToolPluginRegistry"]:
        """Initialize plugin system for extensible tools.

        Args:
            tool_registrar: ToolRegistrar instance for plugin management

        Returns:
            ToolPluginRegistry instance if plugins enabled, None otherwise
        """
        if not getattr(self.settings, "plugin_enabled", True):
            logger.debug("Plugin system disabled")
            return None

        tool_count = tool_registrar.initialize_plugins()
        plugin_manager = tool_registrar.plugin_manager

        if tool_count > 0:
            logger.info(f"Plugins initialized via ToolRegistrar: {tool_count} tools")
        else:
            logger.debug("No plugins loaded")

        return plugin_manager

    def create_tool_result_deduplicator(self) -> "ToolResultDeduplicator":
        """Create ToolResultDeduplicator for file read deduplication."""
        from victor.agent.tool_result_deduplicator import ToolResultDeduplicator

        logger.debug("ToolResultDeduplicator created")
        return ToolResultDeduplicator()

    def create_tool_planner(self) -> "ToolPlannerProtocol":
        """Create ToolPlanner via DI container.

        Returns:
            ToolPlanner instance for tool planning
        """
        from victor.agent.services.protocols import ToolPlanningRuntimeProtocol

        tool_planner = self.container.get(ToolPlanningRuntimeProtocol)
        logger.debug("ToolPlanner created via DI")
        return tool_planner

    def initialize_tool_budget(self, tool_calling_caps: "ToolCallingCapabilities") -> int:
        """Initialize tool call budget with adapter recommendations.

        Args:
            tool_calling_caps: Tool calling capabilities from adapter

        Returns:
            Tool call budget (integer)
        """
        default_budget = tool_calling_caps.recommended_tool_budget
        default_budget = max(default_budget, 50)
        tool_budget = getattr(self.settings, "tool_call_budget", default_budget)

        logger.debug(
            f"Tool budget initialized: {tool_budget} "
            f"(adapter recommended: {tool_calling_caps.recommended_tool_budget}, "
            f"minimum: 50)"
        )
        return tool_budget
