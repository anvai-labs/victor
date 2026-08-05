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

"""Runtime builder methods for OrchestratorFactory.

Provides creation methods for streaming, conversation, provider management,
lifecycle, and response processing components.

Part of CRITICAL-001: Monolithic Orchestrator decomposition.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.config.settings import Settings
    from victor.providers.base import BaseProvider
    from victor.agent.tool_calling import (
        BaseToolCallingAdapter,
        ToolCallingCapabilities,
    )
    from victor.agent.response_sanitizer import ResponseSanitizer
    from victor.agent.conversation.controller import ConversationController
    from victor.agent.tool_pipeline import ToolPipeline
    from victor.agent.streaming_controller import StreamingController
    from victor.agent.context_compactor import ContextCompactor
    from victor.agent.usage_analytics import UsageAnalytics
    from victor.agent.tool_sequence_tracker import ToolSequenceTracker
    from victor.agent.tool_output_formatter import ToolOutputFormatter
    from victor.agent.metrics_collector import MetricsCollector
    from victor.agent.conversation.store import ConversationStore
    from victor.observability.analytics.logger import UsageLogger
    from victor.observability.analytics.streaming_metrics import (
        StreamingMetricsCollector,
    )
    from victor.agent.response_completer import ResponseCompleter
    from victor.agent.message_history import MessageHistory
    from victor.agent.conversation.state_machine import ConversationStateMachine
    from victor.agent.streaming.streaming_coordinator import StreamingCoordinator
    from victor.agent.streaming.handler import StreamingChatHandler
    from victor.agent.services.chat_stream_executor import StreamingChatExecutor
    from victor.agent.lifecycle_manager import LifecycleManager
    from victor.agent.provider_manager import ProviderManager
    from victor.tools.registry import ToolRegistry
    from victor.agent.protocols.provider_protocols import (
        IProviderSwitcher,
        IProviderHealthMonitor,
    )
    from victor.agent.services.protocols import ReminderManagerProtocol

logger = logging.getLogger(__name__)

_CONVERSATION_STORE_INIT_LOCK = threading.Lock()
_CONVERSATION_STORE_LOCK_RETRY_DELAYS = (0.05, 0.2, 0.5)


class RuntimeBuildersMixin:
    """Mixin providing runtime-related factory methods.

    Requires the host class to provide:
        - self.settings: Settings
        - self.provider: BaseProvider
        - self.model: str
        - self.provider_name: Optional[str]
        - self.profile_name: Optional[str]
        - self.container: DI container
    """

    def create_streaming_controller(
        self,
        streaming_metrics_collector: Optional["StreamingMetricsCollector"],
        on_session_complete: Callable,
    ) -> "StreamingController":
        """Create streaming controller for managing streaming sessions and metrics.

        Args:
            streaming_metrics_collector: Optional metrics collector for session tracking
            on_session_complete: Callback invoked when streaming session completes

        Returns:
            StreamingController instance configured for session management
        """
        from victor.agent.streaming_controller import (
            StreamingController,
            StreamingControllerConfig,
        )

        controller = StreamingController(
            config=StreamingControllerConfig(
                max_history=100,
                enable_metrics_collection=streaming_metrics_collector is not None,
            ),
            metrics_collector=streaming_metrics_collector,
            on_session_complete=on_session_complete,
        )
        logger.debug("StreamingController created")
        return controller

    def create_streaming_coordinator(
        self,
        streaming_controller: "StreamingController",
    ) -> "StreamingCoordinator":
        """Create streaming coordinator for response processing.

        Args:
            streaming_controller: StreamingController for session management

        Returns:
            StreamingCoordinator instance for processing streaming responses
        """
        from victor.agent.streaming.streaming_coordinator import StreamingCoordinator

        coordinator = StreamingCoordinator(
            streaming_controller=streaming_controller,
        )
        logger.debug("StreamingCoordinator created")
        return coordinator

    def create_streaming_chat_handler(self, message_adder: Any) -> "StreamingChatHandler":
        """Create streaming chat handler for testable streaming loop logic.

        Args:
            message_adder: Object implementing add_message() interface

        Returns:
            StreamingChatHandler instance configured with settings
        """
        from victor.agent.streaming import StreamingChatHandler

        session_idle_timeout = getattr(self.settings, "session_idle_timeout", 180.0)
        presentation = getattr(message_adder, "_presentation", None)
        handler = StreamingChatHandler(
            settings=self.settings,
            message_adder=message_adder,
            session_idle_timeout=session_idle_timeout,
            presentation=presentation,
        )
        logger.debug(f"StreamingChatHandler created (idle_timeout={session_idle_timeout})")
        return handler

    def create_streaming_chat_adapter(self, runtime_owner: Any) -> Any:
        """Create the canonical service-owned chat-stream adapter."""
        from victor.agent.services.chat_stream_runtime import ServiceStreamingRuntime

        adapter = ServiceStreamingRuntime(runtime_owner)
        logger.debug("Streaming chat adapter created")
        return adapter

    def create_streaming_metrics_collector(
        self,
    ) -> Optional["StreamingMetricsCollector"]:
        """Create streaming metrics collector if enabled."""
        from victor.observability.analytics.streaming_metrics import (
            StreamingMetricsCollector,
        )

        if not getattr(self.settings, "streaming_metrics_enabled", True):
            return None

        history_size = getattr(self.settings, "streaming_metrics_history_size", 1000)
        return StreamingMetricsCollector(max_history=history_size)

    def create_conversation_controller(
        self,
        provider: "BaseProvider",
        model: str,
        conversation: List[Dict[str, Any]],
        conversation_state: "ConversationStateMachine",
        memory_manager: Optional["ConversationStore"],
        memory_session_id: str,
        system_prompt: str,
        context_reminder_manager: Optional[Any] = None,
        hierarchical_manager: Optional[Any] = None,
    ) -> "ConversationController":
        """Create conversation controller for managing message history and state.

        Args:
            provider: LLM provider instance for context calculations
            model: Model identifier for context limits
            conversation: Message history list
            conversation_state: ConversationStateMachine instance
            memory_manager: Memory manager for persistent storage
            memory_session_id: Unique session identifier
            system_prompt: System prompt to set on the controller
            context_reminder_manager: Optional reminder manager for consolidated reminders
            hierarchical_manager: Optional hierarchical compaction manager

        Returns:
            ConversationController instance configured with model-aware settings
        """
        from victor.agent.conversation.controller import (
            ConversationController,
            ConversationConfig,
            CompactionStrategy,
        )
        from victor.agent.orchestrator_utils import calculate_max_context_chars

        model_context_chars = calculate_max_context_chars(self.settings, provider, model)

        compaction_strategy_str = getattr(
            self.settings, "context_compaction_strategy", "tiered"
        ).lower()
        compaction_strategy_map = {
            "simple": CompactionStrategy.SIMPLE,
            "tiered": CompactionStrategy.TIERED,
            "semantic": CompactionStrategy.SEMANTIC,
            "hybrid": CompactionStrategy.HYBRID,
        }
        compaction_strategy = compaction_strategy_map.get(
            compaction_strategy_str, CompactionStrategy.TIERED
        )

        controller = ConversationController(
            config=ConversationConfig(
                max_context_chars=model_context_chars,
                enable_stage_tracking=True,
                enable_context_monitoring=True,
                compaction_strategy=compaction_strategy,
                min_messages_to_keep=getattr(self.settings, "context_min_messages_to_keep", 6),
                tool_result_retention_weight=getattr(
                    self.settings, "context_tool_retention_weight", 1.5
                ),
                recent_message_weight=getattr(self.settings, "context_recency_weight", 2.0),
                semantic_relevance_threshold=getattr(
                    self.settings, "context_semantic_threshold", 0.3
                ),
            ),
            message_history=conversation,
            state_machine=conversation_state,
            conversation_store=memory_manager,
            session_id=memory_session_id,
            context_reminder_manager=context_reminder_manager,
            hierarchical_manager=hierarchical_manager,
        )
        controller.set_system_prompt(system_prompt)

        logger.debug(
            f"ConversationController created with max_context_chars={model_context_chars}, "
            f"compaction_strategy={compaction_strategy}"
        )
        return controller

    def create_conversation_state_machine(self) -> "ConversationStateMachine":
        """Create conversation state machine for intelligent stage detection.

        Returns:
            ConversationStateMachine instance
        """
        from victor.agent.protocols import ConversationStateMachineProtocol

        return self.container.get(ConversationStateMachineProtocol)

    def create_memory_components(
        self, provider_name: str, tool_capable: bool = True
    ) -> Tuple[Optional["ConversationStore"], Optional[str]]:
        """Create conversation memory components.

        Args:
            provider_name: Provider name for session metadata
            tool_capable: Whether the model supports tool calling

        Returns:
            Tuple of (ConversationStore or None, session_id or None)
        """
        if not getattr(self.settings, "conversation_memory_enabled", True):
            self._last_memory_initialization_diagnostics = {
                "status": "disabled",
                "recovered_from_lock": False,
                "lock_retries": 0,
                "db_path": None,
                "session_id": None,
                "last_error": None,
            }
            return None, None

        from victor.config.settings import get_project_paths

        paths = get_project_paths()
        paths.project_victor_dir.mkdir(parents=True, exist_ok=True)
        db_path = paths.project_db
        max_context = getattr(self.settings, "max_context_tokens", 100000)
        response_reserve = getattr(self.settings, "response_token_reserve", 4096)
        diagnostics: Dict[str, Any] = {
            "status": "initializing",
            "recovered_from_lock": False,
            "lock_retries": 0,
            "db_path": str(db_path),
            "session_id": None,
            "last_error": None,
        }
        self._last_memory_initialization_diagnostics = diagnostics

        attempts = len(_CONVERSATION_STORE_LOCK_RETRY_DELAYS) + 1
        for attempt in range(attempts):
            try:
                from victor.agent.conversation.store import ConversationStore

                with _CONVERSATION_STORE_INIT_LOCK:
                    memory_manager = ConversationStore(
                        db_path=db_path,
                        max_context_tokens=max_context,
                        response_reserve=response_reserve,
                    )

                    project_path = str(paths.project_root)
                    session = memory_manager.create_session(
                        project_path=project_path,
                        provider=provider_name,
                        model=self.model,
                        max_tokens=max_context,
                        profile=self.profile_name,
                        tool_capable=tool_capable,
                    )
                session_id = session.session_id
                diagnostics.update(
                    {
                        "status": "initialized",
                        "recovered_from_lock": diagnostics["lock_retries"] > 0,
                        "session_id": session_id,
                    }
                )
                logger.info(
                    f"ConversationStore initialized via factory. "
                    f"Session: {session_id[:8]}..., DB: {db_path}"
                )
                return memory_manager, session_id

            except Exception as e:
                diagnostics["last_error"] = str(e)
                if not self._is_conversation_store_lock_error(e) or attempt == attempts - 1:
                    diagnostics["status"] = "failed"
                    logger.warning(f"Failed to initialize ConversationStore: {e}")
                    return None, None

                diagnostics["lock_retries"] = int(diagnostics["lock_retries"]) + 1
                delay = _CONVERSATION_STORE_LOCK_RETRY_DELAYS[attempt]
                logger.debug(
                    "ConversationStore initialization hit SQLite lock; retrying in %.2fs (%d/%d)",
                    delay,
                    attempt + 1,
                    attempts - 1,
                )
                time.sleep(delay)

        return None, None

    def get_memory_initialization_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostics for the last conversation memory initialization attempt."""
        diagnostics = getattr(self, "_last_memory_initialization_diagnostics", None)
        return dict(diagnostics) if isinstance(diagnostics, dict) else {}

    @staticmethod
    def _is_conversation_store_lock_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "database is locked" in message or "database table is locked" in message

    def create_message_history(self, system_prompt: str) -> "MessageHistory":
        """Create message history for conversation tracking.

        Args:
            system_prompt: System prompt to initialize the conversation with

        Returns:
            MessageHistory instance configured with settings
        """
        from victor.agent.message_history import MessageHistory

        max_history = getattr(self.settings, "max_conversation_history", 100000)
        history = MessageHistory(
            system_prompt=system_prompt,
            max_history_messages=max_history,
        )

        logger.debug(f"MessageHistory created with max_history={max_history}")
        return history

    def create_lifecycle_manager(
        self,
        conversation_controller: "ConversationController",
        metrics_collector: Optional["MetricsCollector"] = None,
        context_compactor: Optional["ContextCompactor"] = None,
        sequence_tracker: Optional["ToolSequenceTracker"] = None,
        usage_analytics: Optional["UsageAnalytics"] = None,
        reminder_manager: Optional["ReminderManagerProtocol"] = None,
    ) -> "LifecycleManager":
        """Create lifecycle manager for session lifecycle and resource cleanup.

        Args:
            conversation_controller: Controller for conversation management
            metrics_collector: Optional metrics collector for stats
            context_compactor: Optional context compactor for cleanup
            sequence_tracker: Optional sequence tracker for pattern learning
            usage_analytics: Optional usage analytics for session tracking
            reminder_manager: Optional reminder manager for context reminders

        Returns:
            LifecycleManager instance for managing lifecycle operations
        """
        from victor.agent.lifecycle_manager import LifecycleManager

        lifecycle_manager = LifecycleManager(
            conversation_controller=conversation_controller,
            metrics_collector=metrics_collector,
            context_compactor=context_compactor,
            sequence_tracker=sequence_tracker,
            usage_analytics=usage_analytics,
            reminder_manager=reminder_manager,
        )
        logger.debug("LifecycleManager created")
        return lifecycle_manager

    def create_provider_manager_with_adapter(
        self,
        provider: "BaseProvider",
        model: str,
        provider_name: str,
    ) -> tuple[
        "ProviderManager",
        "BaseProvider",
        str,
        str,
        "BaseToolCallingAdapter",
        "ToolCallingCapabilities",
    ]:
        """Create ProviderManager and initialize tool adapter.

        Args:
            provider: Initial LLM provider instance
            model: Initial model identifier
            provider_name: Provider name string

        Returns:
            Tuple of (provider_manager, provider, model, provider_name,
                     tool_adapter, tool_calling_caps)
        """
        from victor.agent.provider_manager import ProviderManager, ProviderManagerConfig

        manager = ProviderManager(
            settings=self.settings,
            initial_provider=provider,
            initial_model=model,
            provider_name=provider_name,
            config=ProviderManagerConfig(
                enable_health_checks=getattr(self.settings, "provider_health_checks", True),
                auto_fallback=getattr(self.settings, "provider_auto_fallback", True),
                fallback_providers=getattr(self.settings, "fallback_providers", []),
            ),
        )

        manager.initialize_tool_adapter()

        logger.info(
            f"Tool calling adapter: {manager.tool_adapter.provider_name}, "
            f"native={manager.capabilities.native_tool_calls}, "
            f"format={manager.capabilities.tool_call_format.value}"
        )

        return (
            manager,
            manager.provider,
            manager.model,
            manager.provider_name,
            manager.tool_adapter,
            manager.capabilities,
        )

    def create_response_completer(self) -> "ResponseCompleter":
        """Create response completer for ensuring complete responses after tool calls.

        Returns:
            ResponseCompleter instance
        """
        from victor.agent.response_completer import create_response_completer

        response_completer = create_response_completer(
            provider=self.provider,
            max_retries=getattr(self.settings, "response_completion_retries", 3),
            force_response=getattr(self.settings, "force_response_on_error", True),
        )
        logger.debug("ResponseCompleter created")
        return response_completer
