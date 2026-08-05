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

"""Lifecycle Manager - Manages session lifecycle and resource cleanup.

This module extracts lifecycle management from AgentOrchestrator:
- Session initialization and recovery
- Conversation reset and cleanup
- Graceful shutdown coordination
- Resource cleanup and deallocation

Design Principles:
- Single Responsibility: Manage lifecycle operations only
- Delegation: Use existing components for actual work
- Composable: Works with existing controllers and managers
- Observable: Support for lifecycle hooks

Usage:
    lifecycle_manager = LifecycleManager(
        conversation_controller=controller,
        metrics_collector=metrics,
        context_compactor=compactor,
    )

    # Reset conversation
    lifecycle_manager.reset_conversation()

    # Graceful shutdown
    await lifecycle_manager.graceful_shutdown()
    await lifecycle_manager.shutdown()
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.conversation.controller import ConversationController

logger = logging.getLogger(__name__)


class LifecycleManager:
    """Manage session lifecycle and resource cleanup.

    This coordinator manages the lifecycle of the orchestrator, handling
    session initialization, cleanup, and graceful shutdown.

    Responsibilities:
    - Reset conversation and session state
    - Recover previous sessions
    - Coordinate graceful shutdown
    - Clean up resources (connections, tasks, etc.)
    - Flush analytics and end sessions

    Example:
        lifecycle_manager = LifecycleManager(
            conversation_controller=controller,
            metrics_collector=metrics,
            context_compactor=compactor,
        )

        # Reset conversation
        lifecycle_manager.reset_conversation()

        # Graceful shutdown
        await lifecycle_manager.graceful_shutdown()

        # Full cleanup
        await lifecycle_manager.shutdown()
    """

    def __init__(
        self,
        conversation_controller: "ConversationController",
        metrics_collector: Optional[Any] = None,
        context_compactor: Optional[Any] = None,
        sequence_tracker: Optional[Any] = None,
        usage_analytics: Optional[Any] = None,
        reminder_manager: Optional[Any] = None,
    ):
        """Initialize the lifecycle manager.

        Args:
            conversation_controller: Controller for conversation management
            metrics_collector: Optional metrics collector for stats
            context_compactor: Optional context compactor for cleanup
            sequence_tracker: Optional sequence tracker for pattern learning
            usage_analytics: Optional usage analytics for session tracking
            reminder_manager: Optional reminder manager for context reminders
        """
        self._conversation_controller = conversation_controller
        self._metrics_collector = metrics_collector
        self._context_compactor = context_compactor
        self._sequence_tracker = sequence_tracker
        self._usage_analytics = usage_analytics
        self._reminder_manager = reminder_manager

        # Optional components for shutdown
        self._provider: Optional[Any] = None
        self._code_manager: Optional[Any] = None
        self._semantic_selector: Optional[Any] = None
        self._background_tasks: List[Any] = []
        self._usage_logger: Optional[Any] = None

        # Orchestrator-specific callbacks for shutdown
        self._flush_analytics_callback: Optional[callable] = None
        self._stop_health_monitoring_callback: Optional[callable] = None

    def reset_conversation(self) -> None:
        """Clear conversation history and session state.

        Resets:
        - Conversation history
        - Conversation state machine
        - Context reminder manager
        - Metrics collector stats
        - Context compactor statistics
        - Sequence tracker history (preserves learned patterns)
        - Usage analytics session (ends current, starts fresh)
        """
        # Reset conversation (clear() method doesn't exist, reset() handles it)
        self._conversation_controller.reset()

        # Reset context reminder manager
        if self._reminder_manager is not None:
            self._reminder_manager.reset()

        # Reset metrics collector
        if self._metrics_collector is not None:
            self._metrics_collector.reset_stats()

        # Reset optimization components for clean session
        if self._context_compactor is not None:
            self._context_compactor.reset_statistics()

        if self._sequence_tracker is not None:
            self._sequence_tracker.clear_history()

        if self._usage_analytics is not None:
            # End current session if active, start fresh
            if self._usage_analytics._current_session is not None:
                self._usage_analytics.end_session()
            self._usage_analytics.start_session()

        logger.debug("Conversation and session state reset (including optimization components)")

    async def graceful_shutdown(self) -> Dict[str, bool]:
        """Perform graceful shutdown of all orchestrator components.

        Flushes analytics, stops health monitoring, and cleans up resources.
        Call this before application exit.

        Returns:
            Dictionary with shutdown status for each component
        """
        results: Dict[str, bool] = {}

        # Flush analytics data
        try:
            flush_results = self._flush_analytics()
            results["analytics_flushed"] = all(flush_results.values()) if flush_results else True
        except Exception as e:
            logger.warning(f"Failed to flush analytics during shutdown: {e}")
            results["analytics_flushed"] = False

        # Stop health monitoring
        try:
            results["health_monitoring_stopped"] = await self._stop_health_monitoring()
        except Exception as e:
            logger.warning(f"Failed to stop health monitoring: {e}")
            results["health_monitoring_stopped"] = False

        # End usage analytics session
        if self._usage_analytics is not None:
            try:
                if self._usage_analytics._current_session is not None:
                    self._usage_analytics.end_session()
                results["session_ended"] = True
            except Exception as e:
                logger.warning(f"Failed to end analytics session: {e}")
                results["session_ended"] = False
        else:
            results["session_ended"] = True

        logger.debug(f"Graceful shutdown complete: {results}")
        return results

    async def shutdown(self) -> None:
        """Clean up resources and shutdown gracefully.

        Should be called when the orchestrator is no longer needed.
        Cleans up:
        - Background async tasks
        - Provider connections
        - Code execution manager (Docker containers)
        - Semantic selector resources
        - HTTP clients
        """
        logger.debug("Shutting down LifecycleManager...")

        # Cancel all background tasks first
        if self._background_tasks:
            logger.debug("Cancelling %d background task(s)...", len(self._background_tasks))
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()

            # Wait for all tasks to complete cancellation (only if they're awaitable)
            if self._background_tasks:
                try:
                    # Filter for actual asyncio.Task objects
                    real_tasks = [t for t in self._background_tasks if hasattr(t, "_coroutine")]
                    if real_tasks:
                        await asyncio.gather(*real_tasks, return_exceptions=True)
                except Exception as e:
                    logger.debug(f"Error gathering background tasks: {e}")
            self._background_tasks.clear()
            logger.debug("Background tasks cancelled")

        # Close provider connection
        if self._provider is not None:
            try:
                # Check if provider has a close method
                if hasattr(self._provider, "close") and asyncio.iscoroutinefunction(
                    self._provider.close
                ):
                    await self._provider.close()
                    logger.debug("Provider connection closed")
                elif hasattr(self._provider, "close") and callable(self._provider.close):
                    # Synchronous close method
                    self._provider.close()
                    logger.debug("Provider connection closed (sync)")
                else:
                    logger.debug("Provider has no close method")
            except Exception as e:
                logger.warning("Error closing provider: %s", str(e))

        # Stop code execution manager (cleans up Docker containers)
        if self._code_manager is not None:
            try:
                self._code_manager.stop()
                logger.debug("Code execution manager stopped")
            except Exception as e:
                logger.warning(f"Error stopping code manager: {e}")

        # Close semantic selector
        if self._semantic_selector is not None:
            try:
                # Check if semantic selector has a close method
                if hasattr(self._semantic_selector, "close") and asyncio.iscoroutinefunction(
                    self._semantic_selector.close
                ):
                    await self._semantic_selector.close()
                    logger.debug("Semantic selector closed")
                elif hasattr(self._semantic_selector, "close") and callable(
                    self._semantic_selector.close
                ):
                    # Synchronous close method
                    self._semantic_selector.close()
                    logger.debug("Semantic selector closed (sync)")
                else:
                    logger.debug("Semantic selector has no close method")
            except Exception as e:
                logger.warning(f"Error closing semantic selector: {e}")

        # Signal shutdown to EmbeddingService singleton
        # This prevents post-shutdown embedding operations
        try:
            from victor.storage.embeddings.service import EmbeddingService

            if EmbeddingService._instance is not None:
                EmbeddingService._instance.shutdown()
                logger.debug("EmbeddingService shutdown signaled")
        except Exception as e:
            logger.debug(f"Error signaling EmbeddingService shutdown: {e}")

        # Close cached embedding providers (vector store connections, model handles)
        try:
            from victor.storage.vector_stores.registry import EmbeddingRegistry

            await EmbeddingRegistry.close_all()
            logger.debug("EmbeddingRegistry providers closed")
        except Exception as e:
            logger.debug(f"Error closing EmbeddingRegistry: {e}")

        logger.debug("LifecycleManager shutdown complete")

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for current session.

        Returns:
            Dictionary of session statistics
        """
        message_count = 0
        if hasattr(self._conversation_controller, "message_count"):
            message_count = self._conversation_controller.message_count()

        return {
            "message_count": message_count,
        }

    def recover_session(
        self,
        session_id: str,
        memory_manager: Optional[Any],
    ) -> bool:
        """Recover a previous conversation session.

        Args:
            session_id: ID of the session to recover
            memory_manager: Memory manager for session persistence

        Returns:
            True if session was recovered successfully
        """
        if memory_manager is None:
            logger.warning("Memory manager not provided, cannot recover session")
            return False

        try:
            session = memory_manager.get_session(session_id)
            if not session:
                logger.warning("Session not found: %s", session_id)
                return False

            # Restore messages to in-memory conversation
            self._conversation_controller.reset()
            for msg in session.messages:
                # Add to conversation
                role = getattr(msg, "role", "user")
                content = getattr(msg, "content", "")

                kwargs = {}
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    kwargs["tool_calls"] = msg.tool_calls

                self._conversation_controller.add_message(role, content, **kwargs)

            logger.info(f"Recovered session {session_id} with {len(session.messages)} messages")
            return True

        except Exception as e:
            logger.warning(f"Failed to recover session {session_id}: {e}")
            return False

    # =========================================================================
    # INTERNAL METHODS (for orchestrator integration)
    # =========================================================================

    def _flush_analytics(self) -> Dict[str, bool]:
        """Flush analytics data.

        Calls the orchestrator callback if set.

        Returns:
            Dictionary with flush status for each analytics type
        """
        if self._flush_analytics_callback:
            return self._flush_analytics_callback()
        return {}

    async def _stop_health_monitoring(self) -> bool:
        """Stop health monitoring.

        Calls the orchestrator callback if set.

        Returns:
            True if stopped successfully
        """
        if self._stop_health_monitoring_callback:
            return await self._stop_health_monitoring_callback()
        return True

    # =========================================================================
    # DEPENDENCY SETTERS (called by orchestrator during initialization)
    # =========================================================================

    def set_provider(self, provider: Any) -> None:
        """Set provider for cleanup during shutdown.

        Args:
            provider: LLM provider instance
        """
        self._provider = provider

    def set_code_manager(self, code_manager: Any) -> None:
        """Set code execution manager for cleanup.

        Args:
            code_manager: Code execution manager instance
        """
        self._code_manager = code_manager

    def set_semantic_selector(self, semantic_selector: Any) -> None:
        """Set semantic selector for cleanup.

        Args:
            semantic_selector: Semantic selector instance
        """
        self._semantic_selector = semantic_selector

    def set_background_tasks(self, tasks: List[Any]) -> None:
        """Set background tasks for cancellation during shutdown.

        Args:
            tasks: List of async tasks
        """
        self._background_tasks = tasks

    def set_usage_logger(self, usage_logger: Any) -> None:
        """Set usage logger for final analytics.

        Args:
            usage_logger: Usage logger instance
        """
        self._usage_logger = usage_logger

    def set_flush_analytics_callback(self, callback: callable) -> None:
        """Set callback for flushing analytics during shutdown.

        Args:
            callback: Function to call for analytics flushing
        """
        self._flush_analytics_callback = callback

    def set_stop_health_monitoring_callback(self, callback: callable) -> None:
        """Set callback for stopping health monitoring during shutdown.

        Args:
            callback: Async function to call for stopping health monitoring
        """
        self._stop_health_monitoring_callback = callback


def create_lifecycle_manager(
    conversation_controller: "ConversationController",
    metrics_collector: Optional[Any] = None,
    context_compactor: Optional[Any] = None,
    sequence_tracker: Optional[Any] = None,
    usage_analytics: Optional[Any] = None,
    reminder_manager: Optional[Any] = None,
) -> LifecycleManager:
    """Factory function to create a LifecycleManager.

    Args:
        conversation_controller: Controller for conversation management
        metrics_collector: Optional metrics collector for stats
        context_compactor: Optional context compactor for cleanup
        sequence_tracker: Optional sequence tracker for pattern learning
        usage_analytics: Optional usage analytics for session tracking
        reminder_manager: Optional reminder manager for context reminders

    Returns:
        Configured LifecycleManager instance
    """
    return LifecycleManager(
        conversation_controller=conversation_controller,
        metrics_collector=metrics_collector,
        context_compactor=context_compactor,
        sequence_tracker=sequence_tracker,
        usage_analytics=usage_analytics,
        reminder_manager=reminder_manager,
    )


__all__ = [
    "LifecycleManager",
    "create_lifecycle_manager",
]
