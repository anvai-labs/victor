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

"""Infrastructure builder methods for OrchestratorFactory.

Provides creation methods for observability, tracing, debug logging, metrics,
analytics, budget management, checkpoints, and execution state.

Part of CRITICAL-001: Monolithic Orchestrator decomposition.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.config.settings import Settings
    from victor.providers.base import BaseProvider
    from victor.agent.tool_calling import ToolCallingCapabilities
    from victor.agent.metrics_collector import MetricsCollector
    from victor.agent.usage_analytics import UsageAnalytics
    from victor.agent.tool_sequence_tracker import ToolSequenceTracker
    from victor.agent.unified_task_tracker import UnifiedTaskTracker
    from victor.agent.debug_logger import DebugLogger
    from victor.agent.budget_manager import BudgetManager
    from victor.agent.session_ledger import SessionLedger
    from victor.agent.compaction_summarizer import LedgerAwareCompactionSummarizer
    from victor.agent.compaction_hierarchy import HierarchicalCompactionManager
    from victor.agent.conversation.controller import ConversationController
    from victor.observability.integration import ObservabilityIntegration
    from victor.observability.tracing import ExecutionTracer, ToolCallTracer
    from victor.observability.analytics.logger import UsageLogger
    from victor.observability.analytics.streaming_metrics import (
        StreamingMetricsCollector,
    )
    from victor.storage.checkpoints import ConversationCheckpointManager
    from victor.agent.orchestrator_integration import IntegrationConfig
    from victor.agent.presentation.protocols import PresentationProtocol
    from victor.agent.protocols.infrastructure_protocols import DebugLoggerProtocol

logger = logging.getLogger(__name__)


class InfrastructureBuildersMixin:
    """Mixin providing infrastructure-related factory methods.

    Requires the host class to provide:
        - self.settings: Settings
        - self.provider: BaseProvider
        - self.model: str
        - self.provider_name: Optional[str]
        - self.container: DI container
        - self.mode_controller: property from ModeAwareMixin
    """

    def create_observability(self) -> Optional["ObservabilityIntegration"]:
        """Create observability integration if enabled."""
        from victor.observability.integration import ObservabilityIntegration

        if not getattr(self.settings, "enable_observability", True):
            return None

        observability = ObservabilityIntegration()
        logger.debug("Observability integration created")
        return observability

    def create_debug_logger_configured(self) -> "DebugLogger":
        """Create and configure debug logger for conversation tracking.

        Returns:
            Configured debug logger instance
        """
        import logging as _logging
        from victor.agent.protocols import DebugLoggerProtocol

        debug_logger = self.container.get(DebugLoggerProtocol)

        debug_logger.enabled = (
            getattr(self.settings, "debug_logging", False)
            or _logging.getLogger("victor").level <= _logging.DEBUG
        )

        logger.debug(f"Debug logger initialized: enabled={debug_logger.enabled}")
        return debug_logger

    def create_metrics_collector(
        self,
        streaming_metrics_collector: Optional["StreamingMetricsCollector"],
        usage_logger: "UsageLogger",
        debug_logger: "DebugLogger",
        tool_cost_lookup: Callable[[str], str],
    ) -> "MetricsCollector":
        """Create metrics collector.

        Args:
            streaming_metrics_collector: Optional streaming metrics collector
            usage_logger: Usage logger instance
            debug_logger: Debug logger instance
            tool_cost_lookup: Function to lookup tool cost tier

        Returns:
            Configured MetricsCollector
        """
        from victor.agent.metrics_collector import (
            MetricsCollector,
            MetricsCollectorConfig,
        )

        return MetricsCollector(
            config=MetricsCollectorConfig(
                model=self.model,
                provider=self.provider_name or self.provider.__class__.__name__,
                analytics_enabled=streaming_metrics_collector is not None,
            ),
            usage_logger=usage_logger,
            debug_logger=debug_logger,
            streaming_metrics_collector=streaming_metrics_collector,
            tool_cost_lookup=tool_cost_lookup,
        )

    def create_usage_analytics(self) -> "UsageAnalytics":
        """Create usage analytics (from DI container)."""
        from victor.agent.protocols import UsageAnalyticsProtocol

        return self.container.get(UsageAnalyticsProtocol)

    def create_usage_logger(self) -> "UsageLogger":
        """Create usage logger (DI with fallback).

        Returns:
            UsageLogger instance
        """
        from victor.core.bootstrap import UsageLoggerProtocol
        from victor.analytics.logger import UsageLogger
        from victor.config.settings import get_project_paths

        usage_logger = self.container.get_optional(UsageLoggerProtocol)
        if usage_logger is not None:
            logger.debug("Using enhanced usage logger from DI container")
            return usage_logger

        # Determine analytics log file location
        # Redirect test telemetry to prevent MagicMock leakage to global usage.jsonl
        import os

        if (
            os.getenv("PYTEST_XDIST_WORKER")
            or os.getenv("TEST_MODE")
            or os.getenv("PYTEST_CURRENT_TEST")
        ):
            # Running under pytest - use test-specific log file
            import tempfile

            test_log_dir = Path(tempfile.gettempdir()) / "victor_test_telemetry"
            test_log_dir.mkdir(exist_ok=True)
            analytics_log_file = test_log_dir / "test_usage.jsonl"
        else:
            # Normal operation - use global logs directory
            analytics_log_file = get_project_paths().global_logs_dir / "usage.jsonl"

        usage_logger = UsageLogger(
            analytics_log_file,
            enabled=getattr(self.settings, "analytics_enabled", True),
        )
        logger.debug("Using basic usage logger (enhanced version not available)")
        return usage_logger

    def create_sequence_tracker(self) -> "ToolSequenceTracker":
        """Create tool sequence tracker (from DI container)."""
        from victor.agent.protocols import ToolSequenceTrackerProtocol

        return self.container.get(ToolSequenceTrackerProtocol)

    def create_unified_tracker(
        self, tool_calling_caps: "ToolCallingCapabilities"
    ) -> "UnifiedTaskTracker":
        """Create unified task tracker with model-specific exploration settings.

        Args:
            tool_calling_caps: Tool calling capabilities for model-specific settings

        Returns:
            UnifiedTaskTracker instance configured with model exploration parameters
        """
        from victor.agent.protocols import TaskTrackerProtocol, ModeControllerProtocol

        unified_tracker = self.container.get(TaskTrackerProtocol)

        unified_tracker.set_model_exploration_settings(
            exploration_multiplier=tool_calling_caps.exploration_multiplier,
            continuation_patience=tool_calling_caps.continuation_patience,
        )

        try:
            mode_controller = self.container.get_optional(ModeControllerProtocol)
            if mode_controller:
                from victor.agent.mode_controller import MODE_CONFIGS

                current_mode = mode_controller.current_mode
                mode_config = MODE_CONFIGS.get(current_mode)
                if mode_config and hasattr(mode_config, "exploration_multiplier"):
                    unified_tracker.set_mode_exploration_multiplier(
                        mode_config.exploration_multiplier
                    )
                    logger.info(
                        f"Applied mode exploration multiplier: mode={current_mode.value}, "
                        f"multiplier={mode_config.exploration_multiplier}"
                    )
        except Exception as e:
            logger.debug(f"Could not apply mode exploration multiplier: {e}")

        logger.debug(
            f"UnifiedTaskTracker created with exploration_multiplier="
            f"{tool_calling_caps.exploration_multiplier}, "
            f"continuation_patience={tool_calling_caps.continuation_patience}"
        )

        return unified_tracker

    def create_integration_config(self) -> "IntegrationConfig":
        """Create intelligent pipeline integration configuration.

        Returns:
            IntegrationConfig instance with intelligent pipeline settings
        """
        from victor.agent.orchestrator_integration import IntegrationConfig
        from victor.config.pipeline_settings import resolve_runtime_intelligence_enabled

        config = IntegrationConfig(
            enable_resilient_calls=resolve_runtime_intelligence_enabled(self.settings),
            enable_quality_scoring=getattr(self.settings, "intelligent_quality_scoring", True),
            enable_mode_learning=getattr(self.settings, "intelligent_mode_learning", True),
            enable_prompt_optimization=getattr(
                self.settings, "intelligent_prompt_optimization", True
            ),
            min_quality_threshold=getattr(self.settings, "intelligent_min_quality_threshold", 0.5),
            grounding_confidence_threshold=getattr(
                self.settings, "intelligent_grounding_threshold", 0.7
            ),
        )

        logger.debug(
            f"IntegrationConfig created with resilient_calls={config.enable_resilient_calls}, "
            f"quality_scoring={config.enable_quality_scoring}"
        )
        return config

    def create_checkpoint_manager(self) -> Optional["ConversationCheckpointManager"]:
        """Create ConversationCheckpointManager for time-travel debugging.

        Returns:
            ConversationCheckpointManager instance or None if disabled
        """
        if not getattr(self.settings, "checkpoint_enabled", True):
            logger.debug("Checkpoint system disabled via settings")
            return None

        try:
            from victor.storage.checkpoints import (
                ConversationCheckpointManager,
                SQLiteCheckpointBackend,
            )
            from victor.config.settings import get_project_paths

            paths = get_project_paths()

            backend = SQLiteCheckpointBackend(
                storage_path=paths.project_victor_dir,
                db_name="checkpoints.db",
            )

            auto_interval = getattr(self.settings, "checkpoint_auto_interval", 5)
            max_per_session = getattr(self.settings, "checkpoint_max_per_session", 50)

            manager = ConversationCheckpointManager(
                backend=backend,
                auto_checkpoint_interval=auto_interval,
                max_checkpoints_per_session=max_per_session,
            )

            logger.info(
                f"ConversationCheckpointManager created (auto_interval={auto_interval}, "
                f"max_per_session={max_per_session})"
            )
            return manager

        except Exception as e:
            logger.warning(f"Failed to create ConversationCheckpointManager: {e}")
            return None

    def create_hierarchical_compaction_manager(
        self, summarizer: Optional[Any] = None
    ) -> "HierarchicalCompactionManager":
        """Create HierarchicalCompactionManager for epoch-level compaction.

        Args:
            summarizer: Optional compaction summarizer (kept for backward compat)

        Returns:
            HierarchicalCompactionManager instance
        """
        from victor.agent.compaction_hierarchy import HierarchicalCompactionManager

        return HierarchicalCompactionManager(summarizer=summarizer)

    def create_session_ledger(self) -> "SessionLedger":
        """Create SessionLedger for structured session state tracking."""
        from victor.agent.session_ledger import SessionLedger

        logger.debug("SessionLedger created")
        return SessionLedger()

    def create_compaction_summarizer(
        self, ledger: Optional["SessionLedger"] = None, use_llm: bool = False
    ) -> "LedgerAwareCompactionSummarizer":
        """Create compaction summarizer strategy.

        Args:
            ledger: Optional session ledger
            use_llm: If True, wraps LedgerAwareCompactionSummarizer as fallback
                inside LLMCompactionSummarizer for richer abstractive summaries.
        """
        from victor.agent.compaction_summarizer import LedgerAwareCompactionSummarizer

        fallback = LedgerAwareCompactionSummarizer()

        if use_llm:
            try:
                from victor.agent.llm_compaction_summarizer import (
                    LLMCompactionSummarizer,
                )

                provider = getattr(self, "_provider", None)
                if provider is None:
                    provider = getattr(self, "provider", None)
                if provider is not None:
                    summarizer = LLMCompactionSummarizer(provider=provider, fallback=fallback)
                    logger.debug("LLMCompactionSummarizer created with LedgerAware fallback")
                    return summarizer
            except Exception as e:
                logger.debug(f"LLM summarizer unavailable, using LedgerAware: {e}")

        logger.debug("LedgerAwareCompactionSummarizer created")
        return fallback

    def create_budget_manager(self) -> "BudgetManager":
        """Create BudgetManager for unified budget tracking.

        Returns:
            BudgetManager instance with settings-derived configuration
        """
        from victor.agent.budget_manager import create_budget_manager
        from victor.agent.protocols import BudgetConfig

        base_tool_calls = getattr(self.settings, "tool_budget", 30)
        base_iterations = getattr(self.settings, "max_iterations", 50)
        base_exploration = getattr(self.settings, "max_exploration_iterations", 8)
        base_action = getattr(self.settings, "max_action_iterations", 12)

        config = BudgetConfig(
            base_tool_calls=base_tool_calls,
            base_iterations=base_iterations,
            base_exploration=base_exploration,
            base_action=base_action,
        )

        manager = create_budget_manager(config=config)

        mc = self.mode_controller
        if mc is not None and hasattr(mc.config, "exploration_multiplier"):
            manager.set_mode_multiplier(mc.config.exploration_multiplier)
            logger.debug(
                f"BudgetManager created with mode multiplier: "
                f"{mc.config.exploration_multiplier}"
            )

        return manager

    def initialize_execution_state(self) -> tuple[list, list, set, bool]:
        """Initialize execution state containers.

        Returns:
            Tuple of (observed_files, executed_tools, failed_tool_signatures,
                     tool_capability_warned)
        """
        from typing import List

        observed_files: List[str] = []
        executed_tools: List[str] = []
        failed_tool_signatures: set[tuple[str, str]] = set()
        tool_capability_warned = False

        logger.debug("Execution state containers initialized")
        return (
            observed_files,
            executed_tools,
            failed_tool_signatures,
            tool_capability_warned,
        )

    def create_presentation_adapter(self) -> "PresentationProtocol":
        """Create presentation adapter for icon/emoji rendering.

        Returns:
            PresentationProtocol implementation
        """
        from victor.agent.presentation import PresentationProtocol

        return self.container.get(PresentationProtocol)
