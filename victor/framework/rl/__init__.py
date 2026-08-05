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

"""Reinforcement Learning - Framework-level adaptive learning infrastructure.

This module provides Victor's RL system for learning optimal parameters
from experience across all verticals.

Architecture:
┌──────────────────────────────────────────────────────────────────────┐
│                    HIGH-LEVEL API                                     │
│  RLManager, LearnerType, convenience functions                        │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ wraps
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    RLCoordinator (Singleton)                          │
│  ├─ Learner registry                                                  │
│  ├─ Unified SQLite storage (~/.victor/rl_data/rl.db)                 │
│  ├─ Telemetry collection                                              │
│  └─ Cross-vertical learning                                           │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ manages
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Specialized Learners                               │
│  ├─ ToolSelectorLearner        ├─ ContinuationPatienceLearner        │
│  ├─ ContinuationPromptLearner  ├─ SemanticThresholdLearner           │
│  ├─ ModelSelectorLearner       ├─ GroundingThresholdLearner          │
│  ├─ ModeTransitionLearner      ├─ PromptTemplateLearner              │
│  ├─ QualityWeightsLearner      └─ CacheEvictionLearner               │
└──────────────────────────────────────────────────────────────────────┘

Key Components:
- RLCoordinator: Central coordinator for all learners
- BaseLearner: Abstract base class for all RL learners
- RLManager: High-level API for using RL in applications
- LearnerType: Enum of available learner types

Usage:
    from victor.framework.rl import (
        RLManager,
        LearnerType,
        create_prompt_rollout_experiment,
        get_rl_coordinator,
        get_rl_coordinator_async,
    )

    # Option 1: Use high-level RLManager
    rl = RLManager()
    rl.record_success(
        learner=LearnerType.TOOL_SELECTOR,
        provider="anthropic",
        model="claude-3-opus",
        task_type="analysis",
    )

    rec = rl.get_recommendation(
        learner=LearnerType.CONTINUATION_PATIENCE,
        provider="deepseek",
        model="deepseek-chat",
    )

    # Option 2: Use coordinator directly
    coordinator = get_rl_coordinator()
    coordinator.record_outcome(
        learner_name="continuation_patience",
        outcome=RLOutcome(...),
        vertical="coding",
    )

    # Option 3: Start a prompt rollout for an approved candidate
    experiment_id = create_prompt_rollout_experiment(
        section_name="GROUNDING_RULES",
        provider="anthropic",
        treatment_hash="candidate_hash",
        traffic_split=0.1,
        min_samples_per_variant=50,
    )

    # Option 4: Async callers can access the singleton without blocking
    async_coordinator = await get_rl_coordinator_async()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Union,
)

from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.framework.rl.coordinator import (
    PromptCandidateSuiteWorkflowResult,
    RLCoordinator,
    get_rl_coordinator,
    get_rl_coordinator_async,
)
from victor.framework.rl.option_framework import OptionRegistry
from victor.framework.rl.config import (
    BaseRLConfig,
    DEFAULT_ACTIVE_LEARNERS,
    DEFAULT_PATIENCE_MAP,
    LearnerType,
)
from victor.framework.rl.credit_assignment import (
    ActionMetadata,
    BaseCreditAssigner,
    CreditAssignmentConfig,
    CreditAssignmentIntegration,
    CreditAssignmentProvider,
    CreditGranularity,
    CreditMethodology,
    CreditSignal,
    CriticalActionIdentifier,
    StateGraphCreditMixin,
    TrajectorySegment,
    compute_credit_metrics,
    visualize_credit_assignment,
)
from victor.framework.rl.credit_graph_integration import (
    CreditTracer,
    CreditAwareGraph,
    CompiledCreditAwareGraph,
    Transition,
    ExecutionTrace,
    create_credit_aware_workflow,
)
from victor.framework.rl.credit_visualization import (
    CreditVisualizationBuilder,
    ExportConfig,
    CreditAssignmentExporter,
    CreditAssignmentReport,
    export_credit_report,
    create_interactive_report,
)
from victor.core.constants import DEFAULT_VERTICAL

if TYPE_CHECKING:
    from victor.framework.agent import Agent
    from victor.core.protocols import OrchestratorProtocol as AgentOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class LearnerStats:
    """Statistics for a specific learner.

    Attributes:
        name: Learner name
        total_records: Total number of recorded outcomes
        success_rate: Ratio of successful outcomes
        last_updated: Unix timestamp of last update
        parameters: Current parameter values
    """

    name: str
    total_records: int = 0
    success_rate: float = 0.0
    last_updated: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RLStats:
    """Aggregate statistics for the RL system.

    Attributes:
        total_outcomes: Total outcomes recorded across all learners
        active_learners: Number of active learners
        learner_stats: Per-learner statistics
        database_path: Path to SQLite database
    """

    total_outcomes: int = 0
    active_learners: int = 0
    learner_stats: Dict[str, LearnerStats] = field(default_factory=dict)
    database_path: Optional[str] = None


class RLManager:
    """High-level manager for reinforcement learning operations.

    RLManager provides a user-friendly interface to Victor's RL system,
    wrapping the RLCoordinator with simplified methods.

    Attributes:
        coordinator: Underlying RLCoordinator instance

    Example:
        # Create from agent
        rl = RLManager.from_agent(agent)

        # Record outcome
        rl.record_success(
            learner=LearnerType.TOOL_SELECTOR,
            context={"tool": "code_search"},
        )

        # Get recommendation
        rec = rl.get_recommendation(
            learner=LearnerType.CONTINUATION_PATIENCE,
            provider="anthropic",
        )
    """

    def __init__(self, coordinator: Optional[RLCoordinator] = None):
        """Initialize RLManager.

        Args:
            coordinator: Optional RLCoordinator. If not provided,
                uses the global singleton.
        """
        self._coordinator = coordinator or get_rl_coordinator()

    @classmethod
    def from_agent(cls, agent: "Agent") -> "RLManager":
        """Create RLManager from an Agent instance.

        Args:
            agent: Agent instance

        Returns:
            RLManager configured for the agent
        """
        return cls()

    @classmethod
    def from_orchestrator(cls, orchestrator: "AgentOrchestrator") -> "RLManager":
        """Create RLManager from an AgentOrchestrator.

        Args:
            orchestrator: AgentOrchestrator instance

        Returns:
            RLManager configured for the orchestrator
        """
        return cls()

    @property
    def coordinator(self) -> RLCoordinator:
        """Get underlying RLCoordinator."""
        return self._coordinator

    # =========================================================================
    # Recording Outcomes
    # =========================================================================

    def record_outcome(
        self,
        learner: Union[LearnerType, str],
        outcome: RLOutcome,
        *,
        vertical: str = "general",
    ) -> None:
        """Record an outcome for a specific learner.

        Args:
            learner: Learner type or name
            outcome: RLOutcome with success/metrics data
            vertical: Vertical context (coding, devops, etc.)
        """
        learner_name = learner.value if isinstance(learner, LearnerType) else learner
        self._coordinator.record_outcome(
            learner_name=learner_name,
            outcome=outcome,
            vertical=vertical,
        )

    def record_success(
        self,
        learner: Union[LearnerType, str],
        *,
        provider: str = "unknown",
        model: str = "unknown",
        task_type: str = "general",
        quality_score: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        vertical: str = DEFAULT_VERTICAL,
    ) -> None:
        """Record a successful outcome.

        Convenience method for recording successful outcomes.

        Args:
            learner: Learner type or name
            provider: LLM provider name
            model: Model name
            task_type: Type of task
            quality_score: Quality score 0.0-1.0
            metadata: Additional metadata
            vertical: Vertical context
        """
        outcome = RLOutcome(
            provider=provider,
            model=model,
            task_type=task_type,
            success=True,
            quality_score=quality_score,
            metadata=metadata or {},
            vertical=vertical,
        )
        self.record_outcome(learner, outcome, vertical=vertical)

    def record_failure(
        self,
        learner: Union[LearnerType, str],
        *,
        provider: str = "unknown",
        model: str = "unknown",
        task_type: str = "general",
        quality_score: float = 0.0,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        vertical: str = DEFAULT_VERTICAL,
    ) -> None:
        """Record a failed outcome.

        Convenience method for recording failed outcomes.

        Args:
            learner: Learner type or name
            provider: LLM provider name
            model: Model name
            task_type: Type of task
            quality_score: Quality score 0.0-1.0
            error: Error message
            metadata: Additional metadata
            vertical: Vertical context
        """
        meta = metadata or {}
        if error:
            meta["error"] = error
        outcome = RLOutcome(
            provider=provider,
            model=model,
            task_type=task_type,
            success=False,
            quality_score=quality_score,
            metadata=meta,
            vertical=vertical,
        )
        self.record_outcome(learner, outcome, vertical=vertical)

    # =========================================================================
    # Getting Recommendations
    # =========================================================================

    def get_recommendation(
        self,
        learner: Union[LearnerType, str],
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        task_type: Optional[str] = None,
        vertical: str = "general",
        **context: Any,
    ) -> Optional[RLRecommendation]:
        """Get a recommendation from a specific learner.

        Args:
            learner: Learner type or name
            provider: LLM provider name
            model: Model name
            task_type: Type of task
            vertical: Vertical context
            **context: Additional context for recommendation

        Returns:
            RLRecommendation or None if no recommendation available
        """
        learner_name = learner.value if isinstance(learner, LearnerType) else learner
        return self._coordinator.get_recommendation(
            learner_name=learner_name,
            provider=provider,
            model=model,
            task_type=task_type,
            vertical=vertical,
            **context,
        )

    def get_tool_recommendation(
        self,
        task_type: str,
        *,
        available_tools: Optional[List[str]] = None,
        vertical: str = DEFAULT_VERTICAL,
    ) -> Optional[List[str]]:
        """Get recommended tools for a task type.

        Convenience method for tool selection recommendations.

        Args:
            task_type: Type of task (analysis, implementation, etc.)
            available_tools: List of available tool names
            vertical: Vertical context

        Returns:
            List of recommended tool names or None
        """
        rec = self.get_recommendation(
            LearnerType.TOOL_SELECTOR,
            task_type=task_type,
            vertical=vertical,
            available_tools=available_tools,
        )
        if rec and isinstance(rec.value, list):
            return rec.value
        return None

    def get_patience_recommendation(
        self,
        provider: str,
        model: str,
        *,
        task_type: str = "general",
    ) -> Optional[int]:
        """Get recommended continuation patience.

        Args:
            provider: LLM provider name
            model: Model name
            task_type: Type of task

        Returns:
            Recommended patience value or None
        """
        rec = self.get_recommendation(
            LearnerType.CONTINUATION_PATIENCE,
            provider=provider,
            model=model,
            task_type=task_type,
        )
        if rec and isinstance(rec.value, (int, float)):
            return int(rec.value)
        return None

    def create_prompt_rollout_experiment(
        self,
        *,
        section_name: str,
        provider: str,
        treatment_hash: str,
        control_hash: Optional[str] = None,
        traffic_split: float = 0.1,
        min_samples_per_variant: int = 100,
    ) -> Optional[str]:
        """Create and start a prompt rollout experiment via the coordinator."""
        return self._coordinator.create_prompt_rollout_experiment(
            section_name=section_name,
            provider=provider,
            treatment_hash=treatment_hash,
            control_hash=control_hash,
            traffic_split=traffic_split,
            min_samples_per_variant=min_samples_per_variant,
        )

    async def create_prompt_rollout_experiment_async(
        self,
        *,
        section_name: str,
        provider: str,
        treatment_hash: str,
        control_hash: Optional[str] = None,
        traffic_split: float = 0.1,
        min_samples_per_variant: int = 100,
    ) -> Optional[str]:
        """Async version of create_prompt_rollout_experiment."""
        return await self._coordinator.create_prompt_rollout_experiment_async(
            section_name=section_name,
            provider=provider,
            treatment_hash=treatment_hash,
            control_hash=control_hash,
            traffic_split=traffic_split,
            min_samples_per_variant=min_samples_per_variant,
        )

    def analyze_prompt_rollout_experiment(
        self,
        *,
        section_name: str,
        provider: str,
        treatment_hash: str,
    ) -> Optional[Dict[str, Any]]:
        """Analyze the rollout experiment for a prompt candidate."""
        return self._coordinator.analyze_prompt_rollout_experiment(
            section_name=section_name,
            provider=provider,
            treatment_hash=treatment_hash,
        )

    async def analyze_prompt_rollout_experiment_async(
        self,
        *,
        section_name: str,
        provider: str,
        treatment_hash: str,
    ) -> Optional[Dict[str, Any]]:
        """Async version of analyze_prompt_rollout_experiment."""
        return await self._coordinator.analyze_prompt_rollout_experiment_async(
            section_name=section_name,
            provider=provider,
            treatment_hash=treatment_hash,
        )

    def apply_prompt_rollout_recommendation(
        self,
        *,
        section_name: str,
        provider: str,
        treatment_hash: str,
        dry_run: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Apply the recommended rollout/rollback action for a prompt candidate."""
        return self._coordinator.apply_prompt_rollout_recommendation(
            section_name=section_name,
            provider=provider,
            treatment_hash=treatment_hash,
            dry_run=dry_run,
        )

    async def apply_prompt_rollout_recommendation_async(
        self,
        *,
        section_name: str,
        provider: str,
        treatment_hash: str,
        dry_run: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Async version of apply_prompt_rollout_recommendation."""
        return await self._coordinator.apply_prompt_rollout_recommendation_async(
            section_name=section_name,
            provider=provider,
            treatment_hash=treatment_hash,
            dry_run=dry_run,
        )

    def process_prompt_candidate_evaluation_suite(
        self,
        suite: Any,
        *,
        min_pass_rate: float = 0.5,
        promote_best: bool = False,
        create_rollout: bool = False,
        rollout_control_hash: Optional[str] = None,
        rollout_traffic_split: float = 0.1,
        rollout_min_samples_per_variant: int = 100,
        analyze_rollout: bool = False,
        apply_rollout_decision: bool = False,
        rollout_decision_dry_run: bool = False,
    ) -> PromptCandidateSuiteWorkflowResult:
        """Process a prompt-candidate suite through sync and rollout stages."""
        return self._coordinator.process_prompt_candidate_evaluation_suite(
            suite,
            min_pass_rate=min_pass_rate,
            promote_best=promote_best,
            create_rollout=create_rollout,
            rollout_control_hash=rollout_control_hash,
            rollout_traffic_split=rollout_traffic_split,
            rollout_min_samples_per_variant=rollout_min_samples_per_variant,
            analyze_rollout=analyze_rollout,
            apply_rollout_decision=apply_rollout_decision,
            rollout_decision_dry_run=rollout_decision_dry_run,
        )

    async def process_prompt_candidate_evaluation_suite_async(
        self,
        suite: Any,
        *,
        min_pass_rate: float = 0.5,
        promote_best: bool = False,
        create_rollout: bool = False,
        rollout_control_hash: Optional[str] = None,
        rollout_traffic_split: float = 0.1,
        rollout_min_samples_per_variant: int = 100,
        analyze_rollout: bool = False,
        apply_rollout_decision: bool = False,
        rollout_decision_dry_run: bool = False,
    ) -> PromptCandidateSuiteWorkflowResult:
        """Async version of process_prompt_candidate_evaluation_suite."""
        return await self._coordinator.process_prompt_candidate_evaluation_suite_async(
            suite,
            min_pass_rate=min_pass_rate,
            promote_best=promote_best,
            create_rollout=create_rollout,
            rollout_control_hash=rollout_control_hash,
            rollout_traffic_split=rollout_traffic_split,
            rollout_min_samples_per_variant=rollout_min_samples_per_variant,
            analyze_rollout=analyze_rollout,
            apply_rollout_decision=apply_rollout_decision,
            rollout_decision_dry_run=rollout_decision_dry_run,
        )

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> RLStats:
        """Get aggregate RL statistics.

        Returns:
            RLStats with system-wide statistics
        """
        stats = RLStats(
            database_path=(
                str(self._coordinator.db_path) if hasattr(self._coordinator, "db_path") else None
            ),
        )

        # Get learner stats if available
        if hasattr(self._coordinator, "get_all_stats"):
            learner_data = self._coordinator.get_all_stats()
            for name, data in learner_data.items():
                stats.learner_stats[name] = LearnerStats(
                    name=name,
                    total_records=data.get("total_records", 0),
                    success_rate=data.get("success_rate", 0.0),
                    last_updated=data.get("last_updated"),
                    parameters=data.get("parameters", {}),
                )
                stats.total_outcomes += data.get("total_records", 0)
            stats.active_learners = len(stats.learner_stats)

        return stats

    def get_learner_stats(self, learner: Union[LearnerType, str]) -> Optional[LearnerStats]:
        """Get statistics for a specific learner.

        Args:
            learner: Learner type or name

        Returns:
            LearnerStats or None if learner not found
        """
        learner_name = learner.value if isinstance(learner, LearnerType) else learner
        stats = self.get_stats()
        return stats.learner_stats.get(learner_name)

    # =========================================================================
    # Learner Management
    # =========================================================================

    def list_learners(self) -> List[str]:
        """List all registered learners.

        Returns:
            List of learner names
        """
        if hasattr(self._coordinator, "list_learners"):
            return self._coordinator.list_learners()
        return [lt.value for lt in LearnerType]

    def reset_learner(
        self,
        learner: Union[LearnerType, str],
        *,
        vertical: Optional[str] = None,
    ) -> None:
        """Reset a learner's learned parameters.

        Args:
            learner: Learner type or name
            vertical: Optional vertical to reset (None = all)
        """
        learner_name = learner.value if isinstance(learner, LearnerType) else learner
        if hasattr(self._coordinator, "reset_learner"):
            self._coordinator.reset_learner(learner_name, vertical=vertical)

    def __repr__(self) -> str:
        return f"RLManager(learners={len(self.list_learners())})"


# =============================================================================
# Convenience Functions
# =============================================================================


def create_outcome(
    success: bool,
    *,
    provider: str = "unknown",
    model: str = "unknown",
    task_type: str = "general",
    quality_score: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    vertical: str = DEFAULT_VERTICAL,
) -> RLOutcome:
    """Create an RLOutcome.

    Convenience function for creating outcomes.

    Args:
        success: Whether the action was successful
        provider: LLM provider name
        model: Model name
        task_type: Type of task
        quality_score: Quality score 0.0-1.0 (default 1.0 for success, 0.0 for failure)
        metadata: Additional metadata
        vertical: Vertical context

    Returns:
        RLOutcome instance
    """
    if quality_score is None:
        quality_score = 1.0 if success else 0.0

    return RLOutcome(
        provider=provider,
        model=model,
        task_type=task_type,
        success=success,
        quality_score=quality_score,
        metadata=metadata or {},
        vertical=vertical,
    )


def record_tool_success(
    tool_name: str,
    task_type: str,
    *,
    provider: str = "unknown",
    model: str = "unknown",
    duration_ms: Optional[float] = None,
    vertical: str = DEFAULT_VERTICAL,
) -> None:
    """Record a successful tool execution.

    Convenience function for recording tool successes.

    Args:
        tool_name: Name of the tool
        task_type: Type of task
        provider: LLM provider name
        model: Model name
        duration_ms: Execution duration in milliseconds
        vertical: Vertical context
    """
    coordinator = get_rl_coordinator()
    metadata: Dict[str, Any] = {"tool": tool_name}
    if duration_ms is not None:
        metadata["duration_ms"] = duration_ms

    coordinator.record_outcome(
        learner_name=LearnerType.TOOL_SELECTOR.value,
        outcome=RLOutcome(
            provider=provider,
            model=model,
            task_type=task_type,
            success=True,
            quality_score=1.0,
            metadata=metadata,
            vertical=vertical,
        ),
        vertical=vertical,
    )


def create_prompt_rollout_experiment(
    *,
    section_name: str,
    provider: str,
    treatment_hash: str,
    control_hash: Optional[str] = None,
    traffic_split: float = 0.1,
    min_samples_per_variant: int = 50,
) -> Optional[str]:
    """Create a prompt rollout experiment via the global RL coordinator."""
    coordinator = get_rl_coordinator()
    return coordinator.create_prompt_rollout_experiment(
        section_name=section_name,
        provider=provider,
        treatment_hash=treatment_hash,
        control_hash=control_hash,
        traffic_split=traffic_split,
        min_samples_per_variant=min_samples_per_variant,
    )


async def create_prompt_rollout_experiment_async(
    *,
    section_name: str,
    provider: str,
    treatment_hash: str,
    control_hash: Optional[str] = None,
    traffic_split: float = 0.1,
    min_samples_per_variant: int = 50,
) -> Optional[str]:
    """Async prompt rollout experiment helper via the global RL coordinator."""
    coordinator = await get_rl_coordinator_async()
    return await coordinator.create_prompt_rollout_experiment_async(
        section_name=section_name,
        provider=provider,
        treatment_hash=treatment_hash,
        control_hash=control_hash,
        traffic_split=traffic_split,
        min_samples_per_variant=min_samples_per_variant,
    )


def analyze_prompt_rollout_experiment(
    *,
    section_name: str,
    provider: str,
    treatment_hash: str,
) -> Optional[Dict[str, Any]]:
    """Analyze a prompt rollout experiment via the global RL coordinator."""
    coordinator = get_rl_coordinator()
    return coordinator.analyze_prompt_rollout_experiment(
        section_name=section_name,
        provider=provider,
        treatment_hash=treatment_hash,
    )


async def analyze_prompt_rollout_experiment_async(
    *,
    section_name: str,
    provider: str,
    treatment_hash: str,
) -> Optional[Dict[str, Any]]:
    """Async prompt rollout analysis helper via the global RL coordinator."""
    coordinator = await get_rl_coordinator_async()
    return await coordinator.analyze_prompt_rollout_experiment_async(
        section_name=section_name,
        provider=provider,
        treatment_hash=treatment_hash,
    )


def apply_prompt_rollout_recommendation(
    *,
    section_name: str,
    provider: str,
    treatment_hash: str,
    dry_run: bool = False,
) -> Optional[Dict[str, Any]]:
    """Apply the recommended prompt rollout decision via the global RL coordinator."""
    coordinator = get_rl_coordinator()
    return coordinator.apply_prompt_rollout_recommendation(
        section_name=section_name,
        provider=provider,
        treatment_hash=treatment_hash,
        dry_run=dry_run,
    )


async def apply_prompt_rollout_recommendation_async(
    *,
    section_name: str,
    provider: str,
    treatment_hash: str,
    dry_run: bool = False,
) -> Optional[Dict[str, Any]]:
    """Async prompt rollout decision helper via the global RL coordinator."""
    coordinator = await get_rl_coordinator_async()
    return await coordinator.apply_prompt_rollout_recommendation_async(
        section_name=section_name,
        provider=provider,
        treatment_hash=treatment_hash,
        dry_run=dry_run,
    )


def process_prompt_candidate_evaluation_suite(
    suite: Any,
    *,
    min_pass_rate: float = 0.5,
    promote_best: bool = False,
    create_rollout: bool = False,
    rollout_control_hash: Optional[str] = None,
    rollout_traffic_split: float = 0.1,
    rollout_min_samples_per_variant: int = 100,
    analyze_rollout: bool = False,
    apply_rollout_decision: bool = False,
    rollout_decision_dry_run: bool = False,
) -> PromptCandidateSuiteWorkflowResult:
    """Process a prompt-candidate suite via the global RL coordinator."""
    coordinator = get_rl_coordinator()
    return coordinator.process_prompt_candidate_evaluation_suite(
        suite,
        min_pass_rate=min_pass_rate,
        promote_best=promote_best,
        create_rollout=create_rollout,
        rollout_control_hash=rollout_control_hash,
        rollout_traffic_split=rollout_traffic_split,
        rollout_min_samples_per_variant=rollout_min_samples_per_variant,
        analyze_rollout=analyze_rollout,
        apply_rollout_decision=apply_rollout_decision,
        rollout_decision_dry_run=rollout_decision_dry_run,
    )


async def process_prompt_candidate_evaluation_suite_async(
    suite: Any,
    *,
    min_pass_rate: float = 0.5,
    promote_best: bool = False,
    create_rollout: bool = False,
    rollout_control_hash: Optional[str] = None,
    rollout_traffic_split: float = 0.1,
    rollout_min_samples_per_variant: int = 100,
    analyze_rollout: bool = False,
    apply_rollout_decision: bool = False,
    rollout_decision_dry_run: bool = False,
) -> PromptCandidateSuiteWorkflowResult:
    """Async prompt-candidate suite processing helper via the global coordinator."""
    coordinator = await get_rl_coordinator_async()
    return await coordinator.process_prompt_candidate_evaluation_suite_async(
        suite,
        min_pass_rate=min_pass_rate,
        promote_best=promote_best,
        create_rollout=create_rollout,
        rollout_control_hash=rollout_control_hash,
        rollout_traffic_split=rollout_traffic_split,
        rollout_min_samples_per_variant=rollout_min_samples_per_variant,
        analyze_rollout=analyze_rollout,
        apply_rollout_decision=apply_rollout_decision,
        rollout_decision_dry_run=rollout_decision_dry_run,
    )


__all__ = [
    # Manager
    "RLManager",
    # Types
    "LearnerType",
    "LearnerStats",
    "RLStats",
    # Core infrastructure
    "RLOutcome",
    "RLRecommendation",
    "RLCoordinator",
    "BaseLearner",
    "get_rl_coordinator",
    "get_rl_coordinator_async",
    # Configuration
    "BaseRLConfig",
    "DEFAULT_PATIENCE_MAP",
    "DEFAULT_ACTIVE_LEARNERS",
    # Convenience functions
    "create_outcome",
    "record_tool_success",
    "create_prompt_rollout_experiment",
    "create_prompt_rollout_experiment_async",
    "analyze_prompt_rollout_experiment",
    "analyze_prompt_rollout_experiment_async",
    "apply_prompt_rollout_recommendation",
    "apply_prompt_rollout_recommendation_async",
    "process_prompt_candidate_evaluation_suite",
    "process_prompt_candidate_evaluation_suite_async",
    "PromptCandidateSuiteWorkflowResult",
    # Credit Assignment (arXiv:2604.09459)
    "CreditGranularity",
    "CreditMethodology",
    "ActionMetadata",
    "CreditSignal",
    "TrajectorySegment",
    "CreditAssignmentConfig",
    "CreditAssignmentProvider",
    "BaseCreditAssigner",
    "CreditAssignmentIntegration",
    "CriticalActionIdentifier",
    "StateGraphCreditMixin",
    "compute_credit_metrics",
    "visualize_credit_assignment",
    # Credit Assignment Integration
    "CreditTracer",
    "CreditAwareGraph",
    "CompiledCreditAwareGraph",
    "Transition",
    "ExecutionTrace",
    "create_credit_aware_workflow",
    # Option Framework (hierarchical RL)
    "OptionRegistry",
    # Visualization & Export
    "CreditVisualizationBuilder",
    "ExportConfig",
    "CreditAssignmentExporter",
    "CreditAssignmentReport",
    "export_credit_report",
    "create_interactive_report",
]
