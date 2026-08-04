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

"""Agentic Loop - High-level task lifecycle orchestration.

This module provides a PERCEIVE -> PLAN -> ACT -> EVALUATE -> DECIDE loop
that sits ABOVE the AgentOrchestrator/TurnExecutor layer:

    AgenticLoop (this module)     -- task lifecycle: what to do, is it done?
        └── AgentOrchestrator     -- single-turn execution: LLM call + tools

AgenticLoop owns:
  - Intent perception (ActionIntent + TaskAnalyzer)
  - Execution planning (PlanningCoordinator)
  - Quality evaluation (EvaluationNode, FulfillmentDetector)
  - Adaptive iteration (plateau detection, convergence, extension)
  - Progress tracking (SubSearch intermediate rewards)

AgentOrchestrator owns:
  - LLM provider calls and tool execution
  - Token/context management
  - Provider failover and retry

Design Principle: INTEGRATE existing components, don't build new ones.

Based on research from:
- arXiv:2604.07415 - SubSearch (intermediate rewards / progress tracking)
- arXiv:2604.01681 - Fast-Slow Architecture (adaptive iterations)
- arXiv:2604.00445 - Truth-Aligned Uncertainty (calibrated confidence)

Example:
    from victor.framework.agentic_loop import AgenticLoop

    loop = AgenticLoop(
        orchestrator=orchestrator,
        memory_coordinator=memory,
        max_iterations=5,
    )

    result = await loop.run(
        query="Fix the authentication bug",
        context={"project": "myapp"}
    )
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Union,
    TYPE_CHECKING,
)

from victor.agent.turn_policy import (
    FulfillmentCriteriaBuilder,
    NudgePolicy,
    SpinDetector,
    SpinState,
    TurnEvaluationController,
    TurnObservation,
)
from victor.agent.topology_contract import TopologyAction
from victor.agent.topology_grounder import GroundedTopologyPlan, TopologyGrounder
from victor.agent.topology_selector import TopologySelector
from victor.agent.topology_telemetry import (
    build_topology_telemetry_event,
    emit_topology_telemetry_event,
)
from victor.core.shared_types import TaskPhase
from victor.framework.evaluation_nodes import (
    EvaluationDecision,
    EvaluationResult,
    create_agentic_loop_graph,
)
from victor.framework.fulfillment import FulfillmentDetector, TaskType
from victor.framework.perception_integration import Perception, PerceptionIntegration
from victor.framework.runtime_evaluation_policy import RuntimeEvaluationPolicy
from victor.framework.capabilities.task_hints import TaskTypeHintCapabilityProvider
from victor.agent.paradigm_router import ParadigmRouter, get_paradigm_router
from victor.agent.services.runtime_intelligence import RuntimeIntelligenceService
from victor.framework.enhanced_completion_evaluation import EnhancedCompletionEvaluator
from victor.providers.base import StreamChunk

# StateGraph-backed agentic-loop executor.
# Imported lazily to avoid circular dependencies at module import time.
_AgenticLoopGraphExecutor = None

logger = logging.getLogger(__name__)

RUNTIME_CONTEXT_OVERRIDES_STATE_KEY = "runtime_context_overrides"
LEGACY_TOPOLOGY_OVERRIDES_STATE_KEY = "topology_overrides"

# =============================================================================
# Enums for Agentic Loop Decisions
# =============================================================================


class AdaptiveTerminationDecision(str, Enum):
    """Decision for adaptive termination in agentic loop.

    Inspired by Fast-Slow Architecture (arXiv:2604.01681):
    - "Fast" exit on detected plateau (no progress)
    - "Slow" extension when near completion
    """

    PLATEAU = "plateau"  # Exit early, no progress detected
    EXTEND = "extend"  # Grant more iterations, near completion


# Type alias for backward compatibility
_AdaptiveTerminationDecisionLiteral = Literal["plateau", "extend"]


@dataclass
class StreamingTurnOutcome:
    """Carries the ``TurnResult`` produced by one streaming ACT turn (FEP-0007 Addendum A).

    ``run_streaming``'s ACT is an async generator (it yields ``StreamChunk``s), so it can't
    return a value: the streaming ACT port writes the turn's ``TurnResult`` here for the shared
    EVALUATE phase to consume — the same primitive the buffered ACT (``_act``) produces.
    """

    turn_result: Any = None


class StreamingActProvider(Protocol):
    """Framework-side streaming ACT seam driven by ``AgenticLoop.run_streaming`` (FEP-0007
    Addendum A).

    The streaming counterpart of the buffered ACT (``turn_executor.execute_turn``): given this
    turn's perception/plan/state, run the provider + emit + tool sub-steps, yield ``StreamChunk``s
    for live delivery, and write the produced ``TurnResult`` to ``outcome``. Letting the loop
    depend on this narrow seam keeps PERCEIVE/PLAN/EVALUATE/DECIDE shared while the I/O shape
    differs only in ACT. The concrete implementation — an adapter over the service-layer
    ``StreamingChatExecutor.execute_turn_streaming`` — is wired at cutover; until then
    ``run_streaming`` is unwired (no default provider).
    """

    def stream_turn_act(
        self,
        *,
        query: str,
        state: Dict[str, Any],
        perception: Any,
        plan: Any,
        turn_index: int,
        outcome: StreamingTurnOutcome,
    ) -> AsyncIterator[StreamChunk]:
        """Run one streaming turn's ACT, yielding chunks and writing the ``TurnResult``."""
        ...


if TYPE_CHECKING:
    from victor.agent.services.turn_execution_runtime import (
        TurnExecutor,
        TurnResult,
    )
    from victor.storage.memory.unified import UnifiedMemoryCoordinator
    from victor.agent.services.planning_runtime import PlanningCoordinator
    from victor.framework.agent import Agent
    from victor.framework.verification import VerificationResult


@dataclass
class PlanningGateConfig:
    """Configuration for PlanningGate thresholds.

    Consolidates the two formerly-inconsistent query-length thresholds:
    - ``short_query_threshold`` (was hardcoded ``50`` at line 291)
    - ``learned_threshold_default`` (was hardcoded ``60`` as the ``or 60`` fallback)

    Both had the same semantic (short query → skip LLM planning) but different
    values, meaning the routing-hint override only applied to the 50-60 band.
    Now both read from this config, making the behavior explicit and testable.
    """

    enabled: bool = True
    short_query_threshold: int = 50
    learned_threshold_default: int = 60


class PlanningGate:
    """Fast-Slow Planning Gate for skipping LLM planning on simple tasks.

    Based on arXiv:2604.01681 - Fast-Slow Architecture for Agentic Systems.
    Uses rule-based heuristics to determine if LLM planning is necessary or if
    direct execution fast-path would suffice.

    This reduces LLM calls by 30%+ for simple, straightforward tasks.
    """

    # Patterns that can be handled without LLM planning (fast-path)
    FAST_PATTERNS = {
        "create_simple": True,  # Just write the file
        "action": True,  # Just execute the command
        "search": True,  # Just run the search
        "quick_question": True,  # Direct answer needed
    }

    def __init__(
        self,
        enabled: bool = True,
        config: "Optional[PlanningGateConfig]" = None,
    ):
        """Initialize the planning gate.

        Args:
            enabled: Whether the gate is enabled (for feature flagging)
            config: Optional PlanningGateConfig; defaults to PlanningGateConfig().
        """
        self._config = config if config is not None else PlanningGateConfig(enabled=enabled)
        self.enabled = self._config.enabled
        self._fast_path_count = 0
        self._total_decisions = 0
        self._forced_slow_path_count = 0
        self._learned_fast_path_count = 0

    def should_use_llm_planning(
        self,
        task_type: str,
        tool_budget: int,
        query_complexity: Optional[float] = None,
        query_length: int = 0,
        context: Optional[Dict[str, Any]] = None,
        skip_planning: bool = False,
        routing_hints: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Determine if LLM planning is needed or if rule-based fast-path suffices.

        Args:
            task_type: Detected task type (e.g., "create_simple", "edit")
            tool_budget: Number of tools allowed for this task
            query_complexity: Optional complexity score (0-1, lower is simpler)
            query_length: Length of query in characters
            context: Optional execution context
            skip_planning: Task type hint flag to skip planning
            routing_hints: Optional learned runtime hints that may override heuristics

        Returns:
            False if task can proceed without LLM planning (fast-path)
            True if LLM planning is recommended (slow-path)
        """
        self._total_decisions += 1

        if not self.enabled:
            return True  # Gate disabled, always use LLM planning

        query_lower = context.get("query", "").lower() if context else ""
        action_keywords = [
            "run ",
            "execute",
            "create ",
            "write ",
            "delete ",
            "list ",
            "show ",
            "get ",
            "find ",
            "search ",
        ]
        has_action_keyword = any(keyword in query_lower for keyword in action_keywords)

        if isinstance(routing_hints, dict) and routing_hints.get("planning_force_llm"):
            reason = str(routing_hints.get("planning_force_reason") or "runtime_feedback")
            logger.info(
                f"[PlanningGate] Slow-path override: task_type={task_type} "
                f"reason={reason} (forces LLM planning)"
            )
            self._forced_slow_path_count += 1
            return True

        # Fast Pattern 0: Task type hint explicitly requests skip planning
        if skip_planning:
            logger.info(
                f"[PlanningGate] Fast-path: task_type={task_type} "
                f"skip_planning=True (from TaskTypeHint, skips LLM planning)"
            )
            self._fast_path_count += 1
            return False  # Skip LLM planning

        if isinstance(routing_hints, dict) and routing_hints.get("planning_prefer_fast_path"):
            tuned_tool_budget = max(
                1,
                int(routing_hints.get("planning_fast_path_tool_budget_limit") or 4),
            )
            tuned_query_length = max(
                1,
                int(
                    routing_hints.get("planning_fast_path_query_length_limit")
                    or self._config.learned_threshold_default
                ),
            )
            tuned_complexity = float(
                routing_hints.get("planning_fast_path_complexity_threshold") or 0.3
            )
            reason = str(routing_hints.get("planning_prefer_reason") or "runtime_feedback")
            if task_type in self.FAST_PATTERNS and tool_budget <= tuned_tool_budget:
                logger.info(
                    f"[PlanningGate] Learned fast-path override: task_type={task_type} "
                    f"reason={reason}"
                )
                self._fast_path_count += 1
                self._learned_fast_path_count += 1
                return False
            if (
                query_complexity is not None
                and query_complexity <= tuned_complexity
                and tool_budget <= tuned_tool_budget
            ):
                logger.info(
                    f"[PlanningGate] Learned fast-path override: "
                    f"query_complexity={query_complexity:.2f} reason={reason}"
                )
                self._fast_path_count += 1
                self._learned_fast_path_count += 1
                return False
            if (
                query_length > 0
                and query_length <= tuned_query_length
                and has_action_keyword
                and tool_budget <= tuned_tool_budget
            ):
                logger.info(
                    f"[PlanningGate] Learned fast-path override: short action query "
                    f"(length={query_length}) reason={reason}"
                )
                self._fast_path_count += 1
                self._learned_fast_path_count += 1
                return False

        # Fast Pattern 1: Simple task types with low tool budget
        if task_type in self.FAST_PATTERNS and tool_budget <= 3:
            logger.info(
                f"[PlanningGate] Fast-path: task_type={task_type}, "
                f"tool_budget={tool_budget} (skips LLM planning)"
            )
            self._fast_path_count += 1
            return False  # Skip LLM planning

        # Fast Pattern 2: Low query complexity (if provided)
        if query_complexity is not None and query_complexity < 0.3:
            logger.info(
                f"[PlanningGate] Fast-path: query_complexity={query_complexity:.2f} "
                f"(skips LLM planning)"
            )
            self._fast_path_count += 1
            return False  # Skip LLM planning

        # Fast Pattern 3: Short, direct queries
        if query_length > 0 and query_length < self._config.short_query_threshold:
            if has_action_keyword:
                logger.info(
                    f"[PlanningGate] Fast-path: short action query "
                    f"(length={query_length}, skips LLM planning)"
                )
                self._fast_path_count += 1
                return False  # Skip LLM planning

        # Default: Use LLM planning (slow-path)
        logger.debug(
            f"[PlanningGate] Slow-path: task_type={task_type}, "
            f"tool_budget={tool_budget} (uses LLM planning)"
        )
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about gate decisions.

        Returns:
            Dict with fast_path_count, total_decisions, fast_path_percentage
        """
        fast_path_pct = (
            (self._fast_path_count / self._total_decisions * 100)
            if self._total_decisions > 0
            else 0.0
        )
        return {
            "fast_path_count": self._fast_path_count,
            "forced_slow_path_count": self._forced_slow_path_count,
            "learned_fast_path_count": self._learned_fast_path_count,
            "total_decisions": self._total_decisions,
            "fast_path_percentage": fast_path_pct,
        }


class LoopStage(Enum):
    """Stages in the agentic loop."""

    PERCEIVE = "perceive"
    PLAN = "plan"
    ACT = "act"
    EVALUATE = "evaluate"
    DECIDE = "decide"
    COMPLETE = "complete"


@dataclass
class LoopIteration:
    """Single iteration in the agentic loop.

    Attributes:
        iteration: Iteration number
        stage: Current stage
        perception: Perception result
        plan: Generated plan
        action_result: Result from action execution
        evaluation: Evaluation result
        fulfillment: Fulfillment check result
        timestamp: When iteration occurred
    """

    iteration: int
    stage: LoopStage
    perception: Optional[Perception] = None
    plan: Optional[Any] = None
    action_result: Optional[Any] = None
    evaluation: Optional[EvaluationResult] = None
    fulfillment: Optional[Any] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "iteration": self.iteration,
            "stage": self.stage.value,
            "perception": self.perception.to_dict() if self.perception else None,
            "evaluation": (
                {
                    "decision": str(self.evaluation.decision),
                    "score": self.evaluation.score,
                    "reason": self.evaluation.reason,
                }
                if self.evaluation
                else None
            ),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class LoopResult:
    """Final result from agentic loop execution.

    Attributes:
        success: Whether loop completed successfully
        iterations: All iterations
        final_state: Final state
        total_duration: Total execution time
        metadata: Additional metadata
    """

    success: bool
    iterations: List[LoopIteration]
    final_state: Dict[str, Any]
    total_duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "iterations_count": len(self.iterations),
            "total_duration": self.total_duration,
            "final_state": self.final_state,
            "metadata": self.metadata,
        }


@dataclass
class AgenticLoopConfig:
    """Typed configuration for AgenticLoop, replacing Dict[str, Any] magic-key access.

    Defaults reproduce the original .get(key, default) behavior exactly so that
    existing callers passing dicts or None continue to work via from_dict().
    """

    disable_enhanced_completion: bool = False
    enable_requirement_validation: bool = True
    enable_completion_scoring: bool = True
    enable_context_keywords: bool = True
    enable_calibrated_completion: Optional[bool] = None
    # Completion strategy (ADR-009): "enhanced" (default) | "rubric" | "hybrid" | "legacy".
    # A strategy setting, not a feature flag. Default "enhanced" → zero behavior change until the
    # rubric path is proven to match-or-beat on the A/B + judge-α gate.
    completion_strategy: str = "enhanced"
    # Effect-grounded completion gate (ADR-010 / EVR-4): COMPLETE requires a verifiable effect
    # (workspace delta, verified check, or grounded claim) or is downgraded to RETRY. Opt-in,
    # default off per the flag-graduation policy; a strict no-op when disabled.
    enable_effect_gate: bool = False
    effect_gate_max_downgrades: int = 2
    # Online per-turn auditor (EVR-6 / FEP-0008 Phase C): a continue/alarm check that downgrades a
    # degenerate COMPLETE to RETRY. Opt-in, default off; a strict no-op when disabled.
    enable_per_turn_auditor: bool = False
    per_turn_auditor_max_alarms: int = 2
    enable_planning_gate: bool = True
    enable_paradigm_router: bool = True
    enable_topology_routing: bool = True
    planning_gate: "PlanningGateConfig" = field(default_factory=lambda: PlanningGateConfig())
    # Loop spin guards (per-run). total_timeout: hard wall-clock deadline for a
    # single loop.run in seconds; None (default) = off/unchanged. On breach the
    # run terminates with a FAIL result instead of spinning. perception_backstop_factor:
    # terminate a within-run runaway (perception >> iterations).
    total_timeout: Optional[float] = None
    perception_backstop_factor: int = 8
    extra_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgenticLoopConfig":
        """Backward-compatible factory from a legacy Dict[str, Any] config.

        Known keys populate typed fields; unknown keys are preserved in
        extra_config so downstream consumers (e.g. RuntimeEvaluationPolicy)
        still receive them.
        """
        import dataclasses

        known = {f.name for f in dataclasses.fields(cls)}
        known_vals = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(**known_vals, extra_config=extra)


class AgenticLoop:
    """Complete agentic loop integrating existing Victor components.

    Components:
    - PerceptionIntegration: Uses ActionIntent + TaskAnalyzer
    - PlanningCoordinator: Uses existing SkillPlanner + ToolPlanner
    - StateGraph: Extended with EvaluationNode
    - FulfillmentDetector: Task completion detection

    Loop Flow:
    1. PERCEIVE: Understand user intent
    2. PLAN: Generate execution plan
    3. ACT: Execute plan with tools/agents
    4. EVALUATE: Check progress and quality
    5. DECIDE: Continue, retry, or complete
    6. Repeat until complete or max iterations

    Example:
        loop = AgenticLoop(
            orchestrator=orchestrator,
            memory_coordinator=memory,
        )

        result = await loop.run(
            query="Add user authentication",
            context={"project": "myapp"}
        )

        print(f"Success: {result.success}")
        print(f"Iterations: {len(result.iterations)}")
    """

    def __init__(
        self,
        orchestrator: Optional[Any] = None,  # PlanningContextProtocol
        turn_executor: Optional["TurnExecutor"] = None,
        memory_coordinator: Optional["UnifiedMemoryCoordinator"] = None,
        runtime_intelligence: Optional[Any] = None,
        max_iterations: int = 10,
        enable_fulfillment_check: bool = True,
        enable_adaptive_iterations: bool = True,
        plateau_window: int = 3,
        plateau_tolerance: float = 0.02,
        exploration_settings: Optional[Any] = None,
        config: "Union[AgenticLoopConfig, Dict[str, Any], None]" = None,
        streaming_act_port: Optional["StreamingActProvider"] = None,
        rubric_complete_fn: Optional[Callable[[str], Awaitable[str]]] = None,
        verifier: Optional[Any] = None,
        max_verify_retries: int = 0,
        per_turn_judge: Optional[Any] = None,
    ):
        """Initialize agentic loop.

        Args:
            orchestrator: Victor orchestrator (used for planning + fallback)
            turn_executor: TurnExecutor for single-turn execution.
                If provided, AgenticLoop calls execute_turn() directly
                (one LLM call per iteration). If None, falls back to
                orchestrator.chat() (which runs its own multi-turn loop).
            memory_coordinator: Optional unified memory for context retrieval
            runtime_intelligence: Optional canonical runtime-intelligence service
            max_iterations: Maximum loop iterations (can be adapted at runtime)
            enable_fulfillment_check: Whether to check task fulfillment
            enable_adaptive_iterations: Allow early exit on plateau / extension
                on near-completion (Fast-Slow Architecture, arXiv:2604.01681)
            plateau_window: Iterations to check for progress plateau
            plateau_tolerance: Min score delta to not be a plateau
            config: Additional configuration
        """
        self.orchestrator = orchestrator
        self.turn_executor = turn_executor
        self.memory_coordinator = memory_coordinator
        # FEP-0007 Addendum A: the streaming ACT seam run_streaming() drives. None until
        # wired at cutover, at which point run_streaming becomes the live UI loop body.
        self.streaming_act_port = streaming_act_port
        # FEP-0018: framework verification hook. When set, the loop verifies the
        # agent's work after COMPLETE and re-enters on failure (bounded retries).
        self._verifier = verifier
        self._max_verify_retries = max_verify_retries
        self._verify_retries = 0
        # FEP-0019 Phase 3: proactive LSP context injection. The flag is read off
        # the turn_executor (set by Agent.create from SessionConfig.lsp_perception)
        # so neither AgenticLoop construction site needs a new param.
        self._lsp_context_enabled = bool(getattr(turn_executor, "_lsp_context_enabled", False))
        self._last_lsp_signature: Optional[str] = None
        self.max_iterations = max_iterations
        self.enable_fulfillment_check = enable_fulfillment_check
        self.enable_adaptive_iterations = enable_adaptive_iterations
        self.plateau_window = plateau_window
        self.plateau_tolerance = plateau_tolerance
        # Normalize config to typed AgenticLoopConfig; accept dict or None for backward compat.
        if isinstance(config, AgenticLoopConfig):
            self.config: AgenticLoopConfig = config
        elif isinstance(config, dict):
            self.config = AgenticLoopConfig.from_dict(config)
        else:
            self.config = AgenticLoopConfig()
        _policy_config: Dict[str, Any] = {
            "disable_enhanced_completion": self.config.disable_enhanced_completion,
            "enable_requirement_validation": self.config.enable_requirement_validation,
            "enable_completion_scoring": self.config.enable_completion_scoring,
            "enable_context_keywords": self.config.enable_context_keywords,
            "enable_calibrated_completion": self.config.enable_calibrated_completion,
            "enable_planning_gate": self.config.enable_planning_gate,
            "enable_paradigm_router": self.config.enable_paradigm_router,
            "enable_topology_routing": self.config.enable_topology_routing,
            **self.config.extra_config,
        }
        default_evaluation_policy = RuntimeEvaluationPolicy.from_config(_policy_config)

        # Progress tracking (SubSearch arXiv:2604.07415 intermediate rewards)
        self._progress_scores: List[float] = []

        # Velocity-aware completion tracking (Wave 3 — backslide prevention)
        self._score_history: List[float] = []

        # Shared turn policy (consistent between batch and streaming paths)
        self.spin_detector = SpinDetector()
        self.nudge_policy = NudgePolicy()
        self.criteria_builder = FulfillmentCriteriaBuilder()
        # Shared per-turn content-repetition + nudge decision (also used by the streaming loop).
        # Plateau stays owned by this loop's _check_adaptive_termination (FAIL/extend), so the
        # controller's plateau-nudge is disabled here to preserve current behavior.
        self.turn_evaluation_controller = TurnEvaluationController.from_exploration_settings(
            exploration_settings,
            spin_detector=self.spin_detector,
            nudge_policy=self.nudge_policy,
            enable_plateau_nudge=False,
        )

        # Initialize canonical runtime intelligence boundary
        if runtime_intelligence is None:
            from victor.agent.services.runtime_intelligence import (
                RuntimeIntelligenceService,
            )

            import dataclasses

            _perception_config = {
                **dataclasses.asdict(self.config),
                **self.config.extra_config,
            }
            runtime_intelligence = RuntimeIntelligenceService(
                perception_integration=PerceptionIntegration(
                    memory_coordinator=memory_coordinator,
                    config=_perception_config,
                ),
                evaluation_policy=default_evaluation_policy,
            )
        self.runtime_intelligence = runtime_intelligence
        resolved_policy = getattr(self.runtime_intelligence, "evaluation_policy", None)
        if not isinstance(resolved_policy, RuntimeEvaluationPolicy):
            resolved_policy = default_evaluation_policy
        self._evaluation_policy = resolved_policy
        self.perception = self.runtime_intelligence.perception_integration

        self.fulfillment = FulfillmentDetector() if enable_fulfillment_check else None

        # Initialize enhanced completion evaluator
        # ENABLED BY DEFAULT - use disable_enhanced_completion setting to opt-out if needed
        self.enhanced_completion_evaluator = None
        if not self.config.disable_enhanced_completion:
            self.enhanced_completion_evaluator = EnhancedCompletionEvaluator(
                enable_requirement_validation=self.config.enable_requirement_validation,
                enable_completion_scoring=self.config.enable_completion_scoring,
                enable_context_keywords=self.config.enable_context_keywords,
                enable_calibrated_completion=self.config.enable_calibrated_completion,
                evaluation_policy=self._evaluation_policy,
            )
            logger.info(
                "[EnhancedCompletion] ENABLED by default - requirement-driven completion detection active (use disable_enhanced_completion=True to opt-out)"
            )
        else:
            logger.warning(
                "[EnhancedCompletion] DISABLED via config - using legacy completion detection"
            )

        # Rubric-based completion (ADR-009 / EVR-3) — opt-in via completion_strategy in
        # {"rubric","hybrid"}. Default "enhanced" leaves this None (zero behavior change).
        #
        # ADR-011 fallback contract (flag-graduation policy): rubric completion is trusted
        # ONLY with a calibrated LLM judge. The no-LLM HeuristicRubricJudge scored
        # Krippendorff α=−0.092 in calibration (benchmarks/judge_calibration/FINDINGS.md) —
        # worse than the enhanced baseline — so when no rubric_complete_fn is injected we do
        # NOT silently fall back to it; we leave the evaluator None, which the completion
        # cascade treats as "defer to enhanced". The caller must inject a judge backed by a
        # calibrated model (gemma4:31b α=0.865 on real trajectories, or llama3.3:70b α=1.000
        # — the run-11 / run-10 gate-passers).
        self.rubric_completion_evaluator = self._build_rubric_evaluator(
            self.config.completion_strategy, rubric_complete_fn
        )

        # Effect-grounded completion gate (ADR-010 / EVR-4) — opt-in via enable_effect_gate,
        # default off. Wraps the EVALUATE seam (record → _evaluate_core → apply) so every
        # completion strategy shares the same effect precondition; strict no-op when disabled.
        from victor.framework.effect_gate import EffectGate, EffectGateConfig

        self.effect_gate = EffectGate(
            EffectGateConfig(
                enabled=self.config.enable_effect_gate,
                max_downgrades=self.config.effect_gate_max_downgrades,
            ),
            workspace_resolver=self._resolve_workspace,
        )

        # Online per-turn auditor (EVR-6 / FEP-0008 Phase C) — opt-in via enable_per_turn_auditor,
        # default off. A second post-filter on the same EVALUATE seam that downgrades a degenerate
        # COMPLETE to RETRY; strict no-op when disabled.
        from victor.framework.per_turn_auditor import PerTurnAuditor, PerTurnAuditorConfig

        self.per_turn_auditor = PerTurnAuditor(
            PerTurnAuditorConfig(
                enabled=self.config.enable_per_turn_auditor,
                max_alarms=self.config.per_turn_auditor_max_alarms,
            ),
            judge=per_turn_judge,
        )

        # Initialize planning gate for fast-slow architecture
        self.planning_gate = PlanningGate(enabled=self.config.enable_planning_gate)

        # Initialize task hint provider for enhanced task type hints
        self._task_hint_provider = TaskTypeHintCapabilityProvider()

        # Initialize paradigm router for model selection (arXiv:2604.06753)
        self.paradigm_router = get_paradigm_router()
        self.paradigm_router.enabled = self.config.enable_paradigm_router
        self._topology_enabled = self.config.enable_topology_routing
        self._topology_selector = TopologySelector()
        self._topology_grounder = TopologyGrounder()

        # Try to extract turn_executor from orchestrator if not provided
        if self.turn_executor is None and hasattr(orchestrator, "turn_executor"):
            self.turn_executor = orchestrator.turn_executor

        mode = "single-turn" if self.turn_executor else "orchestrator-fallback"
        logger.info(
            f"AgenticLoop initialized "
            f"(mode={mode}, max_iterations={max_iterations}, "
            f"adaptive={enable_adaptive_iterations}, "
            f"fulfillment={enable_fulfillment_check})"
        )

    async def _analyze_turn(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Perception:
        """Analyze a turn through the canonical runtime-intelligence service."""
        snapshot = await self.runtime_intelligence.analyze_turn(
            query,
            context=context,
            conversation_history=conversation_history,
        )
        if snapshot.perception is not None:
            return snapshot.perception
        raise RuntimeError("Runtime intelligence did not produce a perception snapshot")

    async def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> LoopResult:
        """Run the complete agentic loop.

        Args:
            query: User's natural language query
            context: Additional execution context
            conversation_history: Prior turns for multi-turn context

        Returns:
            LoopResult with all iterations and final state
        """
        import time

        # Phase 1 Consolidation: Use StateGraph-based executor when enabled
        from victor.framework.agentic_loop_executor import use_stategraph_executor

        if use_stategraph_executor():
            return await self._run_with_stategraph(
                query=query,
                context=context,
                conversation_history=conversation_history,
            )

        # Legacy path: original while loop implementation
        start_time = time.time()
        iterations: List[LoopIteration] = []
        state: Dict[str, Any] = {"query": query, **(context or {})}
        self._progress_scores = []
        self._verify_retries = 0  # FEP-0018: reset per run
        self._last_lsp_signature = None  # FEP-0019 Phase 3: reset per run
        self.turn_evaluation_controller.reset()  # resets spin_detector + content-repetition + plateau
        self.criteria_builder.reset()
        # ADR-010: session-scoped effect ledger + downgrade budget (getattr: tests build
        # partial loops via __new__ without the gate; a missing gate means no gating).
        _effect_gate = getattr(self, "effect_gate", None)
        if _effect_gate is not None:
            _effect_gate.reset()
        _auditor = getattr(self, "per_turn_auditor", None)
        if _auditor is not None:
            _auditor.reset()
        effective_max = self.max_iterations

        # Semantic response cache check (arXiv:2508.07675)
        # Only on first query of a loop — multi-turn queries are session-specific
        _sem_cache = None
        from victor.core.feature_flags import FeatureFlag, is_feature_enabled

        if is_feature_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE):
            try:
                from victor.agent.semantic_response_cache import get_semantic_cache

                _sem_cache = get_semantic_cache()
                _cached = _sem_cache.get(query)
                if _cached:
                    logger.info("[SemanticResponseCache] HIT — skipping LLM call")
                    return LoopResult(
                        success=True,
                        iterations=[],
                        final_state={"query": query, "response": _cached["response"]},
                        total_duration=time.time() - start_time,
                        metadata={
                            "source": "semantic_cache",
                            "similarity": _cached.get("similarity", 1.0),
                        },
                    )
            except Exception as _ce:
                logger.debug(f"Semantic cache check skipped: {_ce}")
                _sem_cache = None

        # Spin telemetry (loop guards): per-run counters + timers, threaded to the
        # shared perception seam via a contextvar so both loop paths are covered.
        # Log-only here; the terminating deadline/backstop reuse these counters.
        from victor.framework.loop.guards import (
            LoopGuards,
            LoopSpinError,
            reset_current_guards,
            set_current_guards,
        )

        _loop_guards = LoopGuards()
        _loop_guards_token = set_current_guards(_loop_guards)
        try:
            for i in range(1, effective_max + 1):
                _loop_guards.note_iteration()
                # Spin guards (C2/C3): hard wall-clock deadline + within-run
                # perception backstop. Raises LoopSpinError → caught below → FAIL.
                # Default total_timeout=None = off (no behavior change).
                _loop_guards.enforce(
                    total_timeout=self.config.total_timeout,
                    effective_max=effective_max,
                    backstop_factor=self.config.perception_backstop_factor,
                )
                # FAST-SLOW PLANNING GATE (arXiv:2604.01681)
                # Check if LLM planning is needed or if rule-based fast-path suffices
                # This gate runs BEFORE PERCEIVE to potentially skip the planning stage
                use_llm_planning = True
                skip_reason = None

                if i == 1:  # Only check on first iteration
                    # Extract task information from perception or context
                    # We need to perceive first to get task_type, but we can shortcut
                    # For now, use heuristics from query and context
                    task_type = context.get("task_type", "unknown") if context else "unknown"
                    tool_budget = context.get("tool_budget", 10) if context else 10
                    planning_routing_hints: Dict[str, Any] = {}
                    structured_routing_policy = None

                    # Try to get query complexity from perception if available
                    # Otherwise use simple heuristics
                    query_complexity = None
                    if hasattr(self, "_last_perception"):
                        last_perc = self._last_perception
                        query_complexity = getattr(last_perc, "query_complexity", None)

                    # Extract execution parameters from TaskTypeHint if available
                    skip_planning = False
                    task_hint = None
                    if hasattr(self, "_task_hint_provider"):
                        task_hint = self._task_hint_provider.get_hint(task_type)
                        if task_hint:
                            skip_planning = getattr(task_hint, "skip_planning", False)
                            # Temperature is now resolved at the provider seam from state["task_type"]
                            # via the unified TemperatureResolver (ADR-013), which reads the same
                            # TaskTypeHint.temperature_override constant — no per-turn override here.

                    if hasattr(self.runtime_intelligence, "get_structured_routing_policy"):
                        structured_routing_policy = (
                            self.runtime_intelligence.get_structured_routing_policy(
                                query=query,
                                scope_context={
                                    **(context or {}),
                                    "task_type": task_type,
                                    "tool_budget": tool_budget,
                                },
                            )
                        )
                        if structured_routing_policy is not None and hasattr(
                            structured_routing_policy, "planning_context"
                        ):
                            resolved_planning_hints = structured_routing_policy.planning_context()
                            if isinstance(resolved_planning_hints, dict):
                                planning_routing_hints = resolved_planning_hints
                                state["_structured_routing_policy_obj"] = structured_routing_policy
                                if hasattr(structured_routing_policy, "to_dict"):
                                    serialized_policy = structured_routing_policy.to_dict()
                                    if isinstance(serialized_policy, dict):
                                        state["structured_routing_policy"] = serialized_policy
                    elif hasattr(self.runtime_intelligence, "get_planning_routing_context"):
                        planning_routing_hints = (
                            self.runtime_intelligence.get_planning_routing_context(
                                query=query,
                                scope_context={
                                    **(context or {}),
                                    "task_type": task_type,
                                    "tool_budget": tool_budget,
                                },
                            )
                        )
                    if planning_routing_hints:
                        state["planning_routing_hints"] = dict(planning_routing_hints)

                    # Check with planning gate
                    use_llm_planning = self.planning_gate.should_use_llm_planning(
                        task_type=task_type,
                        tool_budget=tool_budget,
                        query_complexity=query_complexity,
                        query_length=len(query),
                        context=({**context, "query": query} if context else {"query": query}),
                        skip_planning=skip_planning,  # Pass TaskTypeHint skip flag
                        routing_hints=planning_routing_hints,
                    )

                    # Get routing decision from paradigm router (arXiv:2604.06753)
                    routing_decision = self.paradigm_router.route(
                        task_type=task_type,
                        query=query,
                        history_length=(len(conversation_history) if conversation_history else 0),
                        query_complexity=query_complexity,
                        tool_budget=tool_budget,
                        context=context,
                    )

                    # Store routing decision in state for execution
                    state["routing_decision"] = routing_decision.to_dict()

                    # Update execution parameters based on routing decision
                    if routing_decision.skip_planning:
                        use_llm_planning = False  # Override planning gate
                        skip_reason = "paradigm_router"

                    planning_selection_policy = "default_llm_planning"
                    if planning_routing_hints.get("planning_force_llm"):
                        planning_selection_policy = "experiment_forced_slow_path"
                    elif not use_llm_planning and planning_routing_hints.get(
                        "planning_prefer_fast_path"
                    ):
                        planning_selection_policy = "heuristic_fast_path"
                    elif not use_llm_planning and skip_reason == "paradigm_router":
                        planning_selection_policy = "paradigm_router_fast_path"
                    elif not use_llm_planning and skip_planning:
                        planning_selection_policy = "task_hint_fast_path"
                    elif not use_llm_planning:
                        planning_selection_policy = "heuristic_fast_path"

                    state.setdefault("planning_events", []).append(
                        {
                            "selection_policy": planning_selection_policy,
                            "used_llm_planning": bool(use_llm_planning),
                            "task_type": task_type,
                            "skip_reason": skip_reason,
                            "forced_by_runtime_feedback": bool(
                                planning_routing_hints.get("planning_force_llm")
                            ),
                            "force_reason": planning_routing_hints.get("planning_force_reason"),
                            "preference_reason": planning_routing_hints.get(
                                "planning_prefer_reason"
                            ),
                            "preferred_by_runtime_feedback": bool(
                                planning_routing_hints.get("planning_prefer_fast_path")
                            ),
                            "constraint_tags": list(
                                planning_routing_hints.get("planning_constraint_tags") or []
                            ),
                            "experiment_support": planning_routing_hints.get(
                                "planning_experiment_support"
                            ),
                        }
                    )

                    logger.info(
                        f"[Iteration {i}/{effective_max}] ROUTING: "
                        f"paradigm={routing_decision.paradigm.value}, "
                        f"model_tier={routing_decision.model_tier.value}, "
                        f"max_tokens={routing_decision.max_tokens}, "
                        f"confidence={routing_decision.confidence:.2f}"
                    )

                    if not use_llm_planning:
                        skip_reason = skip_reason or "fast_path"
                        logger.info(
                            f"[Iteration {i}/{effective_max}] FAST-PATH DETECTED: "
                            f"Skipping LLM planning, proceeding directly to execution"
                        )

                iteration = LoopIteration(iteration=i, stage=LoopStage.PERCEIVE)

                # PERCEIVE
                logger.info(f"[Iteration {i}/{effective_max}] PERCEIVE")
                perception = await self._analyze_turn(query, context, conversation_history)
                iteration.perception = perception
                self._last_perception = perception  # Cache for next iteration check

                # Update task_type from perception for more accurate gating
                state["perception"] = perception.to_dict()
                state["iteration"] = i
                if hasattr(perception, "task_analysis") and perception.task_analysis:
                    state["task_type"] = getattr(perception.task_analysis, "task_type", "unknown")

                # FEP-0019 Phase 3: proactively inject live LSP symbols + errors for
                # the files being edited, so generation matches the codebase first try.
                await self._maybe_inject_lsp_context(state)

                # RESEARCH TASK PROGRESS REPORTING
                # Report progress every 10 iterations for research tasks
                if state.get("task_type") == "research" and i % 10 == 0:
                    progress = self._calculate_research_progress(i, effective_max)
                    phase = self._get_current_research_phase(state)
                    papers_found = self._count_papers_found(conversation_history)

                    logger.info(
                        f"📊 RESEARCH PROGRESS: {progress}% | "
                        f"Phase: {phase} | "
                        f"Papers found: {papers_found} | "
                        f"Iteration: {i}/{effective_max}"
                    )

                if i == 1:
                    topology_plan = await self._initialize_topology_plan(
                        query=query,
                        perception=perception,
                        state=state,
                        context=context,
                        conversation_history=conversation_history,
                    )
                    if topology_plan and topology_plan.iteration_budget is not None:
                        effective_max = min(
                            effective_max,
                            max(1, int(topology_plan.iteration_budget)),
                        )
                    if topology_plan is not None:
                        topology_preparation = await self._prepare_topology_runtime(
                            topology_plan=topology_plan,
                            query=query,
                            state=state,
                        )
                        if topology_preparation:
                            state["topology_preparation"] = topology_preparation
                    self._capture_provider_degradation_baseline(state)

                # Skip planning stage when fast-path routing chooses direct execution
                if not use_llm_planning:
                    logger.info(
                        f"[Iteration {i}/{effective_max}] SKIP PLAN "
                        f"(reason={skip_reason or 'fast_path'})"
                    )
                    plan = self._build_fast_path_plan(perception, state)
                    iteration.plan = plan
                    state["plan"] = plan
                else:
                    # PLAN (use existing PlanningCoordinator if available)
                    logger.info(f"[Iteration {i}/{effective_max}] PLAN")
                    plan = await self._plan(perception, state)
                    iteration.plan = plan
                    state["plan"] = plan
                state["perception"] = perception.to_dict()

                # DETECT PHASE (for phase-aware context management)
                current_phase = self._detect_phase(i, state, perception)
                state["task_phase"] = current_phase
                logger.debug(f"[Iteration {i}/{effective_max}] Phase: {current_phase.value}")

                # RESEARCH TASK STAGE ENFORCEMENT
                # For research tasks, enforce stage transitions based on research phase
                if state.get("task_type") == "research":
                    from victor.core.shared_types import ConversationStage

                    research_phase = self._get_current_research_phase(state)
                    research_stages = {
                        "discover": (
                            ConversationStage.INITIAL,
                            ConversationStage.PLANNING,
                        ),
                        "search": (ConversationStage.READING,),
                        "analyze": (ConversationStage.ANALYSIS,),
                        "synthesize": (
                            ConversationStage.EXECUTION,
                            ConversationStage.VERIFICATION,
                        ),
                    }

                    target_stages = research_stages.get(research_phase, ())

                    # Get current stage from perception
                    current_stage = None
                    if hasattr(perception, "action_intent") and hasattr(
                        perception.action_intent, "stage"
                    ):
                        current_stage = perception.action_intent.stage

                    # Enforce stage transition if needed
                    if target_stages and current_stage not in target_stages:
                        target_stage = target_stages[0]
                        logger.info(
                            f"[Iteration {i}/{effective_max}] RESEARCH PHASE TRANSITION: "
                            f"{research_phase} → {target_stage.value}"
                        )
                        # Update stage in state (will be picked up by next iteration's perception)
                        state["target_stage"] = target_stage
                        # Update perception's action intent stage for this iteration
                        if hasattr(perception, "action_intent"):
                            perception.action_intent.stage = target_stage

                plan = state.get("plan")
                if plan is None:
                    # PLAN (use existing PlanningCoordinator if available)
                    logger.info(f"[Iteration {i}/{effective_max}] PLAN")
                    plan = await self._plan(perception, state)
                    iteration.plan = plan
                    state["plan"] = plan

                # ACT
                logger.info(f"[Iteration {i}/{effective_max}] ACT")
                action_result = await self._act(plan, state)
                iteration.action_result = action_result
                state["action_result"] = action_result

                # FEP-0023 Phase 1: record the turn onto the live SessionLedger
                # (populates the assembler's <SESSION_STATE>). No-op unless
                # USE_SESSION_LEDGER is enabled.
                self._populate_session_ledger(action_result, i)

                # EVALUATE
                logger.info(f"[Iteration {i}/{effective_max}] EVALUATE")
                evaluation = await self._evaluate(perception, action_result, state)
                iteration.evaluation = evaluation
                iteration.stage = LoopStage.EVALUATE

                # Track progress (SubSearch intermediate rewards)
                self._progress_scores.append(evaluation.score)

                # Content-repetition detection via the shared TurnEvaluationController (the same
                # hash/overlap detector the streaming loop uses — replaces the cruder
                # content-length heuristic). The loop already fed the spin detector in _act, so
                # don't re-record. Plateau/nudge stay owned by this loop below.
                _action_content = self._extract_turn_content(action_result)
                _tool_results = getattr(action_result, "tool_results", None) or []
                _turn_obs = TurnObservation(
                    content=_action_content,
                    productive_count=int(getattr(action_result, "successful_tool_count", 0) or 0),
                    has_tool_calls=bool(getattr(action_result, "has_tool_calls", False)),
                    all_blocked=bool(getattr(action_result, "all_tools_blocked", False)),
                    tool_names={
                        (r.get("tool_name") or r.get("name") or "")
                        for r in _tool_results
                        if isinstance(r, dict)
                    },
                    tool_count=int(getattr(action_result, "tool_calls_count", 0) or 0),
                    tool_signatures=getattr(action_result, "tool_signatures", None),
                    tool_results=_tool_results,
                    iteration=i,
                    max_iterations=effective_max,
                )
                turn_decision = self.turn_evaluation_controller.evaluate(
                    _turn_obs, record_spin=False
                )
                if turn_decision.stop and turn_decision.terminal_success:
                    # Search-saturation (or other clean stop) -> COMPLETE so the loop synthesizes
                    # the gathered context instead of thrashing to the iteration cap.
                    logger.info(
                        "Converged at iteration %d (%s) - completing to synthesize",
                        i,
                        turn_decision.stop_reason,
                    )
                    evaluation = EvaluationResult(
                        decision=EvaluationDecision.COMPLETE,
                        score=evaluation.score,
                        reason=f"Converged: {turn_decision.stop_reason} (iteration {i})",
                    )
                elif turn_decision.stop:
                    state.setdefault("degradation_events", []).append(
                        {
                            "source": "agentic_loop",
                            "kind": turn_decision.stop_reason or "content_repetition",
                            "failure_type": "STUCK_LOOP",
                            "iteration": i,
                            "task_type": state.get("task_type"),
                            "post_degraded": True,
                            "recovered": False,
                            "adaptation_cost": float(i),
                            "degradation_reasons": [
                                turn_decision.stop_reason or "content_repetition"
                            ],
                        }
                    )
                    logger.warning(
                        "Content degradation detected (%s) at iteration %d - stopping loop",
                        turn_decision.stop_reason,
                        i,
                    )
                    evaluation = EvaluationResult(
                        decision=EvaluationDecision.FAIL,
                        score=evaluation.score,
                        reason=f"Content repetition detected (iteration {i})",
                    )
                    # Continue to exit logic below

                # FULFILLMENT CHECK (optional)
                if self.enable_fulfillment_check:
                    logger.info(f"[Iteration {i}/{effective_max}] FULFILLMENT")
                    fulfillment = await self._check_fulfillment(perception, state)
                    iteration.fulfillment = fulfillment
                    state["fulfillment"] = fulfillment

                # ADAPTIVE ITERATION CHECK (Fast-Slow Architecture)
                if self.enable_adaptive_iterations:
                    adaptive_decision = self._check_adaptive_termination(
                        i,
                        evaluation,
                        state=state,
                        action_result=action_result,
                    )
                    if adaptive_decision == "plateau":
                        logger.info("Progress plateaued - exiting loop early")
                        evaluation = EvaluationResult(
                            decision=EvaluationDecision.FAIL,
                            score=evaluation.score,
                            reason=f"Progress plateaued after {i} iterations",
                        )
                        iteration.evaluation = evaluation
                    elif adaptive_decision == "extend" and i >= effective_max:
                        # Near completion — grant extra iterations (up to 50% more)
                        extension = min(3, self.max_iterations // 2)
                        effective_max = min(
                            self.max_iterations + extension, effective_max + extension
                        )
                        logger.info(f"Near completion - extending to {effective_max} iterations")

                if (
                    state.get("_topology_input_obj") is not None
                    and state.get("_topology_decision_obj") is not None
                    and state.get("_topology_plan_obj") is not None
                    and i == state.get("topology_iteration", 1)
                    and not state.get("_topology_telemetry_emitted", False)
                ):
                    outcome = {
                        "iteration": i,
                        "evaluation_decision": evaluation.decision.value,
                        "evaluation_score": evaluation.score,
                        "successful_tool_count": getattr(action_result, "successful_tool_count", 0),
                        "failed_tool_count": getattr(action_result, "failed_tool_count", 0),
                        "has_tool_calls": bool(getattr(action_result, "has_tool_calls", False)),
                    }
                    topology_event = build_topology_telemetry_event(
                        state["_topology_input_obj"],
                        state["_topology_decision_obj"],
                        state["_topology_plan_obj"],
                        outcome=outcome,
                    )
                    state.setdefault("topology_events", []).append(topology_event.to_dict())
                    await emit_topology_telemetry_event(topology_event)
                    if hasattr(self.runtime_intelligence, "record_topology_outcome"):
                        try:
                            tool_call_count = getattr(action_result, "tool_calls_count", None)
                            if tool_call_count is None:
                                tool_call_count = getattr(
                                    action_result, "successful_tool_count", 0
                                ) + getattr(action_result, "failed_tool_count", 0)
                            self.runtime_intelligence.record_topology_outcome(
                                {
                                    "status": self._topology_feedback_status(evaluation.decision),
                                    "completion_score": evaluation.score,
                                    "tool_calls": tool_call_count,
                                    "turns": i,
                                    "topology_events": list(state.get("topology_events", [])),
                                }
                            )
                        except Exception as exc:
                            logger.debug("Failed to record topology runtime outcome: %s", exc)
                    state["_topology_telemetry_emitted"] = True

                # DECIDE
                logger.info(f"[Iteration {i}/{effective_max}] DECIDE: {evaluation.decision}")
                iteration.stage = LoopStage.DECIDE
                iterations.append(iteration)

                # Check termination conditions
                # Apply backslide guard before acting on COMPLETE: a significant score
                # drop from the previous turn means the model regressed and should not
                # be considered done yet.
                if evaluation.decision == EvaluationDecision.COMPLETE:
                    evaluation = self._apply_backslide_guard(evaluation)
                    # FEP-0018: framework verification hook — verify before accepting.
                    if await self._maybe_verify_and_retry(i, evaluation, state, streaming=False):
                        continue  # re-enter the loop (skip break)
                if evaluation.decision == EvaluationDecision.COMPLETE:
                    logger.info("Task complete - exiting loop")
                    break

                if evaluation.decision == EvaluationDecision.FAIL:
                    logger.warning("Task failed - exiting loop")
                    break

                # NUDGE INJECTION via shared NudgePolicy (consistent with streaming).
                # Inject on any non-terminal decision (RETRY or CONTINUE). Shared with
                # run_streaming() so buffered and streaming nudge behavior can't drift.
                self._inject_decide_nudges(evaluation, i, effective_max)

            # Determine success
            success = self._determine_success(iterations)

            # Attribute this turn's outcome to the served prompt candidates so
            # their Thompson posteriors update (closes the prompt-optimization
            # reward loop — FEP-0017). Shared with run_streaming() so buffered
            # and streaming chat train the learner identically; success is
            # derived from the completion SCORE, not the COMPLETE flag.
            self._emit_prompt_reward_outcome(
                iterations[-1].evaluation if iterations else None, state
            )

            self._record_provider_degradation_event(
                state,
                total_iterations=len(iterations),
            )

            # Store cacheable responses in semantic cache for future queries
            if success and _sem_cache is not None:
                try:
                    from victor.agent.semantic_response_cache import (
                        SemanticResponseCache,
                    )

                    _final_action = state.get("action_result")
                    _final_response: Optional[str] = None
                    if isinstance(_final_action, str):
                        _final_response = _final_action
                    elif _final_action is not None and hasattr(_final_action, "response"):
                        if not getattr(_final_action, "has_tool_calls", False):
                            _final_response = _final_action.response
                    if _final_response and SemanticResponseCache.is_cacheable(_final_response):
                        _sem_cache.set(query, _final_response)
                        logger.debug("[SemanticResponseCache] Stored new response")
                except Exception as _se:
                    logger.debug(f"Semantic cache store skipped: {_se}")

            duration = time.time() - start_time

            return LoopResult(
                success=success,
                iterations=iterations,
                final_state=state,
                total_duration=duration,
                metadata={
                    "iterations_completed": len(iterations),
                    "max_iterations_reached": len(iterations) >= self.max_iterations,
                    "effective_max_iterations": effective_max,
                    "progress_scores": list(self._progress_scores),
                    "planning_events": list(state.get("planning_events", [])),
                    "planning_routing_hints": dict(state.get("planning_routing_hints", {})),
                    "structured_routing_policy": dict(state.get("structured_routing_policy", {})),
                    "topology_events": list(state.get("topology_events", [])),
                    "degradation_events": list(state.get("degradation_events", [])),
                },
            )

        except LoopSpinError as spin:
            # A guard terminated the run (wall-clock deadline / perception backstop).
            # Return a FAIL result with the diagnostic reason instead of spinning.
            logger.warning("[AgenticLoop] terminated by spin guard: %s", spin)
            return LoopResult(
                success=False,
                iterations=iterations,
                final_state=state,
                total_duration=time.time() - start_time,
                metadata={
                    "error": str(spin),
                    "terminated_by": "loop_spin_guard",
                    "iterations_completed": len(iterations),
                    "perception_calls": _loop_guards.perception_calls,
                    "progress_scores": list(self._progress_scores),
                },
            )
        except Exception as e:
            logger.error(f"Agentic loop error: {e}", exc_info=True)
            duration = time.time() - start_time
            self._record_provider_degradation_event(
                state,
                total_iterations=len(iterations),
            )

            return LoopResult(
                success=False,
                iterations=iterations,
                final_state=state,
                total_duration=duration,
                metadata={
                    "error": str(e),
                    "progress_scores": list(self._progress_scores),
                    "planning_events": list(state.get("planning_events", [])),
                    "planning_routing_hints": dict(state.get("planning_routing_hints", {})),
                    "structured_routing_policy": dict(state.get("structured_routing_policy", {})),
                    "topology_events": list(state.get("topology_events", [])),
                    "degradation_events": list(state.get("degradation_events", [])),
                },
            )
        finally:
            reset_current_guards(_loop_guards_token)

    @staticmethod
    def _extract_turn_content(action_result: Any) -> str:
        """Return a turn's assistant text as a string for content-repetition evaluation.

        A tool-only turn legitimately has empty content (``""``). The previous
        ``getattr(action_result, "content", None) or getattr(action_result, "response", "")``
        idiom fell through to the ``CompletionResponse`` *object* on such turns, which then
        crashed the content-repetition detector on ``.strip()``. This coerces to ``str``,
        drilling into ``response.content`` only when ``action_result`` is neither a ``TurnResult``
        (string ``.content``) nor a raw string response.
        """
        raw = getattr(action_result, "content", None)
        if isinstance(raw, str):
            return raw
        if isinstance(action_result, str):
            return action_result
        resp = getattr(action_result, "response", "")
        content = resp if isinstance(resp, str) else getattr(resp, "content", "")
        return content if isinstance(content, str) else ""

    def _populate_session_ledger(self, action_result: Any, turn_index: int) -> None:
        """Record a completed turn onto the live SessionLedger (FEP-0023 Phase 1).

        Feeds ``orchestrator._session_ledger`` from the turn's tool results (files
        read/modified) and assistant text (decisions/recommendations), which is what
        the assembler's already-wired ``<SESSION_STATE>`` block renders. Called from
        both the buffered (:meth:`run`) and streaming (:meth:`run_streaming`) loops at
        the same post-ACT point so the two paths cannot drift.

        A no-op when the ``USE_SESSION_LEDGER`` gate is off, when the loop has no
        orchestrator/ledger (e.g. a bare ``__new__``-constructed loop in tests), or
        when extraction raises (population must never break the loop). Tool-result
        keys mirror the ``TurnResult.tool_results`` element shape produced by
        ``tool_execution_runtime`` (``name``/``args``/``result``/``full_result``).
        """
        from victor.core.feature_flags import FeatureFlag, is_feature_enabled

        if not is_feature_enabled(FeatureFlag.USE_SESSION_LEDGER):
            return
        # Double getattr: ``self.orchestrator`` may not even be set on a loop built
        # via ``AgenticLoop.__new__`` (streaming unit tests), so guard the attribute
        # access itself, not just its value.
        ledger = getattr(getattr(self, "orchestrator", None), "_session_ledger", None)
        if ledger is None:
            return
        try:
            for result_entry in getattr(action_result, "tool_results", None) or []:
                if not isinstance(result_entry, dict):
                    continue
                tool_name = str(result_entry.get("tool_name") or result_entry.get("name") or "")
                if not tool_name:
                    continue
                args = result_entry.get("args")
                if not isinstance(args, dict):
                    args = {}
                result_text = str(
                    result_entry.get("full_result")
                    or result_entry.get("result")
                    or result_entry.get("content")
                    or ""
                )
                ledger.update_from_tool_result(tool_name, args, result_text, turn_index)

            content = self._extract_turn_content(action_result)
            if content:
                ledger.update_from_assistant_response(content, turn_index)
        except Exception:  # pragma: no cover - defensive; population must not break the loop
            logger.debug("SessionLedger population failed", exc_info=True)

    def _inject_decide_nudges(
        self,
        evaluation: EvaluationResult,
        iteration: int,
        effective_max: int,
    ) -> None:
        """Inject shared NudgePolicy nudges on a non-terminal DECIDE (RETRY/CONTINUE/ESCALATE).

        Extracted from run() so the buffered and streaming loops share one nudge path and can't
        drift (FEP-0007 Addendum A). The nudge *decision* already comes from the shared
        ``self.nudge_policy`` / ``self.spin_detector``; this is the conversation-injection glue.
        No-op on terminal decisions or when no turn executor / chat context is available.
        """
        if evaluation.decision in (
            EvaluationDecision.COMPLETE,
            EvaluationDecision.FAIL,
        ):
            return
        if self.turn_executor is None:
            return
        chat_ctx = getattr(self.turn_executor, "_chat_context", None)
        if not (chat_ctx and hasattr(chat_ctx, "add_message")):
            return

        from victor.agent.conversation.history_metadata import (
            build_internal_history_metadata,
        )
        from victor.agent.conversation.types import MessageSource

        nudge = self.nudge_policy.evaluate(self.spin_detector)
        if nudge.should_inject:
            chat_ctx.add_message(
                nudge.role,
                nudge.message,
                metadata=build_internal_history_metadata("nudge", source=MessageSource.AGENT_NUDGE),
            )
            logger.info(f"Nudge injected: {nudge.nudge_type.value}")

        budget_nudge = self.nudge_policy.budget_warning(iteration, effective_max)
        if budget_nudge.should_inject:
            chat_ctx.add_message(
                budget_nudge.role,
                budget_nudge.message,
                metadata=build_internal_history_metadata(
                    "budget_nudge", source=MessageSource.AGENT_NUDGE
                ),
            )

        # ADR-010: the effect gate downgraded a COMPLETE to RETRY — tell the model why, so the
        # retry turn produces a verifiable effect (or grounds its claim) instead of restating.
        if evaluation.metadata.get("completion_without_effect"):
            expected = evaluation.metadata.get("expected_effect_class", "effect")
            chat_ctx.add_message(
                "user",
                (
                    "Your completion claim has no verifiable effect on record "
                    f"(expected: {expected}). Use tools to actually perform the work "
                    "(e.g. write/edit files, run the relevant checks) or ground your answer "
                    "in tool evidence before finishing."
                ),
                metadata=build_internal_history_metadata(
                    "effect_gate_nudge", source=MessageSource.AGENT_NUDGE
                ),
            )

    async def _run_verification(self, state: Dict[str, Any]) -> "VerificationResult":
        """Run the framework verification hook (FEP-0018).

        Called after the agent claims COMPLETE. If the verifier is set, it
        checks the agent's work (tests, lint, etc.) and returns a result. The
        caller (the DECIDE gate) injects the feedback and re-enters on failure.
        """
        from victor.framework.verification import VerificationResult

        try:
            workspace = state.get("workspace") or state.get("working_dir")
            return await self._verifier.verify(workspace=workspace, state=state)
        except Exception as exc:
            logger.warning("Verification raised: %s", exc)
            return VerificationResult(
                passed=-1,
                total=-1,
                raw_output=str(exc),
                feedback=f"Verification error: {exc}",
            )

    def _inject_verify_feedback(self, result: "VerificationResult") -> None:
        """Inject the verifier's feedback as a user message for the retry turn."""
        feedback = result.feedback or (
            f"Verification: {result.passed}/{result.total} checks passed.\n"
            f"{result.raw_output[:2000]}"
        )
        try:
            chat_ctx = getattr(self.turn_executor, "_chat_context", None)
            if chat_ctx and hasattr(chat_ctx, "add_message"):
                chat_ctx.add_message(
                    role="user",
                    content=feedback,
                    metadata={"source": "verification"},
                )
        except Exception:
            logger.debug("Failed to inject verify feedback", exc_info=True)

    async def _maybe_verify_and_retry(
        self, turn: int, evaluation: Any, state: Dict[str, Any], *, streaming: bool
    ) -> bool:
        """FEP-0018 verify gate, shared by ``run()`` and ``run_streaming()``.

        After the agent claims COMPLETE (post backslide-guard), run the verifier if
        one is set and retries remain. On failure, inject the feedback, bump the retry
        counter, and return ``True`` so the caller re-enters the loop; otherwise return
        ``False`` and let the caller accept COMPLETE. Extracted so the gate lives in one
        place instead of two byte-drifting copies (FEP-0021 / FP-3 residue).
        """
        if not (
            evaluation.decision == EvaluationDecision.COMPLETE
            and getattr(self, "_verifier", None) is not None
            and getattr(self, "_verify_retries", 0) < getattr(self, "_max_verify_retries", 0)
        ):
            return False
        vr = await self._run_verification(state)
        logger.info(
            "Turn %d verify%s: %d/%d (%s) retries=%d/%d",
            turn,
            " (streaming)" if streaming else "",
            vr.passed,
            vr.total,
            "VERIFIED" if vr.is_verified else "failed",
            self._verify_retries + 1,
            self._max_verify_retries,
        )
        if not vr.is_verified:
            self._inject_verify_feedback(vr)
            self._verify_retries += 1
            return True
        return False

    def _resolve_workspace(self, state: Dict[str, Any]) -> Optional[Path]:
        """Resolve the workspace root via a 3-level fallback (FEP-0019 Phase 3).

        The loop itself has no reliable workspace; check the state, then the
        orchestrator's project context, then the configured project paths.
        """
        ws = state.get("workspace") or state.get("working_dir")
        if ws:
            return Path(ws)
        pc = getattr(self.orchestrator, "project_context", None)
        root = getattr(pc, "root_path", None) if pc is not None else None
        if root:
            return root if isinstance(root, Path) else Path(root)
        try:
            from victor.config.settings import get_project_paths

            return get_project_paths().project_root
        except Exception:
            return None

    async def _maybe_inject_lsp_context(self, state: Dict[str, Any]) -> None:
        """Inject live LSP symbols + errors for the files being edited (FEP-0019 Phase 3).

        Position-free: pulls document symbols + diagnostics for the most-recently
        modified source files and adds them as a per-turn user message, so the
        agent's next generation matches the codebase. Throttled by a content
        signature so identical context is not re-injected. No-op unless
        ``SessionConfig.lsp_perception`` opted in and an LSP capability is set.
        """
        if not getattr(self, "_lsp_context_enabled", False):
            return
        lsp = getattr(self.orchestrator, "lsp", None) if self.orchestrator else None
        workspace = self._resolve_workspace(state)
        if lsp is None or workspace is None:
            return
        try:
            from victor.agent.conversation.history_metadata import build_internal_history_metadata
            from victor.agent.conversation.types import MessageSource
            from victor.framework.lsp_context import build_context

            block, signature = await build_context(
                lsp, workspace, last_signature=self._last_lsp_signature
            )
            if block is None:
                # Nothing new to inject; still remember the signature so a later
                # change re-triggers injection.
                if signature is not None:
                    self._last_lsp_signature = signature
                return
            chat_ctx = getattr(self.turn_executor, "_chat_context", None)
            if chat_ctx and hasattr(chat_ctx, "add_message"):
                chat_ctx.add_message(
                    role="user",
                    content=block,
                    metadata=build_internal_history_metadata(
                        "lsp_context", source=MessageSource.AGENT_GROUNDING
                    ),
                )
                self._last_lsp_signature = signature
        except Exception:
            logger.debug("LSP context injection skipped", exc_info=True)

    def _emit_prompt_reward_outcome(
        self,
        evaluation: Optional[EvaluationResult],
        state: Dict[str, Any],
    ) -> None:
        """Attribute a turn's outcome to the prompt candidates served this turn.

        Shared by ``run()`` and ``run_streaming()`` so buffered and streaming
        chat update the ``prompt_optimizer`` learner identically (FEP-0017
        streaming parity). Non-blocking; no-op when no candidate was served or
        no evaluation exists. Success is derived from the completion score
        inside the emitter, not the COMPLETE flag — a mid-task CONTINUE turn with
        good progress should reward (not penalize) the served candidate.
        """
        if evaluation is None:
            return
        try:
            from victor.agent.services.prompt_optimization_reward import (
                emit_prompt_candidate_outcome,
            )

            emit_prompt_candidate_outcome(
                getattr(self.runtime_intelligence, "_last_served_prompt_identities", []) or [],
                completion_score=float(getattr(evaluation, "score", 0.0) or 0.0),
                task_type=str(state.get("task_type") or "default"),
                session_id=state.get("session_id"),
            )
        except Exception as exc:
            logger.debug("Prompt candidate outcome attribution skipped: %s", exc)

    async def run_streaming(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream the agentic loop — the SAME loop as run(), differing only in I/O.

        FEP-0007 Addendum A (framing B): ``run_streaming`` drives the identical
        PERCEIVE -> PLAN -> ACT -> EVALUATE -> DECIDE sequence as the buffered ``run()``. It
        reuses the same non-yielding ``_analyze_turn`` / ``_plan`` / ``_evaluate`` phase methods
        and the shared ``turn_evaluation_controller`` / ``nudge_policy``. The ONLY difference is
        ACT: instead of a buffered ``_act`` returning a ``TurnResult``, it drives the injected
        streaming ACT port, which yields ``StreamChunk``s for live delivery and produces the same
        ``TurnResult`` the shared EVALUATE phase consumes. Streaming thus becomes a pure I/O mode
        of the one research-rooted loop rather than a separate, thinner loop.

        NOT yet wired into the live streaming path: it requires an injected ``streaming_act_port``
        and currently covers the core phase loop. The run()-only preamble bands (fast-slow
        planning gate, semantic response cache, paradigm/topology routing) and the richer DECIDE
        bands (content-repetition controller, adaptive termination) are reconciled into this shared
        path at cutover; until then ``StreamingChatExecutor.run()`` remains the single live
        streaming loop body.

        Args:
            query: User's natural language query.
            context: Additional execution context.
            conversation_history: Prior turns for multi-turn context.

        Yields:
            StreamChunk: live chunks for each turn's streaming ACT.

        Raises:
            RuntimeError: if no streaming ACT port has been wired.
        """
        if self.streaming_act_port is None:
            raise RuntimeError(
                "AgenticLoop.run_streaming requires a streaming ACT port; it is wired at the "
                "FEP-0007 cutover. Construct AgenticLoop(streaming_act_port=...) to drive it."
            )

        state: Dict[str, Any] = {"query": query, **(context or {})}
        self._progress_scores = []
        self._verify_retries = 0  # FEP-0018: reset per run
        self._last_lsp_signature = None  # FEP-0019 Phase 3: reset per run
        # Resets the shared spin detector + content-repetition + plateau (same as run()).
        self.turn_evaluation_controller.reset()
        self.criteria_builder.reset()
        # ADR-010: session-scoped effect ledger + downgrade budget (getattr: tests build
        # partial loops via __new__ without the gate; a missing gate means no gating).
        _effect_gate = getattr(self, "effect_gate", None)
        if _effect_gate is not None:
            _effect_gate.reset()
        _auditor = getattr(self, "per_turn_auditor", None)
        if _auditor is not None:
            _auditor.reset()
        effective_max = self.max_iterations
        evaluation: Optional[EvaluationResult] = None  # bound per turn; None if loop never runs

        for i in range(1, effective_max + 1):
            # PERCEIVE (shared, non-yielding)
            perception = await self._analyze_turn(query, context, conversation_history)
            self._last_perception = perception
            state["iteration"] = i
            if getattr(perception, "task_analysis", None):
                state["task_type"] = getattr(perception.task_analysis, "task_type", "unknown")
            state["perception"] = (
                perception.to_dict() if hasattr(perception, "to_dict") else perception
            )

            # FEP-0019 Phase 3: proactively inject live LSP symbols + errors for the
            # files being edited (mirrors run()'s perceive→plan injection point).
            await self._maybe_inject_lsp_context(state)

            # PLAN (shared, non-yielding) — planned once, reused across turns like run().
            plan = state.get("plan")
            if plan is None:
                plan = await self._plan(perception, state)
                state["plan"] = plan

            # ACT (streaming) — yields chunks; produces a TurnResult via the outcome holder.
            act_outcome = StreamingTurnOutcome()
            async for chunk in self.streaming_act_port.stream_turn_act(
                query=query,
                state=state,
                perception=perception,
                plan=plan,
                turn_index=i,
                outcome=act_outcome,
            ):
                yield chunk
            action_result = act_outcome.turn_result
            state["action_result"] = action_result

            # FEP-0023 Phase 1: record the turn onto the live SessionLedger
            # (streaming parity with run(); populates <SESSION_STATE>). No-op
            # unless USE_SESSION_LEDGER is enabled.
            self._populate_session_ledger(action_result, i)

            # Update shared spin detector + advance the temperature ratchet — the same per-turn step
            # the buffered _act path runs, so streaming detects spin and escalates temperature
            # identically (closes a latent streaming parity gap; ADR-013).
            self._record_spin_and_ratchet(action_result)

            # EVALUATE (shared, non-yielding)
            evaluation = await self._evaluate(perception, action_result, state)
            self._progress_scores.append(evaluation.score)

            # DECIDE — terminal decisions end the stream; non-terminal nudge + continue.
            if evaluation.decision == EvaluationDecision.COMPLETE:
                evaluation = self._apply_backslide_guard(evaluation)
                # FEP-0018: framework verification hook — verify before accepting.
                if await self._maybe_verify_and_retry(i, evaluation, state, streaming=True):
                    continue  # re-enter (skip return)
            if evaluation.decision in (
                EvaluationDecision.COMPLETE,
                EvaluationDecision.FAIL,
            ):
                # Streaming parity: attribute the turn outcome to served prompt
                # candidates before ending the stream (FEP-0017; mirrors run()).
                self._emit_prompt_reward_outcome(evaluation, state)
                return
            self._inject_decide_nudges(evaluation, i, effective_max)

        # Loop exhausted without a terminal decision — attribute the final turn's
        # evaluation too (FEP-0017 streaming parity; mirrors run()'s epilogue).
        self._emit_prompt_reward_outcome(evaluation, state)

    async def stream(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[LoopIteration]:
        """Stream agentic loop iterations.

        Args:
            query: User's natural language query
            context: Additional execution context
            conversation_history: Prior turns for multi-turn context

        Yields:
            LoopIteration for each iteration
        """
        from victor.framework.agentic_loop_executor import use_stategraph_executor

        if use_stategraph_executor():
            async for iteration in self._stream_with_stategraph(
                query=query,
                context=context,
                conversation_history=conversation_history,
            ):
                yield iteration
            return

        state: Dict[str, Any] = {
            "query": query,
            "conversation_history": conversation_history,
            **(context or {}),
        }

        for i in range(1, self.max_iterations + 1):
            iteration = LoopIteration(iteration=i, stage=LoopStage.PERCEIVE)

            # PERCEIVE
            perception = await self._analyze_turn(query, context, conversation_history)
            iteration.perception = perception
            state["perception"] = perception.to_dict()
            yield iteration

            # PLAN
            plan = await self._plan(perception, state)
            iteration.plan = plan
            state["plan"] = plan
            yield iteration

            # ACT
            action_result = await self._act(plan, state)
            iteration.action_result = action_result
            state["action_result"] = action_result
            yield iteration

            # EVALUATE
            evaluation = await self._evaluate(perception, action_result, state)
            iteration.evaluation = evaluation
            iteration.stage = LoopStage.EVALUATE
            yield iteration

            # Check termination
            if evaluation.decision in (
                EvaluationDecision.COMPLETE,
                EvaluationDecision.FAIL,
            ):
                break

    async def stream_chat(
        self,
        query: str,
        streaming_executor: Any = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Any]:
        """Stream chat with perception and evaluation lifecycle.

        Wraps the canonical StreamingChatExecutor with AgenticLoop's
        PERCEIVE and EVALUATE phases. The streaming executor handles
        the ACT phase (LLM streaming + tool execution + recovery).

        Perception runs concurrently with streaming start to avoid
        blocking the first chunk (TaskAnalyzer cold start is ~5s on
        first call due to embedding model loading).

        Args:
            query: User's natural language query
            streaming_executor: StreamingChatExecutor instance
            context: Additional context

        Yields:
            StreamChunk objects from the streaming executor
        """
        # Reset spin detector for this conversation turn
        self.turn_evaluation_controller.reset()  # resets spin_detector + content-repetition + plateau
        self.criteria_builder.reset()
        # ADR-010: session-scoped effect ledger + downgrade budget (getattr: tests build
        # partial loops via __new__ without the gate; a missing gate means no gating).
        _effect_gate = getattr(self, "effect_gate", None)
        if _effect_gate is not None:
            _effect_gate.reset()
        _auditor = getattr(self, "per_turn_auditor", None)
        if _auditor is not None:
            _auditor.reset()

        # PERCEIVE (before streaming — understands task before LLM call)
        perception = await self._analyze_turn(query, context)
        logger.info(
            f"[stream] Perceived: intent={perception.intent.value}, "
            f"complexity={perception.complexity.value}, "
            f"confidence={perception.confidence:.2f}"
        )

        # ACT via streaming executor (yields chunks token-by-token)
        if streaming_executor is not None:
            try:
                # FEP-0007: use the canonical run_unified() entry point. The
                # legacy run() alias on StreamingChatExecutor is LTS-deprecated;
                # the hasattr fallback keeps the duck-typed test executors
                # (which may only implement run()) working during the cutover.
                if hasattr(streaming_executor, "run_unified"):
                    run_streaming = streaming_executor.run_unified
                else:
                    run_streaming = streaming_executor.run
                async for chunk in run_streaming(query):
                    yield chunk
            except Exception as e:
                if "request format error" in str(e).lower():
                    logger.warning(
                        f"[stream] Streaming failed due to message format error, "
                        f"falling back to non-streaming: {e}"
                    )
                    # Fall back: yield a simple text chunk with error info
                    # In production, you might want to call a non-streaming chat method here
                    from victor.providers.base import StreamChunk

                    yield StreamChunk(
                        content=f"[Streaming error: {str(e)}. The system encountered a message format issue. "
                        f"This has been logged and will be fixed in future updates.]",
                        is_final=True,
                    )
                else:
                    # Re-raise non-format errors
                    raise
        else:
            logger.warning("[stream] No streaming executor provided")
            return

        # EVALUATE (post-hoc — decisions need full response context)
        logger.debug("[stream] Streaming complete, post-hoc evaluation done")

    async def _initialize_topology_plan(
        self,
        query: str,
        perception: Perception,
        state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[GroundedTopologyPlan]:
        """Build and ground a topology plan for the first runtime iteration."""
        if not self._topology_enabled or state.get("_topology_plan_obj") is not None:
            return None

        task_type = str(
            state.get("task_type")
            or getattr(getattr(perception, "task_analysis", None), "task_type", "unknown")
            or "unknown"
        )
        task_classification = state.get("_task_classification")
        default_tool_budget = getattr(task_classification, "tool_budget", None)
        tool_budget = state.get("tool_budget", default_tool_budget if default_tool_budget else 10)
        routing_context = dict(context or {})
        routing_context.setdefault(
            "iteration_budget", state.get("iteration_budget", self.max_iterations)
        )
        routing_context.setdefault("tool_budget", tool_budget)
        routing_context.setdefault("available_team_formations", self._default_team_formations())

        provider_hints = await self._get_topology_provider_hints(
            task_type=task_type,
            context=routing_context,
        )
        if provider_hints:
            state["_topology_provider_hints"] = dict(provider_hints)
            routing_context.update(provider_hints)
            provider_candidates = []
            primary_provider = provider_hints.get("provider_hint")
            if isinstance(primary_provider, str) and primary_provider:
                provider_candidates.append(primary_provider)
            for fallback in provider_hints.get("fallback_chain", []):
                if isinstance(fallback, str) and fallback:
                    provider_candidates.append(fallback)
            if provider_candidates:
                routing_context["provider_candidates"] = list(dict.fromkeys(provider_candidates))
        structured_routing_policy = state.get("_structured_routing_policy_obj")
        if structured_routing_policy is not None and hasattr(
            structured_routing_policy, "selector_context"
        ):
            learned_topology_context = structured_routing_policy.selector_context()
            if isinstance(learned_topology_context, dict) and learned_topology_context:
                routing_context.update(learned_topology_context)
        elif hasattr(self.runtime_intelligence, "get_structured_routing_policy"):
            learned_scope_context = dict(routing_context)
            learned_scope_context.setdefault("task_type", task_type)
            try:
                structured_routing_policy = self.runtime_intelligence.get_structured_routing_policy(
                    query=query,
                    scope_context=learned_scope_context,
                )
            except Exception as exc:
                logger.debug(
                    "Runtime intelligence structured routing policy unavailable: %s",
                    exc,
                )
            else:
                if structured_routing_policy is not None:
                    state["_structured_routing_policy_obj"] = structured_routing_policy
                    if hasattr(structured_routing_policy, "to_dict"):
                        serialized_policy = structured_routing_policy.to_dict()
                        if isinstance(serialized_policy, dict):
                            state["structured_routing_policy"] = serialized_policy
                    learned_topology_context = structured_routing_policy.selector_context()
                    if isinstance(learned_topology_context, dict) and learned_topology_context:
                        routing_context.update(learned_topology_context)
        if hasattr(self.runtime_intelligence, "get_topology_routing_context"):
            learned_scope_context = dict(routing_context)
            learned_scope_context.setdefault("task_type", task_type)
            try:
                learned_topology_context = self.runtime_intelligence.get_topology_routing_context(
                    query=query, scope_context=learned_scope_context
                )
            except Exception as exc:
                logger.debug("Runtime intelligence topology routing hints unavailable: %s", exc)
            else:
                if isinstance(learned_topology_context, dict) and learned_topology_context:
                    routing_context.update(learned_topology_context)

        topology_input = self.paradigm_router.build_topology_input(
            task_type=task_type,
            query=query,
            history_length=len(conversation_history) if conversation_history else 0,
            query_complexity=self._perception_complexity_score(perception),
            tool_budget=int(tool_budget) if tool_budget is not None else None,
            context=routing_context,
        )
        topology_decision = self._topology_selector.select(topology_input)
        topology_plan = self._topology_grounder.ground(topology_decision)
        topology_overrides = topology_plan.to_context_overrides()
        runtime_context_overrides = self._merge_runtime_context_overrides(
            state,
            topology_overrides,
        )

        state["_topology_input_obj"] = topology_input
        state["_topology_decision_obj"] = topology_decision
        state["_topology_plan_obj"] = topology_plan
        state["topology_input"] = topology_input.to_dict()
        state["topology_decision"] = topology_decision.to_dict()
        state["topology_plan"] = topology_plan.to_dict()
        self._set_runtime_context_overrides(state, runtime_context_overrides)
        state["topology_iteration"] = state.get("iteration", 1)
        if topology_plan.tool_budget is not None:
            state["tool_budget"] = topology_plan.tool_budget
        if topology_plan.iteration_budget is not None:
            state["iteration_budget"] = topology_plan.iteration_budget
        if topology_decision.action == TopologyAction.DIRECT_RESPONSE:
            state["_is_qa_task"] = True

        logger.info(
            "[Topology] action=%s topology=%s provider=%s formation=%s",
            topology_decision.action.value,
            topology_decision.topology.value,
            topology_plan.provider,
            topology_plan.formation,
        )
        return topology_plan

    async def _prepare_topology_runtime(
        self,
        topology_plan: GroundedTopologyPlan,
        query: str,
        state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Let the runtime prepare topology-specific execution state once selected."""
        if self.turn_executor is None:
            return None

        prepare_runtime_topology = getattr(self.turn_executor, "prepare_runtime_topology", None)
        if (
            prepare_runtime_topology is None
            or not callable(prepare_runtime_topology)
            or not inspect.iscoroutinefunction(prepare_runtime_topology)
        ):
            return None

        try:
            result = await prepare_runtime_topology(
                topology_plan,
                user_message=query,
                task_classification=state.get("_task_classification"),
            )
        except Exception as exc:
            logger.debug("Topology runtime preparation unavailable: %s", exc)
            return None

        if isinstance(result, dict):
            prepared = dict(result)
            prepared_overrides = prepared.get("runtime_context_overrides")
            if isinstance(prepared_overrides, dict):
                merged_overrides = self._merge_runtime_context_overrides(
                    state,
                    prepared_overrides,
                )
                self._set_runtime_context_overrides(state, merged_overrides)
            return prepared
        return None

    @staticmethod
    def _get_runtime_context_overrides(
        state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Return merged generic runtime overrides with legacy topology fallback."""
        merged: Dict[str, Any] = {}
        legacy_overrides = state.get(LEGACY_TOPOLOGY_OVERRIDES_STATE_KEY)
        if isinstance(legacy_overrides, dict):
            merged.update(legacy_overrides)
        runtime_overrides = state.get(RUNTIME_CONTEXT_OVERRIDES_STATE_KEY)
        if isinstance(runtime_overrides, dict):
            merged.update(runtime_overrides)
        return merged or None

    @classmethod
    def _merge_runtime_context_overrides(
        cls,
        state: Dict[str, Any],
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge new overrides into existing generic runtime override state."""
        merged = dict(cls._get_runtime_context_overrides(state) or {})
        merged.update(overrides)
        return merged

    @staticmethod
    def _set_runtime_context_overrides(
        state: Dict[str, Any],
        overrides: Dict[str, Any],
    ) -> None:
        """Store runtime overrides on the generic key and mirror legacy topology key."""
        state[RUNTIME_CONTEXT_OVERRIDES_STATE_KEY] = dict(overrides)
        state[LEGACY_TOPOLOGY_OVERRIDES_STATE_KEY] = dict(overrides)

    async def _get_topology_provider_hints(
        self,
        task_type: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fetch provider-routing hints when the active provider exposes them."""
        if self.turn_executor is None:
            return {}

        provider_context = getattr(self.turn_executor, "_provider_context", None)
        if provider_context is None:
            return {}

        provider = getattr(provider_context, "provider", None)
        model_hint = getattr(provider_context, "model", None)
        preferred_providers = context.get("preferred_providers")

        hint_sources = [
            getattr(provider, "get_topology_provider_hints", None),
            getattr(getattr(provider, "engine", None), "get_topology_provider_hints", None),
        ]
        for get_hints in hint_sources:
            if not callable(get_hints):
                continue
            try:
                hints = await get_hints(
                    task_type=task_type,
                    model_hint=model_hint,
                    preferred_providers=preferred_providers,
                )
            except Exception as exc:
                logger.debug("Topology provider hints unavailable: %s", exc)
                continue
            if isinstance(hints, dict):
                return hints
        return {}

    @staticmethod
    def _perception_complexity_score(perception: Perception) -> Optional[float]:
        """Convert Perception complexity into a stable numeric score."""
        complexity = getattr(perception, "complexity", None)
        if complexity is None:
            return None
        complexity_value = getattr(complexity, "value", str(complexity)).lower()
        if (
            "action" in complexity_value
            or "complex" in complexity_value
            or "high" in complexity_value
        ):
            return 0.8
        if "medium" in complexity_value or "moderate" in complexity_value:
            return 0.5
        return 0.2

    @staticmethod
    def _default_team_formations() -> List[str]:
        """Return canonical team formation hints for topology grounding."""
        from victor.teams.types import TeamFormation

        return [formation.value for formation in TeamFormation]

    def _capture_provider_degradation_baseline(self, state: Dict[str, Any]) -> None:
        """Capture the pre-execution provider degradation snapshot once per task."""
        if state.get("_provider_degradation_before") is not None:
            return
        tracker, provider_name, model_name = self._resolve_provider_degradation_runtime(state)
        if tracker is None or not provider_name or not hasattr(tracker, "get_degradation_snapshot"):
            return
        try:
            snapshot = tracker.get_degradation_snapshot(provider_name)
        except Exception as exc:
            logger.debug("Failed to capture provider degradation baseline: %s", exc)
            return
        snapshot_payload = snapshot.to_dict() if hasattr(snapshot, "to_dict") else dict(snapshot)
        if model_name and snapshot_payload.get("model") is None:
            snapshot_payload["model"] = model_name
        state["_provider_degradation_before"] = snapshot_payload

    def _record_provider_degradation_event(
        self,
        state: Dict[str, Any],
        *,
        total_iterations: int,
    ) -> None:
        """Emit a stable provider degradation event when runtime evidence exists."""
        if state.get("_provider_degradation_recorded"):
            return
        tracker, provider_name, model_name = self._resolve_provider_degradation_runtime(state)
        if tracker is None or not provider_name or not hasattr(tracker, "get_degradation_snapshot"):
            return
        try:
            snapshot = tracker.get_degradation_snapshot(provider_name)
        except Exception as exc:
            logger.debug("Failed to capture provider degradation summary: %s", exc)
            return
        before = dict(state.get("_provider_degradation_before") or {})
        after = snapshot.to_dict() if hasattr(snapshot, "to_dict") else dict(snapshot)
        if model_name and after.get("model") is None:
            after["model"] = model_name

        pre_degraded = bool(before.get("degraded", False))
        post_degraded = bool(after.get("degraded", False))
        recovered = bool(after.get("recovered_from_recent_incident", False)) or (
            pre_degraded and not post_degraded
        )
        if not (pre_degraded or post_degraded or recovered):
            return

        error_types = dict(after.get("recent_error_types") or {})
        failure_type = "PROVIDER_ERROR"
        if any(
            "rate" in str(name).lower() and "limit" in str(name).lower() for name in error_types
        ):
            failure_type = "RATE_LIMITED"

        event = {
            "source": "provider_performance",
            "kind": (
                "provider_recovered"
                if recovered and not post_degraded
                else "persistent_provider_degradation"
            ),
            "failure_type": failure_type,
            "provider": provider_name,
            "model": after.get("latest_model") or model_name,
            "task_type": state.get("task_type"),
            "pre_degraded": pre_degraded,
            "post_degraded": post_degraded,
            "recovered": recovered,
            "iteration": total_iterations,
            "adaptation_cost": float(
                after.get("recent_incident_failure_count")
                or after.get("failure_streak")
                or before.get("failure_streak")
                or 0.0
            ),
            "time_to_recover_seconds": after.get("time_to_recover_seconds"),
            "degradation_reasons": list(
                after.get("degradation_reasons") or before.get("degradation_reasons") or []
            ),
            "recent_error_types": error_types,
            "score_before": before.get("score"),
            "score_after": after.get("score"),
            "latency_trend_before": before.get("latency_trend"),
            "latency_trend_after": after.get("latency_trend"),
            "failure_streak_before": before.get("failure_streak"),
            "failure_streak_after": after.get("failure_streak"),
            "success_streak_after": after.get("success_streak"),
        }
        state.setdefault("degradation_events", []).append(event)
        state["_provider_degradation_after"] = after
        state["_provider_degradation_recorded"] = True

    def _resolve_provider_degradation_runtime(
        self,
        state: Dict[str, Any],
    ) -> tuple[Optional[Any], Optional[str], Optional[str]]:
        """Resolve the active provider-performance tracker and target provider name."""
        if self.turn_executor is None:
            return None, None, None

        provider_context = getattr(self.turn_executor, "_provider_context", None)
        if provider_context is None:
            return None, None, None

        provider = getattr(provider_context, "provider", None)
        model_name = getattr(provider_context, "model", None)
        tracker = None
        if provider is not None:
            tracker = getattr(provider, "tracker", None)
            if tracker is None:
                tracker = getattr(getattr(provider, "engine", None), "tracker", None)

        provider_name = None
        provider_hints = dict(state.get("_topology_provider_hints") or {})
        hinted_provider = provider_hints.get("provider_hint")
        if isinstance(hinted_provider, str) and hinted_provider:
            provider_name = hinted_provider
        elif provider is not None:
            candidate_name = getattr(provider, "name", None)
            if (
                isinstance(candidate_name, str)
                and candidate_name
                and candidate_name != "smart-router"
            ):
                provider_name = candidate_name

        return tracker, provider_name, model_name

    @staticmethod
    def _topology_feedback_status(decision: EvaluationDecision) -> str:
        """Map loop evaluation decisions into topology feedback statuses."""
        if decision == EvaluationDecision.COMPLETE:
            return "completed"
        if decision == EvaluationDecision.FAIL:
            return "failed"
        return "resolved"

    def _apply_backslide_guard(
        self,
        evaluation: Any,
        backslide_threshold: float = -0.10,
    ) -> Any:
        """Downgrade COMPLETE to CONTINUE if score velocity indicates backslide.

        A significant drop in score from the previous turn (velocity below
        ``backslide_threshold``) means the model regressed and should not be
        considered complete yet.  Uses ``self._score_history`` to compute velocity.

        Args:
            evaluation:          EvaluationResult from the current turn.
            backslide_threshold: Velocity below this value triggers downgrade.

        Returns:
            Possibly-modified EvaluationResult (new object if downgraded).
        """
        from victor.framework.evaluation_nodes import (
            EvaluationDecision,
            EvaluationResult,
        )

        score = getattr(evaluation, "score", None)
        if score is None:
            return evaluation

        self._score_history.append(score)
        if len(self._score_history) < 2:
            return evaluation

        velocity = self._score_history[-1] - self._score_history[-2]
        if evaluation.decision == EvaluationDecision.COMPLETE and velocity < backslide_threshold:
            logger.debug(
                "Backslide guard: score %.2f → %.2f (velocity %.2f < %.2f); "
                "downgrading COMPLETE to CONTINUE",
                self._score_history[-2],
                self._score_history[-1],
                velocity,
                backslide_threshold,
            )
            return EvaluationResult(
                decision=EvaluationDecision.CONTINUE,
                score=score,
                reason=(
                    f"Backslide guard: velocity {velocity:.2f} below threshold "
                    f"{backslide_threshold:.2f} — deferring completion"
                ),
                metrics=getattr(evaluation, "metrics", {}),
                metadata=getattr(evaluation, "metadata", {}),
            )
        return evaluation

    def _detect_phase(
        self,
        iteration: int,
        state: Dict[str, Any],
        perception: Perception,
    ) -> TaskPhase:
        """Detect the current task phase based on iteration and state.

        Args:
            iteration: Current iteration number
            state: Current loop state
            perception: Current perception result

        Returns:
            Detected task phase
        """
        from victor.core.shared_types import ConversationStage

        # Map conversation stage to task phase
        stage = perception.action_intent.stage if hasattr(perception, "action_intent") else None

        # Simple phase detection based on iteration and stage
        if iteration == 1:
            # First iteration is always exploration
            return TaskPhase.EXPLORATION
        elif stage in (ConversationStage.PLANNING,):
            return TaskPhase.PLANNING
        elif stage in (ConversationStage.EXECUTION,):
            return TaskPhase.EXECUTION
        elif stage in (ConversationStage.VERIFICATION, ConversationStage.COMPLETION):
            return TaskPhase.REVIEW
        else:
            # Default to EXECUTION for middle iterations
            return TaskPhase.EXECUTION if iteration > 1 else TaskPhase.EXPLORATION

    def _calculate_research_progress(self, iteration: int, max_iterations: int) -> int:
        """Calculate research progress percentage.

        Research has 4 phases: discover(25%), search(25%), analyze(25%), synthesize(25%)
        Each phase gets roughly equal iterations.

        Args:
            iteration: Current iteration number
            max_iterations: Maximum iterations for this task

        Returns:
            Progress percentage (0-100)
        """
        # Calculate which phase we're in (4 phases)
        phase_size = max_iterations // 4
        phase_num = min(3, (iteration - 1) // phase_size)  # 0-3
        base_progress = phase_num * 25  # 0, 25, 50, 75

        # Calculate progress within current phase
        iteration_in_phase = (iteration - 1) % phase_size
        phase_progress = int((iteration_in_phase / phase_size) * 25) if phase_size > 0 else 0

        return base_progress + phase_progress

    def _get_current_research_phase(self, state: Dict[str, Any]) -> str:
        """Get current research phase based on task phase.

        Args:
            state: Current loop state

        Returns:
            Research phase name (discover, search, analyze, synthesize)
        """
        from victor.core.shared_types import TaskPhase

        task_phase = state.get("task_phase", TaskPhase.EXPLORATION)

        # Map task phases to research phases
        phase_map = {
            TaskPhase.EXPLORATION: "discover",
            TaskPhase.PLANNING: "discover",
            TaskPhase.EXECUTION: "search",  # or "analyze" depending on iteration
            TaskPhase.REVIEW: "synthesize",
        }

        # Refine EXECUTION phase detection
        if task_phase == TaskPhase.EXECUTION:
            iteration = state.get("iteration", 1)
            if iteration > (self.max_iterations // 2):
                return "analyze"  # Second half of execution is analysis

        return phase_map.get(task_phase, "discover")

    def _count_papers_found(
        self, conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """Count unique arXiv papers mentioned in conversation.

        Args:
            conversation_history: Conversation history to search

        Returns:
            Number of unique papers found
        """
        import re
        from collections import defaultdict

        papers = set()

        # Search in conversation history
        if conversation_history:
            for msg in conversation_history:
                content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                # Match arXiv IDs like 1234.56789 or arXiv:1234.56789
                arxiv_pattern = r"\b(\d{4}\.\d{5,})\b"
                for match in re.findall(arxiv_pattern, content):
                    papers.add(match)

        # Also search in tool results if available
        # (This would require accessing the tool results from state)

        return len(papers)

    @staticmethod
    def _build_fast_path_plan(
        perception: Perception,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a minimal execution plan when planning is intentionally skipped."""
        return {
            "planning_skipped": True,
            "query": state.get("query", ""),
            "task_type": state.get("task_type", "unknown"),
            "perception": perception.to_dict(),
        }

    async def _plan(
        self,
        perception: Perception,
        state: Dict[str, Any],
    ) -> Any:
        """Generate execution plan.

        Uses existing PlanningCoordinator if available.
        """
        # Check if orchestrator has planning capabilities
        if hasattr(self.orchestrator, "planning_coordinator"):
            # Use existing PlanningCoordinator.chat_with_planning()
            planning_coordinator = self.orchestrator.planning_coordinator
            result = await planning_coordinator.chat_with_planning(
                state.get("query", ""),
            )
            return result

        # Fallback: Use orchestrator directly
        if hasattr(self.orchestrator, "plan"):
            return await self.orchestrator.plan(
                state.get("query", ""),
                context=state,
            )

        # Minimal fallback: return perception as plan
        return {"perception": perception.to_dict()}

    def _record_spin_and_ratchet(self, turn_result: Any) -> None:
        """Update the shared SpinDetector + advance the temperature ratchet from one turn's result.

        Shared by the buffered (``_act``) and streaming (``run_streaming``) paths so both detect spin
        and escalate temperature identically (ADR-013): a tool-using turn is progress (reset); a
        stalled/spinning turn escalates temperature for the next turn's resolution. The ratchet lives
        on the orchestrator so the resolver reads the same instance.
        """
        if turn_result is None:
            return
        # Defensive reads: in production turn_result is a full TurnResult, but the loop must not crash
        # the turn if a partial result reaches here (a missing field just means "no tool activity").
        has_tool_calls = bool(getattr(turn_result, "has_tool_calls", False))
        tool_results = getattr(turn_result, "tool_results", None)
        tool_names = set()
        if has_tool_calls and tool_results:
            tool_names = {r.get("tool_name", "") for r in tool_results}
        self.spin_detector.record_turn(
            has_tool_calls=has_tool_calls,
            all_blocked=bool(getattr(turn_result, "all_tools_blocked", False)),
            tool_names=tool_names,
            tool_count=getattr(turn_result, "tool_calls_count", 0),
            tool_signatures=getattr(turn_result, "tool_signatures", ()),
        )
        ratchet = getattr(getattr(self, "orchestrator", None), "temperature_ratchet_state", None)
        if ratchet is not None:
            from victor.framework.temperature import SpinSignal

            ratchet.record_turn(
                SpinSignal(
                    spin_state=getattr(self.spin_detector.state, "value", "normal"),
                    consecutive_no_tool_turns=getattr(
                        self.spin_detector, "consecutive_no_tool_turns", 0
                    ),
                    made_progress=has_tool_calls,
                )
            )

    async def _act(
        self,
        plan: Any,
        state: Dict[str, Any],
    ) -> Any:
        """Execute a single turn: one LLM call + tool execution.

        When turn_executor is available, calls execute_turn()
        for true single-turn execution (no nested loops). Falls back to
        orchestrator methods when coordinator is not available.

        Returns:
            TurnResult when using turn_executor, or Any from fallbacks.
        """
        query = state.get("query", "")

        # PRIMARY: Use turn_executor for single-turn (no nested loop)
        if self.turn_executor is not None:
            from victor.agent.services.turn_execution_runtime import TurnResult

            task_classification = state.get("_task_classification")
            is_qa = state.get("_is_qa_task", False)
            runtime_context_overrides = self._get_runtime_context_overrides(state)
            if self._should_execute_team_plan(runtime_context_overrides):
                team_turn = await self._execute_team_plan(
                    plan,
                    state,
                    runtime_context_overrides,
                )
                if team_turn is not None:
                    return team_turn
            enable_thinking = bool(
                state.get("enable_thinking")
                or (
                    isinstance(runtime_context_overrides, dict)
                    and runtime_context_overrides.get("execution_mode") == "escalated_single_agent"
                )
            )

            synthesize_from_tool_evidence = getattr(
                self.turn_executor,
                "synthesize_from_tool_evidence",
                None,
            )
            if (
                state.pop("_force_synthesis_next", False)
                and state.get("_successful_tool_evidence")
                and callable(synthesize_from_tool_evidence)
            ):
                turn_result = await synthesize_from_tool_evidence(
                    recovery_reason="successful tool evidence is ready for synthesis"
                )
            else:
                turn_result = await self.turn_executor.execute_turn(
                    user_message=query,
                    task_classification=task_classification,
                    is_qa_task=is_qa,
                    enable_thinking=enable_thinking,
                    intent=state.get("perception", {}).get("intent"),
                    task_type=state.get("task_type"),
                    runtime_context_overrides=runtime_context_overrides,
                )

            # Update shared spin detector + advance the temperature ratchet (shared with streaming).
            self._record_spin_and_ratchet(turn_result)

            # Record tool results for fulfillment criteria auto-derivation
            if turn_result.tool_results:
                for tr in turn_result.tool_results:
                    self.criteria_builder.record_tool_result(tr)

            return turn_result

        # FALLBACK: Use orchestrator methods (legacy path, nested loops)
        if hasattr(self.orchestrator, "chat"):
            return await self.orchestrator.chat(query)

        if hasattr(self.orchestrator, "execute"):
            return await self.orchestrator.execute(plan, state)

        if hasattr(self.orchestrator, "run"):
            return await self.orchestrator.run(query)

        return {"plan_executed": True, "plan": plan}

    @staticmethod
    def _should_execute_team_plan(
        runtime_context_overrides: Optional[Dict[str, Any]],
    ) -> bool:
        """Check whether topology preparation resolved a concrete team execution."""
        return bool(
            isinstance(runtime_context_overrides, dict)
            and runtime_context_overrides.get("execution_mode") == "team_execution"
            and runtime_context_overrides.get("team_name")
        )

    async def _execute_team_plan(
        self,
        plan: Any,
        state: Dict[str, Any],
        runtime_context_overrides: Optional[Dict[str, Any]],
    ) -> Optional["TurnResult"]:
        """Execute a topology-selected team plan through framework team runtime."""
        from victor.agent.services.turn_execution_runtime import TurnResult
        from victor.framework.team_runtime import run_configured_team
        from victor.providers.base import CompletionResponse

        if not isinstance(runtime_context_overrides, dict):
            return None

        task_classification = state.get("_task_classification")
        complexity_value = getattr(task_classification, "complexity", None)
        if hasattr(complexity_value, "value"):
            complexity_value = complexity_value.value
        if complexity_value is None:
            complexity_value = state.get("task_complexity") or state.get("perception", {}).get(
                "complexity",
                "medium",
            )

        team_execution = await run_configured_team(
            self.orchestrator,
            goal=state.get("query", ""),
            task_type=str(state.get("task_type") or "unknown"),
            complexity=str(complexity_value or "medium"),
            preferred_team=runtime_context_overrides.get("team_name"),
            preferred_formation=runtime_context_overrides.get("formation_hint"),
            max_workers=runtime_context_overrides.get("max_workers"),
            tool_budget=runtime_context_overrides.get("tool_budget"),
            context=self._build_team_execution_context(plan, state, runtime_context_overrides),
        )
        if team_execution is None:
            return None

        resolved_team, team_result = team_execution
        final_output = (
            team_result.final_output.strip() or team_result.error or "Team execution completed."
        )
        response = CompletionResponse(
            content=final_output,
            role="assistant",
            stop_reason="team_execution_completed",
            metadata={
                "execution_mode": "team_execution",
                "team_name": resolved_team.team_name,
                "team_display_name": resolved_team.display_name,
                "formation": resolved_team.formation.value,
                "team_success": bool(team_result.success),
                "member_count": resolved_team.member_count,
                "total_tool_calls": team_result.total_tool_calls,
            },
        )
        return TurnResult(
            response=response,
            tool_results=[],
            has_tool_calls=False,
            tool_calls_count=0,
            all_tools_blocked=not team_result.success,
            is_qa_response=False,
        )

    @staticmethod
    def _build_team_execution_context(
        plan: Any,
        state: Dict[str, Any],
        runtime_context_overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build shared context for framework team execution."""
        context: Dict[str, Any] = {
            "query": state.get("query", ""),
            "plan": plan,
            "task_type": state.get("task_type"),
            "task_complexity": state.get("task_complexity"),
            "perception": state.get("perception"),
            "topology_plan": state.get("topology_plan"),
        }
        for key in (
            "team_name",
            "team_display_name",
            "formation_hint",
            "topology_action",
            "topology_kind",
            "topology_metadata",
            "provider_hint",
            "max_workers",
            "worktree_isolation",
            "materialize_worktrees",
            "dry_run_worktrees",
            "cleanup_worktrees",
        ):
            value = runtime_context_overrides.get(key)
            if value is not None:
                context[key] = value
        return context

    def _is_complete_response(self, response: str) -> bool:
        """Check if response appears to be a complete answer.

        Heuristics for completion:
        - Contains complete code blocks
        - Contains summary/conclusion phrases
        - Direct answer format ("Here is...", "The answer is...")
        - Sufficient length (>150 chars)
        - NOT a continuation request ("Would you like...", "Should I...")

        Args:
            response: The model's response text

        Returns:
            True if response appears complete, False otherwise
        """
        if not response or len(response.strip()) < 150:
            return False

        response_lower = response.lower()

        # Continuation indicators (NOT complete)
        continuation_patterns = [
            "would you like",
            "should i",
            "do you want",
            "shall i",
            "would you prefer",
            "let me know if",
            "would you like me to continue",
        ]

        for pattern in continuation_patterns:
            if pattern in response_lower:
                return False

        # Completion indicators
        completion_patterns = [
            "here is",
            "here's",
            "the answer is",
            "the solution is",
            "in conclusion",
            "to summarize",
            "in summary",
            "the code above",
            "the following code",
            "as shown",
        ]

        has_completion_indicator = any(pattern in response_lower for pattern in completion_patterns)

        # Check for complete code blocks
        has_complete_code = "```" in response and response.count("```") >= 2

        # Check for structured response (headings, lists)
        has_structure = any(marker in response for marker in ["## ", "1.", "-", "*"])

        # Complete if: has indicators OR (has code/structure + sufficient length)
        return (
            has_completion_indicator
            or (has_complete_code and len(response) > 300)
            or (has_structure and len(response) > 400)
        )

    def _is_continuation_request(self, response: str) -> bool:
        """Check if response is asking for continuation direction.

        Args:
            response: The model's response text

        Returns:
            True if response is asking for continuation, False otherwise
        """
        if not response:
            return False

        response_lower = response.lower()

        continuation_patterns = [
            "would you like me to",
            "should i continue",
            "do you want me to",
            "shall i proceed",
            "let me know if you'd like",
            "would you prefer i",
        ]

        return any(pattern in response_lower for pattern in continuation_patterns)

    def _is_intent_only_response(self, response: str) -> bool:
        """Return True when the response is pure future-intent narration.

        Phrases like "I'll now read…" or "Let me analyze…" describe planned
        actions rather than completed work.  Treating them as final answers
        causes the loop to exit before any tools are actually invoked.

        Two checks are applied:
          1. First-line prefix check (preserves legacy behavior) so responses
             that start with intent but contain substantive findings are
             still allowed through.
          2. Meta-deliberation density check across the FULL response. This
             catches the failure mode where the model narrates imminent
             action ("Executing now", "Going now", "Calling now", "Making the
             call", "no more deliberation") without ever invoking a tool.
             Such narration must NOT be treated as a complete answer, or the
             agent loop exits before any tool runs. Only fires when there is
             no substantive payload (no code blocks / result-like content).
        """
        if not response:
            return False
        first_line = response.strip().split("\n")[0].strip().lower()
        intent_prefixes = (
            "i'll now ",
            "i'll ",
            "i will now ",
            "i will ",
            "let me now ",
            "let me ",
            "now i'll ",
            "now i will ",
            "i'm going to ",
            "i am going to ",
            "i'm now ",
            "i am now ",
            "next, i'll ",
            "next i'll ",
        )
        if any(first_line.startswith(p) for p in intent_prefixes):
            return True

        # Meta-deliberation narration density check (full response).
        # Real findings usually carry a payload (a fenced code block or a
        # tool-result-style table). Narration-only responses do not, so we
        # gate the density signal on the absence of such payloads.
        if "```" in response:
            return False
        lowered = response.lower()
        if lowered.count("|") >= 3 and "---" in lowered:
            return False  # Markdown table — looks like a result dump, not narration

        deliberation_markers = (
            "executing now",
            "executing.",
            "going now",
            "going.",
            "calling now",
            "calling.",
            "running now",
            "running.",
            "making the call",
            "making the request",
            "let me make the call",
            "no more deliberation",
            "stop the meta-deliberation",
            "stop deliberating",
            "done deliberating",
            "just execute",
            "executing the",
            "polling",
            "no sleep",
            "pure status read",
            "going. (",
            "done. (",
            "final. (",
            "(no sleep)",
            "(no more deliberation)",
            "(will act on results",
            "(finally.)",
            "(stop. calling.)",
        )
        marker_hits = sum(1 for m in deliberation_markers if m in lowered)
        # 3+ distinct imminent-action markers without a payload is strong
        # evidence of meta-deliberation narration, not a real answer.
        return marker_hits >= 3

    @staticmethod
    def _build_rubric_evaluator(strategy: str, rubric_complete_fn: Any):
        """Select the rubric completion evaluator, enforcing the ADR-011 fallback contract.

        Rubric completion (ADR-009 / EVR-3) is trusted ONLY with a calibrated LLM judge.
        Returns:
          - the LLM-judge evaluator when ``strategy`` is rubric/hybrid AND a
            ``rubric_complete_fn`` is injected (the caller must back it with a calibrated
            model — gemma4:31b α=0.865 real-trajectory, or llama3.3:70b α=1.000; see
            benchmarks/judge_calibration/FINDINGS.md);
          - ``None`` otherwise — which the completion cascade treats as "defer to enhanced".

        Crucially, a rubric/hybrid strategy WITHOUT an LLM judge returns None rather than the
        no-LLM HeuristicRubricJudge: that heuristic scored α=−0.092 in calibration (worse than
        the enhanced baseline), so silently using it would degrade completion decisions.
        """
        if strategy not in ("rubric", "hybrid"):
            return None
        if rubric_complete_fn is None:
            logger.warning(
                "[RubricCompletion] completion_strategy=%s requested but no LLM judge "
                "(rubric_complete_fn) was injected. The heuristic judge is uncalibrated "
                "(a=-0.092); reverting to enhanced completion per the ADR-011 fallback "
                "contract. Inject a calibrated judge to enable rubric completion.",
                strategy,
            )
            return None
        from victor.framework.rubric_completion import (
            AsyncRubricCompletionEvaluator,
            LLMRubricJudge,
        )

        logger.info("[RubricCompletion] ENABLED (LLM judge) — strategy=%s", strategy)
        return AsyncRubricCompletionEvaluator(LLMRubricJudge(rubric_complete_fn))

    async def _rubric_completion_result(
        self, perception: Any, action_result: Any, state: Dict[str, Any]
    ) -> Optional[EvaluationResult]:
        """Rubric-based completion verdict (ADR-009), for completion_strategy in {"rubric","hybrid"}.

        Returns ``None`` to defer to the enhanced/legacy cascade when not applicable: no rubric
        evaluator (other strategy), not a ``TurnResult``, tools still pending, or empty content.
        Duck-types the sync heuristic evaluator (``.evaluate``) vs the async LLM-judge evaluator
        (``.aevaluate``). For ``"hybrid"``, a rubric COMPLETE is confirmed only if the enhanced
        evaluator also says COMPLETE — this cancels the rubric judge's observed over-completion on
        intent statements (calibration κ≈0.5) while keeping its anti-under-scoring on real answers.
        """
        evaluator = self.rubric_completion_evaluator
        if evaluator is None:
            return None
        from victor.agent.services.turn_execution_runtime import TurnResult

        if not isinstance(action_result, TurnResult) or action_result.has_tool_calls:
            return None
        content = (getattr(action_result, "content", "") or "").strip()
        if not content:
            return None
        task_family = str(state.get("task_type") or "general")
        if hasattr(evaluator, "aevaluate"):
            result = await evaluator.aevaluate(
                task_family=task_family, content=content, context=state
            )
        else:
            result = evaluator.evaluate(task_family=task_family, content=content, context=state)

        if (
            self.config.completion_strategy == "hybrid"
            and result.complete
            and self.enhanced_completion_evaluator is not None
        ):
            enhanced = await self.enhanced_completion_evaluator.evaluate(
                perception=perception,
                action_result=action_result,
                state=state,
                fulfillment_detector=self.fulfillment,
                spin_detector=self.spin_detector,
            )
            if enhanced.decision != EvaluationDecision.COMPLETE:
                return EvaluationResult(
                    decision=EvaluationDecision.CONTINUE,
                    score=result.aggregate,
                    reason=f"[hybrid] rubric COMPLETE but enhanced disagreed: {enhanced.reason[:80]}",
                    metadata={"rubric_completion": result.to_dict()},
                )

        decision = EvaluationDecision.COMPLETE if result.complete else EvaluationDecision.CONTINUE
        return EvaluationResult(
            decision=decision,
            score=result.aggregate,
            reason=f"[rubric] {result.reason}",
            metadata={"rubric_completion": result.to_dict()},
        )

    def _is_terminal_answer(self, action_result: Any) -> bool:
        """True when a no-tool turn delivered a genuine final answer (not narration / a question).

        Mirrors the legacy terminal-answer rule: a substantial response (or any response after
        prior tool usage) with no pending tool calls that is neither intent-only narration
        ("I'll now read…") nor a continuation request ("Would you like me to…"). Used to complete
        promptly even when EnhancedCompletionEvaluator under-scores a finished answer — without it
        the loop burns low-confidence retries restating the answer (FEP-0007 cutover verbosity fix).
        """
        from victor.agent.services.turn_execution_runtime import TurnResult

        if not isinstance(action_result, TurnResult):
            return False
        if action_result.has_tool_calls or not action_result.has_content:
            return False
        content = action_result.content.strip()
        had_prior_tool_usage = getattr(self.spin_detector, "total_tool_calls", 0) > 0
        is_substantial = len(content) > 100
        if not (had_prior_tool_usage or is_substantial):
            return False
        # A refusal / "I can't do this" is not a delivered answer. Treating it as
        # terminal would force a high-confidence COMPLETE (success=True) for a turn
        # that declined the task — so exclude it like narration and questions.
        return (
            not self._is_continuation_request(content)
            and not self._is_intent_only_response(content)
            and not self._is_refusal_response(content)
        )

    # Phrases where the model declines/aborts the task itself (not findings like
    # "I cannot find any bugs"). Kept tight to avoid misclassifying real answers.
    _REFUSAL_MARKERS = (
        "i can't read",
        "i cannot read",
        "i can't access",
        "i cannot access",
        "unable to read",
        "unable to access",
        "i'm unable to",
        "i am unable to",
        "cannot comply",
        "can't comply",
        "the information needed",  # "...don't have the information needed..."
        "i can't provide a grounded",
        "can't give a grounded",
        "i'm sorry, but i can't",
        "i am sorry, but i can't",
    )

    def _is_refusal_response(self, content: str) -> bool:
        """True when the final answer declines/aborts the task (a non-answer)."""
        if not content:
            return False
        lowered = content.lower()
        return any(marker in lowered for marker in self._REFUSAL_MARKERS)

    async def _evaluate(
        self,
        perception: Perception,
        action_result: Any,
        state: Dict[str, Any],
    ) -> EvaluationResult:
        """EVALUATE seam wrapper: record effects, run the core cascade, gate COMPLETE.

        The effect gate (ADR-010 / EVR-4) wraps the whole completion cascade as a post-filter —
        the core has ~10 COMPLETE return points across strategies, and the gate must apply to
        all of them as a precondition. With the gate disabled (the default) both calls are
        strict no-ops and the core's result is returned unchanged (ADR-012 parity guarantee).
        """
        gate = getattr(self, "effect_gate", None)
        if gate is not None:
            gate.record(action_result, state)
        evaluation = await self._evaluate_core(perception, action_result, state)
        if gate is not None:
            evaluation = await gate.apply(evaluation, action_result, state)
        # EVR-6: online per-turn auditor — a second post-filter on the same seam (no-op when off).
        auditor = getattr(self, "per_turn_auditor", None)
        if auditor is not None:
            evaluation = auditor.apply(evaluation, action_result, state)
        return evaluation

    async def _evaluate_core(
        self,
        perception: Perception,
        action_result: Any,
        state: Dict[str, Any],
    ) -> EvaluationResult:
        """Evaluate progress and decide next action.

        When action_result is a TurnResult (single-turn mode), uses
        turn-level signals (tool calls, spin detection, Q&A shortcut)
        in addition to fulfillment checks. This replaces the nudge/spin
        logic that was previously inside TurnExecutor's while-loop.

        Enhanced Completion Detection:
        If enable_enhanced_completion is True, uses requirement-driven
        completion detection with multi-signal fusion for more accurate
        and earlier stopping when task requirements are satisfied.
        """
        clarification = self._evaluation_policy.get_clarification_decision(perception)
        if clarification.requires_clarification:
            return EvaluationResult(
                decision=EvaluationDecision.FAIL,
                score=clarification.confidence,
                reason=f"Clarification required: {clarification.reason}",
                metadata={
                    "requires_clarification": True,
                    "clarification_prompt": clarification.prompt,
                },
            )

        try:
            from victor.agent.services.turn_execution_runtime import TurnResult

            if isinstance(action_result, TurnResult):
                raw_response_metadata = getattr(action_result.response, "metadata", None)
                response_metadata = (
                    raw_response_metadata if isinstance(raw_response_metadata, dict) else {}
                )
                # The ACT primitive may signal a HIGH-confidence task completion (no pending
                # tools) — e.g. the streaming executor's task-completion detector. Honor it as an
                # immediate COMPLETE so the loop stops promptly instead of leaning on the
                # under-scoring completion evaluator (FEP-0007 cutover tune-up).
                if response_metadata.get("forced_task_completion") and not getattr(
                    action_result, "has_tool_calls", False
                ):
                    return EvaluationResult(
                        decision=EvaluationDecision.COMPLETE,
                        score=0.95,
                        reason="Task-completion active signal (HIGH confidence, no pending tools)",
                        metadata={"forced_task_completion": True},
                    )
                if (
                    response_metadata.get("deterministic_tool_execution")
                    and action_result.successful_tool_count > 0
                    and action_result.failed_tool_count == 0
                ):
                    return EvaluationResult(
                        decision=EvaluationDecision.COMPLETE,
                        score=0.9,
                        reason=("Deterministic read-only tool execution completed successfully"),
                        metadata={"successful_tool_progress": True},
                    )
                if response_metadata.get("agentic_loop_synthesis") and action_result.has_content:
                    # The forced-synthesis path can fire after just a couple
                    # of tool calls (via `_force_synthesis_next` in evaluate)
                    # before the model has actually written a deliverable.
                    # Failure modes we've seen on this path:
                    #   - short narrative ("Now let me read X")
                    #   - markdown code-block tool-call attempts
                    #     (```` ```graph fan_in``` ````)
                    #   - tool-call pseudocode as inline text
                    #     ("read(path='Cargo.toml', offset=0, limit=80)\n..."),
                    #     which can run 200+ chars without containing any
                    #     real document structure.
                    # A real synthesis (e.g. an init.md) BOTH has markdown
                    # structure (headers) AND substantial length. Require
                    # both — otherwise the loop continues and the model
                    # gets another turn to issue real function calls or
                    # write the deliverable.
                    _synth_content = getattr(action_result.response, "content", "") or ""
                    _stripped = _synth_content.strip()
                    _has_markdown_header = _stripped.startswith("#") or "\n#" in _synth_content
                    _looks_substantive = _has_markdown_header and len(_stripped) >= 200
                    if _looks_substantive:
                        return EvaluationResult(
                            decision=EvaluationDecision.COMPLETE,
                            score=0.9,
                            reason="Synthesized final response from successful tool evidence",
                        )
                    return EvaluationResult(
                        decision=EvaluationDecision.RETRY,
                        score=0.4,
                        reason=(
                            "Forced synthesis did not produce a final document "
                            f"({len(_stripped)} chars, markdown header="
                            f"{_has_markdown_header}); retrying so the model "
                            "can either make real function-call tool calls or "
                            "write the deliverable"
                        ),
                    )
                if response_metadata.get("execution_mode") == "team_execution":
                    if response_metadata.get("team_success", True) and action_result.has_content:
                        return EvaluationResult(
                            decision=EvaluationDecision.COMPLETE,
                            score=0.92,
                            reason="Framework team execution completed with synthesized output",
                        )
                    return EvaluationResult(
                        decision=EvaluationDecision.FAIL,
                        score=0.2,
                        reason="Framework team execution did not produce a usable final output",
                    )
        except Exception:
            pass

        # RUBRIC: task-adaptive rubric completion (ADR-009), opt-in via completion_strategy in
        # {"rubric","hybrid"}. Returns None to defer to the enhanced/legacy cascade when not applicable.
        rubric_result = await self._rubric_completion_result(perception, action_result, state)
        if rubric_result is not None:
            return rubric_result

        # ENHANCED: Use EnhancedCompletionEvaluator if enabled
        if self.enhanced_completion_evaluator is not None and self._should_use_enhanced_evaluation(
            action_result
        ):
            try:
                enhanced_result = await self.enhanced_completion_evaluator.evaluate(
                    perception=perception,
                    action_result=action_result,
                    state=state,
                    fulfillment_detector=self.fulfillment,
                    spin_detector=self.spin_detector,
                )
                # Log enhanced evaluation decision for observability
                logger.info(
                    f"[EnhancedCompletion] Decision: {enhanced_result.decision.value}, "
                    f"Score: {enhanced_result.score:.2f}, "
                    f"Reason: {enhanced_result.reason[:100]}"
                )
                tool_progress_result = self._normalize_successful_tool_progress(
                    enhanced_result,
                    action_result,
                )
                if tool_progress_result is not enhanced_result:
                    logger.info(
                        "[EnhancedCompletion] Converted low-confidence retry to progress "
                        "because successful tools produced execution evidence"
                    )
                    state["_successful_tool_evidence"] = True
                    # NOTE: we intentionally do NOT set `_force_synthesis_next`
                    # here. Setting it after every successful tool batch made
                    # the next iteration call `synthesize_from_tool_evidence`
                    # instead of running a normal model turn — at which point
                    # the recovery prompt ("Please provide your response now
                    # based on the information gathered") was answered by the
                    # model with tool-call pseudocode or narration like
                    # "Let me read more files", because the model had not
                    # actually finished gathering data after a single batch.
                    # The signal `_successful_tool_evidence` already lets the
                    # loop trigger a final synthesis if it terminates without
                    # a written answer; forcing it on every successful turn
                    # is the bug.
                    return tool_progress_result
                # A clear final answer (no pending tools, substantial, not narration/question) is
                # COMPLETE even when the evaluator under-scored it — otherwise the loop retries a
                # finished answer, restating it. Mirrors the legacy terminal-answer rule below
                # (FEP-0007 cutover verbosity fix).
                if enhanced_result.decision in (
                    EvaluationDecision.RETRY,
                    EvaluationDecision.CONTINUE,
                ) and self._is_terminal_answer(action_result):
                    logger.info(
                        "[EnhancedCompletion] Overriding low-confidence %s -> COMPLETE: final "
                        "answer delivered with no pending tools",
                        enhanced_result.decision.value,
                    )
                    return EvaluationResult(
                        decision=EvaluationDecision.COMPLETE,
                        score=max(enhanced_result.score, 0.8),
                        reason=(
                            "Final answer delivered with no pending tools — completing despite "
                            "low completion score"
                        ),
                        metadata={"terminal_answer_override": True},
                    )
                return self._apply_low_confidence_retry_budget(enhanced_result, state)
            except Exception as e:
                # Graceful degradation: fall back to legacy evaluation on error
                logger.warning(
                    f"[EnhancedCompletion] Evaluation failed: {e}, falling back to legacy"
                )

        # LEGACY: Original evaluation logic (preserved for backward compatibility)
        # Check for TurnResult-specific signals (single-turn mode)
        from victor.agent.services.turn_execution_runtime import TurnResult

        if isinstance(action_result, TurnResult):
            turn: TurnResult = action_result

            # Q&A shortcut: model answered without tools on a question task
            if turn.is_qa_response and turn.has_content:
                return EvaluationResult(
                    decision=EvaluationDecision.COMPLETE,
                    score=0.9,
                    reason="Q&A shortcut: accepted direct answer",
                )

            # Use shared SpinDetector for consistent detection
            spin_state = self.spin_detector.state
            if spin_state == SpinState.TERMINATED:
                if self.spin_detector.consecutive_all_blocked > 0:
                    return EvaluationResult(
                        decision=EvaluationDecision.FAIL,
                        score=0.1,
                        reason=(
                            f"Spin detected: {self.spin_detector.consecutive_all_blocked} "
                            "consecutive fully-blocked tool batches"
                        ),
                    )

                # Check for repetition — mirrors SpinDetector.state threshold
                repetition_count = getattr(self.spin_detector, "_repetition_count", 0)
                threshold = getattr(self.spin_detector, "REPETITION_THRESHOLD", 3)
                if repetition_count >= threshold - 1:
                    return EvaluationResult(
                        decision=EvaluationDecision.FAIL,
                        score=0.1,
                        reason=(
                            f"Repetitive loop detected: the model repeated the same tool "
                            f"calls for {repetition_count} consecutive turns"
                        ),
                    )

                return EvaluationResult(
                    decision=EvaluationDecision.FAIL,
                    score=0.2,
                    reason=(
                        f"Agent stuck: {self.spin_detector.consecutive_no_tool_turns} "
                        "turns without tool calls"
                    ),
                )

            # Model used tools = normal progress
            if turn.has_tool_calls and spin_state == SpinState.NORMAL:
                # Check fulfillment if enabled
                if self.fulfillment:
                    try:
                        task_type = self._map_to_task_type(perception)
                        # Use auto-derived criteria from tool results
                        criteria = state.get("criteria", {})
                        auto_criteria = self.criteria_builder.build().to_dict()
                        merged_criteria = {**auto_criteria, **criteria}
                        fulfillment = await self.fulfillment.check_fulfillment(
                            task_type=task_type,
                            criteria=merged_criteria,
                            context=state,
                        )
                        if fulfillment.is_fulfilled:
                            return EvaluationResult(
                                decision=EvaluationDecision.COMPLETE,
                                score=fulfillment.score,
                                reason=f"Fulfilled: {fulfillment.reason}",
                            )
                        # Partial fulfillment — refine with edge model
                        partial_result = EvaluationResult(
                            decision=EvaluationDecision.CONTINUE,
                            score=fulfillment.score,
                            reason=f"Progress: {fulfillment.reason}",
                        )
                        return await self._refine_with_llm(partial_result, action_result, state)
                    except Exception as e:
                        logger.warning(f"Fulfillment check failed: {e}")

                # No fulfillment check — tools ran, so continue
                return EvaluationResult(
                    decision=EvaluationDecision.CONTINUE,
                    score=0.5 + min(turn.successful_tool_count * 0.1, 0.3),
                    reason=f"Tools executed: {turn.successful_tool_count} ok, "
                    f"{turn.failed_tool_count} failed",
                )

            # Detect final answers - both after tools AND direct answers
            response_text = turn.content.strip()
            had_prior_tool_usage = getattr(self.spin_detector, "total_tool_calls", 0) > 0
            is_substantial_answer = len(response_text) > 100
            if (
                not turn.has_tool_calls
                and turn.has_content
                and (had_prior_tool_usage or is_substantial_answer)
            ):
                # Model provided a substantial response without requesting tools
                # This is likely a complete answer

                # Additional check: is this a continuation request?
                is_continuation = self._is_continuation_request(turn.content)
                # Guard: intent-only narration ("I'll now read…") is NOT a final answer
                is_intent_only = self._is_intent_only_response(turn.content)

                if not is_continuation and not is_intent_only:
                    if had_prior_tool_usage and not is_substantial_answer:
                        reason = (
                            "Model provided final answer after prior tool usage "
                            f"({len(turn.content)} chars) without requesting tools"
                        )
                    else:
                        reason = (
                            "Model provided substantial response "
                            f"({len(turn.content)} chars) without requesting tools"
                        )
                    heuristic = EvaluationResult(
                        decision=EvaluationDecision.COMPLETE,
                        score=0.8,
                        reason=reason,
                    )
                    return await self._refine_with_llm(heuristic, action_result, state)
                else:
                    # Model wants to continue (e.g., "Would you like me to...")
                    return EvaluationResult(
                        decision=EvaluationDecision.CONTINUE,
                        score=0.6,
                        reason="Model offered continuation - awaiting user direction",
                    )

            # No tools, no previous tools = needs nudge (continue to retry)
            if not turn.has_tool_calls and spin_state == SpinState.NORMAL:
                return EvaluationResult(
                    decision=EvaluationDecision.CONTINUE,
                    score=0.3,
                    reason="No tools used yet — giving another chance",
                )

            # Nudge territory: inject tool-use prompt
            if spin_state == SpinState.WARNING:
                return EvaluationResult(
                    decision=EvaluationDecision.RETRY,
                    score=0.2,
                    reason="Nudge: model not using tools, retrying with prompt",
                )

        # FALLBACK: Legacy path (orchestrator.chat() returned non-TurnResult)
        if self.fulfillment and hasattr(perception, "task_type"):
            try:
                task_type = self._map_to_task_type(perception)
                fulfillment = await self.fulfillment.check_fulfillment(
                    task_type=task_type,
                    criteria=state.get("criteria", {}),
                    context=state,
                )
                if fulfillment.is_fulfilled:
                    return EvaluationResult(
                        decision=EvaluationDecision.COMPLETE,
                        score=fulfillment.score,
                        reason=f"Fulfilled: {fulfillment.reason}",
                    )
                elif fulfillment.is_partial:
                    return EvaluationResult(
                        decision=EvaluationDecision.CONTINUE,
                        score=fulfillment.score,
                        reason=f"Partial: {fulfillment.reason}",
                    )
                else:
                    return EvaluationResult(
                        decision=EvaluationDecision.RETRY,
                        score=fulfillment.score,
                        reason=f"Not fulfilled: {fulfillment.reason}",
                    )
            except Exception as e:
                logger.warning(f"Fulfillment check failed: {e}")

        # Confidence-based fallback
        return self._evaluation_policy.evaluate_confidence_progress(
            perception.confidence,
            state,
        )

    def _apply_low_confidence_retry_budget(
        self,
        evaluation: EvaluationResult,
        state: Dict[str, Any],
    ) -> EvaluationResult:
        """Bound repeated low-confidence retries in the live loop."""
        return self._evaluation_policy.apply_retry_budget(
            evaluation,
            state,
        )

    def _normalize_successful_tool_progress(
        self,
        evaluation: EvaluationResult,
        action_result: Any,
    ) -> EvaluationResult:
        """Treat successful tool batches as progress, not low-confidence retry debt."""
        if evaluation.decision != EvaluationDecision.RETRY:
            return evaluation

        try:
            from victor.agent.services.turn_execution_runtime import TurnResult

            if not isinstance(action_result, TurnResult):
                return evaluation
            if not action_result.has_tool_calls or action_result.successful_tool_count <= 0:
                return evaluation
            if not self._has_substantive_tool_evidence(action_result):
                return evaluation
        except Exception:
            return evaluation

        metadata = dict(evaluation.metadata)
        metadata["successful_tool_progress"] = True
        metadata["successful_tool_count"] = action_result.successful_tool_count
        return EvaluationResult(
            decision=EvaluationDecision.CONTINUE,
            score=max(evaluation.score, self._evaluation_policy.enhanced_progress_threshold),
            reason=(
                "Successful tools produced execution evidence: "
                f"{action_result.successful_tool_count} ok, "
                f"{action_result.failed_tool_count} failed"
            ),
            metrics=dict(evaluation.metrics),
            metadata=metadata,
        )

    @staticmethod
    def _has_substantive_tool_evidence(action_result: Any) -> bool:
        """Return True when successful tools produced more than navigation context."""
        low_information_tools = {"ls", "pwd", "tree"}
        tool_results = getattr(action_result, "tool_results", []) or []
        successful_names = {
            str(result.get("tool_name") or result.get("name") or "").strip().lower()
            for result in tool_results
            if isinstance(result, dict) and result.get("success")
        }
        successful_names.discard("")
        if not successful_names:
            return False
        return any(tool_name not in low_information_tools for tool_name in successful_names)

    def _should_use_enhanced_evaluation(self, action_result: Any) -> bool:
        """Only run enhanced evaluation when there is a real execution payload."""
        try:
            from victor.agent.services.turn_execution_runtime import TurnResult

            if isinstance(action_result, TurnResult):
                return True
        except Exception:
            pass

        if action_result is None:
            return False
        if isinstance(action_result, str):
            return bool(action_result.strip())
        return hasattr(action_result, "response") or hasattr(action_result, "content")

    async def _check_fulfillment(
        self,
        perception: Perception,
        state: Dict[str, Any],
    ) -> Any:
        """Check task fulfillment.

        Uses FulfillmentDetector.
        """
        if not self.fulfillment:
            return None

        task_type = self._map_to_task_type(perception)
        return await self.fulfillment.check_fulfillment(
            task_type=task_type,
            criteria=state.get("criteria", {}),
            context=state,
        )

    async def _refine_with_llm(
        self,
        heuristic_result: EvaluationResult,
        action_result: Any,
        state: Dict[str, Any],
    ) -> EvaluationResult:
        """Refine evaluation via TieredDecisionService (canonical decision path).

        Uses the same decision service and tier routing as all other LLM
        decisions in Victor. The tier (edge/balanced/performance) is
        determined by TieredDecisionService's per-DecisionType routing,
        which can be escalated/de-escalated based on confidence feedback.

        Falls back to heuristic_result if decision service is unavailable,
        disabled, or errors out. This is a refinement, not a replacement.

        Args:
            heuristic_result: Result from heuristic evaluation
            action_result: TurnResult from execution
            state: Current execution state

        Returns:
            Refined EvaluationResult (may be unchanged)
        """
        # Only refine ambiguous decisions (score 0.4-0.8)
        if heuristic_result.score < 0.4 or heuristic_result.score > 0.8:
            return heuristic_result

        try:
            from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager

            if not get_feature_flag_manager().is_enabled(FeatureFlag.USE_EDGE_MODEL):
                return heuristic_result

            from victor.agent.decisions.schemas import DecisionType
            from victor.agent.services.protocols.decision_service import (
                get_decision_service,
            )
            from victor.core import get_container

            decision_svc = get_decision_service(get_container())
            if decision_svc is None:
                return heuristic_result

            # Build context for TaskCompletionDecision
            response_content = ""
            if hasattr(action_result, "content"):
                response_content = action_result.content or ""

            deliverable_count = 0
            if hasattr(action_result, "successful_tool_count"):
                deliverable_count = action_result.successful_tool_count

            context = {
                "response_tail": response_content[-500:] if response_content else "",
                "deliverable_count": deliverable_count,
                "signal_count": self.spin_detector.total_tool_calls,
            }

            # Call via canonical decision service (tier-aware)
            result = await decision_svc.decide_async(
                decision_type=DecisionType.TASK_COMPLETION,
                context=context,
                heuristic_result=heuristic_result,
                heuristic_confidence=heuristic_result.score,
            )

            # Interpret TaskCompletionDecision
            decision_obj = result.result
            if decision_obj and hasattr(decision_obj, "is_complete"):
                tier = result.source  # "edge", "balanced", "performance", etc.

                if decision_obj.is_complete and decision_obj.confidence >= 0.7:
                    # High confidence from LLM — de-escalate for next call
                    if hasattr(decision_svc, "deescalate_tier"):
                        tier_adjustment = decision_svc.deescalate_tier(
                            DecisionType.TASK_COMPLETION, "high_confidence"
                        )
                        if inspect.isawaitable(tier_adjustment):
                            await tier_adjustment
                    return EvaluationResult(
                        decision=EvaluationDecision.COMPLETE,
                        score=decision_obj.confidence,
                        reason=f"LLM ({tier}): complete (phase={decision_obj.phase})",
                        metadata={"source": tier},
                    )

                elif decision_obj.phase == "stuck":
                    # Stuck — escalate for next call
                    if hasattr(decision_svc, "escalate_tier"):
                        tier_adjustment = decision_svc.escalate_tier(
                            DecisionType.TASK_COMPLETION, "stuck_phase"
                        )
                        if inspect.isawaitable(tier_adjustment):
                            await tier_adjustment
                    return EvaluationResult(
                        decision=EvaluationDecision.RETRY,
                        score=decision_obj.confidence,
                        reason=f"LLM ({tier}): agent appears stuck",
                        metadata={"source": tier},
                    )

                elif decision_obj.confidence < 0.5:
                    # Low confidence from LLM — escalate tier
                    if hasattr(decision_svc, "escalate_tier"):
                        tier_adjustment = decision_svc.escalate_tier(
                            DecisionType.TASK_COMPLETION, "low_confidence"
                        )
                        if inspect.isawaitable(tier_adjustment):
                            await tier_adjustment

            logger.debug(
                "LLM refinement: source=%s, confidence=%.2f",
                result.source,
                result.confidence,
            )

        except Exception as e:
            logger.debug("LLM refinement skipped: %s", e)

        return heuristic_result

    def _check_adaptive_termination(
        self,
        iteration: int,
        evaluation: EvaluationResult,
        state: Optional[Dict[str, Any]] = None,
        action_result: Any = None,
    ) -> Optional[AdaptiveTerminationDecision]:
        """Check if loop should adaptively terminate or extend.

        Inspired by Fast-Slow Architecture (arXiv:2604.01681):
        - "Fast" exit on detected plateau (no progress)
        - "Slow" extension when near completion

        Returns:
            AdaptiveTerminationDecision.PLATEAU if should exit early,
            AdaptiveTerminationDecision.EXTEND if should grant more iterations,
            None otherwise
        """
        scores = self._progress_scores
        task_type = str((state or {}).get("task_type") or "").lower()
        read_heavy_task = task_type in {
            "search",
            "analysis",
            "documentation",
            "research",
        }
        has_successful_tool_activity = False

        try:
            from victor.agent.services.turn_execution_runtime import TurnResult

            if isinstance(action_result, TurnResult):
                has_successful_tool_activity = bool(
                    action_result.has_tool_calls and action_result.successful_tool_count > 0
                )
        except Exception:
            has_successful_tool_activity = False

        min_plateau_iteration = self.plateau_window
        if has_successful_tool_activity:
            if evaluation.metadata.get("successful_tool_progress"):
                logger.debug(
                    "Deferring adaptive plateau because this iteration produced "
                    "successful tool progress"
                )
                return None
            min_plateau_iteration = max(self.plateau_window * 2, 6)
            if not read_heavy_task:
                logger.debug(
                    "Deferring adaptive plateau because successful tools are still producing "
                    "new context for task_type=%s",
                    task_type or "unknown",
                )

        # Need enough history for plateau detection
        if len(scores) >= self.plateau_window and iteration >= min_plateau_iteration:
            recent = scores[-self.plateau_window :]
            improvement = max(recent) - min(recent)
            if improvement < self.plateau_tolerance:
                return AdaptiveTerminationDecision.PLATEAU

        # Check near-completion: score > 0.7 and still improving
        if len(scores) >= 2 and scores[-1] >= 0.7:
            delta = scores[-1] - scores[-2]
            if delta > 0:
                return AdaptiveTerminationDecision.EXTEND

        return None

    def _map_to_task_type(self, perception: Perception) -> TaskType:
        """Map perception to TaskType for fulfillment detection.

        Uses both intent and TaskAnalysis task_type for richer mapping.
        """
        from victor.agent.action_authorizer import ActionIntent

        def _normalize(value: Any) -> str:
            raw = getattr(value, "value", None)
            if raw is None:
                raw = value
            if not isinstance(raw, str):
                name = getattr(raw, "name", None)
                raw = name if isinstance(name, str) else ""
            text = str(raw or "").strip().lower()
            if text.startswith("tasktype."):
                text = text.split(".", 1)[1]
            return text.replace("-", "_")

        def _map_label(label: str) -> Optional[TaskType]:
            if not label:
                return None
            analysis_labels = {
                "analyze",
                "analysis",
                "analysis_deep",
                "architecture",
                "competitive_analysis",
                "correlation_analysis",
                "data_analysis",
                "data_profiling",
                "design",
                "explain",
                "fact_check",
                "general",
                "general_query",
                "literature_review",
                "plan",
                "research",
                "statistical_analysis",
                "technical_research",
                "trend_research",
                "visualization",
            }
            if label in analysis_labels:
                return TaskType.ANALYSIS
            if label in {"doc", "docs", "document", "documentation"}:
                return TaskType.DOCUMENTATION
            if label in {"search", "find", "lookup"}:
                return TaskType.SEARCH
            if label in {"test", "testing"}:
                return TaskType.TESTING
            if label in {"debug", "debugging"}:
                return TaskType.DEBUGGING
            if label in {
                "create",
                "create_simple",
                "code_generation",
                "generation",
            }:
                return TaskType.CODE_GENERATION
            if label in {
                "action",
                "bug",
                "bug_fix",
                "ci_cd",
                "code_modification",
                "docker_compose",
                "dockerfile",
                "edit",
                "implement",
                "infrastructure",
                "issue_resolution",
                "kubernetes",
                "refactor",
                "security",
                "terraform",
            }:
                return TaskType.CODE_MODIFICATION
            if label in {"setup"}:
                return TaskType.SETUP
            if label in {"deploy", "deployment"}:
                return TaskType.DEPLOYMENT
            return None

        # First try TaskAnalysis task_type (more specific)
        if perception.task_analysis:
            for attr in ("task_type", "unified_task_type", "intent_type"):
                mapped = _map_label(_normalize(getattr(perception.task_analysis, attr, None)))
                if mapped:
                    return mapped
            if getattr(perception.task_analysis, "is_analysis_task", False) is True:
                return TaskType.ANALYSIS
            if getattr(perception.task_analysis, "is_generation_task", False) is True:
                return TaskType.CODE_GENERATION
            if (
                getattr(perception.task_analysis, "is_action_task", False) is True
                or getattr(perception.task_analysis, "can_write_files", False) is True
            ):
                return TaskType.CODE_MODIFICATION

        mapped = _map_label(_normalize(getattr(perception, "task_type", "")))
        if mapped:
            return mapped

        # Fallback: map from intent
        intent_to_type = {
            ActionIntent.WRITE_ALLOWED: TaskType.CODE_GENERATION,
            ActionIntent.DISPLAY_ONLY: TaskType.SEARCH,
            ActionIntent.READ_ONLY: TaskType.ANALYSIS,
            ActionIntent.AMBIGUOUS: TaskType.UNKNOWN,
        }
        return intent_to_type.get(perception.intent, TaskType.UNKNOWN)

    def _determine_success(self, iterations: List[LoopIteration]) -> bool:
        """Determine if loop completed successfully."""
        if not iterations:
            return False

        last_iteration = iterations[-1]

        # Check last evaluation
        if last_iteration.evaluation:
            return last_iteration.evaluation.decision == EvaluationDecision.COMPLETE

        # Check fulfillment
        if last_iteration.fulfillment:
            return (
                hasattr(last_iteration.fulfillment, "is_fulfilled")
                and last_iteration.fulfillment.is_fulfilled
            )

        return False

    def _create_stategraph_executor(self) -> Any:
        """Create a configured StateGraph executor with shared service wiring."""
        global _AgenticLoopGraphExecutor

        # Lazy import to avoid circular dependencies
        if _AgenticLoopGraphExecutor is None:
            from victor.framework.agentic_graph.executor import (
                AgenticLoopGraphExecutor as _Executor,
            )

            _AgenticLoopGraphExecutor = _Executor

        execution_context = self._resolve_stategraph_execution_context()
        graph_executor = _AgenticLoopGraphExecutor(
            execution_context=execution_context,
            max_iterations=self.max_iterations,
            enable_fulfillment=self.enable_fulfillment_check,
            enable_adaptive_iterations=self.enable_adaptive_iterations,
        )

        if self.turn_executor is not None:
            graph_executor.turn_executor = self.turn_executor
        if self.runtime_intelligence is not None:
            graph_executor.runtime_intelligence = self.runtime_intelligence
        if hasattr(self.orchestrator, "planning_coordinator"):
            graph_executor.planning_coordinator = self.orchestrator.planning_coordinator
        if self.enhanced_completion_evaluator is not None:
            graph_executor.evaluator = self.enhanced_completion_evaluator
        if self.fulfillment is not None:
            graph_executor.fulfillment_detector = self.fulfillment

        return graph_executor

    @staticmethod
    def _reconstruct_perception_from_state(payload: Any) -> Optional[Perception]:
        """Rebuild a serializable Perception object from graph-state payload."""
        if not isinstance(payload, dict):
            return None

        from victor.agent.action_authorizer import ActionIntent
        from victor.agent.task_analyzer import TaskAnalysis
        from victor.framework.task import TaskComplexity

        intent_raw = payload.get("intent") or ActionIntent.AMBIGUOUS.value
        complexity_raw = payload.get("complexity") or TaskComplexity.MEDIUM.value
        confidence = float(payload.get("confidence") or 0.5)

        try:
            intent = (
                intent_raw if isinstance(intent_raw, ActionIntent) else ActionIntent(intent_raw)
            )
        except Exception:
            intent = ActionIntent.AMBIGUOUS

        try:
            complexity = (
                complexity_raw
                if isinstance(complexity_raw, TaskComplexity)
                else TaskComplexity(complexity_raw)
            )
        except Exception:
            complexity = TaskComplexity.MEDIUM

        task_analysis = TaskAnalysis(
            complexity=complexity,
            tool_budget=int(payload.get("tool_budget") or 10),
            complexity_confidence=confidence,
            task_type=payload.get("task_type"),
        )

        return Perception(
            intent=intent,
            complexity=complexity,
            task_analysis=task_analysis,
            confidence=confidence,
            metadata=dict(payload.get("metadata") or {}),
            needs_clarification=bool(payload.get("needs_clarification", False)),
            clarification_reason=payload.get("clarification_reason"),
            clarification_prompt=payload.get("clarification_prompt"),
        )

    @staticmethod
    def _reconstruct_evaluation_from_state(payload: Any) -> Optional[EvaluationResult]:
        """Rebuild EvaluationResult from serialized graph-state payload."""
        if not isinstance(payload, dict):
            return None

        decision = payload.get("decision", EvaluationDecision.CONTINUE.value)
        try:
            decision = (
                decision
                if isinstance(decision, EvaluationDecision)
                else EvaluationDecision(decision)
            )
        except Exception:
            decision = str(decision)

        return EvaluationResult(
            decision=decision,
            score=float(payload.get("score") or 0.5),
            reason=str(payload.get("reason") or ""),
            metrics=dict(payload.get("metrics") or {}),
            metadata=dict(payload.get("metadata") or {}),
        )

    def _loop_iteration_from_graph_event(
        self,
        node_name: str,
        state: Any,
    ) -> Optional[LoopIteration]:
        """Adapt a StateGraph node-complete event into legacy LoopIteration shape."""
        if node_name == "prompt":
            return None

        if hasattr(state, "to_dict"):
            state_payload = state.to_dict()
        elif isinstance(state, dict):
            state_payload = state
        else:
            return None

        stage_map = {
            "perceive": LoopStage.PERCEIVE,
            "plan": LoopStage.PLAN,
            "act": LoopStage.ACT,
            "evaluate": LoopStage.EVALUATE,
        }
        stage = stage_map.get(node_name)
        if stage is None:
            return None

        return LoopIteration(
            iteration=int(state_payload.get("iteration", 0) or 0),
            stage=stage,
            perception=self._reconstruct_perception_from_state(state_payload.get("perception")),
            plan=state_payload.get("plan"),
            action_result=state_payload.get("action_result"),
            evaluation=self._reconstruct_evaluation_from_state(state_payload.get("evaluation")),
            fulfillment=state_payload.get("fulfillment"),
        )

    def _loop_iterations_from_state_history(self, state_history: Any) -> List[LoopIteration]:
        """Consolidate graph node snapshots into legacy per-iteration results.

        Completed iterations are emitted when an ``evaluate`` snapshot arrives.
        If execution terminates mid-iteration (failure/interruption), the last
        partial iteration is preserved instead of being dropped.
        """
        if not isinstance(state_history, list):
            return []

        partial_iterations: Dict[int, LoopIteration] = {}
        completed_iterations: List[LoopIteration] = []

        for entry in state_history:
            if not isinstance(entry, dict):
                continue

            iteration_update = self._loop_iteration_from_graph_event(
                str(entry.get("node_name") or ""),
                entry.get("state"),
            )
            if iteration_update is None:
                continue

            iteration_id = iteration_update.iteration
            if iteration_id <= 0:
                continue

            iteration = partial_iterations.get(iteration_id)
            if iteration is None:
                iteration = LoopIteration(iteration=iteration_id, stage=iteration_update.stage)
                partial_iterations[iteration_id] = iteration

            iteration.stage = iteration_update.stage
            if iteration_update.perception is not None:
                iteration.perception = iteration_update.perception
            if iteration_update.plan is not None:
                iteration.plan = iteration_update.plan
            if iteration_update.action_result is not None:
                iteration.action_result = iteration_update.action_result
            if iteration_update.evaluation is not None:
                iteration.evaluation = iteration_update.evaluation
            if iteration_update.fulfillment is not None:
                iteration.fulfillment = iteration_update.fulfillment

            if iteration_update.stage == LoopStage.EVALUATE:
                iteration.stage = LoopStage.DECIDE
                completed_iterations.append(iteration)
                partial_iterations.pop(iteration_id, None)

        if partial_iterations:
            completed_iterations.extend(
                iteration for _, iteration in sorted(partial_iterations.items())
            )

        return completed_iterations

    def _stategraph_metadata(
        self,
        graph_result: Any,
        final_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build legacy-compatible metadata for the StateGraph execution path."""
        metadata = {
            "iterations_completed": graph_result.iterations,
            "max_iterations_reached": (graph_result.termination_reason == "max_iterations"),
            "effective_max_iterations": int(
                final_state.get("max_iterations", self.max_iterations) or self.max_iterations
            ),
            "termination_reason": graph_result.termination_reason,
            "progress_scores": list(final_state.get("progress_scores", [])),
            "planning_events": list(final_state.get("planning_events", [])),
            "planning_routing_hints": dict(final_state.get("planning_routing_hints", {})),
            "structured_routing_policy": dict(final_state.get("structured_routing_policy", {})),
            "topology_events": list(final_state.get("topology_events", [])),
            "degradation_events": list(final_state.get("degradation_events", [])),
            "executor_type": "stategraph",
        }

        error = getattr(graph_result, "error", None)
        if error:
            metadata["error"] = error

        return metadata

    async def _run_with_stategraph(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> LoopResult:
        """Run the agentic loop through the StateGraph-backed executor.

        This path is selected only when the framework-side StateGraph execution
        mode is explicitly enabled. The legacy ``AgenticLoop`` API remains the
        compatibility surface; this helper adapts ``AgenticLoopGraphExecutor``
        output into the historical ``LoopResult`` shape.

        Args:
            query: User's natural language query
            context: Additional execution context
            conversation_history: Prior turns for multi-turn context

        Returns:
            Compatibility ``LoopResult`` derived from the graph executor result.
        """
        import time

        start_time = time.time()

        try:
            graph_executor = self._create_stategraph_executor()

            # Run the graph
            graph_result = await graph_executor.run(
                query=query,
                context=context or {},
                conversation_history=conversation_history,
            )

            final_state = graph_result.metadata.get("final_state", {})

            iterations = self._loop_iterations_from_state_history(
                graph_result.metadata.get("state_history")
            )

            # Adapt the graph result into the historical LoopResult shape.
            # The graph executor exposes aggregate iteration counts rather than
            # the older per-iteration object list used by AgenticLoop, so we
            # reconstruct that list from the graph state's node snapshots.
            duration = time.time() - start_time
            return LoopResult(
                success=graph_result.success,
                iterations=iterations,
                final_state=final_state,
                total_duration=duration,
                metadata=self._stategraph_metadata(graph_result, final_state),
            )

        except Exception as e:
            logger.error(f"StateGraph executor error: {e}", exc_info=True)
            duration = time.time() - start_time
            return LoopResult(
                success=False,
                iterations=[],
                final_state={"query": query, "error": str(e)},
                total_duration=duration,
                metadata={
                    "error": str(e),
                    "executor_type": "stategraph",
                },
            )

    async def _stream_with_stategraph(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[LoopIteration]:
        """Stream agentic-loop iterations through the StateGraph executor."""
        graph_executor = self._create_stategraph_executor()

        async for event in graph_executor.stream(
            query=query,
            context=context or {},
            conversation_history=conversation_history,
        ):
            if event.get("event_type") == "error":
                raise RuntimeError(str(event.get("error") or "StateGraph stream failed"))

            iteration = self._loop_iteration_from_graph_event(
                str(event.get("node_name") or ""),
                event.get("state"),
            )
            if iteration is not None:
                yield iteration

    def _resolve_stategraph_execution_context(self) -> Any:
        """Resolve the explicit execution context for StateGraph execution."""
        from victor.runtime.context import resolve_execution_context

        execution_context = resolve_execution_context(self.orchestrator)
        if execution_context is None:
            return self.orchestrator

        try:
            from victor.runtime.context import RuntimeExecutionContext

            if not isinstance(execution_context, RuntimeExecutionContext):
                return execution_context
        except Exception:
            return execution_context

        prompt_orchestrator = getattr(self.orchestrator, "_prompt_orchestrator", None)
        if prompt_orchestrator is None:
            from victor.agent.prompt_orchestrator import get_prompt_orchestrator

            prompt_orchestrator = get_prompt_orchestrator()

        if execution_context.prompt_orchestrator is prompt_orchestrator:
            return execution_context
        return execution_context.with_prompt_orchestrator(prompt_orchestrator)
