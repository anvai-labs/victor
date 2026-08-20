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

"""Runtime-intelligence pipeline integrating all learning components.

This module provides a unified pipeline that integrates:
- IntelligentPromptBuilder: Embedding-based context selection, profile learning
- AdaptiveModeController: Q-learning for mode transitions
- ResponseQualityScorer: Multi-dimensional quality assessment
- GroundingVerifier: Hallucination detection
- ResilientExecutor: Circuit breaker, retry, rate limiting

Architecture (Facade + Chain of Responsibility patterns):

    ┌─────────────────────────────────────────────────────────┐
    │             RuntimeIntelligencePipeline                 │
    │   (Facade coordinating runtime-intelligence components) │
    └─────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
    ┌─────────┐        ┌─────────────┐      ┌──────────────┐
    │ Prompt  │        │    Mode     │      │  Response    │
    │ Builder │        │  Controller │      │  Validator   │
    └─────────┘        └─────────────┘      └──────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────┐        ┌─────────────┐      ┌──────────────┐
    │ Profile │        │  Q-Learning │      │   Quality    │
    │ Learner │        │    Store    │      │   Scorer     │
    └─────────┘        └─────────────┘      └──────────────┘

Usage:
    pipeline = await RuntimeIntelligencePipeline.create(
        provider_name="ollama",
        model="qwen2.5:32b",
        profile_name="local-qwen",
        project_root="/path/to/project",
    )

    # Pre-request: Build optimized prompt and get mode recommendation
    context = await pipeline.prepare_request(
        task="Analyze the auth module",
        task_type="analysis",
        current_mode="explore",
    )

    # Post-response: Validate, score, and learn from the result
    result = await pipeline.process_response(
        response="The auth module uses JWT...",
        tool_calls=5,
        success=True,
    )

    # Get pipeline stats
    stats = pipeline.get_stats()
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.providers.base import BaseProvider

from victor.agent.output_deduplicator import OutputDeduplicator
from victor.protocols.provider_adapter import get_provider_adapter

logger = logging.getLogger(__name__)

# Providers known to have repetition issues requiring deduplication
PROVIDERS_WITH_REPETITION_ISSUES = {"xai", "grok", "x-ai"}


@dataclass
class RequestContext:
    """Context prepared for a request."""

    system_prompt: str
    recommended_tool_budget: int
    recommended_mode: str
    should_continue: bool
    continuation_context: Optional[str] = None
    mode_confidence: float = 0.5
    profile_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseResult:
    """Result of processing a response."""

    is_valid: bool
    quality_score: float
    grounding_score: float
    is_grounded: bool
    quality_details: Dict[str, float] = field(default_factory=dict)
    grounding_issues: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    learning_reward: float = 0.0
    # Grounding failure handling
    should_finalize: bool = False
    """True when grounding retries exceeded - finalize with best-effort response."""
    should_retry: bool = False
    """True when grounding failed but retries remain."""
    finalize_reason: str = ""
    """Reason for forced finalization (e.g., 'grounding failure limit exceeded')."""
    grounding_feedback: str = ""
    """Actionable feedback prompt for correcting grounding issues on retry."""
    grounding_verified_references: List[str] = field(default_factory=list)
    """References directly verified by grounding checks."""
    grounding_unverified_references: List[str] = field(default_factory=list)
    """References that grounding checks could not verify."""
    grounding_evidence_summary: str = ""
    """Concise summary of verified/unverified evidence for downstream UX."""


@dataclass
class PipelineStats:
    """Statistics for the runtime-intelligence pipeline."""

    total_requests: int = 0
    successful_requests: int = 0
    avg_quality_score: float = 0.0
    avg_grounding_score: float = 0.0
    total_learning_reward: float = 0.0
    mode_transitions: int = 0
    circuit_breaker_trips: int = 0
    retry_attempts: int = 0
    cache_state: str = "cold"
    profile_name: str = ""


class RuntimeIntelligencePipeline:
    """Unified pipeline integrating all runtime-intelligence components.

    This facade coordinates:
    1. Pre-request: Prompt building, mode selection, budget optimization
    2. Execution: Resilient provider calls with circuit breaker
    3. Post-response: Quality scoring, grounding verification, learning
    """

    def __init__(
        self,
        provider_name: str,
        model: str,
        profile_name: str,
        project_root: Optional[str] = None,
    ):
        """Initialize the runtime-intelligence pipeline.

        Args:
            provider_name: LLM provider name
            model: Model name/identifier
            profile_name: Profile name for learning tracking
            project_root: Project root for grounding verification
        """
        self.provider_name = provider_name
        self.model = model
        self.profile_name = profile_name
        self.project_root = project_root

        # Components (initialized lazily)
        self._prompt_builder = None
        self._mode_controller = None
        self._quality_scorer = None
        self._grounding_verifier = None
        self._resilient_executor = None
        self._output_deduplicator: Optional[OutputDeduplicator] = None

        # State tracking
        self._current_context: Optional[RequestContext] = None
        self._session_start = datetime.now()
        self._stats = PipelineStats(profile_name=profile_name)

        # Grounding failure handling
        self._grounding_failure_count: int = 0
        self._max_grounding_retries: int = 1  # Only 1 retry, then finalize

        # Provider adapter for capability-based behavior
        self._provider_adapter = get_provider_adapter(provider_name)

        # Provider-aware deduplication
        self._deduplication_enabled = self._provider_adapter.capabilities.output_deduplication

        # Observers for feedback
        self._observers: List[Callable[[ResponseResult], None]] = []

    @classmethod
    async def create(
        cls,
        provider_name: str,
        model: str,
        profile_name: Optional[str] = None,
        project_root: Optional[str] = None,
    ) -> "RuntimeIntelligencePipeline":
        """Factory method to create and initialize the pipeline.

        Args:
            provider_name: LLM provider name
            model: Model name
            profile_name: Optional profile name
            project_root: Optional project root

        Returns:
            Initialized RuntimeIntelligencePipeline
        """
        profile = profile_name or f"{provider_name}:{model}"
        pipeline = cls(
            provider_name=provider_name,
            model=model,
            profile_name=profile,
            project_root=project_root,
        )
        # Skip eager initialization - use lazy loading instead
        return pipeline

    async def _get_prompt_builder(self):
        """Lazy initialize prompt builder."""
        if self._prompt_builder is None:
            try:
                import traceback

                logger.debug(
                    f"[RuntimeIntelligencePipeline] Creating prompt builder with: "
                    f"provider_name={self.provider_name!r} (type={type(self.provider_name).__name__}), "
                    f"model={self.model!r} (type={type(self.model).__name__}), "
                    f"profile_name={self.profile_name!r} (type={type(self.profile_name).__name__})"
                )
                from victor.agent.intelligent_prompt_builder import (
                    IntelligentPromptBuilder,
                )

                self._prompt_builder = await IntelligentPromptBuilder.create(
                    self.provider_name, self.model, self.profile_name
                )
            except Exception as e:
                logger.warning(
                    f"[RuntimeIntelligencePipeline] Prompt builder init failed: {e}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
        return self._prompt_builder

    def _get_mode_controller(self):
        """Lazy initialize mode controller (sync)."""
        if self._mode_controller is None:
            try:
                from victor.agent.adaptive_mode_controller import AdaptiveModeController

                # Get ModeTransitionLearner from RLCoordinator for unified RL
                mode_transition_learner = None
                try:
                    from victor.agent.services.rl_runtime import get_rl_coordinator

                    coordinator = get_rl_coordinator()
                    mode_transition_learner = coordinator.get_learner("mode_transition")
                except Exception as e:
                    logger.debug(
                        f"[RuntimeIntelligencePipeline] Could not get mode_transition learner: {e}"
                    )

                self._mode_controller = AdaptiveModeController(
                    profile_name=self.profile_name,
                    provider_name=self.provider_name,
                    model_name=self.model,
                    provider_adapter=self._provider_adapter,
                    mode_transition_learner=mode_transition_learner,
                )
            except Exception as e:
                logger.warning(f"[RuntimeIntelligencePipeline] Mode controller init failed: {e}")
        return self._mode_controller

    def get_provider_quality_thresholds(self) -> dict:
        """Get provider-specific quality thresholds.

        Returns:
            Dict with 'min_quality' and 'grounding_threshold' for current provider
        """
        # Use provider adapter's capabilities first
        if self._provider_adapter:
            caps = self._provider_adapter.capabilities
            return {
                "min_quality": caps.quality_threshold,
                "grounding_threshold": caps.grounding_strictness,
            }

        # Fall back to mode controller
        controller = self._get_mode_controller()
        if controller:
            return controller.get_quality_thresholds()

        # Default fallback
        return {"min_quality": 0.70, "grounding_threshold": 0.65}

    async def _get_quality_scorer(self):
        """Lazy initialize quality scorer."""
        if self._quality_scorer is None:
            try:
                from victor.agent.response_quality import ResponseQualityScorer

                self._quality_scorer = ResponseQualityScorer()
            except Exception as e:
                logger.warning(f"[RuntimeIntelligencePipeline] Quality scorer init failed: {e}")
        return self._quality_scorer

    async def _get_grounding_verifier(self):
        """Lazy initialize grounding verifier."""
        if self._grounding_verifier is None and self.project_root:
            try:
                from victor.agent.grounding_verifier import GroundingVerifier

                # Get GroundingThresholdLearner from RLCoordinator for adaptive thresholds
                grounding_threshold_learner = None
                try:
                    from victor.agent.services.rl_runtime import get_rl_coordinator

                    coordinator = get_rl_coordinator()
                    grounding_threshold_learner = coordinator.get_learner("grounding_threshold")
                except Exception as e:
                    logger.debug(
                        f"[RuntimeIntelligencePipeline] Could not get grounding_threshold learner: {e}"
                    )

                self._grounding_verifier = GroundingVerifier(
                    project_root=self.project_root,
                    provider_adapter=self._provider_adapter,
                    grounding_threshold_learner=grounding_threshold_learner,
                )
            except Exception as e:
                logger.warning(f"[RuntimeIntelligencePipeline] Grounding verifier init failed: {e}")
        return self._grounding_verifier

    def _emit_grounding_event(
        self,
        is_grounded: bool,
        grounding_score: float,
        task_type: str,
    ) -> None:
        """Emit RL event for grounding verification.

        This activates the grounding_threshold learner to learn optimal
        thresholds based on verification outcomes.

        Args:
            is_grounded: Whether the response passed grounding verification
            grounding_score: Confidence score from grounding verifier
            task_type: Type of task being verified
        """
        try:
            from victor.framework.rl.hooks import get_rl_hooks, RLEvent, RLEventType

            hooks = get_rl_hooks()
            if hooks is None:
                return

            event = RLEvent(
                type=RLEventType.GROUNDING_CHECK,
                success=is_grounded,
                quality_score=grounding_score,
                provider=self.provider_name,
                model=self.model,
                task_type=task_type,
                threshold_value=grounding_score,
                metadata={
                    "is_grounded": is_grounded,
                },
            )

            hooks.emit(event)

        except Exception as e:
            logger.debug(f"[RuntimeIntelligencePipeline] Grounding event emission failed: {e}")

    @staticmethod
    def _is_read_heavy_grounding_task(task_type: str) -> bool:
        """Return whether grounding retries should stay conservative for this task."""
        normalized = str(task_type or "").strip().lower()
        return normalized in {"analysis", "search", "documentation", "research"}

    @staticmethod
    def _has_reference_only_grounding_issues(grounding_result: Any) -> bool:
        """Return whether grounding issues are limited to path/symbol hygiene."""
        issues = getattr(grounding_result, "issues", None) or []
        if not issues:
            return False

        reference_issue_types = {
            "file_not_found",
            "symbol_not_found",
            "path_invalid",
            "unverifiable",
        }

        for issue in issues:
            raw_issue_type = getattr(issue, "issue_type", None)
            issue_value = getattr(raw_issue_type, "value", raw_issue_type)
            if str(issue_value or "").strip().lower() not in reference_issue_types:
                return False

        return True

    def _should_soft_finalize_grounding_retry(
        self,
        *,
        grounding_result: Any,
        response: str,
        quality_score: float,
        tool_calls: int,
        task_type: str,
    ) -> bool:
        """Return whether to keep a substantial answer instead of forcing retry."""
        if not self._is_read_heavy_grounding_task(task_type):
            return False

        if not self._has_reference_only_grounding_issues(grounding_result):
            return False

        response_length = len((response or "").strip())
        return response_length >= 600 or quality_score >= 0.75 or tool_calls >= 3

    @staticmethod
    def _build_grounding_evidence_summary(
        verified_references: List[str],
        unverified_references: List[str],
    ) -> str:
        """Build a compact grounding evidence summary for UX and telemetry."""
        lines: List[str] = []
        if verified_references:
            lines.append(f"- Verified: {', '.join(verified_references[:8])}")
        if unverified_references:
            lines.append(f"- Unverified: {', '.join(unverified_references[:8])}")
        if not lines:
            return ""
        return "Grounding evidence:\n" + "\n".join(lines)

    async def _get_resilient_executor(self):
        """Lazy initialize resilient executor."""
        if self._resilient_executor is None:
            try:
                from victor.agent.resilience import ResilientExecutor

                self._resilient_executor = ResilientExecutor()
            except Exception as e:
                logger.warning(f"[RuntimeIntelligencePipeline] Resilient executor init failed: {e}")
        return self._resilient_executor

    def _get_output_deduplicator(self) -> OutputDeduplicator:
        """Lazy initialize output deduplicator."""
        if self._output_deduplicator is None:
            self._output_deduplicator = OutputDeduplicator(
                min_block_length=50,
                normalize_whitespace=True,
            )
        return self._output_deduplicator

    def _should_enable_deduplication(self) -> bool:
        """Check if deduplication should be enabled for current provider.

        Returns:
            True if the provider is known to have repetition issues, False otherwise
        """
        provider_lower = self.provider_name.lower()
        return provider_lower in PROVIDERS_WITH_REPETITION_ISSUES

    def deduplicate_response(self, response: str) -> tuple[str, dict]:
        """Apply deduplication to response if enabled for provider.

        This method follows the Strategy Pattern - the deduplication strategy
        is selected based on the provider configuration.

        Args:
            response: Raw response from provider

        Returns:
            Tuple of (deduplicated_response, stats_dict)
        """
        if not self._deduplication_enabled or not response:
            return response, {"deduplication_applied": False}

        dedup = self._get_output_deduplicator()
        dedup.reset()  # Reset for fresh processing

        deduplicated = dedup.process(response)
        stats = dedup.get_stats()
        stats["deduplication_applied"] = True
        stats["provider"] = self.provider_name

        if stats.get("duplicates_removed", 0) > 0:
            logger.info(
                f"[RuntimeIntelligencePipeline] Deduplication for {self.provider_name}: "
                f"removed {stats['duplicates_removed']} duplicates, "
                f"saved ~{stats['bytes_saved']} bytes"
            )

        return deduplicated, stats

    async def prepare_request(
        self,
        task: str,
        task_type: str = "general",
        current_mode: str = "explore",
        tool_calls_made: int = 0,
        tool_budget: int = 10,
        iteration_count: int = 0,
        iteration_budget: int = 20,
        quality_score: float = 0.5,
        session_id: Optional[str] = None,
        continuation_context: Optional[str] = None,
    ) -> RequestContext:
        """Prepare context for a request.

        Uses intelligent components to:
        1. Build an optimized system prompt
        2. Get mode transition recommendation
        3. Calculate optimal tool budget

        Args:
            task: Current task/query
            task_type: Detected task type
            current_mode: Current agent mode
            tool_calls_made: Tool calls already made
            tool_budget: Total tool budget
            iteration_count: Current iteration
            iteration_budget: Total iteration budget
            quality_score: Current quality score
            session_id: Session for context retrieval
            continuation_context: Context from continuation

        Returns:
            RequestContext with optimized settings
        """
        start_time = time.perf_counter()

        # Build intelligent prompt (lazy init)
        system_prompt = ""
        if await self._get_prompt_builder():
            system_prompt = await self._prompt_builder.build(
                task=task,
                task_type=task_type,
                current_mode=current_mode,
                tool_budget=tool_budget,
                iteration_budget=iteration_budget,
                session_id=session_id,
                continuation_context=continuation_context,
            )

        # Get mode recommendation
        recommended_mode = current_mode
        mode_confidence = 0.5
        should_continue = True
        recommended_budget = tool_budget

        if self._get_mode_controller():
            action = self._mode_controller.get_recommended_action(
                current_mode=current_mode,
                task_type=task_type,
                tool_calls_made=tool_calls_made,
                tool_budget=tool_budget,
                iteration_count=iteration_count,
                iteration_budget=iteration_budget,
                quality_score=quality_score,
            )
            recommended_mode = action.target_mode.value
            mode_confidence = action.confidence
            should_continue = action.should_continue
            recommended_budget = tool_budget + action.adjust_tool_budget

            # Get learned optimal budget for task type - used as GUIDANCE not hard cap
            # The RL system learns from outcomes:
            # - Success with efficiency → gradually decreases learned budget
            # - Failure/low quality → increases learned budget
            # - Budget exhaustion failures → strongly increases learned budget
            #
            # We use learned budget to potentially INCREASE the recommendation (if
            # history shows this task type needs more), but NEVER decrease below
            # the user's original tool_budget.
            learned_budget = self._mode_controller.get_optimal_tool_budget(task_type)

            # If learned budget suggests we need MORE than user specified, recommend
            # a slight increase (bounded). This helps tasks that historically need more.
            if learned_budget > tool_budget:
                # Recommend up to 20% more, but cap at learned budget
                suggested_increase = min(learned_budget, int(tool_budget * 1.2))
                recommended_budget = max(recommended_budget, suggested_increase)
                logger.debug(
                    f"[RuntimeIntelligencePipeline] RL suggests higher budget for {task_type}: "
                    f"learned={learned_budget}, recommending={recommended_budget}"
                )

            # Floor: never go below user's original budget or minimum viable
            min_budget = max(15, tool_budget)
            recommended_budget = max(recommended_budget, min_budget)

        # Get profile stats
        profile_stats = {}
        if await self._get_prompt_builder():
            profile_stats = self._prompt_builder.get_profile_stats()

        context = RequestContext(
            system_prompt=system_prompt,
            recommended_tool_budget=recommended_budget,
            recommended_mode=recommended_mode,
            should_continue=should_continue,
            continuation_context=continuation_context,
            mode_confidence=mode_confidence,
            profile_stats=profile_stats,
        )

        self._current_context = context
        self._stats.total_requests += 1

        elapsed = time.perf_counter() - start_time
        logger.debug(
            f"[RuntimeIntelligencePipeline] Prepared request in {elapsed*1000:.1f}ms: "
            f"mode={recommended_mode}, budget={recommended_budget}"
        )

        return context

    async def process_response(
        self,
        response: str,
        query: str = "",
        tool_calls: int = 0,
        tool_budget: int = 10,
        success: bool = True,
        task_type: str = "general",
        context: Optional[Dict[str, Any]] = None,
    ) -> ResponseResult:
        """Process and validate a response.

        Uses intelligent components to:
        1. Score response quality
        2. Verify grounding
        3. Record feedback for learning

        Args:
            response: The model's response
            query: Original query
            tool_calls: Number of tool calls made
            tool_budget: Tool budget
            success: Whether the response was successful
            task_type: Task type
            context: Additional context for verification

        Returns:
            ResponseResult with quality and grounding info
        """
        start_time = time.perf_counter()

        # Step 0: Apply deduplication for providers with repetition issues
        # This follows the Decorator pattern - wrapping the response processing
        deduplicated_response, dedup_stats = self.deduplicate_response(response)
        if dedup_stats.get("deduplication_applied"):
            # Use deduplicated response for subsequent processing
            response = deduplicated_response

        # Score quality (lazy init)
        quality_score = 0.5
        quality_details = {}
        improvement_suggestions = []

        if await self._get_quality_scorer() and query:
            quality_result = await self._quality_scorer.score(
                query=query,
                response=response,
                context=context,
            )
            quality_score = quality_result.overall_score
            quality_details = {
                dim.dimension.value: dim.score for dim in quality_result.dimension_scores
            }
            improvement_suggestions = quality_result.improvement_suggestions

        # Verify grounding (lazy init)
        grounding_score = 1.0
        is_grounded = True
        grounding_issues = []
        grounding_result = None  # Store full result for feedback generation
        grounding_verified_references: List[str] = []
        grounding_unverified_references: List[str] = []
        grounding_evidence_summary = ""

        if await self._get_grounding_verifier():
            grounding_result = await self._grounding_verifier.verify(
                response=response,
                context=context,
            )
            grounding_score = grounding_result.confidence
            is_grounded = grounding_result.is_grounded
            grounding_issues = [
                f"{issue.issue_type.value}: {issue.description}"
                for issue in grounding_result.issues
            ]
            grounding_verified_references = list(
                getattr(grounding_result, "verified_references", []) or []
            )
            grounding_unverified_references = list(
                getattr(grounding_result, "unverified_references", []) or []
            )
            grounding_evidence_summary = self._build_grounding_evidence_summary(
                grounding_verified_references,
                grounding_unverified_references,
            )

            # Emit RL event for grounding verification
            self._emit_grounding_event(
                is_grounded=is_grounded,
                grounding_score=grounding_score,
                task_type=task_type,
            )

        # Track grounding failures for best-effort finalize logic
        should_finalize = False
        should_retry = False
        finalize_reason = ""
        grounding_feedback = ""

        if not is_grounded:
            self._grounding_failure_count += 1
            if self._grounding_failure_count > self._max_grounding_retries:
                # Max retries exceeded - force finalize with best-effort response
                should_finalize = True
                finalize_reason = "grounding failure limit exceeded"
                logger.debug(
                    "[RuntimeIntelligencePipeline] Grounding failure count (%d) "
                    "exceeded max retries (%d). Forcing best-effort finalize.",
                    self._grounding_failure_count,
                    self._max_grounding_retries,
                )
            elif grounding_result is not None and self._should_soft_finalize_grounding_retry(
                grounding_result=grounding_result,
                response=response,
                quality_score=quality_score,
                tool_calls=tool_calls,
                task_type=task_type,
            ):
                should_finalize = True
                finalize_reason = "soft grounding finalize for substantial read-heavy answer"
                grounding_feedback = grounding_result.generate_feedback_prompt()
                logger.info(
                    "[RuntimeIntelligencePipeline] Soft-finalizing read-heavy response with "
                    "reference-only grounding issues instead of retrying hidden correction turn."
                )
            else:
                # Can still retry - generate actionable feedback prompt
                should_retry = True
                # Generate feedback from grounding result if available
                if grounding_result is not None:
                    grounding_feedback = grounding_result.generate_feedback_prompt()
                    if grounding_feedback:
                        logger.info(
                            "[RuntimeIntelligencePipeline] Generated grounding feedback for "
                            f"retry: "
                            f"{len(grounding_feedback)} chars"
                        )
                logger.debug(
                    f"[RuntimeIntelligencePipeline] Grounding failed, retry allowed "
                    f"({self._grounding_failure_count}/{self._max_grounding_retries})"
                )
        else:
            # Grounding succeeded - reset counter
            if self._grounding_failure_count > 0:
                logger.debug(
                    "[RuntimeIntelligencePipeline] Grounding succeeded, resetting "
                    f"failure count "
                    f"from {self._grounding_failure_count} to 0"
                )
            self._grounding_failure_count = 0

        # Record feedback for learning
        learning_reward = 0.0
        response_time_ms = (time.perf_counter() - start_time) * 1000

        if await self._get_prompt_builder():
            self._prompt_builder.record_feedback(
                task_type=task_type,
                success=success,
                quality_score=quality_score,
                response_time_ms=response_time_ms,
                tool_calls=tool_calls,
                tool_budget=tool_budget,
                grounded=is_grounded,
            )

        if self._get_mode_controller():
            learning_reward = self._mode_controller.record_outcome(
                success=success,
                quality_score=quality_score,
                user_satisfied=quality_score > 0.6,
                completed=success and quality_score > 0.7,
            )

        # Update stats efficiently
        if success:
            self._stats.successful_requests += 1

        # Use incremental average to avoid floating point drift
        # Guard against division by zero when process_response is called without prepare_request
        n = self._stats.total_requests
        if n == 0:
            # First response without prepare_request - initialize stats
            self._stats.total_requests = 1
            n = 1
        self._stats.avg_quality_score += (quality_score - self._stats.avg_quality_score) / n
        self._stats.avg_grounding_score += (grounding_score - self._stats.avg_grounding_score) / n
        self._stats.total_learning_reward += learning_reward

        result = ResponseResult(
            is_valid=success and is_grounded,
            quality_score=quality_score,
            grounding_score=grounding_score,
            is_grounded=is_grounded,
            quality_details=quality_details,
            grounding_issues=grounding_issues,
            improvement_suggestions=improvement_suggestions,
            learning_reward=learning_reward,
            should_finalize=should_finalize,
            should_retry=should_retry,
            finalize_reason=finalize_reason,
            grounding_feedback=grounding_feedback,
            grounding_verified_references=grounding_verified_references,
            grounding_unverified_references=grounding_unverified_references,
            grounding_evidence_summary=grounding_evidence_summary,
        )

        # Notify observers (skip if none)
        if self._observers:
            for observer in self._observers:
                try:
                    observer(result)
                except Exception as e:
                    logger.warning(f"[RuntimeIntelligencePipeline] Observer error: {e}")

        elapsed = time.perf_counter() - start_time
        logger.debug(
            f"[RuntimeIntelligencePipeline] Processed response in {elapsed*1000:.1f}ms: "
            f"quality={quality_score:.2f}, grounded={is_grounded}"
        )

        return result

    async def execute_with_resilience(
        self,
        provider: "BaseProvider",
        messages: List[Dict[str, Any]],
        circuit_name: Optional[str] = None,
        fallback: Optional[Callable] = None,
        **kwargs,
    ) -> Any:
        """Execute a provider call with resilience (circuit breaker, retry).

        Args:
            provider: The LLM provider
            messages: Messages to send
            circuit_name: Circuit breaker name (default: provider name)
            fallback: Optional fallback function
            **kwargs: Additional provider kwargs

        Returns:
            Provider response

        Raises:
            CircuitOpenError: If circuit is open and no fallback
            Exception: If retries exhausted and no fallback
        """
        if not await self._get_resilient_executor():
            # No resilience - direct call
            return await provider.chat(messages=messages, **kwargs)

        circuit = circuit_name or f"{self.provider_name}:{self.model}"

        async def call_provider():
            return await provider.chat(messages=messages, **kwargs)

        async def fallback_func():
            if fallback:
                return await fallback()
            raise RuntimeError("No fallback provided")

        try:
            result = await self._resilient_executor.execute(
                circuit,  # name (positional)
                call_provider,  # func (positional)
                fallback=fallback_func if fallback else None,
            )
            return result
        except Exception:
            self._stats.circuit_breaker_trips += 1
            raise

    def should_continue(
        self,
        tool_calls_made: int,
        tool_budget: int,
        quality_score: float,
        iteration_count: int,
        iteration_budget: int,
    ) -> tuple[bool, str]:
        """Determine if processing should continue.

        Args:
            tool_calls_made: Tools calls made
            tool_budget: Tool budget
            quality_score: Current quality score
            iteration_count: Iterations completed
            iteration_budget: Iteration budget

        Returns:
            Tuple of (should_continue, reason)
        """
        if self._get_mode_controller():
            return self._mode_controller.should_continue(
                tool_calls_made=tool_calls_made,
                tool_budget=tool_budget,
                quality_score=quality_score,
                iteration_count=iteration_count,
                iteration_budget=iteration_budget,
            )

        # Fallback logic
        if tool_calls_made >= tool_budget:
            return False, "Tool budget exhausted"
        if iteration_count >= iteration_budget:
            return False, "Iteration budget exhausted"
        if quality_score > 0.85:
            return False, "High quality achieved"
        return True, "Continue processing"

    def add_observer(self, observer: Callable[[ResponseResult], None]) -> None:
        """Add observer for response results."""
        self._observers.append(observer)

    def remove_observer(self, observer: Callable[[ResponseResult], None]) -> None:
        """Remove observer."""
        if observer in self._observers:
            self._observers.remove(observer)

    def get_stats(self) -> PipelineStats:
        """Get pipeline statistics."""
        # Lazy update stats only when requested (sync access for existing components)
        if self._prompt_builder:
            self._stats.cache_state = self._prompt_builder.get_profile_stats().get(
                "cache_state", "unknown"
            )
        if self._mode_controller:
            self._stats.mode_transitions = self._mode_controller.get_session_stats().get(
                "mode_transitions", 0
            )
        return self._stats

    def get_optimal_tool_budget(self, task_type: str) -> int:
        """Get the learned optimal tool budget for a task type.

        Args:
            task_type: The type of task to get the budget for

        Returns:
            The learned optimal tool budget, or 0 if mode learning is not enabled
        """
        controller = self._get_mode_controller()
        if controller:
            return controller.get_optimal_tool_budget(task_type)
        return 0

    def reset_session(self) -> None:
        """Reset session tracking."""
        self._session_start = datetime.now()
        self._current_context = None
        self._stats = PipelineStats(profile_name=self.profile_name)

        if self._mode_controller:
            self._mode_controller.reset_session()

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learned behaviors."""
        summary = {
            "profile_name": self.profile_name,
            "total_requests": self._stats.total_requests,
            "success_rate": (self._stats.successful_requests / max(self._stats.total_requests, 1)),
        }

        if self._prompt_builder:
            summary["prompt_profile"] = self._prompt_builder.get_profile_stats()
        if self._mode_controller:
            summary["mode_session"] = self._mode_controller.get_session_stats()

        return summary


# Module-level convenience functions
_runtime_intelligence_pipeline_cache: Dict[tuple, RuntimeIntelligencePipeline] = {}


async def get_runtime_intelligence_pipeline(
    provider_name: str,
    model: str,
    profile_name: Optional[str] = None,
    project_root: Optional[str] = None,
) -> RuntimeIntelligencePipeline:
    """Get or create a runtime-intelligence pipeline instance.

    Args:
        provider_name: Provider name
        model: Model name
        profile_name: Optional profile name
        project_root: Optional project root

    Returns:
        RuntimeIntelligencePipeline instance
    """
    key = (provider_name, model, profile_name or "default", project_root)

    if key not in _runtime_intelligence_pipeline_cache:
        _runtime_intelligence_pipeline_cache[key] = await RuntimeIntelligencePipeline.create(
            provider_name=provider_name,
            model=model,
            profile_name=profile_name,
            project_root=project_root,
        )

    return _runtime_intelligence_pipeline_cache[key]


def clear_runtime_intelligence_pipeline_cache() -> None:
    """Clear the runtime-intelligence pipeline cache."""
    _runtime_intelligence_pipeline_cache.clear()
