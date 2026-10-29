"""Tiered decision service — routes DecisionTypes to different model tiers.

Provider-Agnostic Design:
- Auto-detects active provider from orchestrator
- Resolves model from provider_model_tiers mapping
- Ensures decision service uses same provider as main LLM
- Supports explicit overrides via tier_overrides

Follows the GEPATierManager pattern: edge (fast local), balanced (mid-tier cloud),
performance (frontier). Each decision type is mapped to a tier via settings.
Fallback chain: performance → balanced → edge → heuristic.

Usage:
    from victor.agent.services.tiered_decision_service import TieredDecisionService
    from victor.config.decision_settings import DecisionServiceSettings

    config = DecisionServiceSettings()
    service = TieredDecisionService(config)
    result = service.decide_sync(DecisionType.TOOL_SELECTION, {"message": "fix test"})
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Set

from victor.agent.decisions.schemas import DecisionType
from victor.agent.services.protocols.decision_service import DecisionResult
from victor.config.decision_settings import DecisionServiceSettings
from victor.core.runtime_evaluation_policy import (
    RuntimeEvaluationFeedback,
    RuntimeEvaluationPolicy,
)
from victor.framework.session_config import LOCAL_ENDPOINT_PROVIDERS

logger = logging.getLogger(__name__)

FALLBACK_CHAIN = {
    "performance": ["balanced", "edge"],
    "balanced": ["edge"],
    "edge": [],
}


# =============================================================================
# Classification Triage Data Structures
# =============================================================================


class ClassificationTriage(str, Enum):
    """Classification triage outcome.

    Determines how to handle a classification result based on confidence:
    - ACCEPT: High confidence (≥0.8), use immediately (fast path)
    - VERIFY: Medium confidence (0.5-0.8), verify with edge LLM
    - REJECT: Low confidence (<0.5), reject early (avoid waste)
    """

    ACCEPT = "accept"  # High confidence, use immediately
    VERIFY = "verify"  # Medium confidence, verify with edge LLM
    REJECT = "reject"  # Low confidence, reject early


@dataclass
class EdgeLLMVerificationResult:
    """Result of edge LLM verification for grey-area classifications.

    When a classification falls in the VERIFY band (0.5-0.8 confidence),
    the edge LLM is consulted to verify or correct the classification.
    """

    original_result: Any  # Original classification result
    original_confidence: float  # Original confidence score
    verified_result: Any  # Verified/corrected result from edge LLM
    verification_confidence: float  # Confidence of the verification
    verification_passed: bool  # Whether verification passed (confidence ≥ high threshold)
    latency_ms: float  # Verification latency in milliseconds
    tokens_used: int  # Tokens used for verification

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "original_result": str(self.original_result),
            "original_confidence": self.original_confidence,
            "verified_result": str(self.verified_result),
            "verification_confidence": self.verification_confidence,
            "verification_passed": self.verification_passed,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
        }


@dataclass
class TieredClassificationResult:
    """Result of tiered classification with confidence-based triage.

    Combines base classification with optional edge LLM verification
    based on confidence bands (ACCEPT, VERIFY, REJECT).
    """

    result: Any  # The classification result (task type, intent, etc.)
    confidence: float  # Final confidence score (0.0-1.0)
    triage_outcome: ClassificationTriage  # Triage decision
    source: str  # "keyword", "semantic", "llm", "edge_verification"
    verification_result: Optional[EdgeLLMVerificationResult] = None  # Present if VERIFY
    latency_ms: float = 0.0  # Total latency including verification
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "result": str(self.result),
            "confidence": self.confidence,
            "triage_outcome": self.triage_outcome.value,
            "source": self.source,
            "verification_result": (
                self.verification_result.to_dict() if self.verification_result else None
            ),
            "latency_ms": self.latency_ms,
            "metadata": dict(self.metadata),
        }


class TieredDecisionService:
    """Decision service with per-DecisionType tier routing.

    Provider-Agnostic Behavior:
    - Auto-detects active provider from orchestrator/container
    - Resolves model from provider_model_tiers mapping
    - Falls back gracefully when provider or model unavailable
    - Supports tier_overrides for explicit provider/model selection

    Maintains a pool of LLMDecisionService instances (one per tier).
    Routes each decision type to the appropriate tier via settings.
    Falls back through the chain when a tier is unavailable.
    """

    def __init__(self, config: DecisionServiceSettings) -> None:
        self._config = config
        self._services: Dict[str, Any] = {}  # tier name → LLMDecisionService
        self._tier_routing = dict(config.tier_routing)
        self._failed_tiers: Set[str] = set()
        self._detected_provider: Optional[str] = None  # Cached provider detection

    @staticmethod
    def _normalize_provider_name(provider_name: Optional[str]) -> Optional[str]:
        """Normalize provider names and ignore placeholder values."""
        if not provider_name:
            return None
        normalized = provider_name.strip().lower()
        if normalized in {"", "auto", "none", "unknown"}:
            return None
        return normalized

    def _cache_detected_provider(
        self,
        provider_name: Optional[str],
        *,
        source: str,
    ) -> Optional[str]:
        """Persist a detected provider if it is meaningful."""
        normalized = self._normalize_provider_name(provider_name)
        if normalized is None:
            return None
        self._detected_provider = normalized
        logger.debug(f"Auto-detected provider from {source}: {normalized}")
        return normalized

    def _detect_provider_from_service_container(self) -> Optional[str]:
        """Detect the active provider from DI-managed runtime services."""
        try:
            from victor.agent.services.protocols import ProviderServiceProtocol
            from victor.core import get_container

            container = get_container()

            provider_service = container.get_optional(ProviderServiceProtocol)
            if provider_service is not None:
                try:
                    provider_info = provider_service.get_current_provider_info()
                except Exception:
                    provider_info = None

                provider_name = self._cache_detected_provider(
                    getattr(provider_info, "provider_name", None),
                    source="ProviderServiceProtocol.get_current_provider_info()",
                )
                if provider_name:
                    return provider_name

                provider_name = self._cache_detected_provider(
                    getattr(provider_service, "provider_name", None),
                    source="ProviderServiceProtocol.provider_name",
                )
                if provider_name:
                    return provider_name

            # Try ProviderManager (legacy)
            try:
                from victor.agent.provider_manager import ProviderManager

                provider_mgr = container.get_optional(ProviderManager)
                if provider_mgr is not None:
                    provider_name = self._cache_detected_provider(
                        getattr(provider_mgr, "provider_name", None),
                        source="ProviderManager",
                    )
                    if provider_name:
                        return provider_name
            except Exception:
                pass

        except Exception:
            pass

        return None

    def _detect_active_provider(self) -> Optional[str]:
        """Detect the active provider from the orchestrator.

        Checks multiple sources in order:
        1. Service container (ProviderManager/ProviderService)
        2. Settings.default_provider
        3. Environment variables (VICTOR_PROVIDER)

        Returns:
            Provider name (e.g., "anthropic", "openai") or None
        """
        # Return cached provider if available
        if self._detected_provider:
            return self._detected_provider

        # Try to get from service container
        provider_name = self._detect_provider_from_service_container()
        if provider_name:
            return provider_name

        # Try settings
        try:
            from victor.config.settings import Settings

            settings = Settings()
            default_provider = getattr(
                getattr(settings, "provider", None), "default_provider", None
            )
            provider_name = self._cache_detected_provider(
                default_provider,
                source="settings",
            )
            if provider_name:
                return provider_name
        except Exception:
            pass

        # Try environment variable
        try:
            provider_name = self._cache_detected_provider(
                os.getenv("VICTOR_PROVIDER"),
                source="env var",
            )
            if provider_name:
                return provider_name
        except Exception:
            pass

        logger.debug("Could not auto-detect active provider (normal at bootstrap)")
        return None

    def decide_sync(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> DecisionResult:
        """Route to the correct tier's decision service.

        Args:
            decision_type: What kind of decision to make
            context: Context dict for the decision
            heuristic_result: Fallback result if no LLM available
            heuristic_confidence: Confidence of the heuristic result
        """
        tier = self._tier_routing.get(decision_type.value, "edge")

        # Handle "auto" routing for COMPACTION decision type
        # Auto-route based on complexity: simple→edge, complex→performance
        if tier == "auto" and decision_type == DecisionType.COMPACTION:
            complexity = context.get("complexity", "simple")
            if complexity == "complex":
                tier = "performance"
            else:
                tier = "edge"
            logger.debug(
                "Decision %s: auto-routed to '%s' tier (complexity=%s)",
                decision_type.value,
                tier,
                complexity,
            )

        # Try primary tier
        service = self._get_service(tier)
        if service is not None:
            return service.decide_sync(
                decision_type,
                context,
                heuristic_result=heuristic_result,
                heuristic_confidence=heuristic_confidence,
            )

        # Fallback chain
        for fallback_tier in FALLBACK_CHAIN.get(tier, []):
            if fallback_tier in self._failed_tiers:
                continue
            service = self._get_service(fallback_tier)
            if service is not None:
                logger.debug(
                    "Decision %s: tier '%s' unavailable, using '%s' fallback",
                    decision_type.value,
                    tier,
                    fallback_tier,
                )
                return service.decide_sync(
                    decision_type,
                    context,
                    heuristic_result=heuristic_result,
                    heuristic_confidence=heuristic_confidence,
                )

        # All tiers exhausted — return heuristic
        logger.debug("Decision %s: all tiers unavailable, using heuristic", decision_type.value)
        return DecisionResult(
            decision_type=decision_type,
            result=heuristic_result,
            source="heuristic",
            confidence=heuristic_confidence,
        )

    def _get_service(self, tier: str) -> Optional[Any]:
        """Get or create LLMDecisionService for a tier. Cached."""
        if tier in self._failed_tiers:
            return None
        if tier in self._services:
            return self._services[tier]
        return self._create_service(tier)

    def _create_service(self, tier: str) -> Optional[Any]:
        """Create an LLMDecisionService for the given tier.

        Auto-detects provider from active orchestrator if provider="auto".
        Resolves model from provider_model_tiers mapping if model="auto".
        """
        # Check for tier overrides first
        if tier in self._config.tier_overrides:
            override = self._config.tier_overrides[tier]
            override_provider = override.get("provider", "auto")
            override_model = override.get("model", "auto")

            # Use override values to create a temporary spec
            from victor.config.decision_settings import DecisionModelSpec

            spec = DecisionModelSpec(
                provider=override_provider,
                model=override_model,
                timeout_ms=getattr(self._config, tier).timeout_ms,
                max_tokens=getattr(self._config, tier).max_tokens,
            )

            logger.debug(f"Using override for tier '{tier}': {override_provider}/{override_model}")
        else:
            spec = getattr(self._config, tier, None)

        if spec is None:
            self._failed_tiers.add(tier)
            return None

        # Resolve provider="auto" to actual provider
        provider = spec.provider
        model = spec.model

        if provider == "auto" or model == "auto":
            # Detect active provider from orchestrator
            active_provider = self._detect_active_provider()
            if active_provider is None:
                # Fallbacks for local providers to ensure stability when auto-detection fails
                if tier == "edge":
                    logger.info(
                        "Using hardcoded fallback for 'edge' tier: ollama/qwen2.5-coder:1.5b"
                    )
                    provider = "ollama"
                    model = "qwen2.5-coder:1.5b"
                elif tier == "balanced":
                    logger.info(
                        "Using hardcoded fallback for 'balanced' tier: ollama/qwen2.5-coder:1.5b"
                    )
                    provider = "ollama"
                    model = "qwen2.5-coder:1.5b"
                else:
                    logger.warning(f"Could not detect active provider for tier '{tier}'")
                    self._failed_tiers.add(tier)
                    return None
            else:
                provider = active_provider

        # Cloud providers exceed the 4s edge timeout under load; heuristics are accurate
        # enough for micro-decisions and avoid the retry overhead (~15-70s per iteration).
        if tier == "edge" and provider not in LOCAL_ENDPOINT_PROVIDERS:
            logger.debug(
                "Skipping edge LLM for cloud provider '%s' — heuristics will be used instead",
                provider,
            )
            self._failed_tiers.add(tier)
            return None

        # Resolve model="auto" from provider's tier mapping
        if model == "auto":
            provider_tiers = self._config.provider_model_tiers.get(provider, {})
            model = provider_tiers.get(tier)
            if model is None:
                logger.warning(
                    f"No {tier} model defined for provider '{provider}'. "
                    f"Available tiers: {list(provider_tiers.keys())}"
                )
                self._failed_tiers.add(tier)
                return None

        try:
            from victor.agent.services.decision_service import (
                LLMDecisionService,
                LLMDecisionServiceConfig,
            )
            from victor.providers.registry import ProviderRegistry

            provider_kwargs: Dict[str, Any] = {
                "timeout": spec.timeout_ms // 1000,
            }

            if provider == "ollama":
                provider_kwargs["base_url"] = "http://localhost:11434"
            else:
                try:
                    from victor.config.api_keys import get_api_key

                    api_key = get_api_key(provider)
                    if api_key:
                        provider_kwargs["api_key"] = api_key
                except Exception:
                    pass

            provider_instance = ProviderRegistry.create(provider, **provider_kwargs)

            svc_config = LLMDecisionServiceConfig(
                confidence_threshold=0.7,
                micro_budget=20 if tier == "edge" else 10,
                timeout_ms=spec.timeout_ms,
                cache_ttl=120 if tier == "edge" else 60,
                temperature=0.0,
                max_tokens_override=spec.max_tokens,
            )

            service = LLMDecisionService(provider=provider_instance, model=model, config=svc_config)
            self._services[tier] = service
            logger.info(
                "Created %s decision service: %s/%s (timeout=%dms)",
                tier,
                provider,
                model,
                spec.timeout_ms,
            )
            return service

        except Exception as e:
            logger.warning(
                "Failed to create %s decision service (%s/%s): %s",
                tier,
                provider,
                model,
                e,
            )
            self._failed_tiers.add(tier)
            return None

    async def decide(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> DecisionResult:
        """Async version — delegates to decide_sync for compatibility."""
        return self.decide_sync(
            decision_type,
            context,
            heuristic_result=heuristic_result,
            heuristic_confidence=heuristic_confidence,
        )

    async def decide_async(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> DecisionResult:
        """Async alias for decide()."""
        return await self.decide(
            decision_type,
            context,
            heuristic_result=heuristic_result,
            heuristic_confidence=heuristic_confidence,
        )

    def escalate_tier(
        self,
        decision_type: DecisionType,
        reason: str = "low_confidence",
    ) -> str:
        """Escalate a decision type to a higher tier.

        Called when repeated low-confidence results suggest the current
        tier isn't powerful enough for this decision. Follows the
        GEPATierManager pattern: edge → balanced → performance.

        Args:
            decision_type: Decision type to escalate
            reason: Why escalating (for logging)

        Returns:
            New tier name after escalation
        """
        current = self._tier_routing.get(decision_type.value, "edge")
        escalation = {"edge": "balanced", "balanced": "performance"}
        new_tier = escalation.get(current, current)

        if new_tier != current:
            self._tier_routing[decision_type.value] = new_tier
            logger.info(
                "Decision %s: tier escalated %s → %s (reason: %s)",
                decision_type.value,
                current,
                new_tier,
                reason,
            )

        return new_tier

    def deescalate_tier(
        self,
        decision_type: DecisionType,
        reason: str = "high_confidence",
    ) -> str:
        """De-escalate a decision type to a lower tier.

        Called when consistent high-confidence results suggest a cheaper
        tier is sufficient. performance → balanced → edge.

        Args:
            decision_type: Decision type to de-escalate
            reason: Why de-escalating (for logging)

        Returns:
            New tier name after de-escalation
        """
        current = self._tier_routing.get(decision_type.value, "edge")
        deescalation = {"performance": "balanced", "balanced": "edge"}
        new_tier = deescalation.get(current, current)

        if new_tier != current:
            self._tier_routing[decision_type.value] = new_tier
            logger.info(
                "Decision %s: tier de-escalated %s → %s (reason: %s)",
                decision_type.value,
                current,
                new_tier,
                reason,
            )

        return new_tier

    def classify_with_triage(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
        runtime_policy: Optional[RuntimeEvaluationPolicy] = None,
    ) -> TieredClassificationResult:
        """Classify with confidence-based triage using canonical thresholds.

        Triage Logic:
        - confidence ≥ 0.8: ACCEPT (fast path, ~4-5μs)
        - 0.5 ≤ confidence < 0.8: VERIFY (edge LLM, ~50-200ms)
        - confidence < 0.5: REJECT (early rejection, ~1μs)

        This method implements the core confidence-based triage system for
        all classification types (task type, intent, tool selection, etc.).

        Args:
            decision_type: Type of classification
            context: Classification context
            heuristic_result: Pre-computed heuristic result
            heuristic_confidence: Confidence of the heuristic (0.0-1.0)
            runtime_policy: Optional runtime evaluation policy (uses defaults if None)

        Returns:
            TieredClassificationResult with triage outcome and optional verification
        """
        import time

        start = time.monotonic()

        # Use RuntimeEvaluationPolicy as canonical threshold source
        if runtime_policy is None:
            runtime_policy = RuntimeEvaluationPolicy()

        # Fast-path high-confidence heuristics without paying provider startup or retries.
        if (
            heuristic_result is not None
            and heuristic_confidence >= runtime_policy.high_confidence_threshold
        ):
            return TieredClassificationResult(
                result=heuristic_result,
                confidence=heuristic_confidence,
                triage_outcome=ClassificationTriage.ACCEPT,
                source="heuristic",
                latency_ms=(time.monotonic() - start) * 1000,
                metadata={
                    "fast_path": True,
                    "skipped_decision_service": True,
                },
            )

        # Reject clearly low-confidence heuristics early to avoid unnecessary tier work.
        if (
            heuristic_result is not None
            and heuristic_confidence < runtime_policy.medium_confidence_threshold
        ):
            return TieredClassificationResult(
                result=heuristic_result,
                confidence=heuristic_confidence,
                triage_outcome=ClassificationTriage.REJECT,
                source="heuristic_fallback",
                latency_ms=(time.monotonic() - start) * 1000,
                metadata={
                    "reason": "confidence_below_threshold",
                    "original_source": "heuristic",
                    "original_confidence": heuristic_confidence,
                    "threshold": runtime_policy.medium_confidence_threshold,
                    "skipped_decision_service": True,
                },
            )

        # Get base classification result
        base_result = self.decide_sync(
            decision_type,
            context,
            heuristic_result=heuristic_result,
            heuristic_confidence=heuristic_confidence,
        )

        base_latency_ms = (time.monotonic() - start) * 1000
        confidence = base_result.confidence

        # High confidence: Accept immediately (fast path)
        if confidence >= runtime_policy.high_confidence_threshold:
            logger.debug(
                "Classification %s: high confidence (%.2f), accepting immediately (fast path)",
                decision_type.value,
                confidence,
            )
            return TieredClassificationResult(
                result=base_result.result,
                confidence=confidence,
                triage_outcome=ClassificationTriage.ACCEPT,
                source=base_result.source,
                latency_ms=base_latency_ms,
                metadata={
                    "fast_path": True,
                    "original_source": base_result.source,
                },
            )

        # Medium confidence: Verify with edge LLM
        if confidence >= runtime_policy.medium_confidence_threshold:
            logger.debug(
                "Classification %s: medium confidence (%.2f), triggering verification",
                decision_type.value,
                confidence,
            )
            verification = self._verify_with_edge_llm(
                decision_type,
                context,
                base_result,
                runtime_policy,
            )
            total_latency_ms = base_latency_ms + verification.latency_ms

            # Use verified result if verification passed
            final_result = (
                verification.verified_result
                if verification.verification_passed
                else base_result.result
            )
            final_confidence = (
                verification.verification_confidence
                if verification.verification_passed
                else confidence
            )
            final_source = (
                "edge_verification" if verification.verification_passed else base_result.source
            )

            return TieredClassificationResult(
                result=final_result,
                confidence=final_confidence,
                triage_outcome=ClassificationTriage.VERIFY,
                source=final_source,
                verification_result=verification,
                latency_ms=total_latency_ms,
                metadata={
                    "verification_passed": verification.verification_passed,
                    "original_source": base_result.source,
                    "original_confidence": confidence,
                },
            )

        # Low confidence: Reject early
        logger.debug(
            "Classification %s: low confidence (%.2f), rejecting early",
            decision_type.value,
            confidence,
        )
        return TieredClassificationResult(
            result=heuristic_result,  # Fall back to heuristic
            confidence=heuristic_confidence,
            triage_outcome=ClassificationTriage.REJECT,
            source="heuristic_fallback",
            latency_ms=base_latency_ms,
            metadata={
                "reason": "confidence_below_threshold",
                "original_source": base_result.source,
                "original_confidence": confidence,
                "threshold": runtime_policy.medium_confidence_threshold,
            },
        )

    def _verify_with_edge_llm(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
        original_result: DecisionResult,
        runtime_policy: RuntimeEvaluationPolicy,
    ) -> EdgeLLMVerificationResult:
        """Verify grey-area classification with edge LLM.

        This is a STRATEGY METHOD - can be overridden for different verification strategies.
        Default implementation uses edge tier with verification prompt.

        Args:
            decision_type: Type of decision to verify
            context: Original classification context
            original_result: Original decision result to verify
            runtime_policy: Runtime evaluation policy for thresholds

        Returns:
            EdgeLLMVerificationResult with verification outcome
        """
        import time

        start = time.monotonic()

        # Get edge tier service
        edge_service = self._get_service("edge")
        if edge_service is None:
            # Edge unavailable - conservatively reject
            logger.debug("Edge tier unavailable for verification, conservatively rejecting")
            return EdgeLLMVerificationResult(
                original_result=original_result.result,
                original_confidence=original_result.confidence,
                verified_result=original_result.result,
                verification_confidence=original_result.confidence,
                verification_passed=False,  # Conservative when edge unavailable
                latency_ms=0.0,
                tokens_used=0,
            )

        # Create verification context with original result
        verification_context = {
            **context,
            "original_classification": str(original_result.result),
            "original_confidence": f"{original_result.confidence:.2f}",
            "verification_task": "verify",
        }

        # Call edge LLM for verification
        try:
            verification = edge_service.decide_sync(
                decision_type,
                verification_context,
                heuristic_result=original_result.result,
                heuristic_confidence=original_result.confidence,
            )

            latency_ms = (time.monotonic() - start) * 1000

            # Verification passes if edge LLM confidence ≥ high threshold
            verification_passed = (
                verification.confidence >= runtime_policy.high_confidence_threshold
            )

            logger.debug(
                "Edge LLM verification: passed=%s, confidence=%.2f → %.2f (latency=%.1fms)",
                verification_passed,
                original_result.confidence,
                verification.confidence,
                latency_ms,
            )

            return EdgeLLMVerificationResult(
                original_result=original_result.result,
                original_confidence=original_result.confidence,
                verified_result=verification.result,
                verification_confidence=verification.confidence,
                verification_passed=verification_passed,
                latency_ms=latency_ms,
                tokens_used=getattr(verification, "tokens_used", 0),
            )

        except Exception as e:
            logger.warning("Edge LLM verification failed: %s", e)
            return EdgeLLMVerificationResult(
                original_result=original_result.result,
                original_confidence=original_result.confidence,
                verified_result=original_result.result,
                verification_confidence=original_result.confidence,
                verification_passed=False,  # Conservative on error
                latency_ms=(time.monotonic() - start) * 1000,
                tokens_used=0,
            )

    def get_tier(self, decision_type: DecisionType) -> str:
        """Get current tier for a decision type."""
        return self._tier_routing.get(decision_type.value, "edge")

    def get_runtime_evaluation_feedback(self) -> Optional[RuntimeEvaluationFeedback]:
        """Export runtime feedback from the decision service used for task completion."""
        tier = self.get_tier(DecisionType.TASK_COMPLETION)
        service = self._get_service(tier)
        if service is not None and hasattr(service, "get_runtime_evaluation_feedback"):
            return service.get_runtime_evaluation_feedback()
        return None

    def get_metrics(self) -> Dict[str, Any]:
        """Export metrics from all active tier services."""
        return {
            "active_tiers": list(self._services.keys()),
            "failed_tiers": list(self._failed_tiers),
            "routing": dict(self._tier_routing),
        }

    @property
    def budget_remaining(self) -> int:
        """Number of LLM decision calls remaining in the current turn budget.

        Sums budgets across all active tier services. Returns 0 if no services
        are available or if none track budget.
        """
        total = 0
        for service in self._services.values():
            if hasattr(service, "budget_remaining"):
                try:
                    total += service.budget_remaining
                except Exception:
                    # Service may not support budget tracking
                    pass
        return total

    def reset_budget(self) -> None:
        """Reset the per-turn LLM call budget for all tier services.

        Called at the start of each turn to ensure all services have
        their full budget available.
        """
        for service in self._services.values():
            if hasattr(service, "reset_budget"):
                try:
                    service.reset_budget()
                except Exception as e:
                    logger.warning(f"Failed to reset budget for service: {e}")

    def is_healthy(self) -> bool:
        """Check if the decision service is operational.

        Returns True if at least one tier service is available and not failed.
        """
        # Service is healthy if we have at least one active service
        # and not all tiers have failed
        if not self._services:
            # No services initialized yet - try to detect and initialize
            return self._detect_active_provider() is not None

        # Check if we have any non-failed services
        active_tiers = set(self._services.keys()) - self._failed_tiers
        return len(active_tiers) > 0


def create_tiered_decision_service(
    config: Optional[DecisionServiceSettings] = None,
) -> Optional["TieredDecisionService"]:
    """Factory function to create a TieredDecisionService.

    Returns None if disabled in settings.
    """
    if config is None:
        config = DecisionServiceSettings()

    if not config.enabled:
        return None

    return TieredDecisionService(config)
