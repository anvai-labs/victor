"""LLM Decision Service implementation.

Provides centralized LLM-assisted decision making for the agent orchestration
loop. Heuristics remain the fast path; the LLM is only consulted when
heuristic confidence falls below a configurable threshold.

Key features:
    - Heuristic-first: returns immediately if confidence >= threshold
    - Micro-budget: bounded LLM calls per conversation turn (default 10)
    - Cache: 60s TTL deduplicates rapid-fire calls within a turn
    - Timeout: 2s hard limit with heuristic fallback
    - Works with all 22 providers (no structured output API required)
"""

from __future__ import annotations

import asyncio
import hashlib
from victor.core.json_utils import json_dumps, json_loads
from json import JSONDecodeError
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from victor.agent.decisions.prompts import DECISION_PROMPTS
from victor.agent.decisions.schemas import DecisionType
from victor.agent.services.protocols.decision_service import (
    DecisionMetrics,
    DecisionResult,
)
from victor.core.async_utils import run_sync
from victor.framework.runtime_evaluation_policy import RuntimeEvaluationFeedback
from victor.providers.base import Message

if TYPE_CHECKING:
    from victor.providers.base import BaseProvider

logger = logging.getLogger(__name__)


@dataclass
class LLMDecisionServiceConfig:
    """Configuration for the LLM decision service."""

    confidence_threshold: float = 0.7
    micro_budget: int = 10
    timeout_ms: int = 2000
    cache_ttl: int = 60
    temperature: float = 0.0
    max_tokens_override: Optional[int] = None


class LLMDecisionService:
    """Centralized LLM-assisted decision service.

    Uses the underlying provider as a fallback classifier when heuristic
    confidence is low. Designed for minimal token usage (5-20 tokens per call)
    with budget control, caching, and timeout protection.
    """

    def __init__(
        self,
        provider: BaseProvider,
        model: str,
        config: Optional[LLMDecisionServiceConfig] = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._config = config or LLMDecisionServiceConfig()

        # Per-turn budget tracking
        self._budget_used: int = 0

        # Simple TTL cache: key -> (result, expiry_time)
        self._cache: Dict[str, Tuple[DecisionResult, float]] = {}

        # Metrics
        self._metrics = DecisionMetrics()
        self._auto_disable_warned = False

    def _requires_fresh_provider_for_sync_thread(self) -> bool:
        """Return whether sync thread bridging should clone the provider per call.

        httpx-based OpenAI-compatible providers hold an AsyncClient that is bound
        to the event loop that first uses it. decide_sync() runs async work on a
        fresh thread/loop, so reusing the same client across loops can trigger
        ``RuntimeError: Event loop is closed`` on follow-up calls.
        """

        try:
            from victor.providers.httpx_openai_compat import HttpxOpenAICompatProvider

            return isinstance(self._provider, HttpxOpenAICompatProvider)
        except Exception:
            return False

    async def decide(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> DecisionResult:
        """Make a decision, consulting the LLM if heuristic confidence is low.

        Every path logs the decision (FEP-0012 closed loop) except cache hits,
        which were logged when first computed.
        """
        start = time.monotonic()

        # Fast path: heuristic is confident enough
        if heuristic_confidence >= self._config.confidence_threshold:
            self._metrics.total_calls += 1
            result = DecisionResult(
                decision_type=decision_type,
                result=heuristic_result,
                source="heuristic",
                confidence=heuristic_confidence,
                latency_ms=_elapsed_ms(start),
            )
            self._log_decision(decision_type, context, result)
            return result

        # Cache check — cached results were logged when computed.
        cache_key = self._cache_key(decision_type, context)
        cached = self._get_cached(cache_key)
        if cached is not None:
            self._metrics.total_calls += 1
            self._metrics.cache_hits += 1
            cached.latency_ms = _elapsed_ms(start)
            return cached

        # Budget check
        if self._budget_used >= self._config.micro_budget:
            self._metrics.total_calls += 1
            self._metrics.budget_exhaustions += 1
            logger.debug(
                "LLM decision budget exhausted (%d/%d), using heuristic fallback",
                self._budget_used,
                self._config.micro_budget,
            )
            result = DecisionResult(
                decision_type=decision_type,
                result=heuristic_result,
                source="budget_exhausted",
                confidence=heuristic_confidence,
                latency_ms=_elapsed_ms(start),
            )
            self._log_decision(decision_type, context, result)
            return result

        # LLM call with timeout
        try:
            timeout_s = self._config.timeout_ms / 1000.0
            result = await asyncio.wait_for(
                self._call_llm(decision_type, context),
                timeout=timeout_s,
            )
            self._budget_used += 1
            self._metrics.total_calls += 1
            self._metrics.llm_calls += 1

            result.latency_ms = _elapsed_ms(start)
            self._update_latency(result.latency_ms)

            # Cache the result
            self._cache[cache_key] = (result, time.monotonic() + self._config.cache_ttl)

            logger.debug(
                "LLM decision: type=%s result=%s confidence=%.2f latency=%.0fms",
                decision_type.value,
                result.result,
                result.confidence,
                result.latency_ms,
            )
            self._log_decision(decision_type, context, result)
            return result

        except asyncio.TimeoutError:
            self._metrics.total_calls += 1
            self._metrics.timeouts += 1
            logger.debug(
                "LLM decision timed out after %dms, using heuristic fallback",
                self._config.timeout_ms,
            )
            result = DecisionResult(
                decision_type=decision_type,
                result=heuristic_result,
                source="timeout_fallback",
                confidence=heuristic_confidence,
                latency_ms=_elapsed_ms(start),
            )
            self._log_decision(decision_type, context, result)
            return result

        except Exception:
            self._metrics.total_calls += 1
            logger.debug("LLM decision call failed, using heuristic fallback", exc_info=True)
            result = DecisionResult(
                decision_type=decision_type,
                result=heuristic_result,
                source="heuristic",
                confidence=heuristic_confidence,
                latency_ms=_elapsed_ms(start),
            )
            self._log_decision(decision_type, context, result)
            return result

    def _log_decision(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
        result: DecisionResult,
    ) -> None:
        """Record a decision for RL/fine-tuning data collection (best-effort).

        Called on EVERY decision path so heuristic/budget/auto-disabled/timeout
        decisions are captured too — not just LLM-escalated ones. This is the
        closed-loop data contract: log_decision is the source the classifier
        trainer mines.
        """
        try:
            from victor.agent.decisions.chain import log_decision

            log_decision(
                decision_type=decision_type.value,
                context=context,
                result=str(getattr(result.result, "__dict__", result.result)),
                source=result.source,
                confidence=result.confidence,
            )
        except Exception:
            pass

    def decide_sync(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> DecisionResult:
        """Synchronous version of decide().

        If an event loop is already running, returns the heuristic fallback
        rather than blocking. Every path logs the decision (FEP-0012 closed
        loop) except cache hits, which were logged when first computed.
        """
        # Fast path: heuristic is confident enough
        if heuristic_confidence >= self._config.confidence_threshold:
            self._metrics.total_calls += 1
            result = DecisionResult(
                decision_type=decision_type,
                result=heuristic_result,
                source="heuristic",
                confidence=heuristic_confidence,
            )
            self._log_decision(decision_type, context, result)
            return result

        # Cache check (works in sync) — cached results were logged when computed.
        cache_key = self._cache_key(decision_type, context)
        cached = self._get_cached(cache_key)
        if cached is not None:
            self._metrics.total_calls += 1
            self._metrics.cache_hits += 1
            return cached

        # Budget check
        if self._budget_used >= self._config.micro_budget:
            self._metrics.total_calls += 1
            self._metrics.budget_exhaustions += 1
            result = DecisionResult(
                decision_type=decision_type,
                result=heuristic_result,
                source="budget_exhausted",
                confidence=heuristic_confidence,
            )
            self._log_decision(decision_type, context, result)
            return result

        # Auto-disable if timeout rate is too high (>60% after 10+ calls)
        if (
            self._metrics.total_calls >= 10
            and self._metrics.timeouts / max(self._metrics.total_calls, 1) > 0.6
        ):
            if not self._auto_disable_warned:
                logger.warning(
                    "Edge model auto-disabled: %.0f%% timeout rate (%d/%d calls)",
                    100 * self._metrics.timeouts / self._metrics.total_calls,
                    self._metrics.timeouts,
                    self._metrics.total_calls,
                )
                self._auto_disable_warned = True
            self._metrics.total_calls += 1
            result = DecisionResult(
                decision_type=decision_type,
                result=heuristic_result,
                source="auto_disabled",
                confidence=heuristic_confidence,
            )
            self._log_decision(decision_type, context, result)
            return result

        # Use run_sync_in_thread to handle both cases:
        # 1. No running loop — runs in a thread with its own loop
        # 2. Inside async context — also runs in a thread (avoids blocking)
        # httpx-based OpenAI-compatible providers need a fresh provider/client
        # inside that loop to avoid cross-loop AsyncClient reuse.
        try:
            from victor.core.async_utils import run_sync_in_thread

            if self._requires_fresh_provider_for_sync_thread():
                start = time.monotonic()
                result = run_sync_in_thread(
                    self._call_llm_with_fresh_provider(decision_type, context),
                    timeout=self._config.timeout_ms / 1000.0,
                )
                self._budget_used += 1
                self._metrics.total_calls += 1
                self._metrics.llm_calls += 1
                result.latency_ms = _elapsed_ms(start)
                self._update_latency(result.latency_ms)
                self._cache[cache_key] = (
                    result,
                    time.monotonic() + self._config.cache_ttl,
                )
            else:
                result = run_sync_in_thread(
                    self.decide(
                        decision_type,
                        context,
                        heuristic_result=heuristic_result,
                        heuristic_confidence=heuristic_confidence,
                    ),
                    timeout=self._config.timeout_ms / 1000.0,
                )

            self._log_decision(decision_type, context, result)
            return result
        except (TimeoutError, Exception) as e:
            logger.debug("decide_sync thread execution failed: %s", e)
            self._metrics.total_calls += 1
            if isinstance(e, TimeoutError):
                self._metrics.timeouts += 1
            result = DecisionResult(
                decision_type=decision_type,
                result=heuristic_result,
                source=("timeout_fallback" if isinstance(e, TimeoutError) else "heuristic"),
                confidence=heuristic_confidence,
            )
            self._log_decision(decision_type, context, result)
            return result

    async def decide_async(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> DecisionResult:
        """Async version of decide() — explicit alias for async callers.

        Use this from async context to avoid the thread-spawning overhead
        of decide_sync(). Delegates directly to decide() without wrapping
        in a thread.
        """
        return await self.decide(
            decision_type,
            context,
            heuristic_result=heuristic_result,
            heuristic_confidence=heuristic_confidence,
        )

    @property
    def budget_remaining(self) -> int:
        """Number of LLM decision calls remaining in the current turn budget."""
        return max(0, self._config.micro_budget - self._budget_used)

    def reset_budget(self) -> None:
        """Reset the per-turn LLM call budget and evict expired cache entries."""
        self._budget_used = 0
        self._evict_expired_cache()

    def is_healthy(self) -> bool:
        """Check if the decision service is operational."""
        return self._provider is not None

    def get_metrics(self) -> DecisionMetrics:
        """Get aggregate metrics for monitoring."""
        return self._metrics

    def get_runtime_evaluation_feedback(self) -> RuntimeEvaluationFeedback:
        """Export runtime evaluation thresholds aligned to the service confidence gate."""
        threshold = self._config.confidence_threshold
        return RuntimeEvaluationFeedback(
            completion_threshold=threshold,
            enhanced_progress_threshold=max(0.35, min(threshold, threshold - 0.15)),
            minimum_supported_evidence_score=min(0.95, max(0.55, threshold + 0.05)),
            metadata={
                "source": "llm_decision_service",
                "confidence_threshold": threshold,
                "micro_budget": self._config.micro_budget,
            },
        )

    async def _call_llm(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
        *,
        provider: Optional[BaseProvider] = None,
    ) -> DecisionResult:
        """Build prompt, call provider, parse structured response."""
        prompt_config = DECISION_PROMPTS[decision_type]

        # Build user message from template
        try:
            user_message = prompt_config.user_template.format(**context)
        except KeyError as e:
            logger.warning("Missing context key for %s: %s", decision_type.value, e)
            raise

        messages = [
            Message(role="system", content=prompt_config.system),
            Message(role="user", content=user_message),
        ]

        max_tokens = self._config.max_tokens_override or prompt_config.max_tokens
        provider_instance = provider or self._provider

        # Disable thinking/reasoning for edge decisions — we need raw JSON,
        # not reasoning text. ``thinking`` is Victor's portable transport
        # field; the Sandhi Ollama adapter maps it to native ``think``.
        response = await provider_instance.chat(
            messages=messages,
            model=self._model,
            temperature=self._config.temperature,
            max_tokens=max_tokens,
            tools=None,
            thinking=False,
        )

        # Extract text content from response
        raw_text = self._extract_response_text(response)
        tokens_used = self._extract_token_count(response)

        # Parse JSON response with schema validation
        parsed = self._parse_response(raw_text, prompt_config.schema)

        # Extract confidence from the parsed model if it has one
        confidence = getattr(parsed, "confidence", 0.8)

        return DecisionResult(
            decision_type=decision_type,
            result=parsed,
            source="llm",
            confidence=confidence,
            tokens_used=tokens_used,
        )

    async def _call_llm_with_fresh_provider(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
    ) -> DecisionResult:
        """Run one LLM decision call with a fresh provider inside the worker loop."""

        provider = self._provider
        if provider is None or not self._requires_fresh_provider_for_sync_thread():
            return await self._call_llm(decision_type, context)

        fresh_provider = provider.__class__(
            api_key=provider.api_key,
            base_url=provider.base_url,
            timeout=provider.timeout,
            max_retries=provider.max_retries,
            **provider.extra_config,
        )
        try:
            return await self._call_llm(decision_type, context, provider=fresh_provider)
        finally:
            try:
                await fresh_provider.close()
            except Exception:
                logger.debug("Fresh sync-thread provider cleanup failed", exc_info=True)

    def _extract_response_text(self, response: Any) -> str:
        """Extract text content from a provider response."""
        # Handle CompletionResponse with .content attribute
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, str):
                return content
            # Some providers return content as a list of blocks
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block["text"]
                    if hasattr(block, "text"):
                        return block.text
        # Handle dict responses
        if isinstance(response, dict):
            if "content" in response:
                return str(response["content"])
            if "text" in response:
                return str(response["text"])
        return str(response)

    def _extract_token_count(self, response: Any) -> int:
        """Extract token usage from a provider response."""
        if hasattr(response, "usage"):
            usage = response.usage
            if hasattr(usage, "total_tokens"):
                return usage.total_tokens
            if isinstance(usage, dict):
                return usage.get("total_tokens", 0)
        return 0

    def _parse_response(self, raw_text: str, schema: type) -> Any:
        """Parse raw LLM text into a Pydantic model.

        Handles common edge model output patterns:
        - Markdown code fences (```json ... ```)
        - Embedded JSON within reasoning text (e.g., qwen3 thinking prefix)
        """
        text = raw_text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines[1:] if not line.startswith("```")]
            text = "\n".join(lines).strip()

        # First try: direct parse
        try:
            data = json_loads(text)
            return schema.model_validate(data)
        except JSONDecodeError:
            pass

        # Second try: extract embedded JSON object from mixed text
        # (handles models that prepend reasoning before the JSON)
        brace_pos = text.find("{")
        if brace_pos > 0:
            last_brace = text.rfind("}")
            if last_brace > brace_pos:
                try:
                    data = json_loads(text[brace_pos : last_brace + 1])
                    return schema.model_validate(data)
                except (JSONDecodeError, Exception):
                    pass

        self._metrics.parse_failures += 1
        logger.debug("Failed to parse LLM decision response (raw: %s)", raw_text[:200])
        raise ValueError(f"Could not extract JSON from response: {raw_text[:100]}")

    def _cache_key(self, decision_type: DecisionType, context: Dict[str, Any]) -> str:
        """Generate a cache key from decision type and context."""
        key_data = f"{decision_type.value}:{json_dumps(context, sort_keys=True, default=str)}"
        return hashlib.md5(key_data.encode()).hexdigest()  # noqa: S324

    def _get_cached(self, cache_key: str) -> Optional[DecisionResult]:
        """Get a cached result if it exists and hasn't expired."""
        entry = self._cache.get(cache_key)
        if entry is None:
            return None
        result, expiry = entry
        if time.monotonic() > expiry:
            del self._cache[cache_key]
            return None
        return DecisionResult(
            decision_type=result.decision_type,
            result=result.result,
            source="cache",
            confidence=result.confidence,
            tokens_used=0,
        )

    def _evict_expired_cache(self) -> None:
        """Remove expired cache entries."""
        now = time.monotonic()
        expired = [k for k, (_, expiry) in self._cache.items() if now > expiry]
        for key in expired:
            del self._cache[key]

    def _update_latency(self, latency_ms: float) -> None:
        """Update running average latency."""
        self._metrics._latency_sum += latency_ms
        if self._metrics.llm_calls > 0:
            self._metrics.avg_latency_ms = self._metrics._latency_sum / self._metrics.llm_calls


def _elapsed_ms(start: float) -> float:
    """Calculate elapsed milliseconds since start."""
    return (time.monotonic() - start) * 1000
