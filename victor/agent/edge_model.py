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

"""Edge model provider for micro-decisions.

A lightweight local LLM provider for fast, token-free micro-decisions:
task classification, tool selection, completion detection, etc.

Runs via Ollama with small models (TinyLlama, Qwen2.5-Coder:1.5b) on CPU.
Falls back to heuristic when Ollama is unavailable.

Usage:
    from victor.agent.edge_model import create_edge_decision_service

    service = create_edge_decision_service()
    if service:
        # Use for micro-decisions — zero cloud LLM token cost
        result = service.decide_sync(
            DecisionType.TASK_TYPE_CLASSIFICATION,
            context={"message_excerpt": "Fix the auth bug"},
        )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from victor.agent.prompt_section_registry import build_edge_focus_prompt_options_text

logger = logging.getLogger(__name__)


@dataclass
class EdgeModelConfig:
    """Configuration for the edge model provider.

    Attributes:
        enabled: Whether edge model is active
        provider: Backend provider ("ollama" or "disabled")
        model: Model name (e.g., "qwen2.5-coder:1.5b", "tinyllama")
        timeout_ms: Hard timeout for edge model calls
        max_tokens: Max response tokens (micro-decisions need 30-60)
        cache_ttl: Cache TTL in seconds for identical decisions
        confidence_threshold: Below this, fall back to heuristic
        micro_budget: Maximum uncached decisions before budget fallback
        base_url: Ollama server URL
        tool_selection_enabled: Use edge model for tool ranking
        prompt_focus_enabled: Use edge model for system prompt trimming
        max_tools: Max tools to recommend in tool selection
    """

    enabled: bool = True
    provider: str = "ollama"
    model: str = "qwen3.5:2b"
    timeout_ms: int = 4000
    max_tokens: int = 50
    cache_ttl: int = 120
    confidence_threshold: float = 0.6
    micro_budget: int = 20
    base_url: str = "http://localhost:11434"
    tool_selection_enabled: bool = True
    prompt_focus_enabled: bool = True
    max_tools: int = 6
    # Per-task-type timeouts: simpler decisions get shorter timeouts
    timeout_by_task: Dict[str, int] = field(
        default_factory=lambda: {
            "classification": 2000,
            "completion_check": 2000,
            "prompt_focus": 4000,
            "tool_selection": 6000,
        }
    )
    # Per-task-type cache TTL: stable decisions cached longer
    cache_ttl_by_task: Dict[str, int] = field(
        default_factory=lambda: {
            "classification": 300,
            "completion_check": 300,
            "prompt_focus": 180,
            "tool_selection": 120,
        }
    )

    def get_timeout_for_task(self, task_type: str) -> int:
        """Get the timeout in ms for a specific task type."""
        return self.timeout_by_task.get(task_type, self.timeout_ms)

    def get_cache_ttl_for_task(self, task_type: str) -> int:
        """Get the cache TTL in seconds for a specific task type."""
        return self.cache_ttl_by_task.get(task_type, self.cache_ttl)


# Tool selection prompt — edge model picks most relevant tools
TOOL_SELECTION_PROMPT = """You select tools for an AI coding assistant. Given the user's request, pick the most relevant tools. Respond ONLY with a JSON object.

User request:
{message_excerpt}

Current stage: {stage}
Recent tools used: {recent_tools}

Available tools: {available_tools}

Pick 5-6 most relevant tools for this task.
Respond with JSON: {{"tools": ["tool1", "tool2", ...], "confidence": 0.0-1.0}}"""

# System prompt focus — edge model picks which sections to include
PROMPT_FOCUS_PROMPT = f"""You optimize system prompts for an AI coding assistant. Given the task, select which prompt sections are needed. Respond ONLY with a JSON object.

Task type: {{task_type}}
User request excerpt: {{message_excerpt}}

Available sections:
{build_edge_focus_prompt_options_text()}

Pick only sections needed for this task.
Respond with JSON: {{"sections": ["section1", ...], "confidence": 0.0-1.0}}"""


def create_edge_decision_service(
    config: Optional[EdgeModelConfig] = None,
) -> Optional[Any]:
    """Create an LLMDecisionService backed by a local edge model.

    Uses Ollama with a small model for micro-decisions. Returns None if
    Ollama is unavailable or edge model is disabled.

    Args:
        config: Edge model configuration (uses defaults if None)

    Returns:
        Configured LLMDecisionService, or None if unavailable
    """
    if config is None:
        config = EdgeModelConfig()

    if not config.enabled or config.provider == "disabled":
        logger.debug("Edge model disabled")
        return None

    try:
        from victor.agent.services.decision_service import (
            LLMDecisionService,
            LLMDecisionServiceConfig,
        )
        from victor.providers.registry import ProviderRegistry

        # Create provider using the standard registry — supports any provider type
        # (ollama, deepseek, openai, etc.) via config.provider
        provider_kwargs: Dict[str, Any] = {
            "timeout": config.timeout_ms // 1000,
        }

        if config.provider == "ollama":
            provider_kwargs["base_url"] = config.base_url
            # Verify model is available for Ollama
            if not _check_model_available(config.base_url, config.model):
                logger.warning(
                    "Edge model '%s' not available in Ollama. " "Pull it with: ollama pull %s",
                    config.model,
                    config.model,
                )
                return None
        else:
            # For cloud providers, get API key
            try:
                from victor.config.api_keys import get_api_key

                api_key = get_api_key(config.provider)
                if api_key:
                    provider_kwargs["api_key"] = api_key
            except Exception:
                pass

        provider = ProviderRegistry.create(config.provider, **provider_kwargs)

        decision_config = LLMDecisionServiceConfig(
            confidence_threshold=config.confidence_threshold,
            micro_budget=config.micro_budget,
            timeout_ms=config.timeout_ms,
            cache_ttl=config.cache_ttl,
            temperature=0.0,
            max_tokens_override=config.max_tokens,
        )

        service = LLMDecisionService(
            provider=provider,
            model=config.model,
            config=decision_config,
        )
        logger.info(
            "Edge model decision service created: model=%s, timeout=%dms",
            config.model,
            config.timeout_ms,
        )
        return service

    except Exception as e:
        logger.debug("Failed to create edge decision service: %s", e)
        return None


def select_tools_with_edge_model(
    service: Any,
    user_message: str,
    available_tools: List[str],
    stage: str = "initial",
    recent_tools: Optional[List[str]] = None,
    max_tools: int = 6,
) -> Optional[List[str]]:
    """Use edge model to select relevant tools from the full set.

    Uses the decision service's decide_sync() which handles async-in-sync
    context safely (returns heuristic fallback when inside an event loop).

    Args:
        service: LLMDecisionService (edge-backed)
        user_message: The user's request
        available_tools: All available tool names
        stage: Current conversation stage
        recent_tools: Recently used tool names
        max_tools: Maximum tools to return

    Returns:
        List of selected tool names, or None if edge model unavailable
    """
    try:
        from victor.agent.decisions.schemas import DecisionType

        decision = service.decide_sync(
            DecisionType.TOOL_SELECTION,
            context={
                "message_excerpt": user_message[:200],
                "stage": stage,
                "available_tools": ", ".join(available_tools),
            },
            heuristic_confidence=0.0,
        )

        if decision.source in ("heuristic", "budget_exhausted", "timeout_fallback"):
            return None

        result = decision.result
        if not hasattr(result, "tools"):
            return None

        valid = [t for t in result.tools if t in set(available_tools)]
        if valid:
            logger.info(
                "Edge tool selection: %d tools selected (from %d available)",
                len(valid),
                len(available_tools),
            )
            return valid[:max_tools]

    except Exception as e:
        logger.debug("Edge tool selection failed: %s", e)

    return None


def select_prompt_sections_with_edge_model(
    service: Any,
    user_message: str,
    task_type: str,
    available_sections: List[str],
) -> Optional[List[str]]:
    """Use edge model to select which system prompt sections to include.

    Args:
        service: LLMDecisionService (edge-backed)
        user_message: The user's request
        task_type: Classified task type
        available_sections: All available prompt section names

    Returns:
        List of section names to include, or None if unavailable
    """
    try:
        from victor.agent.decisions.chain import should_use_llm

        if not should_use_llm("prompt_focus"):
            return None

        from victor.agent.decisions.schemas import DecisionType

        decision = service.decide_sync(
            DecisionType.PROMPT_FOCUS,
            context={
                "task_type": task_type,
                "message_excerpt": user_message[:200],
                "available_sections": ", ".join(available_sections),
            },
            heuristic_confidence=0.0,
        )

        if decision.source in ("heuristic", "budget_exhausted", "timeout_fallback"):
            return None

        result = decision.result
        if not hasattr(result, "sections"):
            return None

        valid = [s for s in result.sections if s in set(available_sections)]
        if valid:
            logger.info(
                "Edge prompt focus: %d/%d sections selected",
                len(valid),
                len(available_sections),
            )
            return valid

    except Exception as e:
        logger.debug("Edge prompt focus failed: %s", e)

    return None


def _check_model_available(base_url: str, model: str) -> bool:
    """Check if a model is available in Ollama."""
    try:
        import httpx

        resp = httpx.get(f"{base_url}/api/tags", timeout=2.0)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            model_names = {m.get("name", "").split(":")[0] for m in models}
            # Also check full name with tag
            model_full_names = {m.get("name", "") for m in models}
            return model in model_names or model in model_full_names
    except Exception:
        pass
    return False


# =============================================================================
# Classification Verification (Tiered Triage Support)
# =============================================================================


@dataclass
class EdgeLLMVerificationResult:
    """Result of edge LLM verification for grey-area classifications.

    When a classification falls in the VERIFY band (0.5-0.8 confidence),
    the edge LLM is consulted to verify or correct the classification.

    This is a duplicate of the dataclass in tiered_decision_service.py to
    avoid circular imports. The two should be kept in sync.
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


def verify_classification_with_edge_llm(
    service: Any,
    decision_type: Any,
    original_result: Any,
    original_confidence: float,
    context: Dict[str, Any],
    high_confidence_threshold: float = 0.8,
) -> EdgeLLMVerificationResult:
    """Verify a grey-area classification using edge LLM.

    This function provides a standalone verification capability that can be
    used independently of TieredDecisionService. It's useful for callers
    that already have an edge decision service instance.

    Args:
        service: LLMDecisionService (edge-backed)
        decision_type: DecisionType enum value
        original_result: Original classification result
        original_confidence: Original confidence score
        context: Classification context
        high_confidence_threshold: Threshold for verification passing (default 0.8)

    Returns:
        EdgeLLMVerificationResult with verification outcome
    """
    import time

    from victor.framework.runtime_evaluation_policy import RuntimeEvaluationPolicy

    start = time.monotonic()

    if service is None:
        return EdgeLLMVerificationResult(
            original_result=original_result,
            original_confidence=original_confidence,
            verified_result=original_result,
            verification_confidence=original_confidence,
            verification_passed=False,
            latency_ms=0.0,
            tokens_used=0,
        )

    # Create verification context
    verification_context = {
        **context,
        "original_classification": str(original_result),
        "original_confidence": f"{original_confidence:.2f}",
        "verification_task": "verify",
    }

    # Call edge LLM
    try:
        decision = service.decide_sync(
            decision_type,
            verification_context,
            heuristic_result=original_result,
            heuristic_confidence=original_confidence,
        )

        latency_ms = (time.monotonic() - start) * 1000

        # Verification passes if confidence ≥ high threshold
        verification_passed = decision.confidence >= high_confidence_threshold

        logger.debug(
            "Edge LLM verification: passed=%s, confidence=%.2f → %.2f (latency=%.1fms)",
            verification_passed,
            original_confidence,
            decision.confidence,
            latency_ms,
        )

        return EdgeLLMVerificationResult(
            original_result=original_result,
            original_confidence=original_confidence,
            verified_result=decision.result,
            verification_confidence=decision.confidence,
            verification_passed=verification_passed,
            latency_ms=latency_ms,
            tokens_used=getattr(decision, "tokens_used", 0),
        )

    except Exception as e:
        logger.warning("Edge LLM verification failed: %s", e)
        return EdgeLLMVerificationResult(
            original_result=original_result,
            original_confidence=original_confidence,
            verified_result=original_result,
            verification_confidence=original_confidence,
            verification_passed=False,
            latency_ms=(time.monotonic() - start) * 1000,
            tokens_used=0,
        )
