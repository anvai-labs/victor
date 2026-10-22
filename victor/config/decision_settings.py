"""Tiered decision service configuration.

Routes different DecisionTypes to different model tiers:
- edge: Fast model for micro-decisions (~5-50ms)
- balanced: Mid-tier model for moderate decisions (~500ms)
- performance: Frontier model for complex decisions (~2s)

Provider-Agnostic Design:
- Tiers use provider="auto" to auto-detect from active orchestrator
- Each provider defines their own edge/balanced/performance models
- Ensures decision service uses same provider as main LLM
- Supports explicit overrides via tier_overrides

Follows the same pattern as GEPASettings/GEPATierManager.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from victor.config.decision_backend import DecisionBackend


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean env var (true/1/yes → True)."""
    return os.getenv(name, "true" if default else "false").strip().lower() in (
        "true",
        "1",
        "yes",
    )


class DecisionModelSpec(BaseModel):
    """Provider/model specification for one decision tier.

    Supports provider="auto" and model="auto" for provider-agnostic behavior.
    """

    provider: str = "auto"
    model: str = "auto"
    timeout_ms: int = 4000
    max_tokens: int = 50


class DecisionServiceSettings(BaseModel):
    """Tiered decision service configuration with provider-agnostic tiers.

    Each tier defines a provider/model pair. Decision types are
    routed to tiers via the tier_routing map. Fallback chain:
    performance → balanced → edge → heuristic.

    Provider-Agnostic Behavior:
    - provider="auto": Auto-detects from active orchestrator/provider
    - model="auto": Resolves from provider_model_tiers mapping
    - Ensures decision service uses same provider as main LLM
    - No cross-provider API key management needed

    Explicit Overrides:
    - Use tier_overrides to force specific provider/model per tier
    - Example: {"edge": {"provider": "ollama", "model": "phi3:mini"}}
    """

    enabled: bool = True

    # FEP-0012: ONE config value selects the micro-decision backend (replaces the
    # USE_EDGE_MODEL / USE_LLM_DECISION_SERVICE flag tangle). AUTO (default)
    # auto-upgrades to the shipped local classifier once its artifact is bundled,
    # else falls back to the legacy flag-based selection. See DecisionBackend.
    decision_backend: DecisionBackend = Field(
        default_factory=lambda: DecisionBackend.parse(os.getenv("VICTOR_DECISION_BACKEND", "auto")),
        description="Micro-decision backend: auto|local_classifier|edge|llm|heuristic. "
        "Env: VICTOR_DECISION_BACKEND",
    )

    # FEP-0012 Phase 6: per-project online RL personalization delta. On each
    # session outcome, a reward-weighted per-label nudge is written to the
    # project DB's local_classifier_delta and blended into the shipped head's
    # logits at predict time (alpha is the model's own blend knob). Local-only,
    # bounded (top-K) and L2-decayed so the universal model re-asserts, never
    # uploaded. Default ON: degrades to a no-op when no outcomes exist.
    local_learning_enabled: bool = Field(
        default_factory=lambda: _env_bool("VICTOR_LOCAL_LEARNING_ENABLED", True),
        description="Enable the per-project RL delta overlay. Env: VICTOR_LOCAL_LEARNING_ENABLED",
    )
    local_learning_lr: float = Field(default=0.1, description="Per-label SGD step size.")
    local_learning_top_k: int = Field(
        default=2000,
        description="Max rows kept per (decision_type, label) after trimming.",
    )
    local_learning_decay: float = Field(
        default=0.995, description="L2 decay applied to all delta weights each update."
    )

    # FEP-0012 acceptance #1: allow the shipped local classifier's task_completion
    # head to SIGNAL completion (so the agentic loop can stop on it). The head
    # predicts reward buckets (pass/partial/fail); a confident "pass" -> stop.
    # Default OFF: trusting the classifier to stop the loop is a behavioral change
    # that wants Phase-7 A/B validation before it defaults on. Flip to opt in.
    # Env: VICTOR_LOCAL_CLASSIFIER_COMPLETION_SIGNAL
    local_classifier_completion_signal: bool = Field(
        default_factory=lambda: _env_bool("VICTOR_LOCAL_CLASSIFIER_COMPLETION_SIGNAL", False),
        description="Let the shipped classifier's task_completion head signal completion. "
        "Default off (Phase-7 A/B pending). Env: VICTOR_LOCAL_CLASSIFIER_COMPLETION_SIGNAL",
    )

    # Classification triage settings (confidence-based routing)
    enable_classification_triage: bool = True
    triage_verification_timeout_ms: int = 2000
    triage_use_runtime_policy: bool = True  # Use RuntimeEvaluationPolicy thresholds

    # Optional: Override thresholds (if not using RuntimeEvaluationPolicy)
    # These are only used if triage_use_runtime_policy is False
    triage_high_threshold: Optional[float] = None  # Default: 0.8
    triage_medium_threshold: Optional[float] = None  # Default: 0.5

    # Tier definitions (provider-agnostic by default)
    edge: DecisionModelSpec = Field(
        default_factory=lambda: DecisionModelSpec(
            provider="auto", model="auto", timeout_ms=4000, max_tokens=50
        )
    )
    balanced: DecisionModelSpec = Field(
        default_factory=lambda: DecisionModelSpec(
            provider="auto", model="auto", timeout_ms=8000, max_tokens=200
        )
    )
    performance: DecisionModelSpec = Field(
        default_factory=lambda: DecisionModelSpec(
            provider="auto", model="auto", timeout_ms=15000, max_tokens=500
        )
    )

    # Provider-specific model mappings (provider → tier → model)
    # Defines which model to use for each tier per provider
    #
    # NOTE: Update these with the actual API model IDs as new models are released.
    # The version numbers below (e.g., "4.7", "5.4") should be replaced with
    # the actual model IDs returned by the provider APIs.
    provider_model_tiers: Dict[str, Dict[str, str]] = Field(
        default_factory=lambda: {
            # Anthropic Claude models
            "anthropic": {
                "edge": "claude-haiku-4-5-20251001",
                "balanced": "claude-sonnet-4-6",
                "performance": "claude-opus-4-7",
            },
            # OpenAI GPT models
            "openai": {
                "edge": "gpt-5.4-mini",  # Fast, cost-effective mini version
                "balanced": "gpt-5.4",  # General purpose, latest frontier model
                "performance": "gpt-5.4-pro",  # Maximum performance for complex tasks
            },
            # Google Gemini models
            "google": {
                "edge": "gemini-3.1-lite",  # Fast, lightweight
                "balanced": "gemini-3.1-flash",  # Balanced speed and capability
                "performance": "gemini-3.1-pro",  # Full capability
            },
            # xAI Grok models
            "xai": {
                "edge": "grok-4-1-fast",  # Fast, cost-effective for tool calling
                "balanced": "grok-4-1-fast",  # High-performance agentic tool calling
                "performance": "grok-4.20",  # Newest flagship model
            },
            # DeepSeek models
            "deepseek": {
                "edge": "deepseek-chat",
                "balanced": "deepseek-chat",
                "performance": "deepseek-chat",
            },
            # Mistral models
            "mistral": {
                "edge": "ministral-8b-latest",
                "balanced": "mistral-small-latest",
                "performance": "mistral-large-latest",
            },
            # Z.AI (Zhipu) GLM models
            "zai": {
                "edge": "glm-4.7-flash",  # Fast, low-cost valid flash tier
                "balanced": "glm-5-turbo",  # Faster flagship tier for routine decisions
                "performance": "glm-5.1",  # Highest-quality generally available GLM tier
            },
            # Moonshot (Kimi) models
            "moonshot": {
                "edge": "kimi-k2-instruct",
                "balanced": "kimi-k2-instruct",
                "performance": "kimi-k2-thinking",
            },
            # Local providers (Ollama)
            "ollama": {
                "edge": "qwen3.5:2b",  # Fast, lightweight (2.7GB)
                "balanced": "qwen2.5-coder:14b",  # Balanced performance (9GB)
                "performance": "qwen3.5:27b-q4_K_M",  # Full capability (17GB)
            },
            # LMStudio (local)
            "lmstudio": {
                "edge": "deepseek-r1-14b",  # Fast, lightweight reasoning
                "balanced": "qwen25-coder-tools-14b-64k",  # Balanced performance with tools
                "performance": "qwen3-coder-tools-30b-128k",  # Full capability with 128K context
            },
            # vLLM (local)
            "vllm": {
                "edge": "Qwen/Qwen2.5-3B-Instruct",
                "balanced": "Qwen/Qwen2.5-7B-Instruct",
                "performance": "meta-llama/Llama-3.1-70B-Instruct",
            },
            # GroqCloud
            "groqcloud": {
                "edge": "llama-3.1-8b-instant",
                "balanced": "llama-3.3-70b-versatile",
                "performance": "openai/gpt-oss-1b-preview",
            },
            # Cerebras
            "cerebras": {
                "edge": "llama-3.1-8b",
                "balanced": "llama-3.1-70b",
                "performance": "llama-3.1-70b",
            },
            # Together AI
            "together": {
                "edge": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                "balanced": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                "performance": "Qwen/Qwen2.5-72B-Instruct",
            },
            # Fireworks AI
            "fireworks": {
                "edge": "accounts/fireworks/models/llama-3.1-8b-instruct",
                "balanced": "accounts/fireworks/models/llama-3.1-70b-instruct",
                "performance": "accounts/fireworks/models/qwen-72b",
            },
            # Azure OpenAI
            "azure": {
                "edge": "gpt-4o-mini",
                "balanced": "gpt-4o",
                "performance": "o3-mini",
            },
            # AWS Bedrock
            "bedrock": {
                "edge": "anthropic.claude-3-5-haiku-20241022-v1:0",
                "balanced": "anthropic.claude-sonnet-4-20250514-v1:0",
                "performance": "anthropic.claude-opus-4-5-20251101-v1:0",
            },
            # Google Vertex AI
            "vertex": {
                "edge": "gemini-3.1-lite",  # Fast, lightweight
                "balanced": "gemini-3.1-flash",  # Balanced speed and capability
                "performance": "gemini-3.1-pro",  # Full capability
            },
        }
    )

    # Override capability: explicit provider/model per tier
    # Example: { "edge": {"provider": "ollama", "model": "phi3:mini"} }
    tier_overrides: Dict[str, Dict[str, str]] = Field(default_factory=dict)

    # Decision type → tier mapping
    tier_routing: Dict[str, str] = Field(
        default_factory=lambda: {
            "tool_selection": "edge",
            "skill_selection": "edge",
            "stage_detection": "edge",
            "intent_classification": "edge",
            "task_completion": "edge",
            "error_classification": "edge",
            "continuation_action": "edge",
            "loop_detection": "edge",
            "prompt_focus": "edge",
            "question_classification": "edge",
            "task_type_classification": "balanced",
            "multi_skill_decomposition": "balanced",
            "compaction": "auto",  # Auto-select based on complexity (simple→edge, complex→performance)
        }
    )

    def validate_provider_tiers(self) -> Dict[str, Any]:
        """Validate provider tier configurations.

        Returns:
            Dict with validation results:
            - valid: bool - overall validation status
            - warnings: List[str] - validation warnings
            - errors: List[str] - validation errors
        """
        result: Dict[str, Any] = {"valid": True, "warnings": [], "errors": []}
        required_tiers = {"edge", "balanced", "performance"}

        # Check each provider has all required tiers
        for provider, tiers in self.provider_model_tiers.items():
            missing_tiers = required_tiers - set(tiers.keys())
            if missing_tiers:
                result["warnings"].append(
                    f"Provider '{provider}' missing tiers: {sorted(missing_tiers)}"
                )
                result["valid"] = False

            # Check for unknown tiers
            unknown_tiers = set(tiers.keys()) - required_tiers
            if unknown_tiers:
                result["warnings"].append(
                    f"Provider '{provider}' has unknown tiers: {sorted(unknown_tiers)}"
                )

        # Validate tier overrides reference valid tiers
        for tier_name, override in self.tier_overrides.items():
            if tier_name not in required_tiers:
                result["errors"].append(f"Tier override references unknown tier: '{tier_name}'")
                result["valid"] = False

            # Check override has provider and model
            if "provider" not in override:
                result["errors"].append(f"Tier override for '{tier_name}' missing 'provider' key")
                result["valid"] = False
            if "model" not in override:
                result["errors"].append(f"Tier override for '{tier_name}' missing 'model' key")
                result["valid"] = False

        return result


# Rebuild the model to resolve forward references
DecisionServiceSettings.model_rebuild()
