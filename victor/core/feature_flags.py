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

"""Feature flag system for gradual rollout of new features.

This module provides a comprehensive feature flag system that supports:
- Environment variable configuration
- YAML configuration file loading
- Runtime flag enable/disable
- Thread-safe flag management

Feature flags enable gradual rollout of new architecture components
without breaking existing functionality.

Example:
    from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag

    manager = get_feature_flag_manager()

    # Check if a feature is enabled
    if manager.is_enabled(FeatureFlag.USE_SMART_ROUTING):
        # Use smart routing
        pass
    else:
        # Use default provider
        pass

    # Enable a feature at runtime
    manager.enable(FeatureFlag.USE_SMART_ROUTING)

Environment Variables:
    VICTOR_USE_SMART_ROUTING=true
    VICTOR_USE_EDGE_MODEL=false
    VICTOR_USE_COMPOSITION_OVER_INHERITANCE=true

YAML Configuration (~/.victor/features.yaml):
    features:
        use_smart_routing: true
        use_edge_model: false
        use_composition_over_inheritance: true
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from victor.core.yaml_utils import safe_load as yaml_safe_load

logger = logging.getLogger(__name__)


class FeatureFlag(Enum):
    """Feature flags for gradual rollout of new features.

    Active flags guard real behavior in the codebase. Removed flags that
    no longer guard behavior (Phase 3 service layer flags removed in W3 cleanup).

    Usage:
        Set via environment variable: VICTOR_USE_SMART_ROUTING=true
        Set via YAML config: features.use_smart_routing: true
        Enable at runtime: manager.enable(FeatureFlag.USE_SMART_ROUTING)
    """

    # Phase 4 - Vertical Composition
    USE_COMPOSITION_OVER_INHERITANCE = "use_composition_over_inheritance"

    # Phase 5 - Tool Registration
    USE_STRATEGY_BASED_TOOL_REGISTRATION = "use_strategy_based_tool_registration"

    # Phase 7 - LLM Decision Service
    USE_LLM_DECISION_SERVICE = "use_llm_decision_service"

    # Phase 8 - Edge Model for micro-decisions
    USE_EDGE_MODEL = "use_edge_model"

    # Optimization flags (default: False — opt-in only)
    USE_SEMANTIC_RESPONSE_CACHE = "use_semantic_response_cache"

    # Phase 11 - Smart Model Routing (automatic local→cloud fallback)
    USE_SMART_ROUTING = "use_smart_routing"

    # Priority 4 - Learning from Execution (meta-learning, user feedback, explainability)
    # Default: enabled. Roll back with VICTOR_USE_LEARNING_FROM_EXECUTION=false.
    USE_LEARNING_FROM_EXECUTION = "use_learning_from_execution"

    # Tool Broadcasting Optimization (context-aware, economy-first).
    # Read by string in orchestrator._is_tool_strategy_v2_enabled(); the former
    # settings field duplicate was removed in #324 (this flag is the real gate).
    TOOL_STRATEGY_V2 = "tool_strategy_v2"

    # Phase 13 - Agentic rollout controls (default: False — opt-in only)
    USE_CALIBRATED_COMPLETION = "use_calibrated_completion"
    USE_PROMPT_COMPLETENESS_GUARD = "use_prompt_completeness_guard"
    USE_PROMPT_DICTIONARY_COMPRESSION = "use_prompt_dictionary_compression"
    USE_PRIME_MEMORY_EVOLUTION = "use_prime_memory_evolution"

    # Phase 9 - Prompt Optimization (controlled via settings.prompt_optimization)
    # USE_PROMPT_OPTIMIZER → settings.prompt_optimization.enabled
    # USE_GEPA_V2          → settings.prompt_optimization.gepa.enabled
    # USE_GEPA_TRACE_ENRICHMENT → settings.prompt_optimization.gepa.capture_reasoning

    # Phase 12 - Rich Formatting (unified formatter system with preview integration)
    # USE_RICH_FORMATTING → settings.rich_formatting_enabled (master switch)
    # This flag controls the entire Rich formatter system including:
    # - Tool output formatting with Rich markup
    # - Preview strategy integration
    # - Performance guards and error handling
    USE_RICH_FORMATTING = "use_rich_formatting"

    # Phase 14 - Graph-Based Enhancements (unified graph schema, CCG, Graph RAG)
    # These flags enable the new graph-based code intelligence features.
    # NOTE: USE_GRAPH_RAG + USE_MULTI_HOP_RETRIEVAL are read by victor-coding's
    # code_search_tool (and documented as VICTOR_USE_GRAPH_RAG /
    # VICTOR_USE_MULTI_HOP_RETRIEVAL env vars) — do NOT remove without checking
    # verticals/. They were wrongly classified dead by a root-only audit (#539).
    USE_GRAPH_RAG = "use_graph_rag"
    USE_CCG = "use_ccg"
    USE_GRAPH_QUERY_TOOL = "use_graph_query_tool"
    USE_MULTI_HOP_RETRIEVAL = "use_multi_hop_retrieval"
    USE_GRAPH_ENHANCED_CONTEXT = "use_graph_enhanced_context"
    # Gates graph-node embedding generation during BACKGROUND refreshes
    # (GraphManager); `victor graph index --embeddings` works regardless.
    # Default off: background embedding costs model inference on every refresh.
    USE_GRAPH_EMBEDDINGS = "use_graph_embeddings"

    # Phase 15 - Architecture Consolidation (Framework Primitives)
    # USE_STATEGRAPH_AGENTIC_LOOP gates the StateGraph executor inside
    # AgenticLoop.run()/stream() (see victor/framework/agentic_loop.py). The other
    # Phase-15 placeholders (USE_FRAMEWORK_TEAMS, USE_FRAMEWORK_COORDINATORS,
    # USE_CONTEXT_SERVICE_INJECTION) were removed — they were defined but
    # never read by production code. Coordinator/team consolidation now ships
    # unconditionally; teams use UnifiedTeamCoordinator directly.
    USE_STATEGRAPH_AGENTIC_LOOP = "use_stategraph_agentic_loop"

    # Phase 16 - Stage Transition Coordination (Streaming Path Optimization)
    # USE_STAGE_TRANSITION_COORDINATOR enables batching of tool executions
    # and Phase 1 optimizations (cooldown, high confidence skip) in the
    # streaming path. Fixes the issue where the legacy streaming runtime bypasses
    # Phase 1 optimizations and causes stage thrashing.
    # Default: False (opt-in for gradual rollout)
    USE_STAGE_TRANSITION_COORDINATOR = "use_stage_transition_coordinator"

    # Phase 17 - Tiered Classification (confidence-based triage for classifiers)
    # Adds confidence-band routing to TaskTypeClassifier, IntentClassifier, and
    # SemanticToolSelector:
    #   HIGH (≥0.8): Accept immediately (fast path, ~4-5μs)
    #   MEDIUM (0.5-0.8): Verify with edge LLM (~50-200ms)
    #   LOW (<0.5): Reject early, return safe fallback
    # Reuses TieredDecisionService and RuntimeEvaluationPolicy — no new infra.
    # Default: False (opt-in, enable with VICTOR_USE_TIERED_CLASSIFICATION=true)
    USE_TIERED_CLASSIFICATION = "use_tiered_classification"

    # Phase 18 - Governance Policy Engine (ALLOW/DENY/ASK over tool execution)
    # Gates wiring of PolicyEngineMiddleware into the tool pipeline (see
    # victor/framework/policies/ and settings.governance). Enabling requires
    # both this flag and settings.governance.enabled=True.
    # Default: False (opt-in, enable with VICTOR_USE_POLICY_ENGINE=true)
    USE_POLICY_ENGINE = "use_policy_engine"

    # Phase 19 - E3-TIR Experience-Replay Tool Exploration
    # Gates the E3-TIR post-selection reranker (victor/tools/e3_tir_selector.py),
    # which reranks the per-turn tool set using experience-based exploration phases
    # (demonstration warm-up -> self-play -> targeted exploration) with mode-collapse
    # prevention. Applied only for non-KV/per-turn providers (it reorders tools per
    # turn, which conflicts with KV-prefix-stable tool ordering). Requires
    # USE_LEARNING_FROM_EXECUTION to also be enabled.
    # Default: False (opt-in, enable with VICTOR_USE_E3_TIR_EXPLORATION=true)
    USE_E3_TIR_EXPLORATION = "use_e3_tir_exploration"

    # Phase 20 - Context Management Subsystem Activation (FEP-0023, Phase 1)
    # Gates population of the live SessionLedger (orchestrator._session_ledger)
    # from tool results and assistant responses at the per-turn boundary. When
    # ON, the assembler's already-wired <SESSION_STATE> block is populated with
    # files read/modified, decisions, and recommendations (persistent session
    # memory that survives compaction). When OFF the ledger stays empty and the
    # block is inert. Keystone for FEP-0023 Phases 2-3.
    #
    # Default: ON. Graduated from opt-in on the FEP-0023 P1 measurement: over
    # 12.7k real assistant messages the (hardened) extractor holds ~100% labeled
    # precision, and <SESSION_STATE> costs ~400 tokens/turn (hard-capped at
    # ~750). Disable with VICTOR_USE_SESSION_LEDGER=false. The one dimension not
    # yet measured is task-accuracy A/B (EVR-gated); this flag retro-gates it.
    USE_SESSION_LEDGER = "use_session_ledger"

    # Phase 20 - Context Management Subsystem Activation (FEP-0023, Phase 2)
    # Gates the ToolResultDeduplicator as a *view stage* inside the context
    # assembler: older duplicate tool outputs (identical file re-reads) in the
    # assembled context copy are replaced with compact stubs to cut tokens. It
    # operates on the assembler's throwaway copy and NEVER mutates the
    # source-of-truth history. When OFF the assembler passes context through
    # unchanged.
    #
    # Default: ON. Graduated from opt-in on the FEP-0023 P2 measurement: across 9
    # real victor sessions ~3% of large (>=500 char) tool outputs recurred
    # identically (~460+ chars saved per stub; ~7k tokens in-sample), and this is
    # complementary to — not redundant with — the cross-turn idempotent-call cache
    # (a different layer: execution vs context tokens, and those duplicates reached
    # history despite it). Zero accuracy risk: exact-content dups only, newest copy
    # retained, view-only. Disable with VICTOR_USE_TOOL_RESULT_DEDUP=false.
    USE_TOOL_RESULT_DEDUP = "use_tool_result_dedup"

    def get_env_var_name(self) -> str:
        """Get the environment variable name for this flag.

        Returns:
            Environment variable name (e.g., VICTOR_USE_SMART_ROUTING)
        """
        return f"VICTOR_{self.value.upper()}"

    def get_yaml_key(self) -> str:
        """Get the YAML configuration key for this flag.

        Returns:
            YAML key (e.g., use_edge_model)
        """
        return self.value

    def is_opt_in_by_default(self) -> bool:
        """Whether the flag should stay disabled unless explicitly enabled."""
        return self in {
            FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE,
            FeatureFlag.TOOL_STRATEGY_V2,
            FeatureFlag.USE_CALIBRATED_COMPLETION,
            FeatureFlag.USE_PROMPT_COMPLETENESS_GUARD,
            FeatureFlag.USE_PROMPT_DICTIONARY_COMPRESSION,
            FeatureFlag.USE_PRIME_MEMORY_EVOLUTION,
            # Phase 15: Architecture Consolidation (opt-in for safety)
            FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP,
            # Phase 16: Stage Transition Coordination (opt-in for gradual rollout)
            FeatureFlag.USE_STAGE_TRANSITION_COORDINATOR,
            # Phase 17: Tiered Classification (opt-in, off by default)
            FeatureFlag.USE_TIERED_CLASSIFICATION,
            # Phase 18: Governance Policy Engine (opt-in, off by default)
            FeatureFlag.USE_POLICY_ENGINE,
            # Phase 19: E3-TIR experience-replay tool exploration (opt-in, off by default)
            FeatureFlag.USE_E3_TIR_EXPLORATION,
            # Background graph-node embedding generation (opt-in: costs model
            # inference on every background refresh)
            FeatureFlag.USE_GRAPH_EMBEDDINGS,
        }

    def get_default_enabled(self, fallback: bool) -> bool:
        """Return the effective default state for this flag."""
        if self.is_opt_in_by_default():
            return False
        return fallback


def get_flag_manifest() -> List[Dict[str, object]]:
    """Return the code-level default state of every ``FeatureFlag``, sorted by name.

    This is the single source of truth for the generated flag inventory
    (``docs/architecture/feature-flags.md``) and the ``feature_flags`` block of
    ``victor capabilities --json``. A flag's **code default** is OFF iff it is in
    ``is_opt_in_by_default()`` — computed here with ``fallback=True`` (the default
    ``FeatureFlagConfig.default_enabled``), i.e. assuming no YAML/env override.

    Prose in docs should cite this manifest rather than restating per-flag defaults
    (which drift — see F-016/TD-17); the regen guard test fails if this drifts from
    the enum.

    Returns:
        A list of ``{"name": str, "opt_in": bool, "default": bool}`` entries.
    """
    manifest = [
        {
            "name": flag.value,
            "opt_in": flag.is_opt_in_by_default(),
            "default": flag.get_default_enabled(True),
        }
        for flag in FeatureFlag
    ]
    manifest.sort(key=lambda entry: str(entry["name"]))
    return manifest


@dataclass
class FeatureFlagConfig:
    """Configuration for feature flag loading.

    Attributes:
        config_path: Path to YAML configuration file
        env_prefix: Prefix for environment variables
        default_enabled: Default state for flags not explicitly configured
        strict_mode: If True, raises errors on invalid config; if False, logs warnings
    """

    config_path: Optional[Path] = None
    env_prefix: str = "VICTOR_"
    default_enabled: bool = True
    strict_mode: bool = False


class FeatureFlagManager:
    """Manage feature flags with environment variable and YAML config support.

    The manager loads flag values from multiple sources in order of precedence:
    1. Runtime enable() calls (highest priority)
    2. Environment variables
    3. YAML configuration file
    4. Default value (lowest priority)

    Thread-safe for concurrent access to flag state.

    Example:
        manager = FeatureFlagManager()

        # Check if a feature is enabled
        if manager.is_enabled(FeatureFlag.USE_SMART_ROUTING):
            use_smart_routing()

        # Enable a feature at runtime
        manager.enable(FeatureFlag.USE_SMART_ROUTING)

        # Get all enabled flags
        enabled_flags = manager.get_enabled_flags()
    """

    def __init__(self, config: Optional[FeatureFlagConfig] = None) -> None:
        """Initialize feature flag manager.

        Args:
            config: Optional configuration for flag loading behavior
        """
        self._config = config or FeatureFlagConfig()
        self._flags: Dict[FeatureFlag, bool] = {}
        self._runtime_overrides: Dict[FeatureFlag, bool] = {}
        self._lock = threading.RLock()
        self._load_from_config()

    def is_enabled(self, flag: FeatureFlag) -> bool:
        """Check if a feature flag is enabled.

        Checks in order of precedence:
        1. Runtime overrides (from enable() calls)
        2. Environment variables
        3. YAML configuration
        4. Default value

        Args:
            flag: Feature flag to check

        Returns:
            True if the flag is enabled, False otherwise
        """
        with self._lock:
            # Check runtime override first
            if flag in self._runtime_overrides:
                return self._runtime_overrides[flag]

            # Check environment variable
            env_var = flag.get_env_var_name()
            if env_var in os.environ:
                return os.environ[env_var].lower() in ("true", "1", "yes", "on")

            # Check loaded configuration
            if flag in self._flags:
                return self._flags[flag]

            # Return default
            return flag.get_default_enabled(self._config.default_enabled)

    def enable(self, flag: FeatureFlag) -> None:
        """Enable a feature flag at runtime.

        Runtime overrides take precedence over environment variables
        and configuration files.

        Args:
            flag: Feature flag to enable
        """
        with self._lock:
            self._runtime_overrides[flag] = True
            logger.debug(f"Enabled feature flag: {flag.value}")

    def disable(self, flag: FeatureFlag) -> None:
        """Disable a feature flag at runtime.

        Runtime overrides take precedence over environment variables
        and configuration files.

        Args:
            flag: Feature flag to disable
        """
        with self._lock:
            self._runtime_overrides[flag] = False
            logger.debug(f"Disabled feature flag: {flag.value}")

    def clear_runtime_override(self, flag: FeatureFlag) -> None:
        """Clear runtime override for a feature flag.

        After clearing, the flag value will be determined by
        environment variables or configuration files.

        Args:
            flag: Feature flag to clear override for
        """
        with self._lock:
            self._runtime_overrides.pop(flag, None)
            logger.debug(f"Cleared runtime override for: {flag.value}")

    def set(self, flag: FeatureFlag, enabled: bool) -> None:
        """Set a feature flag state at runtime.

        Convenience method that calls enable() or disable().

        Args:
            flag: Feature flag to set
            enabled: Whether to enable or disable the flag
        """
        if enabled:
            self.enable(flag)
        else:
            self.disable(flag)

    def get_enabled_flags(self) -> Dict[FeatureFlag, bool]:
        """Get all feature flags and their current states.

        Returns:
            Dictionary mapping feature flags to their enabled state
        """
        with self._lock:
            return {flag: self.is_enabled(flag) for flag in FeatureFlag}

    def reload_config(self) -> None:
        """Reload configuration from environment variables and YAML file.

        Clears runtime overrides and reloads flag values from configuration sources.
        """
        with self._lock:
            self._runtime_overrides.clear()
            self._flags.clear()
            self._load_from_config()
            logger.info("Reloaded feature flag configuration")

    def _load_from_config(self) -> None:
        """Load flag values from environment variables and YAML configuration."""
        # Load from YAML file if path is specified
        if self._config.config_path and self._config.config_path.exists():
            self._load_from_yaml()

        # Environment variables are checked directly in is_enabled()
        # No need to preload them

    def _load_from_yaml(self) -> None:
        """Load flag values from YAML configuration file.

        Expected YAML format:
            features:
                use_edge_model: true
                use_llm_decision_service: false
                use_composition_over_inheritance: true
        """
        try:
            with open(self._config.config_path, "r") as f:
                data = yaml_safe_load(f)

            if not data or not isinstance(data, dict):
                logger.warning(f"Invalid feature flag config: {self._config.config_path}")
                return

            features = data.get("features", {})
            if not isinstance(features, dict):
                logger.warning("Invalid 'features' section in config")
                return

            # Map YAML keys to FeatureFlag enums
            yaml_key_to_flag = {flag.get_yaml_key(): flag for flag in FeatureFlag}

            for key, value in features.items():
                if key in yaml_key_to_flag:
                    flag = yaml_key_to_flag[key]
                    if isinstance(value, bool):
                        self._flags[flag] = value
                    elif self._config.strict_mode:
                        raise ValueError(
                            f"Invalid value for feature flag '{key}': "
                            f"expected bool, got {type(value).__name__}"
                        )
                    else:
                        logger.warning(
                            f"Invalid value for feature flag '{key}': "
                            f"expected bool, got {type(value).__name__}"
                        )

        except Exception as e:
            if self._config.strict_mode:
                raise
            logger.warning(f"Failed to load feature flags from YAML: {e}")


# =============================================================================
# Global Manager
# =============================================================================

_global_manager: Optional[FeatureFlagManager] = None
_global_lock = threading.Lock()


def get_feature_flag_manager(
    config: Optional[FeatureFlagConfig] = None,
    force_reload: bool = False,
) -> FeatureFlagManager:
    """Get the global feature flag manager.

    Creates a new manager if one doesn't exist or if force_reload is True.

    Args:
        config: Optional configuration for the manager
        force_reload: If True, recreate the manager even if one exists

    Returns:
        Global feature flag manager instance
    """
    global _global_manager

    with _global_lock:
        if _global_manager is None or force_reload:
            # Determine default config path
            if config is None or config.config_path is None:
                from victor.config.settings import GLOBAL_VICTOR_DIR

                default_config_path = GLOBAL_VICTOR_DIR / "features.yaml"
                if config is None:
                    config = FeatureFlagConfig(config_path=default_config_path)
                else:
                    # Update existing config with default path
                    config = FeatureFlagConfig(
                        config_path=default_config_path,
                        env_prefix=config.env_prefix,
                        default_enabled=config.default_enabled,
                        strict_mode=config.strict_mode,
                    )

            _global_manager = FeatureFlagManager(config)

        return _global_manager


def reset_feature_flag_manager() -> None:
    """Reset the global feature flag manager.

    Primarily useful for testing when you need to clear flag state.
    """
    global _global_manager
    with _global_lock:
        _global_manager = None


def is_feature_enabled(flag: FeatureFlag) -> bool:
    """Convenience function to check if a feature flag is enabled.

    Args:
        flag: Feature flag to check

    Returns:
        True if the flag is enabled, False otherwise
    """
    return get_feature_flag_manager().is_enabled(flag)


def enable_feature(flag: FeatureFlag) -> None:
    """Convenience function to enable a feature flag.

    Args:
        flag: Feature flag to enable
    """
    get_feature_flag_manager().enable(flag)


def disable_feature(flag: FeatureFlag) -> None:
    """Convenience function to disable a feature flag.

    Args:
        flag: Feature flag to disable
    """
    get_feature_flag_manager().disable(flag)
