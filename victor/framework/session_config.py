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

"""Session configuration — immutable capture of CLI/runtime overrides.

This module provides ``SessionConfig``, a single dataclass that captures
all runtime overrides (CLI flags, environment variables, programmatic args)
in an **immutable** container.  The framework consumes it via
``Agent.create(session_config=...)`` instead of mutating ``Settings`` directly.

Design Rationale:
    - **Immutability**: CLI flags produce a ``SessionConfig`` once; the
      framework reads it but never mutates it.
    - **Traceability**: You can inspect ``session_config`` to see exactly
      what the caller overrode — no hidden ``settings.xxx = yyy`` mutations.
    - **Thread safety**: ``frozen=True`` dataclass is safe to share across
      coroutines without locking.

Migration Path:
    **OLD (deprecated):**
        settings.tool_settings.tool_output_preview_enabled = False
        settings.smart_routing_enabled = True

    **NEW:**
        from victor.framework.session_config import SessionConfig

        config = SessionConfig(
            tool_output_preview=False,
            smart_routing=True,
        )
        agent = await Agent.create(session_config=config)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from victor.framework.bayesian_config import BayesianConfig

DEFAULT_PROVIDER_MODELS = {
    "anthropic": "claude-3-5-sonnet-20241022",
    "openai": "gpt-4o",
    "google": "gemini-2.0-flash-exp",
    "ollama": "qwen2.5-coder:7b",
    "lmstudio": "local-model",
    "vllm": "local-model",
    "deepseek": "deepseek-chat",
    "xai": "grok-beta",
    "zai": "glm-4.7",
    "cohere": "command-r-plus",
    "azure": "gpt-4o",
}

LOCAL_ENDPOINT_PROVIDERS = frozenset({"ollama", "lmstudio", "vllm", "mlx", "llama.cpp"})


def _default_model_for_provider(provider: str) -> Optional[str]:
    """Resolve Victor-owned model policy without duplicating configured defaults."""
    default = DEFAULT_PROVIDER_MODELS.get(provider)
    if default is not None:
        return default
    try:
        from victor.providers.openai_compat_model_policy import get_openai_compat_provider_spec

        return get_openai_compat_provider_spec(provider).default_model
    except KeyError:
        return None


@dataclass(frozen=True)
class CompactionConfig:
    """Compaction threshold overrides for a session.

    Attributes:
        threshold: Compaction threshold (0.1-0.95). Lower = earlier compaction.
        adaptive: Enable adaptive threshold based on conversation patterns.
        min_threshold: Minimum adaptive threshold (0.1-0.8).
        max_threshold: Maximum adaptive threshold (0.2-0.95).
    """

    threshold: Optional[float] = None
    adaptive: Optional[bool] = None
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None


@dataclass(frozen=True)
class SmartRoutingConfig:
    """Smart provider routing overrides for a session.

    Attributes:
        enabled: Enable smart routing across providers.
        profile: Routing profile (e.g. 'balanced', 'performance', 'cost').
        fallback_chain: Comma-separated fallback provider chain.
    """

    enabled: bool = False
    profile: str = "balanced"
    fallback_chain: Optional[str] = None


@dataclass(frozen=True)
class ToolOutputConfig:
    """Tool output preview/pruning overrides for a session.

    Attributes:
        preview_enabled: Show tool output previews.
        pruning_enabled: Enable tool output pruning.
        pruning_safe_only: Only prune safe (read-heavy) tool outputs.
    """

    preview_enabled: bool = True
    pruning_enabled: bool = False
    pruning_safe_only: bool = True


@dataclass(frozen=True)
class ProviderOverrideConfig:
    """Explicit provider/model/runtime override state for a session."""

    provider: Optional[str] = None
    model: Optional[str] = None
    endpoint: Optional[str] = None
    auth_mode: Optional[str] = None
    timeout: Optional[int] = None
    coding_plan: bool = False

    @property
    def is_active(self) -> bool:
        """Return True when any explicit provider override is present."""
        return any(
            (
                self.provider is not None,
                self.model is not None,
                self.endpoint is not None,
                self.auth_mode is not None,
                self.timeout is not None,
                self.coding_plan,
            )
        )

    @classmethod
    def from_cli(
        cls,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        auth_mode: Optional[str] = None,
        timeout: Optional[int] = None,
        coding_plan: bool = False,
    ) -> "ProviderOverrideConfig":
        """Normalize provider override flags from CLI-friendly inputs."""
        normalized_provider = provider.lower() if provider else None
        normalized_model = model
        normalized_auth_mode = auth_mode.lower() if auth_mode else None

        if endpoint and normalized_provider and normalized_provider not in LOCAL_ENDPOINT_PROVIDERS:
            raise ValueError(
                "--endpoint is only supported for local providers "
                "(ollama, lmstudio, vllm, mlx, llama.cpp)."
            )
        if timeout is not None and timeout <= 0:
            raise ValueError("--provider-timeout must be greater than 0.")

        if normalized_provider and not normalized_model:
            normalized_model = _default_model_for_provider(normalized_provider)
            if normalized_model is None:
                raise ValueError(
                    f"No default model known for provider '{normalized_provider}'. "
                    "Please specify --model."
                )

        # Preserve existing CLI behavior: auth/coding-plan only participate in the
        # explicit override path when at least one provider-selection flag is set.
        has_selection = any((normalized_provider, normalized_model, endpoint))
        if not has_selection:
            normalized_auth_mode = None
            coding_plan = False

        return cls(
            provider=normalized_provider,
            model=normalized_model,
            endpoint=endpoint,
            auth_mode=normalized_auth_mode,
            timeout=timeout,
            coding_plan=coding_plan,
        )

    def to_profile_overrides(self) -> dict[str, object]:
        """Return extra profile payload fields for AgentFactory synthesis."""
        overrides: dict[str, object] = {}
        if self.endpoint:
            overrides["base_url"] = self.endpoint
        if self.auth_mode:
            overrides["auth_mode"] = self.auth_mode
        if self.timeout is not None:
            overrides["timeout"] = self.timeout
        if self.coding_plan:
            overrides["coding_plan"] = True
        return overrides


@dataclass(frozen=True)
class ToolApprovalConfig:
    """Human-in-the-loop tool-approval overrides for a session.

    When ``enabled`` is set, ``apply_to_settings`` turns on the governance policy
    engine (``governance.enabled`` + the ``USE_POLICY_ENGINE`` flag) and routes the
    named tools through the ASK path, so a surface that has registered a
    ``PolicyApprovalHandler`` is prompted before those tools run. This is the
    framework seam any non-TTY surface (web chat, TUI, API) consumes — it does not
    by itself register a handler.

    Attributes:
        enabled: Gate ASK-based tool approval for this session.
        ask_on_tools: Tool names that must be approved before executing.
        ask_fallback: Decision when no handler answers ("deny" | "allow").
        durable: Arm durable pause (FEP-0029): on ASK, park the turn as
            ``awaiting_approval`` (resumable) instead of blocking the inline modal.
            Opt-in (default False); intended on for headless/API surfaces.
    """

    enabled: bool = False
    ask_on_tools: tuple = ()
    ask_fallback: str = "deny"
    durable: bool = False


@dataclass(frozen=True)
class ShellSafetyConfig:
    """Shell safety policy overrides for a session (FEP-0013).

    Configures the damage-scoped :class:`ShellSafetyPolicy` wired to the shell
    tool. The default profile ``"legacy"`` preserves the shell tool's existing
    inline readonly allowlist (no behaviour change). Selecting a non-legacy
    profile (``"strict"``, ``"benchmark"``, ``"autonomous"``) installs a
    damage-scoped policy that allows in-workspace work (``pip install``,
    ``xargs``, ``sed -i``, redirects) while denying workspace escapes and
    protected-asset writes.

    Attributes:
        profile: Safety profile — ``"legacy"`` | ``"strict"`` | ``"benchmark"``
            | ``"autonomous"`` | ``"custom"``.
        workspace_root: Workspace boundary the agent may write within. ``None``
            resolves to the session working directory.
        protected_paths: Extra protected paths (``~/.victor`` and system dirs
            are always implicit).
        allow_network: ``True``/``False`` to force-allow/force-deny network
            egress; ``None`` defers to the profile default.
        extra_allow_patterns: Regex patterns that short-circuit to ALLOW.
        deny_patterns: Regex patterns that force DENY.
    """

    profile: str = "legacy"
    workspace_root: Optional[str] = None
    protected_paths: tuple = ()
    allow_network: Optional[bool] = None
    extra_allow_patterns: tuple = ()
    deny_patterns: tuple = ()

    @classmethod
    def from_cli(
        cls,
        *,
        profile: Optional[str] = None,
        workspace_root: Optional[str] = None,
        allow_network: Optional[bool] = None,
    ) -> "ShellSafetyConfig":
        """Normalize shell-safety flags from CLI-friendly inputs."""
        valid = {"legacy", "strict", "benchmark", "autonomous", "custom"}
        resolved = (profile or "legacy").lower()
        if resolved not in valid:
            raise ValueError(
                f"shell_safety_profile must be one of {sorted(valid)}, got '{profile}'."
            )
        return cls(
            profile=resolved,
            workspace_root=workspace_root,
            allow_network=allow_network,
        )


@dataclass(frozen=True)
class SessionConfig:
    """Immutable capture of all CLI/runtime session overrides.

    This is the **single config object** that the CLI (or any client)
    produces and passes to ``Agent.create()``.  The framework reads it
    but never mutates it — eliminating scattered ``settings.xxx = yyy``
    mutations throughout the codebase.

    Attributes:
        agent_profile: Agent profile name from ~/.victor/profiles.yaml (e.g., 'zai-coding', 'default').
        tool_budget: Override tool call budget for this session.
        max_iterations: Override maximum iterations for this session.
        compaction: Compaction threshold overrides.
        smart_routing: Smart provider routing overrides.
        tool_output: Tool output preview/pruning overrides.
        planning_enabled: Enable structured planning for complex tasks.
        planning_model: Override model for planning tasks.
        mode: Initial agent mode ('build', 'plan', 'explore').
        show_reasoning: Show LLM reasoning/thinking content.
        provider_override: Explicit provider/model/endpoint override state.
        tool_preview: Shorthand to disable tool output preview.
        enable_pruning: Shorthand to enable broader tool output pruning.
        enable_smart_routing: Shorthand to enable smart routing.
        routing_profile: Shorthand for routing profile.
        fallback_chain: Shorthand for fallback provider chain.
        compaction_threshold: Shorthand for compaction threshold.
        adaptive_threshold: Shorthand for adaptive compaction toggle.
        compaction_min_threshold: Shorthand for adaptive min threshold.
        compaction_max_threshold: Shorthand for adaptive max threshold.
    """

    # Agent profile (from ~/.victor/profiles.yaml)
    agent_profile: Optional[str] = None

    # Agent behaviour
    tool_budget: Optional[int] = None
    max_iterations: Optional[int] = None
    planning_enabled: Optional[bool] = None
    planning_model: Optional[str] = None
    mode: Optional[str] = None
    show_reasoning: bool = False
    observability_logging: Optional[bool] = None
    auto_skill_enabled: Optional[bool] = None
    one_shot_mode: Optional[bool] = None
    headless_mode: bool = False
    verify_mode: str = "none"
    lsp_feedback: str = "errors"
    lsp_perception: bool = False

    # Composed sub-configs (for structured access)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    smart_routing: SmartRoutingConfig = field(default_factory=SmartRoutingConfig)
    tool_output: ToolOutputConfig = field(default_factory=ToolOutputConfig)
    provider_override: ProviderOverrideConfig = field(default_factory=ProviderOverrideConfig)
    bayesian: BayesianConfig = field(default_factory=BayesianConfig)
    tool_approval: ToolApprovalConfig = field(default_factory=ToolApprovalConfig)
    shell_safety: ShellSafetyConfig = field(default_factory=ShellSafetyConfig)

    # --- Convenience shorthands (populate sub-configs) ---

    def __post_init__(self) -> None:
        """Merge shorthand flags into composed sub-configs and validate ranges.

        Because the dataclass is frozen, we use ``object.__setattr__``
        to populate the composed configs from flat shorthand fields.
        """
        # Validate numeric ranges
        if self.tool_budget is not None and self.tool_budget < 1:
            raise ValueError(f"tool_budget must be >= 1, got {self.tool_budget}")
        if self.max_iterations is not None and self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")

        # Compaction threshold validation (via composed config)
        if self.compaction.threshold is not None and not (0.0 <= self.compaction.threshold <= 1.0):
            raise ValueError(
                f"compaction.threshold must be in [0.0, 1.0], got {self.compaction.threshold}"
            )

        # Bayesian threshold validation (via composed config)
        if not (0.0 <= self.bayesian.simple_threshold <= 1.0):
            raise ValueError(
                f"bayesian.simple_threshold must be in [0.0, 1.0], got {self.bayesian.simple_threshold}"
            )
        if not (0.0 <= self.bayesian.complex_threshold <= 1.0):
            raise ValueError(
                f"bayesian.complex_threshold must be in [0.0, 1.0], got {self.bayesian.complex_threshold}"
            )

    @classmethod
    def from_cli_flags(
        cls,
        *,
        agent_profile: Optional[str] = None,
        tool_budget: Optional[int] = None,
        max_iterations: Optional[int] = None,
        compaction_threshold: Optional[float] = None,
        adaptive_threshold: Optional[bool] = None,
        compaction_min_threshold: Optional[float] = None,
        compaction_max_threshold: Optional[float] = None,
        enable_smart_routing: bool = False,
        routing_profile: str = "balanced",
        fallback_chain: Optional[str] = None,
        tool_preview: bool = True,
        enable_pruning: bool = False,
        planning_enabled: Optional[bool] = None,
        planning_model: Optional[str] = None,
        mode: Optional[str] = None,
        show_reasoning: bool = False,
        observability_logging: Optional[bool] = None,
        auto_skill_enabled: Optional[bool] = None,
        one_shot_mode: Optional[bool] = None,
        headless_mode: bool = False,
        verify_mode: str = "none",
        lsp_feedback: str = "errors",
        lsp_perception: bool = False,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        auth_mode: Optional[str] = None,
        provider_timeout: Optional[int] = None,
        coding_plan: bool = False,
        # Bayesian orchestration flags
        enable_bayesian: bool = True,
        force_bayesian: bool = False,
        simple_threshold: float = 0.3,
        complex_threshold: float = 0.7,
        enable_voi: bool = True,
        enable_correlation: bool = True,
        min_agents_for_bayesian: int = 2,
        # Human-in-the-loop tool approval
        tool_approval_enabled: bool = False,
        ask_on_tools: Optional[List[str]] = None,
        ask_fallback: str = "deny",
        # Shell safety policy (FEP-0013)
        shell_safety_profile: Optional[str] = None,
        shell_workspace_root: Optional[str] = None,
        shell_allow_network: Optional[bool] = None,
    ) -> "SessionConfig":
        """Create a ``SessionConfig`` from flat CLI flags.

        This is the primary factory for CLI code — collect all Typer
        options and pass them here to get an immutable config object.

        Args:
            agent_profile: Agent profile name from ~/.victor/profiles.yaml.
            tool_budget: Override tool call budget.
            max_iterations: Override max iterations.
            compaction_threshold: Compaction threshold (0.1-0.95).
            adaptive_threshold: Enable adaptive compaction.
            compaction_min_threshold: Adaptive min threshold.
            compaction_max_threshold: Adaptive max threshold.
            enable_smart_routing: Enable smart routing.
            routing_profile: Routing profile name.
            fallback_chain: Fallback provider chain.
            tool_preview: Show tool output previews.
            enable_pruning: Enable broader tool output pruning.
            planning_enabled: Enable structured planning.
            planning_model: Override model for planning.
            mode: Agent mode (build/plan/explore).
            show_reasoning: Show LLM reasoning.
            observability_logging: Enable event/observability logging for this session.
            auto_skill_enabled: Override skill auto-selection for this session.
            one_shot_mode: Override headless one-shot execution mode.
            provider: Override provider for this session.
            model: Override model for this session.
            endpoint: Override endpoint for local providers.
            auth_mode: Override provider auth mode.
            provider_timeout: Override provider request timeout in seconds.
            coding_plan: Enable provider-specific coding-plan endpoint mode.

        Returns:
            Immutable ``SessionConfig`` instance.

        Example::

            config = SessionConfig.from_cli_flags(
                tool_budget=50,
                enable_smart_routing=True,
                tool_preview=False,
            )
            agent = await Agent.create(session_config=config)
        """
        return cls(
            agent_profile=agent_profile,
            tool_budget=tool_budget,
            max_iterations=max_iterations,
            planning_enabled=planning_enabled,
            planning_model=planning_model,
            mode=mode,
            show_reasoning=show_reasoning,
            observability_logging=observability_logging,
            auto_skill_enabled=auto_skill_enabled,
            one_shot_mode=one_shot_mode,
            headless_mode=headless_mode,
            verify_mode=verify_mode,
            lsp_feedback=lsp_feedback,
            lsp_perception=lsp_perception,
            compaction=CompactionConfig(
                threshold=compaction_threshold,
                adaptive=adaptive_threshold,
                min_threshold=compaction_min_threshold,
                max_threshold=compaction_max_threshold,
            ),
            smart_routing=SmartRoutingConfig(
                enabled=enable_smart_routing,
                profile=routing_profile,
                fallback_chain=fallback_chain,
            ),
            tool_output=ToolOutputConfig(
                preview_enabled=tool_preview,
                pruning_enabled=enable_pruning,
                pruning_safe_only=not enable_pruning,
            ),
            provider_override=ProviderOverrideConfig.from_cli(
                provider=provider,
                model=model,
                endpoint=endpoint,
                auth_mode=auth_mode,
                timeout=provider_timeout,
                coding_plan=coding_plan,
            ),
            bayesian=BayesianConfig.from_cli_flags(
                enable_bayesian=enable_bayesian,
                force_bayesian=force_bayesian,
                simple_threshold=simple_threshold,
                complex_threshold=complex_threshold,
                enable_voi=enable_voi,
                enable_correlation=enable_correlation,
                min_agents_for_bayesian=min_agents_for_bayesian,
            ),
            tool_approval=ToolApprovalConfig(
                enabled=tool_approval_enabled,
                ask_on_tools=tuple(ask_on_tools or ()),
                ask_fallback=ask_fallback,
            ),
            shell_safety=ShellSafetyConfig.from_cli(
                profile=shell_safety_profile,
                workspace_root=shell_workspace_root,
                allow_network=shell_allow_network,
            ),
        )

    def apply_to_settings(self, settings: object) -> None:
        """Apply session overrides to a Settings object.

        This is the **only** place where Settings mutation should happen
        from session config.  All CLI code should call this method instead
        of directly mutating ``settings.xxx = yyy``.

        Args:
            settings: Application Settings instance.
        """
        # Tool output settings
        tool_settings = getattr(settings, "tool_settings", None)
        if tool_settings is not None:
            if hasattr(tool_settings, "tool_output_preview_enabled"):
                object.__setattr__(
                    tool_settings,
                    "tool_output_preview_enabled",
                    self.tool_output.preview_enabled,
                )
            if hasattr(tool_settings, "tool_output_pruning_enabled"):
                object.__setattr__(
                    tool_settings,
                    "tool_output_pruning_enabled",
                    self.tool_output.pruning_enabled,
                )
            if hasattr(tool_settings, "tool_output_pruning_safe_only"):
                object.__setattr__(
                    tool_settings,
                    "tool_output_pruning_safe_only",
                    self.tool_output.pruning_safe_only,
                )

        # Tool budget override (canonical consumer field is ``settings.tools``;
        # ``/system`` and the calibration seam both read
        # ``settings.tools.tool_call_budget``). Prior to this, ``tool_budget``
        # was validated by __post_init__ but never applied here, so an explicit
        # ``--tool-budget`` override was silently dropped. This also unblocks
        # the FEP-0002 calibration precedence path (explicit overrides win).
        if self.tool_budget is not None:
            tools_group = getattr(settings, "tools", None)
            if tools_group is not None and hasattr(tools_group, "tool_call_budget"):
                object.__setattr__(tools_group, "tool_call_budget", self.tool_budget)

        observability_settings = getattr(settings, "observability", None)
        if self.observability_logging is not None:
            if observability_settings is not None and hasattr(
                observability_settings, "enable_observability_logging"
            ):
                object.__setattr__(
                    observability_settings,
                    "enable_observability_logging",
                    self.observability_logging,
                )
            if hasattr(settings, "enable_observability_logging"):
                object.__setattr__(
                    settings,
                    "enable_observability_logging",
                    self.observability_logging,
                )

        if self.auto_skill_enabled is not None and hasattr(settings, "skill_auto_select_enabled"):
            object.__setattr__(
                settings,
                "skill_auto_select_enabled",
                self.auto_skill_enabled,
            )

        if self.one_shot_mode is not None:
            automation = getattr(settings, "automation", None)
            if automation is not None and hasattr(automation, "one_shot_mode"):
                object.__setattr__(automation, "one_shot_mode", self.one_shot_mode)
            if hasattr(settings, "one_shot_mode"):
                object.__setattr__(settings, "one_shot_mode", self.one_shot_mode)

        if self.headless_mode:
            automation = getattr(settings, "headless", None)
            if automation is not None and hasattr(automation, "headless_mode"):
                object.__setattr__(automation, "headless_mode", True)
            if hasattr(settings, "headless_mode"):
                object.__setattr__(settings, "headless_mode", True)

        provider_override = self.provider_override
        provider_settings = getattr(settings, "provider", None)
        if provider_settings is not None:
            if provider_override.provider and hasattr(provider_settings, "default_provider"):
                object.__setattr__(
                    provider_settings,
                    "default_provider",
                    provider_override.provider,
                )
            if provider_override.model and hasattr(provider_settings, "default_model"):
                object.__setattr__(
                    provider_settings,
                    "default_model",
                    provider_override.model,
                )
            if provider_override.timeout is not None and hasattr(provider_settings, "timeout"):
                object.__setattr__(
                    provider_settings,
                    "timeout",
                    provider_override.timeout,
                )
            if provider_override.endpoint:
                if provider_override.provider == "ollama" and hasattr(
                    provider_settings, "ollama_base_url"
                ):
                    object.__setattr__(
                        provider_settings,
                        "ollama_base_url",
                        provider_override.endpoint,
                    )
                elif provider_override.provider == "lmstudio" and hasattr(
                    provider_settings, "lmstudio_base_urls"
                ):
                    object.__setattr__(
                        provider_settings,
                        "lmstudio_base_urls",
                        [provider_override.endpoint],
                    )
                elif provider_override.provider == "vllm" and hasattr(
                    provider_settings, "vllm_base_url"
                ):
                    object.__setattr__(
                        provider_settings,
                        "vllm_base_url",
                        provider_override.endpoint,
                    )

        # Smart routing settings
        if self.smart_routing.enabled:
            if hasattr(settings, "smart_routing_enabled"):
                object.__setattr__(settings, "smart_routing_enabled", True)
            routing = getattr(settings, "routing", None)
            if routing is not None:
                if hasattr(routing, "profile"):
                    object.__setattr__(routing, "profile", self.smart_routing.profile)
                if hasattr(routing, "fallback_chain") and self.smart_routing.fallback_chain:
                    object.__setattr__(routing, "fallback_chain", self.smart_routing.fallback_chain)

        # Human-in-the-loop tool approval: turn on the governance policy engine and
        # route the named tools through the ASK path. A surface still has to register a
        # PolicyApprovalHandler in the container to answer the prompt; without one, ASK
        # resolves via ask_fallback (default "deny").
        if self.tool_approval.enabled:
            governance = getattr(settings, "governance", None)
            if governance is not None:
                if hasattr(governance, "enabled"):
                    object.__setattr__(governance, "enabled", True)
                if hasattr(governance, "ask_fallback"):
                    object.__setattr__(governance, "ask_fallback", self.tool_approval.ask_fallback)
                if hasattr(governance, "ask_on_tools"):
                    # Union with any pre-configured tools, preserving order.
                    existing = list(getattr(governance, "ask_on_tools", []) or [])
                    merged = existing + [
                        t for t in self.tool_approval.ask_on_tools if t not in existing
                    ]
                    object.__setattr__(governance, "ask_on_tools", merged)
                # FEP-0029: durable pause opt-in — park ASK as awaiting_approval instead of blocking.
                if hasattr(governance, "durable"):
                    object.__setattr__(governance, "durable", self.tool_approval.durable)
                # The policy engine is feature-flag gated; requesting approval implies it.
                try:
                    from victor.core.feature_flags import FeatureFlag, enable_feature

                    enable_feature(FeatureFlag.USE_POLICY_ENGINE)
                except Exception:  # pragma: no cover - defensive
                    pass
