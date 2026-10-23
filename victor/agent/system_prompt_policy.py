"""System prompt policy and fallbacks for Victor agents.

This module centralizes guardrails for constructing system prompts so that
PromptCoordinator can enforce consistent sections, deduplicate content, and
provide safe fallbacks whenever prompt assembly fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence

from victor.agent.prompting.builder import PromptBuilder

if TYPE_CHECKING:
    from victor.agent.services.prompt_runtime import PromptRuntimeContext as TaskContext

# Default prompt fragments used when contributors fail to populate sections.
DEFAULT_VICTOR_IDENTITY = (
    "You are Victor, the open-source agentic framework orchestrator. "
    "You coordinate tools, teams, and workflows to deliver reliable software results."
)

DEFAULT_VICTOR_GUIDELINES = (
    "- Always explain your plan before taking actions.\n"
    "- Prefer Victor tooling (agents, workflows, evaluators) over ad-hoc shell commands.\n"
    "- Call out limitations and ask clarifying questions when requirements are incomplete."
)

DEFAULT_OPERATION_PREAMBLE = (
    "Operating Mode: {stage}\n"
    "Task Focus: {task_type}\n"
    "Model: {model}\n"
    "Provider: {provider}"
)

DEFAULT_FALLBACK_PROMPT = (
    "You are Victor, a safety-focused software agent orchestrator. "
    "Primary task type: {task_type}. Current user request: {message}.\n"
    "Summarize constraints, outline a tool-driven plan, and highlight any missing context."
)


@dataclass
class SystemPromptPolicyConfig:
    """Configuration for SystemPromptPolicy."""

    enforce_identity: bool = True
    enforce_guidelines: bool = True
    enforce_operating_preamble: bool = True
    enforce_unique_sections: bool = True
    protected_sections: Sequence[str] = ("identity", "guidelines", "operating_mode")
    max_section_chars: int = 18000
    fallback_identity: str = DEFAULT_VICTOR_IDENTITY
    fallback_guidelines: str = DEFAULT_VICTOR_GUIDELINES
    fallback_operating_template: str = DEFAULT_OPERATION_PREAMBLE
    fallback_prompt_template: str = DEFAULT_FALLBACK_PROMPT


class SystemPromptPolicy:
    """Applies guardrails to PromptBuilder instances."""

    def __init__(self, config: Optional[SystemPromptPolicyConfig] = None) -> None:
        self._config = config or SystemPromptPolicyConfig()

    def enforce(self, builder: PromptBuilder, context: Optional["TaskContext"] = None) -> None:
        """Ensure required sections exist and trim oversized prompts."""

        if self._config.enforce_identity:
            builder.ensure_section(
                "identity",
                self._config.fallback_identity,
                priority=PromptBuilder.PRIORITY_IDENTITY,
                header="",
            )

        if self._config.enforce_guidelines:
            builder.ensure_section(
                "guidelines",
                self._config.fallback_guidelines,
                priority=PromptBuilder.PRIORITY_GUIDELINES,
            )

        if self._config.enforce_operating_preamble:
            operating = self._render_operating_preamble(context)
            builder.ensure_section(
                "operating_mode",
                operating,
                priority=PromptBuilder.PRIORITY_CAPABILITIES,
            )

        if self._config.enforce_unique_sections:
            builder.deduplicate_sections()

        if self._config.max_section_chars:
            builder.trim_sections_by_priority(
                max_total_chars=self._config.max_section_chars,
                protected_sections=self._config.protected_sections,
                min_priority=PromptBuilder.PRIORITY_CONTEXT,
            )

    def build_fallback_prompt(self, context: Optional["TaskContext"] = None) -> str:
        """Return a safe fallback prompt when assembly fails."""
        task_type = (context.task_type if context and context.task_type else "general").strip()
        message = (context.message if context and context.message else "").strip()
        if not message:
            message = "No user message captured."

        return self._config.fallback_prompt_template.format(
            task_type=task_type,
            message=message,
        )

    def _render_operating_preamble(self, context: Optional["TaskContext"]) -> str:
        """Create an operating mode section tailored to context."""
        task_type = (context.task_type if context else "general").upper()
        stage = context.stage if context and context.stage else "global"
        model = context.model if context and context.model else "unspecified"
        provider = context.provider if context and context.provider else "unspecified"
        return self._config.fallback_operating_template.format(
            stage=stage,
            task_type=task_type,
            model=model,
            provider=provider,
        )


def create_policy_from_settings(settings: Optional[Any]) -> SystemPromptPolicy:
    """Create a SystemPromptPolicy using VictorSettings or similar config objects."""

    base_config = SystemPromptPolicyConfig()
    if not settings:
        return SystemPromptPolicy(base_config)

    def _get(attr: str, default: Any) -> Any:
        value = getattr(settings, attr, default)
        return default if value is None else value

    protected_sections_override = getattr(settings, "prompt_policy_protected_sections", None)
    if protected_sections_override is None:
        protected_sections = base_config.protected_sections
    else:
        protected_sections = tuple(protected_sections_override)

    config = SystemPromptPolicyConfig(
        enforce_identity=_get("prompt_policy_enforce_identity", base_config.enforce_identity),
        enforce_guidelines=_get("prompt_policy_enforce_guidelines", base_config.enforce_guidelines),
        enforce_operating_preamble=_get(
            "prompt_policy_enforce_operating_preamble",
            base_config.enforce_operating_preamble,
        ),
        enforce_unique_sections=_get(
            "prompt_policy_enforce_unique_sections", base_config.enforce_unique_sections
        ),
        protected_sections=protected_sections,
        max_section_chars=_get("prompt_policy_max_section_chars", base_config.max_section_chars),
        fallback_identity=_get("prompt_policy_identity", base_config.fallback_identity),
        fallback_guidelines=_get("prompt_policy_guidelines", base_config.fallback_guidelines),
        fallback_operating_template=_get(
            "prompt_policy_operating_template", base_config.fallback_operating_template
        ),
        fallback_prompt_template=_get(
            "prompt_policy_fallback_template", base_config.fallback_prompt_template
        ),
    )

    return SystemPromptPolicy(config=config)


__all__ = [
    "SystemPromptPolicy",
    "SystemPromptPolicyConfig",
    "DEFAULT_VICTOR_IDENTITY",
    "DEFAULT_VICTOR_GUIDELINES",
    "DEFAULT_OPERATION_PREAMBLE",
    "DEFAULT_FALLBACK_PROMPT",
    "create_policy_from_settings",
]
