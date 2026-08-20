# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Canonical runtime adapter for prompt coordination protocol consumers.

This module provides the non-deprecated implementation behind
``PromptRuntimeProtocol``. It preserves the narrower mutable prompt-coordination
contract used by DI/runtime seams without routing through the deprecated
``PromptCoordinator`` shim.

``UnifiedPromptPipeline`` remains the canonical owner of orchestrator-managed
live prompt assembly. This adapter exists for protocol consumers that still
depend on task hints, runtime sections, and grounding-mode mutation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from victor.agent.system_prompt_policy import SystemPromptPolicy
from victor.framework.prompt_builder import PromptBuilder

if TYPE_CHECKING:
    from victor.agent.vertical_context import VerticalContext

logger = logging.getLogger(__name__)


@dataclass
class PromptRuntimeContext:
    """Context for protocol-driven prompt building."""

    message: str
    task_type: str = "unknown"
    complexity: str = "medium"
    stage: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptRuntimeConfig:
    """Configuration for the canonical prompt runtime adapter."""

    default_grounding_mode: str = "minimal"
    enable_task_hints: bool = True
    enable_vertical_sections: bool = True
    enable_safety_rules: bool = True
    max_context_tokens: int = 2000


class PromptRuntimeAdapter:
    """Canonical implementation of the mutable prompt runtime contract."""

    def __init__(
        self,
        prompt_builder: Optional[PromptBuilder] = None,
        vertical_context: Optional["VerticalContext"] = None,
        config: Optional[PromptRuntimeConfig] = None,
        base_identity: Optional[str] = None,
        on_prompt_built: Optional[Callable[[str, PromptRuntimeContext], None]] = None,
        policy: Optional[SystemPromptPolicy] = None,
    ) -> None:
        self._builder = prompt_builder or PromptBuilder()
        self._vertical_context = vertical_context
        self._config = config or PromptRuntimeConfig()
        self._base_identity = base_identity
        self._on_prompt_built = on_prompt_built
        self._policy = policy or SystemPromptPolicy()

        self._task_hints: Dict[str, str] = {}
        self._additional_sections: Dict[str, str] = {}
        self._safety_rules: List[str] = []

        logger.debug(
            "PromptRuntimeAdapter initialized with grounding_mode=%s",
            self._config.default_grounding_mode,
        )

    @property
    def vertical_context(self) -> Optional["VerticalContext"]:
        """Get the current vertical context."""
        return self._vertical_context

    @vertical_context.setter
    def vertical_context(self, context: Optional["VerticalContext"]) -> None:
        """Set the vertical context."""
        self._vertical_context = context

    def build_system_prompt(
        self,
        context: PromptRuntimeContext,
        include_hints: bool = True,
    ) -> str:
        """Build the complete system prompt for the provided task context."""
        builder = self._prepare_builder(context, include_hints=include_hints)

        try:
            prompt = self._build_with_pipeline(builder)
        except Exception:
            logger.exception("Failed to build system prompt via UnifiedPromptPipeline.")
            try:
                prompt = builder.build()
            except Exception:
                logger.exception("Failed to build system prompt. Using fallback.")
                prompt = self._policy.build_fallback_prompt(context)
            else:
                if not prompt.strip():
                    logger.warning("Empty system prompt produced; using fallback string.")
                    prompt = self._policy.build_fallback_prompt(context)

        if self._on_prompt_built:
            self._on_prompt_built(prompt, context)

        logger.debug(
            "Built system prompt for task_type=%s, length=%d chars",
            context.task_type,
            len(prompt),
        )
        return prompt

    def _prepare_builder(
        self,
        context: PromptRuntimeContext,
        *,
        include_hints: bool,
    ) -> PromptBuilder:
        """Assemble mutable prompt state into a fresh builder instance."""
        builder = PromptBuilder()

        if self._base_identity:
            builder.add_section(
                "identity",
                self._base_identity,
                priority=PromptBuilder.PRIORITY_IDENTITY,
                header="",
            )

        if self._config.enable_vertical_sections and self._vertical_context:
            self._add_vertical_sections(builder, context)

        if include_hints and self._config.enable_task_hints:
            self._add_task_hint(builder, context)

        for name, content in self._additional_sections.items():
            builder.add_section(name, content)

        if self._config.enable_safety_rules and self._safety_rules:
            builder.add_safety_rules(self._safety_rules)

        if context.additional_context:
            remaining_budget = self._config.max_context_tokens
            for key, value in context.additional_context.items():
                if isinstance(value, str):
                    chunk = f"{key}: {value}".strip()
                    if not chunk:
                        continue

                    if remaining_budget is not None:
                        if remaining_budget <= 0:
                            logger.debug(
                                "Context budget exceeded for system prompt; skipping remainder."
                            )
                            break
                        chunk_length = len(chunk)
                        if chunk_length > remaining_budget:
                            chunk = chunk[:remaining_budget]
                            remaining_budget = 0
                        else:
                            remaining_budget -= chunk_length

                    builder.add_context(chunk)

        builder.set_grounding_mode(self._config.default_grounding_mode)

        try:
            if self._policy:
                self._policy.enforce(builder, context)
        except Exception:
            logger.exception("System prompt policy enforcement failed. Continuing without policy.")

        return builder

    def _build_with_pipeline(self, builder: PromptBuilder) -> str:
        """Delegate canonical prompt assembly to UnifiedPromptPipeline."""
        from victor.agent.prompt_pipeline import UnifiedPromptPipeline

        pipeline = UnifiedPromptPipeline(
            provider=None,
            builder=builder,
        )
        prompt = pipeline.build_system_prompt()
        if not prompt.strip():
            raise ValueError("UnifiedPromptPipeline returned an empty system prompt")
        return prompt

    def _add_vertical_sections(
        self,
        builder: PromptBuilder,
        context: PromptRuntimeContext,
    ) -> None:
        """Add vertical-specific sections to the builder."""
        del context
        if not self._vertical_context:
            return

        prompt_ext = self._vertical_context.get_prompt_extensions()
        if not prompt_ext:
            return

        sections = prompt_ext.get_combined_system_prompt_sections()
        if sections:
            builder.add_section(
                "vertical_guidelines",
                sections,
                priority=PromptBuilder.PRIORITY_GUIDELINES + 5,
            )

        grounding = prompt_ext.get_grounding_rules()
        if grounding:
            builder.set_custom_grounding(grounding)

    def _add_task_hint(
        self,
        builder: PromptBuilder,
        context: PromptRuntimeContext,
    ) -> None:
        """Add task-type-specific hint to the builder."""
        task_type = context.task_type.lower()

        if task_type in self._task_hints:
            builder.add_section(
                "task_hint",
                self._task_hints[task_type],
                priority=PromptBuilder.PRIORITY_TASK_HINTS,
                header="",
            )
            return

        if self._vertical_context:
            prompt_ext = self._vertical_context.get_prompt_extensions()
            if prompt_ext:
                hint = prompt_ext.get_hint_for_task(task_type)
                if hint:
                    hint_text = hint.hint if hasattr(hint, "hint") else str(hint)
                    builder.add_section(
                        "task_hint",
                        hint_text,
                        priority=PromptBuilder.PRIORITY_TASK_HINTS,
                        header="",
                    )

                    if hasattr(hint, "priority_tools") and hint.priority_tools:
                        for tool in hint.priority_tools:
                            builder.add_tool_hint(
                                tool,
                                f"Prioritized for {task_type} tasks",
                                priority_boost=0.2,
                            )

    def add_task_hint(self, task_type: str, hint: str) -> None:
        """Add or update a task-type hint."""
        self._task_hints[task_type.lower()] = hint
        logger.debug("Added task hint for %s", task_type)

    def remove_task_hint(self, task_type: str) -> None:
        """Remove a task-type hint."""
        self._task_hints.pop(task_type.lower(), None)

    def get_task_hint(self, task_type: str) -> Optional[str]:
        """Get the hint for a task type."""
        return self._task_hints.get(task_type.lower())

    def add_section(
        self,
        name: str,
        content: str,
        priority: Optional[int] = None,
    ) -> None:
        """Add a runtime section to be included in prompts."""
        del priority
        self._additional_sections[name] = content
        logger.debug("Added section '%s'", name)

    def remove_section(self, name: str) -> None:
        """Remove a runtime section."""
        self._additional_sections.pop(name, None)

    def add_safety_rule(self, rule: str) -> None:
        """Add a safety rule."""
        if rule not in self._safety_rules:
            self._safety_rules.append(rule)

    def clear_safety_rules(self) -> None:
        """Clear all safety rules."""
        self._safety_rules.clear()

    def set_grounding_mode(self, mode: str) -> None:
        """Set the grounding rules mode."""
        if mode in ("minimal", "extended"):
            self._config.default_grounding_mode = mode
        else:
            logger.warning("Invalid grounding mode: %s", mode)

    def set_base_identity(self, identity: str) -> None:
        """Set the base identity section."""
        self._base_identity = identity

    def get_all_task_hints(self) -> Dict[str, str]:
        """Get all configured task hints."""
        return dict(self._task_hints)

    def clear(self) -> None:
        """Clear all custom sections, hints, and safety rules."""
        self._task_hints.clear()
        self._additional_sections.clear()
        self._safety_rules.clear()
        logger.debug("PromptRuntimeAdapter cleared")


def create_prompt_coordinator(
    config: Optional[PromptRuntimeConfig] = None,
    base_identity: Optional[str] = None,
    prompt_builder: Optional[PromptBuilder] = None,
    vertical_context: Optional["VerticalContext"] = None,
    on_prompt_built: Optional[Callable[[str, PromptRuntimeContext], None]] = None,
    policy: Optional[SystemPromptPolicy] = None,
) -> PromptRuntimeAdapter:
    """Factory function to create a PromptRuntimeAdapter.

    Args:
        config: Optional configuration for the coordinator
        base_identity: Optional base identity string
        prompt_builder: Optional custom prompt builder
        vertical_context: Optional vertical context
        on_prompt_built: Optional callback for when prompts are built
        policy: Optional system prompt policy

    Returns:
        A new PromptRuntimeAdapter instance
    """
    return PromptRuntimeAdapter(
        prompt_builder=prompt_builder,
        vertical_context=vertical_context,
        config=config,
        base_identity=base_identity,
        on_prompt_built=on_prompt_built,
        policy=policy,
    )


__all__ = [
    "PromptRuntimeAdapter",
    "PromptRuntimeConfig",
    "PromptRuntimeContext",
    "create_prompt_coordinator",
]
