"""Prompt orchestrator for unified prompt construction.

Provides a unified entry point for prompt construction that works
across legacy and StateGraph execution paths.

Research basis:
- Facade pattern — Unified API for complex subsystems
- arXiv:2601.06007 — System-prompt-only caching is optimal (41-80% cost reduction)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.agent.content_registry import ContentRegistry
    from victor.agent.optimization_injector import OptimizationInjector


@dataclass
class OrchestratorConfig:
    """Configuration for PromptOrchestrator.

    Attributes:
        use_evolved_content: Whether to inject evolved content when available
        enable_constraint_activation: Whether to activate constraints
        fallback_to_static: Whether to fall back to static content if evolution fails
        cache_evolved_content: Whether to cache evolved content resolution
    """

    use_evolved_content: bool = True
    enable_constraint_activation: bool = True
    fallback_to_static: bool = True
    cache_evolved_content: bool = True


class PromptOrchestrator:
    """Unified orchestrator for prompt construction.

    Responsibilities:
    - Coordinate between legacy and framework builders
    - Inject evolved content when available
    - Activate constraints for all execution paths
    - Provide consistent API regardless of execution path

    This is a facade that coordinates existing components without
    reimplementing their logic.
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        optimization_injector: Optional[OptimizationInjector] = None,
        content_registry: Optional[ContentRegistry] = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            config: Orchestrator configuration
            optimization_injector: Optional injector for evolved content
            content_registry: Optional content registry
        """
        self._config = config or OrchestratorConfig()
        self._injector = optimization_injector
        self._registry = content_registry

        # Lazy-loaded components
        self._resolver: Optional[Any] = None
        self._constraint_activator: Optional[Any] = None

        logger.debug(
            f"PromptOrchestrator initialized (evolved_content={self._config.use_evolved_content}, "
            f"constraints={self._config.enable_constraint_activation})"
        )

    def build_system_prompt(
        self,
        builder_type: str = "auto",  # "legacy", "framework", or "auto"
        provider: str = "",
        model: str = "",
        task_type: str = "default",
        **builder_kwargs: Any,
    ) -> str:
        """Build a system prompt with evolved content injection.

        Args:
            builder_type: Which builder to use ("legacy", "framework", or "auto")
            provider: Provider name
            model: Model name
            task_type: Task type for context
            **builder_kwargs: Additional arguments for the builder

        Returns:
            Complete system prompt string
        """
        # Auto-detect builder type
        if builder_type == "auto":
            builder_type = self._detect_builder_type(**builder_kwargs)
            logger.debug(f"Auto-detected builder type: {builder_type}")

        # Get the appropriate builder
        if builder_type == "legacy":
            prompt = self._build_with_legacy(provider, model, task_type, **builder_kwargs)
        else:
            prompt = self._build_with_framework(provider, model, task_type, **builder_kwargs)

        return prompt

    def activate_constraints(
        self,
        constraints: Any,
        vertical: str = "coding",
    ) -> bool:
        """Activate constraints for current execution.

        Args:
            constraints: Task constraints
            vertical: Vertical name for defaults

        Returns:
            True if activation succeeded
        """
        if not self._config.enable_constraint_activation:
            return True  # Disabled, treat as success

        activator = self._get_constraint_activator()
        result = activator.activate_constraints(constraints, vertical)
        return result.success

    def deactivate_constraints(self) -> None:
        """Deactivate constraints after execution."""
        activator = self._get_constraint_activator()
        activator.deactivate_constraints()

    def _detect_builder_type(self, **kwargs: Any) -> str:
        """Auto-detect which builder to use based on arguments.

        Args:
            **kwargs: Builder arguments

        Returns:
            "legacy" or "framework"
        """
        # If has prompt_contributors, use legacy
        if "prompt_contributors" in kwargs or "builder" in kwargs or "legacy_builder" in kwargs:
            return "legacy"

        # If has base_prompt, use framework
        if "base_prompt" in kwargs:
            return "framework"

        # Default to legacy for backward compatibility
        return "legacy"

    def _build_with_legacy(
        self,
        provider: str,
        model: str,
        task_type: str,
        **kwargs: Any,
    ) -> str:
        """Build using legacy SystemPromptBuilder.

        Args:
            provider: Provider name
            model: Model name
            task_type: Task type
            **kwargs: Additional builder arguments

        Returns:
            System prompt string
        """
        from victor.agent.prompt_builder import SystemPromptBuilder

        builder = kwargs.pop("builder", kwargs.pop("legacy_builder", None))
        get_context_window = kwargs.pop("get_context_window", None)
        on_prompt_built = kwargs.pop("on_prompt_built", None)

        if builder is None:
            builder = SystemPromptBuilder(
                provider_name=provider,
                model=model,
                task_type=task_type,
                **kwargs,
            )

        prompt = builder.build()

        if get_context_window is not None:
            context_window = get_context_window()
            if context_window >= 32768:
                from victor.agent.context_compactor import (
                    calculate_parallel_read_budget,
                )

                budget = calculate_parallel_read_budget(context_window)
                prompt = f"{prompt}\n\n{budget.to_prompt_hint()}"

        if on_prompt_built is not None:
            on_prompt_built(prompt)

        return prompt

    def _build_with_framework(
        self,
        provider: str,
        model: str,
        task_type: str,
        **kwargs: Any,
    ) -> str:
        """Build using framework PromptBuilder with evolved content.

        Args:
            provider: Provider name
            model: Model name
            task_type: Task type
            **kwargs: Additional builder arguments

        Returns:
            System prompt string
        """
        from victor.agent.prompting.builder import PromptBuilder

        builder = PromptBuilder()

        # Inject evolved content if enabled
        if self._config.use_evolved_content:
            self._inject_evolved_content(builder, provider, model, task_type)

        # Add base sections from kwargs
        if "base_prompt" in kwargs:
            builder.add_section("base", kwargs["base_prompt"], priority=10)

        return builder.build()

    def _inject_evolved_content(
        self,
        builder: Any,
        provider: str,
        model: str,
        task_type: str,
    ) -> None:
        """Inject evolved content into framework builder.

        Args:
            builder: PromptBuilder instance
            provider: Provider name
            model: Model name
            task_type: Task type
        """
        resolver = self._get_resolver()
        sections_to_inject = self._get_framework_evolved_sections()
        fallback_map = self._get_fallback_texts([section.name for section in sections_to_inject])

        # Resolve evolved content
        resolved_list = resolver.resolve_multiple(
            section_names=[section.name for section in sections_to_inject],
            provider=provider,
            model=model,
            task_type=task_type,
            fallback_map=fallback_map,
        )

        # Inject into builder
        for resolved, section in zip(resolved_list, sections_to_inject):
            if resolved.text:
                builder.add_section(
                    name=resolved.section_name.lower(),
                    content=resolved.text,
                    priority=section.priority,
                )
                logger.debug(
                    f"Injected section '{resolved.section_name}' "
                    f"(source={resolved.source}, priority={section.priority})"
                )

    def _get_framework_evolved_sections(self) -> list[Any]:
        """Return the canonical framework prompt sections that evolve in place."""
        from victor.agent.prompt_section_registry import get_required_evolvable_sections

        return get_required_evolvable_sections()

    def _get_fallback_texts(self, section_names: list[str]) -> dict[str, str]:
        """Get static fallback texts for sections.

        Returns:
            Map of section_name -> fallback_text
        """
        fallback_map: dict[str, str] = {}

        if self._registry is not None:
            for name in section_names:
                item = self._registry.get(name)
                if item is not None and item.default_text:
                    fallback_map[name] = item.default_text

        from victor.agent.prompt_section_registry import build_fallback_map

        for name, text in build_fallback_map(section_names).items():
            fallback_map.setdefault(name, text)

        return fallback_map

    def _get_resolver(self) -> Any:
        """Get or create evolved content resolver.

        Returns:
            EvolvedContentResolver instance
        """
        if self._resolver is None:
            from victor.agent.evolved_content_resolver import EvolvedContentResolver

            self._resolver = EvolvedContentResolver(optimization_injector=self._injector)
        return self._resolver

    def _get_constraint_activator(self) -> Any:
        """Get or create constraint activator.

        Returns:
            ConstraintActivationService instance
        """
        if self._constraint_activator is None:
            from victor.agent.constraint_activation_service import (
                get_constraint_activator,
            )

            self._constraint_activator = get_constraint_activator()
        return self._constraint_activator


# Singleton instance for global access
_orchestrator: Optional[PromptOrchestrator] = None


def get_prompt_orchestrator(
    config: Optional[OrchestratorConfig] = None,
    **kwargs: Any,
) -> PromptOrchestrator:
    """Get the global prompt orchestrator instance.

    Args:
        config: Optional orchestrator configuration
        **kwargs: Additional arguments for orchestrator initialization

    Returns:
        PromptOrchestrator instance

    Note:
        If the orchestrator already exists and config is provided,
        the existing orchestrator is returned (config is ignored).
        To update config, create a new orchestrator instance directly.
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PromptOrchestrator(config=config, **kwargs)
        logger.debug("Global PromptOrchestrator created")
    return _orchestrator
