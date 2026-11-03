"""Skill and system-prompt runtime accessors for the orchestrator.

Moved verbatim from ``victor/agent/orchestrator.py`` (facade decomposition):
skill runtime delegation and the system-prompt build/refresh path.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OrchestratorPromptingMixin:
    """Moved verbatim from orchestrator.py; see module docstring."""

    def _get_skill_runtime(self) -> Any:
        """Get the canonical service-owned skill runtime helper."""
        runtime = getattr(self, "_skill_runtime", None)
        if runtime is None:
            from victor.agent.services.skill_runtime import SkillRuntime

            runtime = SkillRuntime(self.protocol_adapter)
            self._skill_runtime = runtime
        return runtime

    def get_last_skill_match_info(self) -> Optional[Dict[str, Any]]:
        """Return metadata about the last skill match for response attachment."""
        return self._get_skill_runtime().get_last_skill_match_info()

    def _get_prompt_builder_runtime(self) -> Any:
        """Get the canonical service-owned prompt-builder runtime helper."""
        runtime = getattr(self, "_prompt_builder_runtime", None)
        if runtime is None:
            from victor.agent.services.prompt_builder_runtime import (
                PromptBuilderRuntime,
            )

            runtime = PromptBuilderRuntime(self.protocol_adapter)
            self._prompt_builder_runtime = runtime
        return runtime

    def clear_active_skills(self) -> None:
        """Remove any active skill injection.

        Called at the start of each turn to ensure skills don't accumulate.
        When cache_optimization enabled: just clears _active_skill_prompt
        (system prompt was never touched). Otherwise: restores base prompt.
        """
        self._get_skill_runtime().clear_active_skills()

    def get_skill_user_prefix(self) -> str:
        """Get active skill prompt as user message prefix (cache-friendly).

        When cache_optimization is enabled, skills are injected into the
        user message instead of mutating the system prompt. This keeps
        the system prompt byte-identical across turns for prefix caching.

        Returns:
            Skill prompt prefix string, or empty string if no active skill.
        """
        return self._get_skill_runtime().get_skill_user_prefix()

    def inject_skill(self, skill: Any) -> None:
        """Inject a skill's prompt fragment.

        When cache_optimization is enabled: stores skill text for user
        message injection (via get_skill_user_prefix). System prompt
        stays frozen.

        When disabled: legacy behavior — prepends to system prompt.

        Args:
            skill: A SkillDefinition with name, description, prompt_fragment.
        """
        self._get_skill_runtime().inject_skill(skill)

    def inject_skills(self, skills: List[Any]) -> None:
        """Inject multiple skills' prompt fragments.

        Cache-friendly: when cache_optimization enabled, stores for user
        message injection. Otherwise, legacy system prompt mutation.

        Args:
            skills: List of (SkillDefinition, score) tuples.
        """
        self._get_skill_runtime().inject_skills(skills)

    def update_system_prompt_for_query(self, query_classification=None) -> None:
        """Rebuild system prompt with query-specific classification context.

        Called by coordinators when a query classification is available,
        injecting task-aware guidance and tool constraints into the prompt.
        Updates both the orchestrator's cached prompt AND the conversation's
        live system message so the provider receives the updated prompt.

        When cache_optimization_enabled is True and the system prompt has
        already been built, this is a no-op to preserve prefix cache stability.
        Query-specific guidance should be injected via user messages instead.

        Args:
            query_classification: Optional QueryClassification for task-aware prompting
        """
        self._get_prompt_builder_runtime().update_system_prompt_for_query(
            query_classification=query_classification
        )

    def refresh_system_prompt(
        self,
        query_classification=None,
        *,
        preserve_existing_classification: bool = True,
    ) -> None:
        """Force a system prompt rebuild after an explicit runtime change."""
        self._get_prompt_builder_runtime().refresh_system_prompt(
            query_classification=query_classification,
            preserve_existing_classification=preserve_existing_classification,
        )

    def _sync_prompt_builder_runtime_state(self) -> None:
        """Align prompt-builder state with current mode and enabled tools."""
        self._get_prompt_builder_runtime().sync_prompt_builder_runtime_state()

    def build_system_prompt(self) -> str:
        """Build the system prompt via the canonical prompt runtime surface.

        Delegates to UnifiedPromptPipeline.build_system_prompt() when
        available. Falls back to the prompt-builder runtime helper during
        early initialization or compatibility-only pipeline failures.
        """
        return self._get_prompt_builder_runtime().build_system_prompt()

    def _emit_prompt_used_event(self, prompt: str) -> None:
        """Emit PROMPT_USED event for RL prompt template learner.

        Delegates to UnifiedPromptPipeline.
        """
        pipeline = getattr(self, "_prompt_pipeline", None)
        if pipeline is not None:
            pipeline._emit_prompt_used_event(prompt)

    def _resolve_shell_variant(self, tool_name: str) -> str:
        """Resolve shell aliases to the appropriate enabled shell variant.

        Delegates to UnifiedPromptPipeline, then falls back to direct runtime
        dependency resolution.
        """
        pipeline = getattr(self, "_prompt_pipeline", None)
        if pipeline is not None:
            return pipeline.resolve_shell_variant(tool_name)

        try:
            from victor.agent.shell_resolver import resolve_shell_variant

            return resolve_shell_variant(
                tool_name,
                getattr(self, "tools", None),
                getattr(self, "mode_controller", None),
            )
        except Exception as exc:
            logger.debug("Shell resolver unavailable during runtime fallback: %s", exc)
            return tool_name

    def _get_thinking_disabled_prompt(self, base_prompt: str) -> str:
        """Prefix a prompt with the thinking disable prefix if supported.

        IMPORTANT: This should ONLY be used in RECOVERY scenarios where:
        - Model returned empty response (stuck in thinking)
        - Context overflow forced completion
        - Iteration limit forced completion

        Normal model calls should NOT use this - thinking mode produces
        better quality results. This is a last-resort recovery mechanism.

        For models with thinking mode (e.g., Qwen3), prepends the configured
        disable prefix (e.g., "/no_think") to get direct responses without
        internal reasoning overhead.

        Args:
            base_prompt: The base prompt text

        Returns:
            Prompt with thinking disable prefix if available, otherwise base_prompt
        """
        prefix = getattr(self.tool_calling_caps, "thinking_disable_prefix", None)
        if prefix:
            return f"{prefix}\n{base_prompt}"
        return base_prompt
