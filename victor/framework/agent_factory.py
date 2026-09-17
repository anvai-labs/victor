"""Unified AgentFactory — single authority for all agent creation paths.

Replaces older divergent initialization paths with one Facade:
- CLI and one-shot session handlers → AgentCreationFactory → AgentFactory
- CLI tools → direct from_settings (legacy internal path)
- API server → direct from_settings (legacy internal path)
- Agent.create() → AgentFactory

All entry points now use:
    factory = AgentFactory(settings, profile=..., vertical=..., ...)
    orchestrator = await factory.create()
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

from victor.core.errors import VictorError
from victor.framework.events import EventSeverity

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)


class InitializationError(VictorError):
    """Structured error from agent initialization with user guidance.

    Attributes:
        stage: Which step failed (profile/credentials/provider/vertical/bootstrap)
        suggestions: List of commands or actions the user can try
        run_command: Primary fix command (e.g., "victor doctor")
    """

    def __init__(
        self,
        stage: str,
        message: str,
        suggestions: Optional[List[str]] = None,
        run_command: Optional[str] = None,
    ):
        self.stage = stage
        self.suggestions = suggestions or []
        self.run_command = run_command
        super().__init__(message)


class AgentFactory:
    """Unified agent creation — single authority for all entry points.

    Applies Facade pattern to consolidate initialization:
    1. Validate configuration (fail-fast with guidance)
    2. Bootstrap DI container with vertical
    3. Create orchestrator via from_settings
    4. Apply vertical configuration
    5. Wire observability
    6. Apply overrides (tool budget, iterations)

    Example:
        factory = AgentFactory(settings, profile="groq-fast", vertical="coding")
        orchestrator = await factory.create()
    """

    def __init__(
        self,
        settings: "Settings",
        profile: str = "default",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        vertical: Optional[Union[Type, str]] = None,
        thinking: bool = False,
        session_id: Optional[str] = None,
        enable_observability: bool = True,
        tool_budget: Optional[int] = None,
        max_iterations: Optional[int] = None,
        profile_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._settings = settings
        self._profile = profile
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._thinking = thinking
        self._session_id = session_id or str(uuid.uuid4())
        self._enable_observability = enable_observability
        self._tool_budget = tool_budget
        self._max_iterations = max_iterations
        self._profile_overrides = dict(profile_overrides or {})
        self._orchestrator: Optional["AgentOrchestrator"] = None

        # Store original vertical name for error reporting
        self._vertical_name = vertical if isinstance(vertical, str) else None

        # Resolve vertical from string or class
        self._vertical = self._resolve_vertical(vertical)

    def _apply_profile_overrides(self) -> None:
        """Make explicit SDK overrides visible to the profile-based creation path."""
        if (
            self._provider is None
            and self._model is None
            and self._temperature is None
            and self._max_tokens is None
            and not self._profile_overrides
        ):
            return

        from victor.config.settings import ProfileConfig

        profiles = dict(self._settings.load_profiles())
        base_profile = profiles.get(self._profile)
        profile_payload: Dict[str, Any] = (
            base_profile.model_dump() if base_profile is not None else {}
        )

        effective_provider = self._provider or profile_payload.get("provider")
        effective_model = (
            self._model
            or profile_payload.get("model")
            or getattr(getattr(self._settings, "provider", None), "default_model", None)
        )

        if not effective_provider or not effective_model:
            raise InitializationError(
                stage="profile",
                message="Explicit provider/model overrides require a resolvable provider and model.",
                suggestions=["Pass both provider and model explicitly."],
            )

        profile_payload.update(
            {
                "provider": effective_provider,
                "model": effective_model,
                "temperature": (
                    self._temperature
                    if self._temperature is not None
                    else profile_payload.get("temperature", 0.7)
                ),
                "max_tokens": (
                    self._max_tokens
                    if self._max_tokens is not None
                    else profile_payload.get("max_tokens", 4096)
                ),
            }
        )
        profile_payload.update(self._profile_overrides)

        profiles[self._profile] = ProfileConfig(**profile_payload)
        self._settings.load_profiles = lambda: profiles

    async def create(self) -> "AgentOrchestrator":
        """Create a fully configured orchestrator.

        Returns:
            Configured AgentOrchestrator instance.

        Raises:
            InitializationError: With stage, suggestions, and run_command.
        """
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.core.bootstrap import ensure_bootstrapped

        self._apply_profile_overrides()

        # Step 1: Pre-flight validation
        issues = self.validate()
        errors = [i for i in issues if i.get("severity") == EventSeverity.ERROR.value]
        if errors:
            raise InitializationError(
                stage=errors[0].get("stage", "validation"),
                message=errors[0].get("message", "Validation failed"),
                suggestions=errors[0].get("suggestions", []),
                run_command="victor doctor",
            )

        # Step 2: Bootstrap container with vertical
        try:
            # Use stored vertical name (preserves original name even if vertical not found)
            vertical_name = self._vertical_name or (self._vertical.name if self._vertical else None)
            ensure_bootstrapped(self._settings, vertical=vertical_name)
        except Exception as e:
            raise InitializationError(
                stage="bootstrap",
                message=str(e),
                suggestions=["victor doctor", "victor init"],
                run_command="victor doctor",
            ) from e

        # Step 3: Create orchestrator
        try:
            self._orchestrator = await AgentOrchestrator.from_settings(
                self._settings,
                profile_name=self._profile,
                thinking=self._thinking,
            )
        except ValueError as e:
            # Profile not found, credentials missing, etc.
            raise InitializationError(
                stage="profile",
                message=str(e),
                suggestions=["victor profiles list", "victor init"],
                run_command="victor init",
            ) from e
        except Exception as e:
            if "api" in str(e).lower() or "key" in str(e).lower():
                raise InitializationError(
                    stage="credentials",
                    message=str(e),
                    suggestions=[
                        "Check API key environment variables",
                        "victor config validate",
                    ],
                    run_command="victor doctor",
                ) from e
            raise InitializationError(
                stage="orchestrator",
                message=str(e),
                suggestions=["victor doctor"],
            ) from e

        # Step 4: Apply vertical configuration
        if self._vertical:
            try:
                self._apply_vertical()
            except Exception as e:
                logger.warning(f"Vertical application failed: {e}")

        # Step 5: Wire observability
        if self._enable_observability:
            self._wire_observability()

        # Step 6: Apply overrides
        if self._tool_budget is not None and hasattr(self._orchestrator, "unified_tracker"):
            self._orchestrator.unified_tracker.set_tool_budget(
                self._tool_budget, user_override=True
            )
        if self._max_iterations is not None and hasattr(self._orchestrator, "unified_tracker"):
            self._orchestrator.unified_tracker.set_max_iterations(
                self._max_iterations, user_override=True
            )

        # Step 7: Initialize skill matcher
        await self._initialize_skill_matcher()

        # Step 8: Background RL warm-up — evolve stale prompt candidates
        # Non-blocking: spawns a daemon thread if candidates are >7 days old.
        self._trigger_background_evolution()

        logger.info(
            "AgentFactory created orchestrator: profile=%s, vertical=%s, session=%s",
            self._profile,
            self._vertical.name if self._vertical else None,
            self._session_id,
        )

        return self._orchestrator

    def validate(self) -> List[Dict[str, Any]]:
        """Pre-flight validation without creating orchestrator.

        Returns list of issues, each with: stage, severity, message, suggestions.
        """
        import os
        from pathlib import Path

        issues: List[Dict[str, Any]] = []

        # Check config directory
        config_dir = Path.home() / ".victor"
        if not config_dir.exists():
            issues.append(
                {
                    "stage": "config",
                    "severity": "warning",
                    "message": "No ~/.victor directory found. Run 'victor init' to set up.",
                    "suggestions": ["victor init"],
                }
            )

        # Check common API key env vars for the target provider
        provider = self._provider or "openai"
        key_vars = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "xai": "XAI_API_KEY",
            "cohere": "COHERE_API_KEY",
        }
        if provider in key_vars and not os.environ.get(key_vars[provider]):
            issues.append(
                {
                    "stage": "credentials",
                    "severity": "warning",
                    "message": f"{key_vars[provider]} not set for provider '{provider}'",
                    "suggestions": [
                        f"export {key_vars[provider]}=...",
                        "victor doctor",
                    ],
                }
            )

        return issues

    def _resolve_vertical(self, vertical: Optional[Union[Type, str]]) -> Optional[Any]:
        """Resolve vertical from string name or class."""
        if vertical is None:
            return None
        if isinstance(vertical, str):
            try:
                from victor.core.verticals.base import VerticalRegistry

                return VerticalRegistry.get(vertical)
            except Exception:
                logger.debug(f"Vertical '{vertical}' not found in registry")
                return None
        return vertical

    def _apply_vertical(self) -> None:
        """Apply vertical configuration to the orchestrator."""
        if not self._vertical or not self._orchestrator:
            return
        try:
            from victor.framework.vertical_service import apply_vertical_configuration

            apply_vertical_configuration(self._orchestrator, self._vertical)
        except ImportError:
            logger.debug("vertical_service not available, skipping vertical application")
        except Exception as e:
            logger.warning(f"Vertical application failed: {e}")

    def _wire_observability(self) -> None:
        """Wire observability integration."""
        if not self._orchestrator:
            return
        try:
            from victor.framework._internal import setup_observability_integration

            setup_observability_integration(self._orchestrator, session_id=self._session_id)
        except Exception as e:
            logger.debug(f"Observability wiring failed: {e}")

    async def _initialize_skill_matcher(self) -> None:
        """Initialize skill auto-selection.

        Populates the shared ``SkillRegistry`` (vertical skills are loaded by
        ``FrameworkStepHandler.apply_skills`` during integration; entry-point +
        user skills are loaded here once) and passes it to the matcher.
        Previously this passed the orchestrator, which has no ``list_all()`` —
        so the matcher silently failed to load any skills.
        """
        if not self._orchestrator:
            return
        try:
            from victor.framework.skill_matcher import SkillMatcher
            from victor.framework.skills import get_skill_registry

            registry = get_skill_registry()
            # apply_skills (step handler) covers the integrated vertical; load
            # the external sources once so they're available to the matcher.
            if not getattr(registry, "_external_loaded", False):
                registry.from_entry_points()
                registry.from_user_skills()
                registry._external_loaded = True

            matcher = SkillMatcher()
            # Attach first, initialize in the background: every consumer
            # guards on the matcher's readiness (skill_runtime, planning
            # prompt enrichment, chat preview, /skills), so a not-yet-ready
            # matcher degrades to "no skill auto-selection" for the first
            # turns instead of stalling bootstrap by ~6s (the
            # sentence-transformers import + model load).
            self._orchestrator._skill_matcher = matcher

            async def _initialize() -> None:
                await matcher.initialize(registry)

            init_task = asyncio.ensure_future(_initialize())
            # Keep a strong reference so the task cannot be garbage-collected
            # mid-flight; surfaced for tests/shutdown ordering.
            self._skill_matcher_init_task = init_task
        except Exception as e:
            logger.debug(f"Skill matcher initialization skipped: {e}")

    def _trigger_background_evolution(self) -> None:
        """Kick off background GEPA evolution if candidates are stale.

        Non-blocking: the coordinator spawns a daemon thread internally.
        Resolves provider/model from current settings for evolution context.
        """
        try:
            from victor.framework.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            provider = getattr(self._settings, "provider", None)
            provider_name = getattr(provider, "default_provider", "") if provider else ""
            model_name = getattr(provider, "default_model", "") if provider else ""
            coordinator.maybe_background_evolve(provider_name, model_name)
        except Exception as e:
            logger.debug(f"Background RL evolution skipped: {e}")
