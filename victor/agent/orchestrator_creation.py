"""Factory functions for creating AgentOrchestrator from settings/profiles.

Extracted from AgentOrchestrator.from_settings() to reduce orchestrator LOC.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)


async def create_orchestrator_from_settings(
    cls: type["AgentOrchestrator"],
    settings: "Settings",
    profile_name: str = "default",
    thinking: bool = False,
) -> "AgentOrchestrator":
    """Create orchestrator from settings.

    Args:
        cls: The AgentOrchestrator class
        settings: Application settings
        profile_name: Profile to use
        thinking: Enable extended thinking mode (Claude models only)

    Returns:
        Configured AgentOrchestrator instance
    """
    from victor.providers.registry import ProviderRegistry

    # Load profile
    profiles = settings.load_profiles()
    profile = profiles.get(profile_name)

    if not profile:
        available = list(profiles.keys())
        import difflib

        suggestions = difflib.get_close_matches(profile_name, available, n=3, cutoff=0.4)

        error_msg = f"Profile not found: '{profile_name}'"
        if suggestions:
            error_msg += f"\n  Did you mean: {', '.join(suggestions)}?"
        if available:
            error_msg += f"\n  Available profiles: {', '.join(sorted(available))}"
        else:
            error_msg += (
                "\n  No profiles configured. Run 'victor init' or create ~/.victor/profiles.yaml"
            )
        raise ValueError(error_msg)

    # Get provider-level settings.
    # Thread the profile's credential-identity account through resolution so the
    # profile's own key/endpoint are used (not the provider default). getattr covers
    # both the first-class ProfileConfig.account field and any residual pydantic-extra.
    profile_extras = profile.__pydantic_extra__ if hasattr(profile, "__pydantic_extra__") else {}
    account_name = getattr(profile, "account", None) or profile_extras.get("account")
    provider_settings = settings.get_provider_settings(
        profile.provider, profile_extras, account_name=account_name
    )

    if profile_extras:
        logger.debug(
            f"Profile '{profile_name}' extras passed to provider config: "
            f"{list(profile_extras.keys())}"
        )

    # Apply timeout multiplier from model capabilities
    from victor.agent.tool_calling.capabilities import ModelCapabilityLoader

    cap_loader = ModelCapabilityLoader()
    caps = cap_loader.get_capabilities(profile.provider, profile.model)
    if caps and caps.timeout_multiplier > 1.0:
        base_timeout = provider_settings.get("timeout", 300)
        adjusted_timeout = int(base_timeout * caps.timeout_multiplier)
        provider_settings["timeout"] = adjusted_timeout
        logger.info(
            f"Adjusted timeout for {profile.provider}/{profile.model}: "
            f"{base_timeout}s -> {adjusted_timeout}s (multiplier: {caps.timeout_multiplier}x)"
        )

    # Create provider instance
    provider = ProviderRegistry.create(profile.provider, **provider_settings)

    orchestrator = cls(
        settings=settings,
        provider=provider,
        model=profile.model,
        temperature=profile.temperature,
        max_tokens=profile.max_tokens,
        tool_selection=profile.tool_selection,
        thinking=thinking,
        provider_name=profile.provider,
        profile_name=profile_name,
    )

    # Setup JSONL exporter if enabled
    if getattr(settings, "enable_observability_logging", False):
        from victor.observability.bridge import ObservabilityBridge

        try:
            bridge = ObservabilityBridge.get_instance()
            log_path = getattr(settings, "observability_log_path", None)
            bridge.setup_jsonl_exporter(log_path=log_path)
            logger.info(
                f"JSONL event logging enabled: {log_path or '~/.victor/metrics/victor.jsonl'}"
            )
        except Exception as e:
            logger.warning(f"Failed to setup JSONL exporter: {e}")

    return orchestrator
