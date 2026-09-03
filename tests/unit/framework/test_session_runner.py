from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.framework.session_config import SessionConfig
from victor.framework.session_runner import FrameworkSessionRunner


def _build_settings() -> SimpleNamespace:
    return SimpleNamespace(
        provider=SimpleNamespace(
            default_provider="zai",
            default_model="glm-5.1",
        ),
        tool_settings=SimpleNamespace(
            tool_output_preview_enabled=True,
            tool_output_pruning_enabled=False,
            tool_output_pruning_safe_only=True,
        ),
        observability=SimpleNamespace(enable_observability_logging=False),
        automation=SimpleNamespace(one_shot_mode=False),
        enable_observability_logging=False,
        skill_auto_select_enabled=True,
        one_shot_mode=False,
    )


def test_prepare_state_applies_config_and_resolves_reasoning() -> None:
    settings = _build_settings()
    config = SessionConfig.from_cli_flags(
        planning_enabled=True,
        observability_logging=True,
        auto_skill_enabled=False,
    )
    runner = FrameworkSessionRunner(settings, config)

    with patch(
        "victor.agent.tool_calling.capabilities.ModelCapabilityLoader.get_capabilities",
        return_value=SimpleNamespace(thinking_mode=True),
    ):
        prepared = runner.prepare_state(
            one_shot_mode=True,
            stream=True,
            show_reasoning=False,
        )

    assert prepared.config.one_shot_mode is True
    assert prepared.show_reasoning is True
    assert prepared.use_streaming is False
    # Overrides land on the runner's PRIVATE copy (adversarial-review fix:
    # the caller's instance — often the shared process snapshot — must not
    # receive sticky session overrides that bleed into later runs).
    runner_settings = runner.settings
    assert runner_settings.observability.enable_observability_logging is True
    assert runner_settings.enable_observability_logging is True
    assert runner_settings.skill_auto_select_enabled is False
    assert runner_settings.automation.one_shot_mode is True
    assert runner_settings.one_shot_mode is True
    # Negative: the caller's settings stay untouched.
    assert settings.observability.enable_observability_logging is not True
    assert settings.one_shot_mode is not True


def test_prepare_state_preserves_explicit_reasoning_and_interactive_mode() -> None:
    settings = _build_settings()
    config = SessionConfig.from_cli_flags()
    runner = FrameworkSessionRunner(settings, config)

    prepared = runner.prepare_state(
        one_shot_mode=False,
        stream=True,
        show_reasoning=True,
    )

    assert prepared.config.one_shot_mode is False
    assert prepared.show_reasoning is True
    assert prepared.use_streaming is True
    assert settings.automation.one_shot_mode is False
    assert settings.one_shot_mode is False


def test_validate_configuration_uses_canonical_framework_validator() -> None:
    settings = _build_settings()
    config = SessionConfig()
    runner = FrameworkSessionRunner(settings, config)
    validation_result = SimpleNamespace(is_valid=lambda: True)

    with patch(
        "victor.config.validation.validate_configuration",
        return_value=validation_result,
    ) as validate:
        result = runner.validate_configuration()

    assert result is validation_result
    validate.assert_called_once_with(settings)


@pytest.mark.asyncio
async def test_initialize_client_applies_planning_override() -> None:
    settings = _build_settings()
    runner = FrameworkSessionRunner(settings, SessionConfig())
    agent = SimpleNamespace()
    client = SimpleNamespace(initialize=AsyncMock(return_value=agent))

    initialized = await runner.initialize_client(client, planning_model="planner-x")

    assert initialized is agent
    assert agent._planning_model_override == "planner-x"
    client.initialize.assert_awaited_once_with()


def test_create_client_uses_framework_client_for_prepared_config() -> None:
    settings = _build_settings()
    base_config = SessionConfig()
    prepared_config = SessionConfig.from_cli_flags(one_shot_mode=True)
    runner = FrameworkSessionRunner(settings, base_config)
    sentinel_client = MagicMock()

    with patch(
        "victor.framework.session_runner.create_victor_client",
        return_value=sentinel_client,
    ) as client_factory:
        client = runner.create_client(prepared_config)

    assert client is sentinel_client
    client_factory.assert_called_once_with(prepared_config)


def test_create_client_uses_injected_client_factory() -> None:
    settings = _build_settings()
    config = SessionConfig()
    sentinel_client = MagicMock()
    factory = MagicMock(return_value=sentinel_client)
    runner = FrameworkSessionRunner(settings, config, client_factory=factory)

    client = runner.create_client()

    assert client is sentinel_client
    factory.assert_called_once_with(config)
