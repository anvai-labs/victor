import pytest
from victor.config.settings import ProfileConfig
from victor.framework.session_config import ProviderOverrideConfig, SessionConfig
from victor.integrations.api.routes.classify_routes import ClassifyRequest


@pytest.mark.parametrize(
    "effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"]
)
def test_model_effort_shared_across_profile_session_and_classify(effort):
    profile = ProfileConfig(provider="openai", model="gpt-6-luna", reasoning_effort=effort)
    request = ClassifyRequest(input="hello", preset="triage.v1", reasoning_effort=effort)
    config = SessionConfig.from_cli_flags(model="gpt-6-luna", reasoning_effort=effort)
    assert profile.reasoning_effort == request.reasoning_effort == effort
    assert config.provider_override.to_profile_overrides()["reasoning_effort"] == effort
    assert config.provider_override.is_active


@pytest.mark.parametrize("effort", ["typo", True, 3])
def test_session_rejects_invalid_effort(effort):
    with pytest.raises(ValueError):
        ProviderOverrideConfig(reasoning_effort=effort)


def test_unset_effort_preserves_profile_default():
    assert "reasoning_effort" not in ProviderOverrideConfig().to_profile_overrides()
