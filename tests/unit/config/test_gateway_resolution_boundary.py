"""Gateway configuration must resolve before direct accounts, secrets or discovery."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest

from victor.config.provider_config_registry import (
    DefaultProviderConfig,
    LMStudioConfig,
    OpenAIConfig,
    ProviderConfigRegistry,
    ZAIConfig,
)
from victor.config.settings import Settings

GATEWAY = {
    "url": "https://synthetic-gateway.example.test/v1",
    "virtual_key": "synthetic-virtual-key",
    "protocol": "responses",
}
POLICY = {
    "timeout": 17.0,
    "max_tokens": 123,
    "temperature": 0.2,
    "context_window": 8192,
    "stream_idle_timeout_secs": 19,
    "cache_intent": {"mode": "auto_stable_prefix", "lifetime": "5m"},
    "auth_mode": "oauth",
}


def fixture_settings(config):
    return SimpleNamespace(
        load_provider_config=lambda _: SimpleNamespace(model_dump=lambda **_: dict(config)),
        provider=SimpleNamespace(
            lmstudio_base_urls=["http://synthetic-upstream.invalid"], openai_api_key=None
        ),
    )


def registry():
    value = ProviderConfigRegistry()
    for strategy in (
        LMStudioConfig(),
        OpenAIConfig(),
        ZAIConfig(),
        DefaultProviderConfig("anthropic"),
    ):
        value.register(strategy)
    return value


@pytest.mark.parametrize("provider", ["lmstudio", "openai", "zai", "anthropic"])
@pytest.mark.parametrize("source", ["provider", "profile", "environment"])
@pytest.mark.parametrize("named", [False, True])
def test_settings_gateway_skips_upstream_resolution_and_preserves_policy(provider, source, named):
    config = dict(POLICY)
    overrides = {}
    env = {}
    if source == "provider":
        config["gateway"] = dict(GATEWAY)
    elif source == "profile":
        overrides["gateway"] = dict(GATEWAY)
    else:
        env = {
            "SANDHI_GATEWAY_URL": GATEWAY["url"],
            "SANDHI_GATEWAY_VIRTUAL_KEY": GATEWAY["virtual_key"],
        }
    with (
        patch.dict("os.environ", env, clear=True),
        patch(
            "victor.config.provider_config_registry.get_provider_config_registry",
            return_value=registry(),
        ),
        patch(
            "victor.config.accounts.get_account_manager",
            side_effect=AssertionError("account lookup forbidden"),
        ) as account,
        patch(
            "victor.config.api_keys.get_api_key",
            side_effect=AssertionError("upstream key forbidden"),
        ) as key,
        patch("httpx.get", side_effect=AssertionError("upstream discovery forbidden")) as probe,
    ):
        result = Settings.get_provider_settings(
            fixture_settings(config),
            provider,
            overrides,
            account_name="absent-upstream" if named else None,
        )
    account.assert_not_called()
    key.assert_not_called()
    probe.assert_not_called()
    assert all(result[name] == value for name, value in POLICY.items())
    assert result["gateway"]["url"] == "https://synthetic-gateway.example.test"
    assert result["gateway"]["virtual_key"] == GATEWAY["virtual_key"]
    if source != "environment":
        assert result["gateway"]["protocol"] == "responses"


@pytest.mark.parametrize("provider", ["lmstudio", "openai", "zai", "anthropic"])
@pytest.mark.parametrize(
    "gateway",
    [
        {"url": "http://synthetic-upstream.invalid", "virtual_key": "synthetic"},
        {"url": "https://synthetic-gateway.example.test"},
        {**GATEWAY, "protocol": "chatgpt_responses"},
        {"virtual_key": "synthetic"},
        {**GATEWAY, "virtual_key": 123},
        {**GATEWAY, "url": "https://synthetic-gateway.example.test/v1/v1"},
        {**GATEWAY, "url": "https://synthetic-gateway.example.test:bad"},
        {**GATEWAY, "url": "https://synthetic-gateway.example.test:0"},
        "malformed",
    ],
)
def test_malformed_gateway_refuses_before_any_upstream_resolution(provider, gateway):
    with (
        patch.dict("os.environ", {}, clear=True),
        patch(
            "victor.config.provider_config_registry.get_provider_config_registry",
            return_value=registry(),
        ),
        patch(
            "victor.config.accounts.get_account_manager",
            side_effect=AssertionError("account forbidden"),
        ) as account,
        patch(
            "victor.config.api_keys.get_api_key", side_effect=AssertionError("key forbidden")
        ) as key,
        patch("httpx.get", side_effect=AssertionError("probe forbidden")) as probe,
    ):
        with pytest.raises(ValueError, match="[Gg]ateway"):
            Settings.get_provider_settings(fixture_settings({"gateway": gateway}), provider)
    account.assert_not_called()
    key.assert_not_called()
    probe.assert_not_called()


def test_direct_settings_retains_account_and_strategy_precedence():
    manager = Mock()
    manager.config_path.exists.return_value = True
    manager.get_account.return_value = SimpleNamespace(
        auth=SimpleNamespace(method="api_key"), endpoint="http://synthetic-account.invalid"
    )
    manager.resolve_provider_config.return_value = {"api_key": "synthetic-account-key"}
    with (
        patch.dict("os.environ", {}, clear=True),
        patch(
            "victor.config.provider_config_registry.get_provider_config_registry",
            return_value=registry(),
        ),
        patch("victor.config.accounts.get_account_manager", return_value=manager),
        patch("httpx.get", return_value=httpx.Response(200)) as probe,
    ):
        result = Settings.get_provider_settings(
            fixture_settings(POLICY), "lmstudio", {"timeout": 23}, account_name="selected-upstream"
        )
    manager.get_account.assert_called_once_with(name="selected-upstream")
    manager.resolve_provider_config.assert_called_once()
    probe.assert_called_once_with("http://synthetic-account.invalid/v1/models", timeout=1.5)
    assert result["api_key"] == "synthetic-account-key"
    assert result["timeout"] == 23
    assert result["auth_mode"] == "oauth"


@pytest.mark.parametrize("kind", ["oidc", "credential"])
def test_named_openai_gateway_renewal_preserves_identity_without_reading_token(kind):
    class Credential:
        async def get_token(self, *scopes):
            raise AssertionError("token acquisition belongs to dispatch")

    gateway = {"url": GATEWAY["url"], "protocol": "responses", "grant": "selected-grant"}
    if kind == "oidc":
        gateway["oidc"] = {
            "token_file": "/tmp/synthetic-absent-gateway-token.json",
            "issuer": "https://synthetic-issuer.example.test",
            "audience": "selected-audience",
            "subject": "selected-subject",
        }
    else:
        gateway.update(credential=Credential(), audience="selected-audience")
    with (
        patch.dict("os.environ", {"SANDHI_GATEWAY_VIRTUAL_KEY": "unrelated-env-key"}, clear=True),
        patch(
            "victor.config.provider_config_registry.get_provider_config_registry",
            return_value=registry(),
        ),
        patch(
            "victor.config.accounts.get_account_manager",
            side_effect=AssertionError("account forbidden"),
        ) as account,
        patch(
            "victor.core.identity.gateway.GatewayTokenFileCredential._read",
            side_effect=AssertionError("token read forbidden"),
        ) as read,
    ):
        result = Settings.get_provider_settings(
            fixture_settings({**POLICY, "gateway": gateway}),
            "openai",
            account_name="unrelated-upstream-account",
        )
    account.assert_not_called()
    read.assert_not_called()
    assert result["gateway"]["virtual_key"] == ""
    assert result["gateway"]["grant"] == "selected-grant"
    assert result["gateway"][kind] == gateway[kind]
    assert result["auth_mode"] == "oauth"


@pytest.mark.parametrize(
    "gateway",
    [
        {**GATEWAY, "oidc": {}},
        {"url": GATEWAY["url"], "oidc": {"token_file": "/tmp/synthetic"}},
        {"url": GATEWAY["url"], "credential": object(), "audience": "synthetic"},
    ],
)
def test_invalid_gateway_identity_rejected_without_account_resolution(gateway):
    with (
        patch.dict("os.environ", {}, clear=True),
        patch(
            "victor.config.provider_config_registry.get_provider_config_registry",
            return_value=registry(),
        ),
        patch("victor.config.accounts.get_account_manager") as account,
    ):
        with pytest.raises(ValueError, match="[Gg]ateway"):
            Settings.get_provider_settings(fixture_settings({"gateway": gateway}), "openai")
    account.assert_not_called()


def test_registry_gateway_bypasses_lazy_account_and_direct_strategy():
    value = registry()
    callback = Mock(side_effect=AssertionError("lazy account forbidden"))
    with (
        patch.dict("os.environ", {}, clear=True),
        patch.object(
            value._strategies["lmstudio"],
            "get_settings",
            side_effect=AssertionError("direct policy forbidden"),
        ) as strategy,
    ):
        result = value.get_settings(
            "lmstudio", fixture_settings({"gateway": GATEWAY}), account_data_factory=callback
        )
    assert result["gateway"]["virtual_key"] == GATEWAY["virtual_key"]
    callback.assert_not_called()
    strategy.assert_not_called()


@pytest.mark.parametrize("gateway", [{**GATEWAY, "protocol": "codex"}, "malformed"])
def test_actual_provider_file_rejects_invalid_gateway_without_direct_fallback(tmp_path, gateway):
    import yaml

    config = tmp_path / "profiles.yaml"
    config.write_text(yaml.safe_dump({"providers": {"openai": {"gateway": gateway}}}))
    settings = Settings()
    with (
        patch.dict("os.environ", {}, clear=True),
        patch.object(Settings, "get_config_dir", return_value=tmp_path),
        patch.object(Settings, "_central_config_layers", return_value=[]),
        patch("victor.config.accounts.get_account_manager") as account,
        patch("victor.config.api_keys.get_api_key") as key,
        patch("httpx.get") as probe,
    ):
        with pytest.raises(ValueError, match="[Gg]ateway") as error:
            settings.get_provider_settings("openai")
    assert "synthetic-virtual-key" not in str(error.value)
    account.assert_not_called()
    key.assert_not_called()
    probe.assert_not_called()


@pytest.mark.parametrize("renewable", [False, True])
def test_actual_provider_file_gateway_preserves_policy_before_named_account(tmp_path, renewable):
    import yaml

    gateway = dict(GATEWAY)
    if renewable:
        gateway.pop("virtual_key")
        gateway["oidc"] = {
            "token_file": str(tmp_path / "absent-token.json"),
            "issuer": "https://synthetic-issuer.example.test",
            "audience": "selected-audience",
            "subject": "selected-subject",
        }
    (tmp_path / "profiles.yaml").write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "openai": {
                        "timeout": int(POLICY["timeout"]),
                        "stream_idle_timeout_secs": POLICY["stream_idle_timeout_secs"],
                        "gateway": gateway,
                    }
                }
            }
        )
    )
    settings = Settings()
    with (
        patch.dict("os.environ", {}, clear=True),
        patch.object(Settings, "get_config_dir", return_value=tmp_path),
        patch.object(Settings, "_central_config_layers", return_value=[]),
        patch("victor.config.accounts.get_account_manager") as account,
        patch("victor.config.api_keys.get_api_key") as key,
        patch("httpx.get") as probe,
    ):
        result = settings.get_provider_settings("openai", POLICY, account_name="absent-upstream")
    account.assert_not_called()
    key.assert_not_called()
    probe.assert_not_called()
    assert all(result[name] == value for name, value in POLICY.items())
    assert result["gateway"]["protocol"] == "responses"
    if renewable:
        assert result["gateway"]["oidc"] == gateway["oidc"]
    else:
        assert result["gateway"]["virtual_key"] == gateway["virtual_key"]
