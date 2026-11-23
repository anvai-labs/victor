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
@pytest.mark.parametrize("source", ["provider", "profile"])
@pytest.mark.parametrize("named", [False, True])
def test_settings_gateway_skips_upstream_resolution_and_preserves_policy(provider, source, named):
    """Gateway from provider or profile config activates gateway mode."""
    config = dict(POLICY)
    overrides = {}
    if source == "provider":
        config["gateway"] = dict(GATEWAY)
    else:
        overrides["gateway"] = dict(GATEWAY)
    with (
        patch.dict("os.environ", {}, clear=True),
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
    assert result["gateway"]["protocol"] == "responses"


@pytest.mark.parametrize("provider", ["lmstudio", "openai", "zai", "anthropic"])
def test_settings_env_vars_alone_do_not_activate_gateway(provider):
    """Env vars SANDHI_GATEWAY_URL + SANDHI_GATEWAY_VIRTUAL_KEY alone do NOT activate gateway mode.

    The env var must be explicitly referenced in config (e.g., `url: ${SANDHI_GATEWAY_URL}`).
    """
    config = dict(POLICY)
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
    ):
        result = Settings.get_provider_settings(
            fixture_settings(config),
            provider,
            {},
        )
    # Env vars alone don't activate gateway — direct mode
    assert result.get("gateway") is None


@pytest.mark.parametrize("provider", ["lmstudio", "openai", "zai", "anthropic"])
def test_actual_provider_file_expands_nested_gateway_url_reference(tmp_path, provider):
    """An explicit nested environment reference activates the configured gateway."""
    import yaml

    (tmp_path / "profiles.yaml").write_text(
        yaml.safe_dump(
            {
                "providers": {
                    provider: {
                        "gateway": {
                            "url": "${SANDHI_GATEWAY_URL}",
                            "virtual_key": "synthetic-virtual-key",
                            "protocol": "responses",
                        }
                    }
                }
            }
        )
    )
    settings = Settings()
    with (
        patch.dict(
            "os.environ",
            {"SANDHI_GATEWAY_URL": "https://synthetic-gateway.example.test/v1"},
            clear=True,
        ),
        patch.object(Settings, "get_config_dir", return_value=tmp_path),
        patch.object(Settings, "_central_config_layers", return_value=[]),
        patch("victor.config.accounts.get_account_manager") as account,
        patch("victor.config.api_keys.get_api_key") as key,
        patch("httpx.get") as probe,
    ):
        result = settings.get_provider_settings(provider)

    assert result["gateway"]["url"] == "https://synthetic-gateway.example.test"
    assert result["gateway"]["virtual_key"] == "synthetic-virtual-key"
    account.assert_not_called()
    key.assert_not_called()
    probe.assert_not_called()


def test_missing_nested_gateway_url_reference_refuses_without_direct_fallback(tmp_path):
    """A referenced but unset environment variable cannot become literal/direct mode."""
    import yaml

    (tmp_path / "profiles.yaml").write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "openai": {
                        "gateway": {
                            "url": "${SANDHI_GATEWAY_URL}",
                            "virtual_key": "synthetic-virtual-key",
                        }
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
        with pytest.raises(ValueError, match="SANDHI_GATEWAY_URL"):
            settings.get_provider_settings("openai")

    account.assert_not_called()
    key.assert_not_called()
    probe.assert_not_called()


@pytest.mark.parametrize("provider", ["lmstudio", "openai", "zai", "anthropic"])
@pytest.mark.parametrize(
    "gateway",
    [
        # Genuinely malformed gateways — should error
        {**GATEWAY, "protocol": "chatgpt_responses"},  # invalid protocol
        {**GATEWAY, "virtual_key": 123},  # virtual_key must be string
        {**GATEWAY, "url": "https://synthetic-gateway.example.test/v1/v1"},  # double /v1
        {**GATEWAY, "url": "https://synthetic-gateway.example.test:bad"},  # invalid port
        {**GATEWAY, "url": "https://synthetic-gateway.example.test:0"},  # port 0
        "malformed",  # not a dict
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


@pytest.mark.parametrize("provider", ["lmstudio", "openai", "zai", "anthropic"])
@pytest.mark.parametrize(
    "gateway",
    [
        # All-empty values carry no gateway intent and select direct mode.
        {"virtual_key": None},
        {"oidc": None},
        {"credential": None},
        {"auth_mode": None},
        {"url": None},
    ],
)
def test_empty_gateway_configuration_selects_direct_mode(provider, gateway):
    """Only an absent, null, or all-empty gateway block selects direct mode."""
    with (
        patch.dict("os.environ", {}, clear=True),
        patch(
            "victor.config.provider_config_registry.get_provider_config_registry",
            return_value=registry(),
        ),
    ):
        result = Settings.get_provider_settings(fixture_settings({"gateway": gateway}), provider)
    # Gateway is dropped because there's no valid URL — direct mode
    assert result.get("gateway") is None


@pytest.mark.parametrize(
    "gateway",
    [
        {"virtual_key": "synthetic"},
        {"oidc": {"token_file": "/private/access.json"}},
        {"credential": object(), "audience": "synthetic"},
        {"grant": "owner"},
        {"protocol": "responses"},
        {"auth_mode": "tokens"},
        {"api_key": "misplaced-credential"},
        {"url": "", "virtual_key": "synthetic"},
        {"url": None, "virtual_key": "synthetic"},
    ],
)
def test_partial_gateway_credentials_without_url_refuse_direct_fallback(gateway):
    """Explicit gateway intent cannot silently bypass the gateway/provider boundary."""
    with (
        patch.dict("os.environ", {}, clear=True),
        patch(
            "victor.config.provider_config_registry.get_provider_config_registry",
            return_value=registry(),
        ),
        patch("victor.config.accounts.get_account_manager") as account,
        patch("victor.config.api_keys.get_api_key") as key,
        patch("httpx.get") as probe,
    ):
        with pytest.raises(
            ValueError,
            match="Gateway configuration (requires|contains unsupported)",
        ):
            Settings.get_provider_settings(fixture_settings({"gateway": gateway}), "openai")

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
    assert "virtual_key" not in result["gateway"]
    assert result["gateway"]["auth_mode"] == "oidc"
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


@pytest.mark.parametrize(
    "gateway",
    [
        {**GATEWAY, "protocol": "codex"},
        "malformed",
        {**GATEWAY, "virtual_key": ""},
        {**GATEWAY, "virtual_key": None},
        {"url": GATEWAY["url"], "oidc": None},
        {**GATEWAY, "auth_mode": "none"},
    ],
)
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


@pytest.mark.parametrize("mode", ["tokens", "oidc", "none"])
def test_actual_provider_file_gateway_preserves_policy_before_named_account(tmp_path, mode):
    import yaml

    gateway = dict(GATEWAY)
    if mode != "tokens":
        gateway.pop("virtual_key")
    if mode == "oidc":
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
    if mode == "oidc":
        assert result["gateway"]["oidc"] == gateway["oidc"]
    elif mode == "none":
        assert "virtual_key" not in result["gateway"]
        assert result["gateway"]["auth_mode"] == "none"
    else:
        assert result["gateway"]["virtual_key"] == gateway["virtual_key"]


@pytest.mark.parametrize("provider", ["lmstudio", "openai", "zai", "anthropic", "google"])
@pytest.mark.parametrize("source", ["provider", "profile"])
def test_url_only_gateway_is_selected_without_upstream_discovery(provider, source):
    """Gateway URL from provider or profile config activates gateway mode."""
    config = dict(POLICY)
    overrides = {}
    gateway = {"url": GATEWAY["url"]}
    if source == "provider":
        config["gateway"] = gateway
    else:
        overrides["gateway"] = gateway
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("victor.config.accounts.get_account_manager", side_effect=AssertionError("account")),
        patch("victor.config.api_keys.get_api_key", side_effect=AssertionError("upstream key")),
        patch("httpx.get", side_effect=AssertionError("upstream probe")),
    ):
        result = Settings.get_provider_settings(
            fixture_settings(config), provider, overrides, account_name="unused-upstream"
        )
    assert result["gateway"] == {
        "url": "https://synthetic-gateway.example.test",
        "auth_mode": "none",
    }


@pytest.mark.parametrize("value", ["", " ", "invalid\nkey"])
def test_invalid_explicit_gateway_key_cannot_fall_back_to_environment(value):
    from victor.config.provider_config_registry import resolve_provider_gateway

    config = {"gateway": {"url": GATEWAY["url"], "virtual_key": value}}
    with patch.dict("os.environ", {"SANDHI_GATEWAY_VIRTUAL_KEY": "valid-env-key"}, clear=True):
        with pytest.raises(ValueError, match="gateway"):
            resolve_provider_gateway(config, "openai")


def test_explicit_noauth_does_not_import_environment_credentials():
    from victor.config.provider_config_registry import resolve_provider_gateway

    config = {"gateway": {"url": GATEWAY["url"], "auth_mode": "none"}}
    with patch.dict("os.environ", {"SANDHI_GATEWAY_VIRTUAL_KEY": "unrelated-env-key"}, clear=True):
        resolve_provider_gateway(config, "openai")
    assert config["gateway"]["auth_mode"] == "none"
    assert "virtual_key" not in config["gateway"]


def test_declared_token_mode_resolves_the_documented_environment_key():
    from victor.config.provider_config_registry import resolve_provider_gateway
    from victor.config.settings import ProviderGatewayConfig

    config = {"gateway": ProviderGatewayConfig(url=GATEWAY["url"], auth_mode="tokens")}
    with patch.dict("os.environ", {"SANDHI_GATEWAY_VIRTUAL_KEY": "configured-env-key"}, clear=True):
        resolve_provider_gateway(config, "openai")
    assert config["gateway"]["auth_mode"] == "tokens"
    assert config["gateway"]["virtual_key"] == "configured-env-key"


@pytest.mark.parametrize(
    "gateway",
    [
        {"virtual_key": None},
        {"oidc": None},
        {"credential": None},
        {"auth_mode": None},
        {"url": None},
    ],
)
def test_raw_registry_all_empty_gateway_selects_direct_mode(gateway):
    """All-empty gateway values carry no intent and allow the direct account path."""
    callback = Mock(return_value={"api_key": "sk-account-key"})
    with patch.dict("os.environ", {}, clear=True):
        result = registry().get_settings(
            "openai",
            fixture_settings({}),
            profile_overrides={"gateway": gateway},
            account_data_factory=callback,
        )
    # Gateway is dropped because there's no URL — direct mode
    assert result.get("gateway") is None
    # Account callback should be called in direct mode
    callback.assert_called_once()


def test_intentionally_empty_gateway_block_keeps_direct_mode():
    from victor.config.provider_config_registry import resolve_provider_gateway

    settings = {"gateway": {}}
    with patch.dict("os.environ", {}, clear=True):
        resolve_provider_gateway(settings, "openai")
    assert settings == {}


@pytest.mark.parametrize("provider", ["openai", "zai"])
@pytest.mark.parametrize("source", ["profile", "account", "shared"])
def test_direct_api_key_keeps_its_explicit_source(provider, source):
    endpoint = "https://upstream.example.test/v1"
    overrides = {"gateway": None, "base_url": endpoint}
    account_data = {}
    if source == "profile":
        account_data["api_key"] = "account-key"
        overrides["api_key"] = "profile-key"
    elif source == "account":
        account_data["api_key"] = "account-key"

    with patch("victor.config.api_keys.get_api_key", return_value="shared-key") as shared:
        result = registry().get_settings(
            provider,
            fixture_settings({"gateway": GATEWAY}),
            profile_overrides=overrides,
            account_data=account_data,
        )

    assert result.get("gateway") is None
    assert result["base_url"] == endpoint
    assert result["api_key"] == f"{source}-key"
    if source == "shared":
        shared.assert_called_once_with(provider)
    else:
        shared.assert_not_called()
