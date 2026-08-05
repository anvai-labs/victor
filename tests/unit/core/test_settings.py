# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for config/settings.py module."""

import os
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

import pytest
from pydantic import SecretStr

from victor.config.settings import (
    ProviderConfig,
    ProfileConfig,
    Settings,
    load_settings,
)


class TestProviderConfig:
    """Tests for ProviderConfig model."""

    def test_provider_config_defaults(self):
        """Test ProviderConfig with default values."""
        config = ProviderConfig()

        assert config.api_key is None
        assert config.base_url is None
        assert config.timeout == 300
        assert config.max_retries == 3
        assert config.organization is None

    def test_provider_config_with_values(self):
        """Test ProviderConfig with custom values."""
        config = ProviderConfig(
            api_key="test_key",
            base_url="https://api.example.com",
            timeout=120,
            max_retries=5,
            organization="test_org",
        )

        assert isinstance(config.api_key, SecretStr)
        assert config.api_key.get_secret_value() == "test_key"
        assert config.api_key_value == "test_key"
        assert config.base_url == "https://api.example.com"
        assert config.timeout == 120
        assert config.max_retries == 5
        assert config.organization == "test_org"


class TestProfileConfig:
    """Tests for ProfileConfig model."""

    def test_profile_config_creation(self):
        """Test creating a ProfileConfig."""
        config = ProfileConfig(provider="lmstudio", model="qwen2.5-coder:7b")

        assert config.provider == "lmstudio"
        assert config.model == "qwen2.5-coder:7b"
        assert config.temperature == 0.6  # ADR-013 default flip
        assert config.max_tokens == 4096

    def test_profile_config_custom_values(self):
        """Test ProfileConfig with custom values."""
        config = ProfileConfig(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            temperature=0.5,
            max_tokens=8192,
        )

        assert config.provider == "anthropic"
        assert config.model == "claude-3-5-sonnet-20241022"
        assert config.temperature == 0.5
        assert config.max_tokens == 8192

    def test_profile_config_provider_tuning_defaults(self):
        """Test ProfileConfig provider tuning options default to None."""
        config = ProfileConfig(provider="deepseek", model="deepseek-chat")

        # All tuning options should default to None
        assert config.loop_repeat_threshold is None
        assert config.max_continuation_prompts is None
        assert config.quality_threshold is None
        assert config.grounding_threshold is None
        assert config.max_tool_calls_per_turn is None
        assert config.tool_cache_enabled is None
        assert config.tool_deduplication_enabled is None
        assert config.session_idle_timeout is None
        assert config.timeout is None

    def test_profile_config_provider_tuning_custom_values(self):
        """Test ProfileConfig with custom provider tuning options."""
        config = ProfileConfig(
            provider="deepseek",
            model="deepseek-chat",
            loop_repeat_threshold=2,
            max_continuation_prompts=4,
            quality_threshold=0.6,
            grounding_threshold=0.8,
            max_tool_calls_per_turn=5,
            tool_cache_enabled=True,
            tool_deduplication_enabled=True,
            session_idle_timeout=300,
            timeout=120,
        )

        assert config.loop_repeat_threshold == 2
        assert config.max_continuation_prompts == 4
        assert config.quality_threshold == 0.6
        assert config.grounding_threshold == 0.8
        assert config.max_tool_calls_per_turn == 5
        assert config.tool_cache_enabled is True
        assert config.tool_deduplication_enabled is True
        assert config.session_idle_timeout == 300
        assert config.timeout == 120

    def test_profile_config_xai_tuning(self):
        """Test ProfileConfig with xAI/Grok tuning options."""
        config = ProfileConfig(
            provider="xai",
            model="grok-code-fast-1",
            max_continuation_prompts=5,
            quality_threshold=0.5,
            session_idle_timeout=300,
        )

        assert config.provider == "xai"
        assert config.max_continuation_prompts == 5
        assert config.quality_threshold == 0.5
        assert config.session_idle_timeout == 300


class TestSettings:
    """Tests for Settings class."""

    @pytest.fixture(autouse=True)
    def _bypass_account_manager(self, tmp_path):
        """Force get_provider_settings to use legacy ProviderConfigRegistry.

        On developer machines ~/.victor/config.yaml exists, causing
        AccountManager to return a different dict shape than the legacy
        path these tests exercise.
        """
        fake_config = tmp_path / "nonexistent_config.yaml"
        with patch("victor.config.accounts.get_account_manager") as mock_get_am:
            mock_am = MagicMock()
            mock_am.config_path = fake_config  # .exists() → False
            mock_get_am.return_value = mock_am
            yield

    def test_settings_defaults(self):
        """Test Settings with default values."""
        settings = Settings()

        # Note: default_provider is "ollama" in Settings class
        assert settings.provider.default_provider == "ollama"
        assert settings.provider.default_model == "qwen3.5:27b-q4_K_M"
        assert settings.provider.default_temperature == 0.6  # ADR-013 default flip
        assert settings.provider.default_max_tokens == 4096
        assert settings.log_level == "INFO"
        assert settings.airgapped_mode is False
        assert settings.ui.stream_responses is True
        assert settings.event_queue_maxsize == 10000
        assert settings.event_queue_overflow_policy == "drop_newest"
        assert settings.event_queue_overflow_block_timeout_ms == 50.0
        assert (
            settings.event_queue_overflow_topic_policies["vertical.applied"] == "block_with_timeout"
        )
        assert settings.event_queue_overflow_topic_block_timeout_ms["lifecycle.session.*"] == 150.0
        assert settings.extension_loader_warn_queue_threshold == 24
        assert settings.extension_loader_error_queue_threshold == 32
        assert settings.extension_loader_warn_in_flight_threshold == 6
        assert settings.extension_loader_error_in_flight_threshold == 8
        assert settings.extension_loader_pressure_cooldown_seconds == 5.0
        assert settings.extension_loader_metrics_reporter_enabled is False
        assert settings.extension_loader_metrics_reporter_interval_seconds == 60.0
        assert settings.tools.generic_result_cache_enabled is False
        assert settings.cache.http_connection_pool_enabled is False
        assert settings.framework_preload_enabled is True  # Enabled by default for performance
        assert settings.framework_private_fallback_strict_mode is False
        assert settings.framework_protocol_fallback_strict_mode is False

    def test_settings_event_overflow_policy_validation(self):
        """event_queue_overflow_policy should validate and normalize values."""
        import pytest

        settings = Settings(event_queue_overflow_policy="Drop_Oldest")
        assert settings.event_queue_overflow_policy == "drop_oldest"

        with pytest.raises(ValueError, match="event_queue_overflow_policy must be one of"):
            Settings(event_queue_overflow_policy="reject")

    def test_settings_event_topic_overflow_policy_validation(self):
        """Per-topic overflow policy maps should validate and normalize values."""
        import pytest

        settings = Settings(
            event_queue_overflow_topic_policies={" lifecycle.session.* ": "Drop_Oldest"}
        )
        assert settings.event_queue_overflow_topic_policies == {
            "lifecycle.session.*": "drop_oldest"
        }

        with pytest.raises(
            ValueError,
            match="event_queue_overflow_topic_policies values must be one of",
        ):
            Settings(event_queue_overflow_topic_policies={"lifecycle.session.*": "reject"})

    def test_settings_event_topic_block_timeout_validation(self):
        """Per-topic block timeout overrides must be numeric and >= 0."""
        import pytest

        settings = Settings(event_queue_overflow_topic_block_timeout_ms={"error.*": 125})
        assert settings.event_queue_overflow_topic_block_timeout_ms == {"error.*": 125.0}

        with pytest.raises(
            ValueError,
            match="Input should be a valid number",
        ):
            Settings(event_queue_overflow_topic_block_timeout_ms={"error.*": "abc"})

        with pytest.raises(
            ValueError,
            match="event_queue_overflow_topic_block_timeout_ms values must be >=",
        ):
            Settings(event_queue_overflow_topic_block_timeout_ms={"error.*": -1})

    def test_settings_extension_loader_threshold_relationship_validation(self):
        """Error thresholds must be >= warning thresholds."""
        import pytest

        with pytest.raises(
            ValueError,
            match="extension_loader_error_queue_threshold must be >=",
        ):
            Settings(
                extension_loader_warn_queue_threshold=20,
                extension_loader_error_queue_threshold=19,
            )

        with pytest.raises(
            ValueError,
            match="extension_loader_error_in_flight_threshold must be >=",
        ):
            Settings(
                extension_loader_warn_in_flight_threshold=8,
                extension_loader_error_in_flight_threshold=7,
            )

    def test_settings_extension_loader_reporter_interval_validation(self):
        """extension_loader_metrics_reporter_interval_seconds must be > 0."""
        import pytest

        with pytest.raises(
            ValueError,
            match="extension_loader_metrics_reporter_interval_seconds must be > 0",
        ):
            Settings(extension_loader_metrics_reporter_interval_seconds=0)

    def test_settings_runtime_infra_validation(self):
        """Runtime infra thresholds must be valid (fields live in the cache nested group)."""
        import pytest

        with pytest.raises(ValueError, match="http_connection_pool_max_connections must be >= 1"):
            Settings(cache={"http_connection_pool_max_connections": 0})

        with pytest.raises(
            ValueError,
            match="http_connection_pool_max_connections_per_host must be >= 1",
        ):
            Settings(cache={"http_connection_pool_max_connections_per_host": 0})

        with pytest.raises(ValueError, match="http_connection_pool_total_timeout must be > 0"):
            Settings(cache={"http_connection_pool_total_timeout": 0})

    def test_get_config_dir(self):
        """Test getting config directory uses GLOBAL_VICTOR_DIR."""
        mock_dir = Path("/mock/home/.victor")
        with patch("victor.config.settings.GLOBAL_VICTOR_DIR", mock_dir):
            with patch.object(Path, "mkdir") as mock_mkdir:
                config_dir = Settings.get_config_dir()

                assert config_dir == mock_dir
                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_load_profiles_no_file_seeds_bundled_defaults(self, tmp_path):
        """A missing user file is seeded from the bundled defaults."""
        with patch.object(Settings, "get_config_dir", classmethod(lambda cls: tmp_path)):
            profiles = Settings.load_profiles()

        assert (tmp_path / "profiles.yaml").exists()
        assert "default" in profiles
        assert profiles["default"].provider == "ollama"

    def test_load_profiles_no_file_legacy_lmstudio_fallback(self):
        """Without the bundled resource (unusual install), the legacy
        LM Studio probe still provides a default profile."""
        with patch.object(Settings, "get_config_dir") as mock_get_config_dir:
            mock_dir = MagicMock()
            mock_profiles_file = MagicMock()
            mock_profiles_file.exists.return_value = False
            mock_dir.__truediv__.return_value = mock_profiles_file
            mock_get_config_dir.return_value = mock_dir

            with (
                patch.object(Settings, "_bundled_default_profiles_text", return_value=None),
                patch.object(
                    Settings,
                    "_choose_default_lmstudio_model",
                    return_value="qwen2.5-coder:7b",
                ),
            ):
                profiles = Settings.load_profiles()

                assert "default" in profiles
                assert profiles["default"].provider == "lmstudio"
                assert profiles["default"].model == "qwen2.5-coder:7b"

    def test_load_profiles_with_file(self):
        """Test loading profiles from YAML file."""
        yaml_content = """
profiles:
  dev:
    provider: ollama
    model: qwen2.5-coder:7b
    temperature: 0.8
    max_tokens: 2048
  prod:
    provider: anthropic
    model: claude-3-5-sonnet-20241022
    temperature: 0.5
    max_tokens: 8192
"""

        with patch.object(Settings, "get_config_dir") as mock_get_config_dir:
            mock_dir = MagicMock()
            mock_profiles_file = MagicMock()
            mock_profiles_file.exists.return_value = True
            mock_profiles_file.read_text.return_value = yaml_content
            mock_dir.__truediv__.return_value = mock_profiles_file
            mock_get_config_dir.return_value = mock_dir

            profiles = Settings.load_profiles()

            assert "dev" in profiles
            assert "prod" in profiles
            assert profiles["dev"].provider == "ollama"
            assert profiles["prod"].provider == "anthropic"
            assert profiles["prod"].temperature == 0.5

    def test_load_profiles_error_handling(self):
        """Test error handling when loading profiles fails."""
        with patch.object(Settings, "get_config_dir") as mock_get_config_dir:
            mock_dir = MagicMock()
            mock_profiles_file = MagicMock()
            mock_profiles_file.exists.return_value = True
            mock_dir.__truediv__.return_value = mock_profiles_file
            mock_get_config_dir.return_value = mock_dir

            mock_profiles_file.read_text.side_effect = IOError("Read error")
            profiles = Settings.load_profiles()

            # Should return empty dict on error
            assert profiles == {}

    def test_load_provider_config_no_file(self):
        """Test loading provider config when file doesn't exist."""
        with patch.object(Settings, "get_config_dir") as mock_get_config_dir:
            mock_dir = MagicMock()
            mock_profiles_file = MagicMock()
            mock_profiles_file.exists.return_value = False
            mock_dir.__truediv__.return_value = mock_profiles_file
            mock_get_config_dir.return_value = mock_dir

            config = Settings.load_provider_config("anthropic")

            assert config is None

    def test_load_provider_config_with_file(self):
        """Test loading provider config from YAML file."""
        yaml_content = """
providers:
  anthropic:
    api_key: test_key_123
    timeout: 90
    max_retries: 5
  openai:
    api_key: openai_key
    base_url: https://api.openai.com/v1
"""

        with patch.object(Settings, "get_config_dir") as mock_get_config_dir:
            mock_dir = MagicMock()
            mock_profiles_file = MagicMock()
            mock_profiles_file.exists.return_value = True
            mock_dir.__truediv__.return_value = mock_profiles_file
            mock_get_config_dir.return_value = mock_dir

            with patch("builtins.open", mock_open(read_data=yaml_content)):
                config = Settings.load_provider_config("anthropic")

                assert config is not None
                assert config.api_key is not None
                assert config.api_key.get_secret_value() == "test_key_123"
                assert config.timeout == 90
                assert config.max_retries == 5

    def test_load_provider_config_with_env_vars(self):
        """Test loading provider config with environment variable expansion."""
        yaml_content = """
providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    timeout: 60
"""

        with patch.object(Settings, "get_config_dir") as mock_get_config_dir:
            mock_dir = MagicMock()
            mock_profiles_file = MagicMock()
            mock_profiles_file.exists.return_value = True
            mock_dir.__truediv__.return_value = mock_profiles_file
            mock_get_config_dir.return_value = mock_dir

            with patch("builtins.open", mock_open(read_data=yaml_content)):
                with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env_key_456"}):
                    config = Settings.load_provider_config("anthropic")

                    assert config is not None
                    assert config.api_key is not None
                    assert config.api_key.get_secret_value() == "env_key_456"

    def test_load_provider_config_error_handling(self):
        """Test error handling when loading provider config fails."""
        with patch.object(Settings, "get_config_dir") as mock_get_config_dir:
            mock_dir = MagicMock()
            mock_profiles_file = MagicMock()
            mock_profiles_file.exists.return_value = True
            mock_dir.__truediv__.return_value = mock_profiles_file
            mock_get_config_dir.return_value = mock_dir

            with patch("builtins.open", side_effect=IOError("Read error")):
                config = Settings.load_provider_config("anthropic")

                # Should return None on error
                assert config is None

    def test_get_provider_settings_anthropic(self):
        """Test getting settings for Anthropic provider."""
        settings = Settings(anthropic_api_key="test_anthropic_key")

        provider_settings = settings.get_provider_settings("anthropic")

        assert provider_settings["api_key"] == "test_anthropic_key"
        assert provider_settings["base_url"] == "https://api.anthropic.com"

    def test_get_provider_settings_openai(self):
        """Test getting settings for OpenAI provider."""
        settings = Settings(openai_api_key="test_openai_key")

        provider_settings = settings.get_provider_settings("openai")

        assert provider_settings["api_key"] == "test_openai_key"
        assert provider_settings["base_url"] == "https://api.openai.com/v1"

    def test_get_provider_settings_google(self):
        """Test getting settings for Google provider."""
        settings = Settings(google_api_key="test_google_key")

        provider_settings = settings.get_provider_settings("google")

        assert provider_settings["api_key"] == "test_google_key"

    def test_get_provider_settings_ollama(self):
        """Test getting settings for Ollama provider."""
        settings = Settings(ollama_base_url="http://localhost:11434")

        with patch.object(Settings, "load_provider_config", return_value=None):
            provider_settings = settings.get_provider_settings("ollama")

        assert provider_settings["base_url"] == "http://localhost:11434"

    def test_choose_default_lmstudio_model_prefers_available(self):
        """Pick preferred model if advertised by reachable LMStudio server."""
        urls = ["http://192.168.1.126:1234"]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"id": "random-model"}, {"id": "qwen2.5-coder:14b"}]
        }

        with patch("httpx.get", return_value=mock_response):
            model = Settings._choose_default_lmstudio_model(urls)

        assert model == "qwen2.5-coder:14b"

    def test_choose_default_lmstudio_model_respects_vram(self):
        """Select most capable model within detected VRAM budget."""
        urls = ["http://127.0.0.1:1234"]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "qwen3-coder-30b"},
                {"id": "qwen2.5-coder-7b"},
            ]
        }

        with patch("httpx.get", return_value=mock_response):
            with patch.object(Settings, "_detect_vram_gb", return_value=12.0):
                model = Settings._choose_default_lmstudio_model(urls)
        assert model == "qwen2.5-coder-7b"

        with patch("httpx.get", return_value=mock_response):
            with patch.object(Settings, "_detect_vram_gb", return_value=48.0):
                model = Settings._choose_default_lmstudio_model(urls)
        # Picks most capable coder model within budget
        assert model == "qwen3-coder-30b"

    def test_choose_default_lmstudio_model_respects_config_cap(self):
        """Use configured max_vram cap when larger GPUs are available."""
        urls = ["http://127.0.0.1:1234"]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "qwen3-coder-30b"},
                {"id": "qwen2.5-coder-7b"},
            ]
        }

        with patch("httpx.get", return_value=mock_response):
            with patch.object(Settings, "_detect_vram_gb", return_value=48.0):
                model = Settings._choose_default_lmstudio_model(urls, max_vram_gb=10.0)

        # Cap forces smaller model when budget is limited
        assert model == "qwen2.5-coder-7b"

    def test_get_provider_settings_lmstudio(self):
        """Test getting settings for LMStudio provider."""
        settings = Settings(lmstudio_base_urls=["http://localhost:1234"])

        with patch.object(Settings, "load_provider_config", return_value=None):
            provider_settings = settings.get_provider_settings("lmstudio")

        assert provider_settings["base_url"].startswith("http://localhost:1234")

    def test_get_provider_settings_vllm(self):
        """Test getting settings for vLLM provider."""
        settings = Settings(vllm_base_url="http://localhost:8000")

        provider_settings = settings.get_provider_settings("vllm")

        assert provider_settings["base_url"] == "http://localhost:8000"

    def test_get_provider_settings_with_config(self):
        """Test getting settings with provider config from YAML."""
        yaml_content = """
providers:
  anthropic:
    timeout: 120
    max_retries: 10
"""

        with patch.object(Settings, "get_config_dir") as mock_get_config_dir:
            mock_dir = MagicMock()
            mock_profiles_file = MagicMock()
            mock_profiles_file.exists.return_value = True
            mock_dir.__truediv__.return_value = mock_profiles_file
            mock_get_config_dir.return_value = mock_dir

            with patch("builtins.open", mock_open(read_data=yaml_content)):
                settings = Settings(anthropic_api_key="test_key")
                provider_settings = settings.get_provider_settings("anthropic")

                assert provider_settings["api_key"] == "test_key"
                assert provider_settings["timeout"] == 120
                assert provider_settings["max_retries"] == 10

    def test_server_secrets_use_secretstr(self):
        """Server-facing secrets should be stored as SecretStr values."""
        settings = Settings(
            server_api_key="server-token",
            server_session_secret="session-secret",
        )

        assert isinstance(settings.server.server_api_key, SecretStr)
        assert settings.server.server_api_key.get_secret_value() == "server-token"
        assert isinstance(settings.server.server_session_secret, SecretStr)
        assert settings.server.server_session_secret.get_secret_value() == "session-secret"


class TestLoadSettings:
    """Tests for load_settings function."""

    def test_load_settings(self):
        """Test loading settings."""
        settings = load_settings()

        assert isinstance(settings, Settings)
        # Note: default_provider is "ollama" in Settings class
        assert settings.provider.default_provider == "ollama"


class TestToolSelectionValidation:
    """Tests for ProfileConfig tool_selection validation."""

    def test_tool_selection_none(self):
        """Test tool_selection with None value (covers line 65-66)."""
        config = ProfileConfig(
            provider="ollama",
            model="test",
            tool_selection=None,
        )
        assert config.tool_selection is None

    def test_tool_selection_tier_preset_tiny(self):
        """Test tool_selection with tiny tier preset (covers lines 69-84)."""
        config = ProfileConfig(
            provider="ollama",
            model="test",
            tool_selection={"model_size_tier": "tiny"},
        )
        assert config.tool_selection["base_threshold"] == 0.35
        assert config.tool_selection["base_max_tools"] == 5

    def test_tool_selection_tier_preset_small(self):
        """Test tool_selection with small tier preset."""
        config = ProfileConfig(
            provider="ollama",
            model="test",
            tool_selection={"model_size_tier": "small"},
        )
        assert config.tool_selection["base_threshold"] == 0.25
        assert config.tool_selection["base_max_tools"] == 7

    def test_tool_selection_tier_preset_medium(self):
        """Test tool_selection with medium tier preset."""
        config = ProfileConfig(
            provider="ollama",
            model="test",
            tool_selection={"model_size_tier": "medium"},
        )
        assert config.tool_selection["base_threshold"] == 0.20
        assert config.tool_selection["base_max_tools"] == 10

    def test_tool_selection_tier_preset_large(self):
        """Test tool_selection with large tier preset."""
        config = ProfileConfig(
            provider="ollama",
            model="test",
            tool_selection={"model_size_tier": "large"},
        )
        assert config.tool_selection["base_threshold"] == 0.15
        assert config.tool_selection["base_max_tools"] == 12

    def test_tool_selection_tier_preset_cloud(self):
        """Test tool_selection with cloud tier preset."""
        config = ProfileConfig(
            provider="anthropic",
            model="claude",
            tool_selection={"model_size_tier": "cloud"},
        )
        assert config.tool_selection["base_threshold"] == 0.18
        assert config.tool_selection["base_max_tools"] == 10

    def test_tool_selection_tier_preset_with_override(self):
        """Test tier preset values can be overridden manually (covers line 83-84)."""
        config = ProfileConfig(
            provider="ollama",
            model="test",
            tool_selection={
                "model_size_tier": "small",
                "base_threshold": 0.30,  # Override preset
            },
        )
        # Override should take precedence
        assert config.tool_selection["base_threshold"] == 0.30
        assert config.tool_selection["base_max_tools"] == 7  # From preset

    def test_tool_selection_invalid_threshold_type(self):
        """Test tool_selection with invalid threshold type (covers lines 89-90)."""
        import pytest

        with pytest.raises(ValueError, match="base_threshold must be a number"):
            ProfileConfig(
                provider="ollama",
                model="test",
                tool_selection={"base_threshold": "not_a_number"},
            )

    def test_tool_selection_threshold_out_of_range(self):
        """Test tool_selection with threshold out of range (covers lines 91-92)."""
        import pytest

        with pytest.raises(ValueError, match="base_threshold must be between 0.0 and 1.0"):
            ProfileConfig(
                provider="ollama",
                model="test",
                tool_selection={"base_threshold": 1.5},
            )

    def test_tool_selection_invalid_max_tools_type(self):
        """Test tool_selection with invalid max_tools type (covers lines 97-98)."""
        import pytest

        with pytest.raises(ValueError, match="base_max_tools must be an integer"):
            ProfileConfig(
                provider="ollama",
                model="test",
                tool_selection={"base_max_tools": "not_an_int"},
            )

    def test_tool_selection_max_tools_negative(self):
        """Test tool_selection with negative max_tools (covers lines 99-100)."""
        import pytest

        with pytest.raises(ValueError, match="base_max_tools must be positive"):
            ProfileConfig(
                provider="ollama",
                model="test",
                tool_selection={"base_max_tools": 0},
            )

    def test_tool_selection_valid_custom_values(self):
        """Test tool_selection with valid custom values (covers lines 87-88, 94-96)."""
        config = ProfileConfig(
            provider="ollama",
            model="test",
            tool_selection={
                "base_threshold": 0.5,
                "base_max_tools": 8,
            },
        )
        assert config.tool_selection["base_threshold"] == 0.5
        assert config.tool_selection["base_max_tools"] == 8


class TestSettingsExtra:
    """Additional Settings tests for coverage."""

    def test_settings_semantic_tool_selection(self):
        """Test semantic tool selection settings."""
        settings = Settings(use_semantic_tool_selection=True)
        assert settings.tool_selection.use_semantic_tool_selection is True

    def test_settings_airgapped_mode(self):
        """Test airgapped mode settings."""
        settings = Settings(airgapped_mode=True)
        assert settings.airgapped_mode is True

    def test_settings_tool_cache_settings(self):
        """Test tool cache settings — stored in tools nested group."""
        settings = Settings(
            tool_cache_enabled=True,
            tool_cache_ttl=600,
        )
        assert settings.tools.tool_cache_enabled is True
        assert settings.tools.tool_cache_ttl == 600

    def test_settings_analytics_disabled(self):
        """Test analytics disabled setting (analytics_enabled lives in analytics nested group)."""
        settings = Settings(analytics_enabled=False)
        assert settings.analytics.analytics_enabled is False
