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

"""Tests for victor.providers.resolution module."""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest

from victor.providers.resolution import (
    UnifiedApiKeyResolver,
    APIKeyResult,
    APIKeyNotFoundError,
    KeySource,
    get_api_key_with_resolution,
    _get_provider_env_var,
    _get_key_from_sentinelpass,
)


class TestGetProviderEnvVar:
    """Tests for _get_provider_env_var function."""

    def test_deepseek_env_var(self):
        assert _get_provider_env_var("deepseek") == "DEEPSEEK_API_KEY"

    def test_anthropic_env_var(self):
        assert _get_provider_env_var("anthropic") == "ANTHROPIC_API_KEY"

    def test_openai_env_var(self):
        assert _get_provider_env_var("openai") == "OPENAI_API_KEY"

    def test_case_insensitive(self):
        assert _get_provider_env_var("DEEPSEEK") == "DEEPSEEK_API_KEY"
        assert _get_provider_env_var("DeepSeek") == "DEEPSEEK_API_KEY"

    def test_unknown_provider(self):
        assert _get_provider_env_var("unknown") is None


class TestKeySource:
    """Tests for KeySource dataclass."""

    def test_key_source_str_found(self):
        source = KeySource(
            source="environment",
            description="DEEPSEEK_API_KEY environment variable",
            found=True,
            value_preview="sk-abc123...",
            interactive_required=False,
        )
        assert "✓" in str(source)
        assert "DEEPSEEK_API_KEY" in str(source)

    def test_key_source_str_not_found(self):
        source = KeySource(
            source="keyring",
            description="System keyring",
            found=False,
            interactive_required=True,
        )
        assert "✗" in str(source)
        assert "System keyring" in str(source)


class TestUnifiedApiKeyResolver:
    """Tests for UnifiedApiKeyResolver class."""

    def test_detect_non_interactive_from_env_var(self):
        """Test VICTOR_NONINTERACTIVE=true detection."""
        with patch.dict(os.environ, {"VICTOR_NONINTERACTIVE": "true"}):
            resolver = UnifiedApiKeyResolver()
            assert resolver.non_interactive is True

    def test_detect_non_interactive_from_ci(self):
        """Test CI environment detection."""
        with patch.dict(os.environ, {"CI": "true"}):
            resolver = UnifiedApiKeyResolver()
            assert resolver.non_interactive is True

    def test_detect_non_interactive_from_kubernetes(self):
        """Test Kubernetes environment detection."""
        with patch.dict(os.environ, {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}):
            resolver = UnifiedApiKeyResolver()
            assert resolver.non_interactive is True

    def test_detect_non_interactive_from_container(self):
        """Test container environment detection."""
        with patch.dict(os.environ, {"container": "docker"}):
            resolver = UnifiedApiKeyResolver()
            assert resolver.non_interactive is True

    def test_detect_interactive_mode(self):
        """Test interactive mode when no indicators present."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("sys.stdin.isatty", return_value=True):
                resolver = UnifiedApiKeyResolver()
                assert resolver.non_interactive is False

    def test_explicit_non_interactive_override(self):
        """Test explicit non_interactive parameter overrides detection."""
        with patch.dict(os.environ, {}, clear=True):
            resolver = UnifiedApiKeyResolver(non_interactive=True)
            assert resolver.non_interactive is True

    def test_explicit_interactive_override(self):
        """Test explicit interactive parameter overrides detection."""
        with patch.dict(os.environ, {"CI": "true"}):
            resolver = UnifiedApiKeyResolver(non_interactive=False)
            assert resolver.non_interactive is False

    def test_get_api_key_explicit_parameter(self):
        """Test explicit api_key parameter takes priority."""
        resolver = UnifiedApiKeyResolver()
        result = resolver.get_api_key("deepseek", explicit_key="sk-test123")

        assert result.key == "sk-test123"
        assert result.source == "explicit"
        assert result.source_detail == "Explicit api_key parameter"
        assert result.confidence == "high"
        assert len(result.sources_attempted) == 1
        assert result.sources_attempted[0].found is True

    def test_get_api_key_from_environment(self):
        """Test getting API key from environment variable."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-env-key-456"}):
            resolver = UnifiedApiKeyResolver()
            result = resolver.get_api_key("deepseek")

            assert result.key == "sk-env-key-456"
            assert result.source == "environment"
            assert "DEEPSEEK_API_KEY" in result.source_detail
            assert result.confidence == "high"

    def test_get_api_key_prefers_environment_over_sentinelpass(self):
        """Environment variables must win over SentinelPass lookups."""
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "sk-ant-env-key",
                "VICTOR_SENTINELPASS_ENABLED": "true",
            },
            clear=True,
        ):
            with patch(
                "victor.providers.resolution._get_key_from_sentinelpass"
            ) as mock_sentinelpass:
                resolver = UnifiedApiKeyResolver(non_interactive=True)
                result = resolver.get_api_key("anthropic")

                assert result.key == "sk-ant-env-key"
                assert result.source == "environment"
                mock_sentinelpass.assert_not_called()

    def test_get_api_key_from_sentinelpass_when_enabled(self):
        """SentinelPass is used after env vars and before keyring/file fallback."""
        with patch.dict(os.environ, {"VICTOR_SENTINELPASS_ENABLED": "true"}, clear=True):
            with patch(
                "victor.providers.resolution._get_key_from_sentinelpass",
                return_value="sk-ant-sentinelpass",
            ):
                resolver = UnifiedApiKeyResolver(non_interactive=True)
                result = resolver.get_api_key("anthropic")

                assert result.key == "sk-ant-sentinelpass"
                assert result.source == "sentinelpass"
                assert result.source_detail == "SentinelPass local vault"
                assert any(source.source == "sentinelpass" for source in result.sources_attempted)

    def test_get_key_from_sentinelpass_invokes_secret_get_without_logging_value(self):
        """SentinelPass helper shells out to `sentinelpass secret-get` and returns stdout."""
        completed = MagicMock()
        completed.returncode = 0
        completed.stdout = "sk-ant-from-vault\n"
        completed.stderr = ""

        with patch.dict(os.environ, {"VICTOR_SENTINELPASS_ENABLED": "true"}, clear=True):
            with patch(
                "victor.providers.resolution.shutil.which",
                return_value="/usr/bin/sentinelpass",
            ):
                with patch(
                    "victor.providers.resolution.subprocess.run", return_value=completed
                ) as run:
                    key = _get_key_from_sentinelpass("anthropic")

        assert key == "sk-ant-from-vault"
        cmd = run.call_args.args[0]
        assert cmd[:2] == ["/usr/bin/sentinelpass", "secret-get"]
        assert "--domain" in cmd
        assert "anthropic" in cmd
        assert "sk-ant-from-vault" not in cmd

    def test_get_api_key_not_found(self):
        """Test API key not found."""
        with patch.dict(os.environ, {}, clear=True):
            resolver = UnifiedApiKeyResolver()
            result = resolver.get_api_key("deepseek")

            assert result.key is None
            assert result.source == "none"
            assert result.confidence == "low"
            # Should have attempted multiple sources
            assert len(result.sources_attempted) >= 2

    def test_get_api_key_keyring_skipped_in_non_interactive(self):
        """Test keyring is skipped in non-interactive mode."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("victor.providers.resolution.is_keyring_available", return_value=True):
                resolver = UnifiedApiKeyResolver(non_interactive=True)
                result = resolver.get_api_key("deepseek")

                # Check that keyring source shows "skipped"
                keyring_sources = [s for s in result.sources_attempted if s.source == "keyring"]
                assert len(keyring_sources) > 0
                assert "skipped" in keyring_sources[0].description

    def test_get_api_key_caching(self):
        """Test that results are cached."""
        resolver = UnifiedApiKeyResolver()

        # First call
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}):
            result1 = resolver.get_api_key("deepseek")
            assert result1.key == "sk-test"

        # Second call should use cache
        with patch.dict(os.environ, {}, clear=True):
            result2 = resolver.get_api_key("deepseek")
            # Still returns the cached value even though env var is gone
            assert result2.key == "sk-test"

    def test_clear_cache(self):
        """Test cache clearing."""
        resolver = UnifiedApiKeyResolver()

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}):
            resolver.get_api_key("deepseek")

        resolver.clear_cache()

        with patch.dict(os.environ, {}, clear=True):
            result = resolver.get_api_key("deepseek")
            assert result.key is None

    def test_preview_key_short(self):
        """Test key preview for short keys."""
        resolver = UnifiedApiKeyResolver()
        assert resolver._preview_key("sk") == "(empty)"

    def test_preview_key_medium(self):
        """Test key preview for medium keys."""
        resolver = UnifiedApiKeyResolver()
        assert resolver._preview_key("sk-1234") == "sk-12..."

    def test_preview_key_long(self):
        """Test key preview for long keys."""
        resolver = UnifiedApiKeyResolver()
        assert resolver._preview_key("sk-1234567890abcdef") == "sk-12345678..."


class TestAPIKeyNotFoundError:
    """Tests for APIKeyNotFoundError exception."""

    def test_error_message_format(self):
        """Test error message is formatted correctly."""
        sources = [
            KeySource(
                source="explicit",
                description="Explicit api_key parameter",
                found=False,
                interactive_required=False,
            ),
            KeySource(
                source="environment",
                description="DEEPSEEK_API_KEY environment variable",
                found=False,
                interactive_required=False,
            ),
        ]

        error = APIKeyNotFoundError(
            provider="deepseek",
            sources_attempted=sources,
            non_interactive=True,
            model="deepseek-chat",
        )

        error_str = str(error)
        assert "DEEPSEEK API key not found" in error_str
        assert "Tried 2 source" in error_str
        assert "DEEPSEEK_API_KEY environment variable" in error_str
        assert "Solutions:" in error_str
        assert "non-interactive mode" in error_str

    def test_error_message_non_interactive_suggests_env_var(self):
        """Test error message suggests env var in non-interactive mode."""
        sources = [
            KeySource(
                source="environment",
                description="DEEPSEEK_API_KEY environment variable",
                found=False,
                interactive_required=False,
            ),
        ]

        error = APIKeyNotFoundError(
            provider="deepseek",
            sources_attempted=sources,
            non_interactive=True,
        )

        error_str = str(error)
        assert "DEEPSEEK_API_KEY environment variable" in error_str
        assert "(recommended for servers/containers/CI)" in error_str

    def test_error_message_interactive_suggests_keyring(self):
        """Test error message suggests keyring in interactive mode."""
        sources = [
            KeySource(
                source="keyring",
                description="System keyring",
                found=False,
                interactive_required=True,
            ),
        ]

        error = APIKeyNotFoundError(
            provider="anthropic",
            sources_attempted=sources,
            non_interactive=False,
        )

        error_str = str(error)
        assert "victor keys set anthropic --keyring" in error_str

    def test_to_dict(self):
        """Test conversion to dictionary."""
        sources = [
            KeySource(
                source="environment",
                description="ANTHROPIC_API_KEY",
                found=False,
                interactive_required=False,
            ),
        ]

        error = APIKeyNotFoundError(
            provider="anthropic",
            sources_attempted=sources,
            non_interactive=True,
            model="claude-3-5-haiku",
        )

        d = error.to_dict()
        assert d["error_type"] == "APIKeyNotFound"
        assert d["provider"] == "anthropic"
        assert d["model"] == "claude-3-5-haiku"
        assert d["non_interactive"] is True
        assert len(d["sources_attempted"]) == 1


class TestGetApiKeyWithResolution:
    """Tests for get_api_key_with_resolution convenience function."""

    def test_returns_key_when_found(self):
        """Test function returns key when found."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}):
            key = get_api_key_with_resolution("deepseek")
            assert key == "sk-test"

    def test_returns_none_when_not_found_and_no_raise(self):
        """Test function returns None when not found and raise_on_not_found=False."""
        with patch.dict(os.environ, {}, clear=True):
            key = get_api_key_with_resolution(
                "deepseek",
                raise_on_not_found=False,
            )
            assert key is None

    def test_raises_when_not_found(self):
        """Test function raises when not found and raise_on_not_found=True."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(APIKeyNotFoundError) as exc_info:
                get_api_key_with_resolution("deepseek")

            assert exc_info.value.provider == "deepseek"

    def test_passes_explicit_key(self):
        """Test explicit key parameter works."""
        key = get_api_key_with_resolution("deepseek", api_key="sk-explicit")
        assert key == "sk-explicit"

    def test_respects_non_interactive_flag(self):
        """Test non_interactive flag is respected."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}):
            key = get_api_key_with_resolution("deepseek", non_interactive=True)
            assert key == "sk-test"


class TestAPIKeyResult:
    """Tests for APIKeyResult dataclass."""

    def test_result_with_key(self):
        """Test result when key is found."""
        sources = [
            KeySource(
                source="environment",
                description="DEEPSEEK_API_KEY",
                found=True,
                value_preview="sk-...",
            ),
        ]

        result = APIKeyResult(
            key="sk-test123",
            source="environment",
            source_detail="DEEPSEEK_API_KEY",
            sources_attempted=sources,
            non_interactive=True,
            confidence="high",
        )

        assert result.key == "sk-test123"
        assert result.source == "environment"
        assert result.confidence == "high"

    def test_result_without_key(self):
        """Test result when key is not found."""
        sources = [
            KeySource(
                source="environment",
                description="DEEPSEEK_API_KEY",
                found=False,
            ),
        ]

        result = APIKeyResult(
            key=None,
            source="none",
            source_detail="No key found",
            sources_attempted=sources,
            non_interactive=True,
            confidence="low",
        )

        assert result.key is None
        assert result.source == "none"
        assert result.confidence == "low"
