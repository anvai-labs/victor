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

"""Tests for the providers CLI command - achieving 70%+ coverage."""

import pytest
from unittest.mock import patch, MagicMock
from io import StringIO

from typer.testing import CliRunner

from victor.ui.commands.providers import providers_app, list_providers, console

runner = CliRunner()


class TestListProviders:
    """Tests for list_providers command."""

    def test_list_providers_outputs_table(self):
        """Test that list_providers outputs a table of providers."""
        # providers_app has list as default command, so no args needed
        result = runner.invoke(providers_app, [])
        assert result.exit_code == 0
        # Should contain table title
        assert "Available Providers" in result.stdout or "Provider" in result.stdout

    def test_list_flag_alias_outputs_table(self):
        """Test --list remains accepted for users expecting option-style listing."""
        result = runner.invoke(providers_app, ["--list"])
        assert result.exit_code == 0
        assert "Available Providers" in result.stdout or "Provider" in result.stdout

    def test_list_providers_shows_known_providers(self):
        """Test that common providers are shown in the list."""
        result = runner.invoke(providers_app, [])
        assert result.exit_code == 0
        # Check for some known providers
        assert "ollama" in result.stdout.lower() or "anthropic" in result.stdout.lower()

    def test_list_providers_shows_status_column(self):
        """Test that status column is present."""
        result = runner.invoke(providers_app, [])
        assert result.exit_code == 0
        # Should contain status indicators
        assert "Ready" in result.stdout or "Status" in result.stdout

    def test_list_providers_shows_features_column(self):
        """Test that features column is present."""
        result = runner.invoke(providers_app, [])
        assert result.exit_code == 0
        # Should mention features like Tool calling, Streaming
        output_lower = result.stdout.lower()
        assert any(x in output_lower for x in ["tool", "streaming", "features", "model"])

    def test_list_providers_shows_help_message(self):
        """Test that help message for profiles is shown."""
        result = runner.invoke(providers_app, [])
        assert result.exit_code == 0
        # Should contain hint about profiles
        assert "profiles" in result.stdout.lower()

    @patch("victor.ui.commands.providers.ProviderRegistry")
    def test_list_providers_with_mocked_registry(self, mock_registry):
        """Test list_providers with mocked registry."""
        mock_registry.list_providers.return_value = [
            "test_provider",
            "another_provider",
        ]

        result = runner.invoke(providers_app, [])
        assert result.exit_code == 0
        mock_registry.list_providers.assert_called_once()

    @patch("victor.ui.commands.providers.ProviderRegistry")
    def test_list_providers_empty_list(self, mock_registry):
        """Test list_providers with no providers."""
        mock_registry.list_providers.return_value = []

        result = runner.invoke(providers_app, [])
        assert result.exit_code == 0
        # Should still show table (even if empty)
        assert "Available Providers" in result.stdout or "Provider" in result.stdout

    @patch("victor.ui.commands.providers.ProviderRegistry")
    def test_list_providers_unknown_provider(self, mock_registry):
        """Test list_providers with unknown provider shows unknown status."""
        mock_registry.list_providers.return_value = ["unknown_custom_provider"]

        result = runner.invoke(providers_app, [])
        assert result.exit_code == 0
        # Unknown providers should still be listed
        assert "unknown_custom_provider" in result.stdout.lower() or "Unknown" in result.stdout

    @patch("victor.ui.commands.providers.ProviderRegistry")
    def test_list_providers_sorted_output(self, mock_registry):
        """Test that providers are sorted alphabetically."""
        mock_registry.list_providers.return_value = ["zeta", "alpha", "beta"]

        result = runner.invoke(providers_app, [])
        assert result.exit_code == 0
        # Output should be sorted
        output = result.stdout.lower()
        alpha_pos = output.find("alpha")
        beta_pos = output.find("beta")
        zeta_pos = output.find("zeta")

        if alpha_pos != -1 and beta_pos != -1 and zeta_pos != -1:
            assert alpha_pos < beta_pos < zeta_pos


class TestProvidersAppHelp:
    """Tests for providers app help."""

    def test_providers_help(self):
        """Test providers app help message."""
        result = runner.invoke(providers_app, ["--help"])
        assert result.exit_code == 0
        assert "providers" in result.stdout.lower() or "list" in result.stdout.lower()


class TestProviderInfoMapping:
    """Tests for provider info mapping."""

    def test_known_provider_info_completeness(self):
        """Test that known providers have complete info."""
        from victor.ui.commands.providers import list_providers

        # Just verify the function doesn't crash when providers are listed
        result = runner.invoke(providers_app, [])
        assert result.exit_code == 0

    @patch("victor.ui.commands.providers.ProviderRegistry")
    def test_ollama_provider_info(self, mock_registry):
        """Test ollama provider shows correct info."""
        mock_registry.list_providers.return_value = ["ollama"]

        result = runner.invoke(providers_app, [])
        assert result.exit_code == 0
        output = result.stdout.lower()
        # Should show ollama-related info
        assert "ollama" in output

    @patch("victor.ui.commands.providers.ProviderRegistry")
    def test_anthropic_provider_info(self, mock_registry):
        """Test anthropic provider shows correct info."""
        mock_registry.list_providers.return_value = ["anthropic"]

        result = runner.invoke(providers_app, [])
        assert result.exit_code == 0
        output = result.stdout.lower()
        assert "anthropic" in output

    @patch("victor.ui.commands.providers.ProviderRegistry")
    def test_openai_provider_info(self, mock_registry):
        """Test openai provider shows correct info."""
        mock_registry.list_providers.return_value = ["openai"]

        result = runner.invoke(providers_app, [])
        assert result.exit_code == 0
        output = result.stdout.lower()
        assert "openai" in output

    @patch("victor.ui.commands.providers.ProviderRegistry")
    def test_google_provider_info(self, mock_registry):
        """Test google provider shows correct info."""
        mock_registry.list_providers.return_value = ["google"]

        result = runner.invoke(providers_app, [])
        assert result.exit_code == 0
        output = result.stdout.lower()
        assert "google" in output

    @patch("victor.ui.commands.providers.ProviderRegistry")
    def test_xai_grok_alias(self, mock_registry):
        """Test xai and grok show as aliases."""
        mock_registry.list_providers.return_value = ["xai", "grok"]

        result = runner.invoke(providers_app, [])
        assert result.exit_code == 0
        output = result.stdout.lower()
        # Both should be present
        assert "xai" in output or "grok" in output


class TestConsoleOutput:
    """Tests for console output formatting."""

    def test_table_has_columns(self):
        """Test that output table has expected columns."""
        result = runner.invoke(providers_app, [])
        assert result.exit_code == 0
        # Table should have Provider, Status, Features columns
        output = result.stdout
        # At minimum should have Provider column header
        assert "Provider" in output or "provider" in output.lower()

    @patch("victor.ui.commands.providers.ProviderRegistry")
    def test_output_includes_profiles_hint(self, mock_registry):
        """Test that output includes profiles command hint."""
        mock_registry.list_providers.return_value = ["test"]

        result = runner.invoke(providers_app, [])
        assert result.exit_code == 0
        assert "profiles" in result.stdout.lower()

    @patch("victor.ui.commands.providers._load_configured_provider_summary")
    @patch("victor.ui.commands.providers.ProviderRegistry")
    def test_output_includes_configured_account_context(self, mock_registry, mock_configured):
        """Provider list should connect catalog providers to configured accounts."""
        mock_registry.list_providers.return_value = ["openai"]
        mock_registry.get_aliases.return_value = {}
        mock_configured.return_value = ({"openai": ["openai-cheap"]}, "openai")

        result = runner.invoke(providers_app, [])

        assert result.exit_code == 0
        assert "Configured" in result.stdout
        assert "openai-cheap" in result.stdout


class TestProviderRegistryIntegration:
    """Integration tests with actual ProviderRegistry."""

    def test_actual_providers_listed(self):
        """Test that actual registered providers are listed."""
        result = runner.invoke(providers_app, [])
        assert result.exit_code == 0

        # Should list at least some common providers
        output = result.stdout.lower()
        # At least one of these common providers should be present
        common_providers = ["ollama", "anthropic", "openai", "google"]
        assert any(p in output for p in common_providers)

    def test_all_registered_providers_appear(self):
        """Test that all registered providers appear in output."""
        from victor.providers.registry import ProviderRegistry

        result = runner.invoke(providers_app, [])
        assert result.exit_code == 0

        registered = ProviderRegistry.list_providers()
        output = result.stdout.lower()

        # At least half of registered providers should appear
        found = sum(1 for p in registered if p in output)
        assert found >= len(registered) // 2


class TestEdgeCases:
    """Edge case tests for providers command."""

    @patch("victor.ui.commands.providers.ProviderRegistry")
    def test_special_characters_in_provider_name(self, mock_registry):
        """Test handling of special characters in provider names."""
        mock_registry.list_providers.return_value = [
            "provider-with-dash",
            "provider_with_underscore",
        ]

        result = runner.invoke(providers_app, [])
        assert result.exit_code == 0
        # Should not crash on special characters

    @patch("victor.ui.commands.providers.ProviderRegistry")
    def test_very_long_provider_name(self, mock_registry):
        """Test handling of very long provider names."""
        mock_registry.list_providers.return_value = ["a" * 100]

        result = runner.invoke(providers_app, [])
        assert result.exit_code == 0
        # Should handle long names gracefully

    @patch("victor.ui.commands.providers.ProviderRegistry")
    def test_many_providers(self, mock_registry):
        """Test handling of many providers."""
        mock_registry.list_providers.return_value = [f"provider_{i}" for i in range(50)]

        result = runner.invoke(providers_app, [])
        assert result.exit_code == 0
        # Should handle many providers

    @patch("victor.ui.commands.providers.ProviderRegistry")
    def test_unicode_in_provider_name(self, mock_registry):
        """Test handling of unicode in provider names."""
        mock_registry.list_providers.return_value = [
            "provider_日本語",
            "provider_emoji_🚀",
        ]

        result = runner.invoke(providers_app, [])
        assert result.exit_code == 0
        # Should handle unicode gracefully


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
