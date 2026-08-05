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

"""Tests for feature flag integration with Rich formatter system."""

import pytest

from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager
from victor.tools.formatters import format_test_results, format_search_results


class TestFeatureFlagIntegration:
    """Test that USE_RICH_FORMATTING feature flag controls formatter behavior."""

    def test_feature_flag_exists(self):
        """Test that USE_RICH_FORMATTING feature flag is defined."""
        flag = FeatureFlag.USE_RICH_FORMATTING
        assert flag.value == "use_rich_formatting"
        assert flag.get_env_var_name() == "VICTOR_USE_RICH_FORMATTING"

    def test_feature_flag_enabled_by_default(self):
        """Test that Rich formatting is enabled by default."""
        manager = get_feature_flag_manager()
        # Should be enabled by default
        assert manager.is_enabled(FeatureFlag.USE_RICH_FORMATTING)

    def test_format_with_feature_flag_enabled(self):
        """Test that formatters produce Rich markup when feature flag is enabled."""
        manager = get_feature_flag_manager()
        manager.enable(FeatureFlag.USE_RICH_FORMATTING)

        test_data = {
            "summary": {"total_tests": 10, "passed": 8, "failed": 2, "skipped": 0},
            "failures": [
                {
                    "test_name": "test_foo",
                    "error_message": "AssertionError",
                }
            ],
        }

        result = format_test_results(test_data)

        # Should contain Rich markup
        assert result.contains_markup is True
        assert "[green]" in result.content or "[red]" in result.content

    def test_format_with_feature_flag_disabled(self):
        """Test that formatters produce plain text when feature flag is disabled."""
        manager = get_feature_flag_manager()
        manager.disable(FeatureFlag.USE_RICH_FORMATTING)

        test_data = {
            "summary": {"total_tests": 10, "passed": 8, "failed": 2, "skipped": 0},
            "failures": [
                {
                    "test_name": "test_foo",
                    "error_message": "AssertionError",
                }
            ],
        }

        result = format_test_results(test_data)

        # Should NOT contain Rich markup
        assert result.contains_markup is False
        # Check for actual Rich markup tags, not just [ characters
        assert "[green]" not in result.content
        assert "[red]" not in result.content
        assert "[/" not in result.content  # No closing tags
        assert result.format_type == "plain"

    def test_feature_flag_disabled_for_non_whitelisted_tool(self):
        """Test that tools not in whitelist get plain text even when flag is enabled."""
        from victor.tools.formatters.registry import format_tool_output

        manager = get_feature_flag_manager()
        manager.enable(FeatureFlag.USE_RICH_FORMATTING)

        # "fake_tool" is not in the allowed_tools whitelist
        fake_data = {"results": []}
        result = format_tool_output("fake_tool", fake_data)

        # Should return plain text
        assert result.contains_markup is False
        assert result.format_type == "plain"

    def test_feature_flag_environment_variable(self, monkeypatch):
        """Test that environment variable VICTOR_USE_RICH_FORMATTING controls the flag."""
        # Reset manager to pick up new env var
        from victor.core.feature_flags import reset_feature_flag_manager

        # Test with env var set to false
        monkeypatch.setenv("VICTOR_USE_RICH_FORMATTING", "false")
        reset_feature_flag_manager()

        manager = get_feature_flag_manager()
        assert manager.is_enabled(FeatureFlag.USE_RICH_FORMATTING) is False

        # Test with env var set to true
        monkeypatch.setenv("VICTOR_USE_RICH_FORMATTING", "true")
        reset_feature_flag_manager()

        manager = get_feature_flag_manager()
        assert manager.is_enabled(FeatureFlag.USE_RICH_FORMATTING) is True

    def test_clear_runtime_override_restores_default(self):
        """Test that clearing runtime override restores default behavior."""
        manager = get_feature_flag_manager()

        # Disable the flag
        manager.disable(FeatureFlag.USE_RICH_FORMATTING)
        assert manager.is_enabled(FeatureFlag.USE_RICH_FORMATTING) is False

        # Clear override - should return to default (True)
        manager.clear_runtime_override(FeatureFlag.USE_RICH_FORMATTING)
        assert manager.is_enabled(FeatureFlag.USE_RICH_FORMATTING) is True

    def test_multiple_tools_respect_feature_flag(self):
        """Test that multiple formatters respect the feature flag consistently."""
        manager = get_feature_flag_manager()

        # Disable flag
        manager.disable(FeatureFlag.USE_RICH_FORMATTING)

        # Test multiple tools
        test_data = {
            "summary": {"total_tests": 5, "passed": 5, "failed": 0, "skipped": 0},
            "failures": [],
        }
        test_result = format_test_results(test_data)
        assert test_result.contains_markup is False

        search_data = {
            "results": [
                {
                    "path": "foo.py",
                    "line": 42,
                    "score": 10,
                    "snippet": "def foo():",
                }
            ],
            "count": 1,
        }
        search_result = format_search_results(search_data)
        assert search_result.contains_markup is False

        # Re-enable flag
        manager.enable(FeatureFlag.USE_RICH_FORMATTING)

        # Both should now produce markup
        test_result = format_test_results(test_data)
        assert test_result.contains_markup is True

        search_result = format_search_results(search_data)
        assert search_result.contains_markup is True
