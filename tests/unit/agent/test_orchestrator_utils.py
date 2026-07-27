# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Tests for orchestrator_utils module.

Tests the utility functions for the AgentOrchestrator including:
- Context size calculation
- Git operation inference from aliases
- Tool status message generation
"""

import pytest
from unittest.mock import MagicMock, patch

from victor.agent.orchestrator_utils import (
    calculate_max_context_chars,
    infer_git_operation,
    get_tool_status_message,
    _calculate_max_context_chars,
    _infer_git_operation,
    _get_tool_status_message,
)
from victor.agent.presentation import NullPresentationAdapter

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_settings():
    """Create mock settings object."""
    settings = MagicMock()
    settings.max_context_chars = None
    return settings


@pytest.fixture
def mock_settings_with_override():
    """Create mock settings with max_context_chars override."""
    settings = MagicMock()
    settings.max_context_chars = 500000
    return settings


@pytest.fixture
def mock_provider():
    """Create mock provider object."""
    provider = MagicMock()
    provider.name = "anthropic"
    return provider


@pytest.fixture
def mock_provider_unnamed():
    """Create mock provider without name attribute."""
    provider = MagicMock(spec=[])  # Empty spec means no attributes
    return provider


@pytest.fixture
def mock_provider_limits():
    """Create mock ProviderLimits object."""
    limits = MagicMock()
    limits.context_window = 200000
    return limits


# =============================================================================
# Tests for calculate_max_context_chars
# =============================================================================


class TestCalculateMaxContextChars:
    """Tests for calculate_max_context_chars function."""

    def test_settings_override_takes_precedence(self, mock_settings_with_override, mock_provider):
        """Test that settings.max_context_chars override takes precedence."""
        result = calculate_max_context_chars(
            settings=mock_settings_with_override,
            provider=mock_provider,
            model="claude-3-sonnet",
        )
        assert result == 500000

    def test_settings_zero_not_used(self, mock_provider):
        """Test that zero value in settings is not used as override."""
        settings = MagicMock()
        settings.max_context_chars = 0

        with patch("victor.config.config_loaders.get_provider_limits") as mock_get_limits:
            mock_limits = MagicMock()
            mock_limits.context_window = 128000
            mock_get_limits.return_value = mock_limits

            result = calculate_max_context_chars(
                settings=settings, provider=mock_provider, model="gpt-4"
            )

            # Should use YAML config, not the zero value
            # 128000 tokens * 3.5 chars/token * 0.8 safety = 358400
            assert result == int(128000 * 3.5 * 0.8)

    def test_settings_negative_not_used(self, mock_provider):
        """Test that negative value in settings is not used as override."""
        settings = MagicMock()
        settings.max_context_chars = -100

        with patch("victor.config.config_loaders.get_provider_limits") as mock_get_limits:
            mock_limits = MagicMock()
            mock_limits.context_window = 128000
            mock_get_limits.return_value = mock_limits

            result = calculate_max_context_chars(
                settings=settings, provider=mock_provider, model="gpt-4"
            )

            # Should use YAML config
            assert result == int(128000 * 3.5 * 0.8)

    def test_uses_yaml_config_when_no_settings_override(self, mock_settings, mock_provider):
        """Test that YAML config is used when no settings override."""
        with patch("victor.config.config_loaders.get_provider_limits") as mock_get_limits:
            mock_limits = MagicMock()
            mock_limits.context_window = 200000
            mock_get_limits.return_value = mock_limits

            result = calculate_max_context_chars(
                settings=mock_settings, provider=mock_provider, model="claude-3-opus"
            )

            mock_get_limits.assert_called_once_with("anthropic", "claude-3-opus")
            # 200000 tokens * 3.5 chars/token * 0.8 safety = 560000
            assert result == int(200000 * 3.5 * 0.8)

    def test_provider_name_lowercase(self, mock_settings):
        """Test that provider name is lowercased for lookup."""
        provider = MagicMock()
        provider.name = "ANTHROPIC"

        with patch("victor.config.config_loaders.get_provider_limits") as mock_get_limits:
            mock_limits = MagicMock()
            mock_limits.context_window = 128000
            mock_get_limits.return_value = mock_limits

            calculate_max_context_chars(
                settings=mock_settings, provider=provider, model="test-model"
            )

            mock_get_limits.assert_called_once_with("anthropic", "test-model")

    def test_provider_without_name_attribute(self, mock_settings, mock_provider_unnamed):
        """Test handling provider without name attribute."""
        with patch("victor.config.config_loaders.get_provider_limits") as mock_get_limits:
            mock_limits = MagicMock()
            mock_limits.context_window = 128000
            mock_get_limits.return_value = mock_limits

            result = calculate_max_context_chars(
                settings=mock_settings, provider=mock_provider_unnamed, model="test"
            )

            # Should use empty string for provider name
            mock_get_limits.assert_called_once_with("", "test")
            assert result == int(128000 * 3.5 * 0.8)

    def test_fallback_on_config_load_error(self, mock_settings, mock_provider):
        """Test fallback to default when config loading fails."""
        with patch("victor.config.config_loaders.get_provider_limits") as mock_get_limits:
            mock_get_limits.side_effect = Exception("Config file not found")

            result = calculate_max_context_chars(
                settings=mock_settings, provider=mock_provider, model="test-model"
            )

            # Fallback: 128000 tokens * 3.5 * 0.8 = 358400
            assert result == int(128000 * 3.5 * 0.8)

    def test_non_numeric_context_tokens_fallback(self, mock_settings, mock_provider):
        """Test fallback when context_window returns non-numeric value."""
        with patch("victor.config.config_loaders.get_provider_limits") as mock_get_limits:
            mock_limits = MagicMock()
            mock_limits.context_window = "invalid"  # Non-numeric
            mock_get_limits.return_value = mock_limits

            result = calculate_max_context_chars(
                settings=mock_settings, provider=mock_provider, model="test"
            )

            # Should fallback to 100k token calculation
            assert result == int(100000 * 3.5 * 0.8)

    def test_string_numeric_context_tokens(self, mock_settings, mock_provider):
        """Test handling of string numeric context token values."""
        with patch("victor.config.config_loaders.get_provider_limits") as mock_get_limits:
            mock_limits = MagicMock()
            mock_limits.context_window = "150000"  # String numeric
            mock_get_limits.return_value = mock_limits

            result = calculate_max_context_chars(
                settings=mock_settings, provider=mock_provider, model="test"
            )

            # Should parse string and calculate: 150000 * 3.5 * 0.8 = 420000
            assert result == int(150000 * 3.5 * 0.8)

    def test_float_context_tokens(self, mock_settings, mock_provider):
        """Test handling of float context token values."""
        with patch("victor.config.config_loaders.get_provider_limits") as mock_get_limits:
            mock_limits = MagicMock()
            mock_limits.context_window = 128000.5
            mock_get_limits.return_value = mock_limits

            result = calculate_max_context_chars(
                settings=mock_settings, provider=mock_provider, model="test"
            )

            assert result == int(128000.5 * 3.5 * 0.8)

    def test_calculation_formula(self, mock_settings, mock_provider):
        """Test the exact calculation formula: tokens * 3.5 * 0.8."""
        with patch("victor.config.config_loaders.get_provider_limits") as mock_get_limits:
            mock_limits = MagicMock()
            mock_limits.context_window = 100000
            mock_get_limits.return_value = mock_limits

            result = calculate_max_context_chars(
                settings=mock_settings, provider=mock_provider, model="test"
            )

            # Exact formula: tokens * 3.5 chars/token * 0.8 safety margin
            expected = int(100000 * 3.5 * 0.8)
            assert result == expected
            assert result == 280000

    def test_different_providers(self, mock_settings):
        """Test with different provider names."""
        providers_and_models = [
            ("anthropic", "claude-3-sonnet"),
            ("openai", "gpt-4-turbo"),
            ("google", "gemini-pro"),
            ("ollama", "llama2"),
        ]

        with patch("victor.config.config_loaders.get_provider_limits") as mock_get_limits:
            mock_limits = MagicMock()
            mock_limits.context_window = 128000
            mock_get_limits.return_value = mock_limits

            for provider_name, model_name in providers_and_models:
                provider = MagicMock()
                provider.name = provider_name

                calculate_max_context_chars(
                    settings=mock_settings, provider=provider, model=model_name
                )

                mock_get_limits.assert_called_with(provider_name, model_name)


# =============================================================================
# Tests for infer_git_operation
# =============================================================================


class TestInferGitOperation:
    """Tests for infer_git_operation function."""

    def test_non_git_tool_returns_unchanged(self):
        """Test that non-git tools return args unchanged."""
        args = {"query": "test"}
        result = infer_git_operation("code_search", "code_search", args)
        assert result == args
        assert result is args  # Should be same object

    def test_git_tool_with_operation_already_set(self):
        """Test that existing operation is preserved."""
        args = {"operation": "status", "path": "/some/path"}
        result = infer_git_operation("git_log", "git", args)
        assert result == args
        assert result["operation"] == "status"

    def test_git_status_inference(self):
        """Test inference of status operation from git_status alias."""
        args = {"path": "."}
        result = infer_git_operation("git_status", "git", args)
        assert result["operation"] == "status"
        assert result["path"] == "."

    def test_git_diff_inference(self):
        """Test inference of diff operation from git_diff alias."""
        args = {}
        result = infer_git_operation("git_diff", "git", args)
        assert result["operation"] == "diff"

    def test_git_log_inference(self):
        """Test inference of log operation from git_log alias."""
        args = {"num_commits": 10}
        result = infer_git_operation("git_log", "git", args)
        assert result["operation"] == "log"
        assert result["num_commits"] == 10

    def test_git_commit_inference(self):
        """Test inference of commit operation from git_commit alias."""
        args = {"message": "Initial commit"}
        result = infer_git_operation("git_commit", "git", args)
        assert result["operation"] == "commit"
        assert result["message"] == "Initial commit"

    def test_git_branch_inference(self):
        """Test inference of branch operation from git_branch alias."""
        args = {"branch_name": "feature-branch"}
        result = infer_git_operation("git_branch", "git", args)
        assert result["operation"] == "branch"
        assert result["branch_name"] == "feature-branch"

    def test_git_stage_inference(self):
        """Test inference of stage operation from git_stage alias."""
        args = {"files": ["file1.py", "file2.py"]}
        result = infer_git_operation("git_stage", "git", args)
        assert result["operation"] == "stage"
        assert result["files"] == ["file1.py", "file2.py"]

    def test_unknown_git_alias_no_inference(self):
        """Test that unknown git aliases don't add operation."""
        args = {"path": "."}
        result = infer_git_operation("git_unknown", "git", args)
        assert "operation" not in result

    def test_args_not_mutated(self):
        """Test that original args dict is not mutated."""
        original_args = {"path": "."}
        result = infer_git_operation("git_status", "git", original_args)

        # Result should have operation, but original should not
        assert result["operation"] == "status"
        assert "operation" not in original_args

    def test_canonical_name_must_be_git(self):
        """Test that canonical_name must be 'git' for inference."""
        args = {"path": "."}

        # Even if original_name looks like git alias, canonical must be 'git'
        result = infer_git_operation("git_status", "not_git", args)
        assert result == args
        assert "operation" not in result

    def test_empty_args(self):
        """Test inference with empty args dictionary."""
        result = infer_git_operation("git_status", "git", {})
        assert result == {"operation": "status"}

    def test_preserves_all_original_args(self):
        """Test that all original args are preserved after inference."""
        args = {
            "path": "/home/user/project",
            "files": ["a.py", "b.py"],
            "message": "test",
            "extra": "value",
        }
        result = infer_git_operation("git_commit", "git", args)

        assert result["operation"] == "commit"
        assert result["path"] == "/home/user/project"
        assert result["files"] == ["a.py", "b.py"]
        assert result["message"] == "test"
        assert result["extra"] == "value"


# =============================================================================
# Tests for get_tool_status_message
# =============================================================================


class TestGetToolStatusMessage:
    """Tests for get_tool_status_message function."""

    # Icon prefix used in tool status messages (using NullPresentationAdapter)
    ICON_PREFIX = "*"  # Running icon from NullPresentationAdapter

    @pytest.fixture(autouse=True)
    def setup_presentation(self):
        """Set up the NullPresentationAdapter for all tests."""
        self.presentation = NullPresentationAdapter()

    def test_execute_bash_with_short_command(self):
        """Test status message for execute_bash with short command."""
        result = get_tool_status_message(
            "execute_bash", {"command": "ls -la"}, presentation=self.presentation
        )
        assert result == f"{self.ICON_PREFIX} Running execute_bash: `ls -la`"

    def test_execute_bash_with_long_command_truncates(self):
        """Test that long commands are truncated at 80 chars."""
        long_command = "a" * 100  # 100 characters
        result = get_tool_status_message(
            "execute_bash", {"command": long_command}, presentation=self.presentation
        )

        # Should truncate to 80 chars + "..."
        expected = f"{self.ICON_PREFIX} Running execute_bash: `{'a' * 80}...`"
        assert result == expected
        assert len("a" * 80) == 80

    def test_execute_bash_exactly_80_chars(self):
        """Test command exactly at 80 chars boundary."""
        command = "x" * 80
        result = get_tool_status_message(
            "execute_bash", {"command": command}, presentation=self.presentation
        )
        assert result == f"{self.ICON_PREFIX} Running execute_bash: `{command}`"
        assert "..." not in result

    def test_execute_bash_81_chars_truncates(self):
        """Test command at 81 chars gets truncated."""
        command = "y" * 81
        result = get_tool_status_message(
            "execute_bash", {"command": command}, presentation=self.presentation
        )
        assert result == f"{self.ICON_PREFIX} Running execute_bash: `{'y' * 80}...`"

    def test_non_dict_args_returns_generic_message(self):
        """Non-dict arguments (e.g. a model emitting a JSON array) must not raise.

        Regression: glm-5.2 emitted `write` arguments as a list; the `.get()`
        call raised AttributeError and killed the whole streaming turn.
        """
        for bad_args in ([{"path": "a.py"}], ["a", "b"], "raw string", 7, None, True):
            for tool_name in ("write", "read", "shell", "edit", "ls"):
                result = get_tool_status_message(
                    tool_name, bad_args, presentation=self.presentation
                )
                assert result == f"{self.ICON_PREFIX} Running {tool_name}..."

    def test_execute_bash_no_command(self):
        """Test execute_bash without command key falls back to default."""
        result = get_tool_status_message("execute_bash", {}, presentation=self.presentation)
        assert result == f"{self.ICON_PREFIX} Running execute_bash..."

    def test_shell_with_cmd(self):
        """Test canonical shell tool with cmd argument."""
        result = get_tool_status_message("shell", {"cmd": "ls -la"}, presentation=self.presentation)
        assert result == f"{self.ICON_PREFIX} Running shell: `ls -la`"

    def test_list_directory_with_path(self):
        """Test status message for list_directory with path."""
        result = get_tool_status_message(
            "list_directory", {"path": "/home/user"}, presentation=self.presentation
        )
        assert result == f"{self.ICON_PREFIX} Listing directory: /home/user"

    def test_list_directory_without_path(self):
        """Test list_directory defaults to '.' when no path."""
        result = get_tool_status_message("list_directory", {}, presentation=self.presentation)
        assert result == f"{self.ICON_PREFIX} Listing directory: ."

    def test_ls_with_path(self):
        """Test canonical ls tool with path."""
        result = get_tool_status_message(
            "ls", {"path": "/home/user"}, presentation=self.presentation
        )
        assert result == f"{self.ICON_PREFIX} Listing directory: /home/user"

    def test_read_with_path(self):
        """Test status message for read tool with path."""
        result = get_tool_status_message(
            "read", {"path": "/etc/config.yaml"}, presentation=self.presentation
        )
        assert result == f"{self.ICON_PREFIX} Reading file: /etc/config.yaml"

    def test_read_without_path(self):
        """Test read defaults to 'file' when no path."""
        result = get_tool_status_message("read", {}, presentation=self.presentation)
        assert result == f"{self.ICON_PREFIX} Reading file: file"

    def test_edit_files_single_file(self):
        """Test status message for edit_files with single file."""
        result = get_tool_status_message(
            "edit_files",
            {"files": [{"path": "main.py"}]},
            presentation=self.presentation,
        )
        assert result == f"{self.ICON_PREFIX} Editing: main.py"

    def test_edit_files_multiple_files(self):
        """Test status message for edit_files with multiple files."""
        files = [{"path": "a.py"}, {"path": "b.py"}, {"path": "c.py"}]
        result = get_tool_status_message(
            "edit_files", {"files": files}, presentation=self.presentation
        )
        assert result == f"{self.ICON_PREFIX} Editing: a.py, b.py, c.py"

    def test_edit_files_more_than_three(self):
        """Test that edit_files shows +N more for files beyond 3."""
        files = [
            {"path": "a.py"},
            {"path": "b.py"},
            {"path": "c.py"},
            {"path": "d.py"},
            {"path": "e.py"},
        ]
        result = get_tool_status_message(
            "edit_files", {"files": files}, presentation=self.presentation
        )
        assert result == f"{self.ICON_PREFIX} Editing: a.py, b.py, c.py (+2 more)"

    def test_edit_files_empty_list(self):
        """Test edit_files with empty files list."""
        result = get_tool_status_message(
            "edit_files", {"files": []}, presentation=self.presentation
        )
        assert result == f"{self.ICON_PREFIX} Running edit_files..."

    def test_edit_files_no_files_key(self):
        """Test edit_files without files key."""
        result = get_tool_status_message("edit_files", {}, presentation=self.presentation)
        assert result == f"{self.ICON_PREFIX} Running edit_files..."

    def test_edit_with_ops(self):
        """Test canonical edit tool with ops list."""
        result = get_tool_status_message(
            "edit",
            {
                "ops": [
                    {
                        "type": "replace",
                        "path": "main.py",
                        "old_str": "a",
                        "new_str": "b",
                    }
                ]
            },
            presentation=self.presentation,
        )
        assert result == f"{self.ICON_PREFIX} Editing: main.py"

    def test_edit_files_non_list_files(self):
        """Test edit_files with non-list files value."""
        result = get_tool_status_message(
            "edit_files", {"files": "not a list"}, presentation=self.presentation
        )
        assert result == f"{self.ICON_PREFIX} Running edit_files..."

    def test_edit_files_missing_path_in_file(self):
        """Test edit_files when file entry is missing path."""
        files = [{"content": "data"}, {"path": "b.py"}, {}]
        result = get_tool_status_message(
            "edit_files", {"files": files}, presentation=self.presentation
        )
        assert result == f"{self.ICON_PREFIX} Editing: ?, b.py, ?"

    def test_write_with_path(self):
        """Test status message for write tool with path."""
        result = get_tool_status_message(
            "write", {"path": "/tmp/output.txt"}, presentation=self.presentation
        )
        assert result == f"{self.ICON_PREFIX} Writing file: /tmp/output.txt"

    def test_write_without_path(self):
        """Test write defaults to 'file' when no path."""
        result = get_tool_status_message("write", {}, presentation=self.presentation)
        assert result == f"{self.ICON_PREFIX} Writing file: file"

    def test_code_search_with_short_query(self):
        """Test status message for code_search with short query."""
        result = get_tool_status_message(
            "code_search", {"query": "def main"}, presentation=self.presentation
        )
        assert result == f"{self.ICON_PREFIX} Searching: def main"

    def test_code_search_with_long_query_truncates(self):
        """Test that long queries are truncated at 50 chars."""
        long_query = "q" * 60
        result = get_tool_status_message(
            "code_search", {"query": long_query}, presentation=self.presentation
        )
        assert result == f"{self.ICON_PREFIX} Searching: {'q' * 50}..."

    def test_code_search_exactly_50_chars(self):
        """Test query exactly at 50 chars boundary."""
        query = "z" * 50
        result = get_tool_status_message(
            "code_search", {"query": query}, presentation=self.presentation
        )
        assert result == f"{self.ICON_PREFIX} Searching: {query}"
        assert "..." not in result

    def test_code_search_51_chars_truncates(self):
        """Test query at 51 chars gets truncated."""
        query = "w" * 51
        result = get_tool_status_message(
            "code_search", {"query": query}, presentation=self.presentation
        )
        assert result == f"{self.ICON_PREFIX} Searching: {'w' * 50}..."

    def test_code_search_no_query(self):
        """Test code_search without query key."""
        result = get_tool_status_message("code_search", {}, presentation=self.presentation)
        assert result == f"{self.ICON_PREFIX} Searching: "

    def test_unknown_tool_default_message(self):
        """Test default message for unknown tools."""
        result = get_tool_status_message(
            "unknown_tool", {"any": "args"}, presentation=self.presentation
        )
        assert result == f"{self.ICON_PREFIX} Running unknown_tool..."

    def test_default_message_for_various_tools(self):
        """Test default message format for various unknown tools."""
        tools = ["custom_tool", "my_tool", "analyze_code", "deploy"]
        for tool in tools:
            result = get_tool_status_message(tool, {}, presentation=self.presentation)
            assert result == f"{self.ICON_PREFIX} Running {tool}..."

    def test_empty_tool_name(self):
        """Test handling of empty tool name."""
        result = get_tool_status_message("", {}, presentation=self.presentation)
        assert result == f"{self.ICON_PREFIX} Running ..."

    def test_special_characters_in_command(self):
        """Test handling special characters in bash command."""
        command = 'echo "hello world" | grep -E "pattern"'
        result = get_tool_status_message(
            "execute_bash", {"command": command}, presentation=self.presentation
        )
        assert result == f"{self.ICON_PREFIX} Running execute_bash: `{command}`"

    def test_newlines_in_command(self):
        """Test handling newlines in bash command."""
        command = "echo 'line1\nline2'"
        result = get_tool_status_message(
            "execute_bash", {"command": command}, presentation=self.presentation
        )
        assert result == f"{self.ICON_PREFIX} Running execute_bash: `{command}`"

    def test_message_starts_with_emoji(self):
        """Test that all messages start with the wrench emoji."""
        test_cases = [
            ("execute_bash", {"command": "ls"}),
            ("list_directory", {"path": "/tmp"}),
            ("read", {"path": "file.txt"}),
            ("write", {"path": "out.txt"}),
            ("edit_files", {"files": [{"path": "x.py"}]}),
            ("code_search", {"query": "test"}),
            ("unknown", {}),
        ]
        for tool_name, args in test_cases:
            result = get_tool_status_message(tool_name, args, presentation=self.presentation)
            assert result.startswith(self.ICON_PREFIX), f"Failed for {tool_name}"


# =============================================================================
# Tests for Backward Compatibility Aliases
# =============================================================================


class TestBackwardCompatibilityAliases:
    """Tests for backward compatibility aliases."""

    @pytest.fixture(autouse=True)
    def setup_presentation(self):
        """Set up the NullPresentationAdapter for all tests."""
        self.presentation = NullPresentationAdapter()

    def test_calculate_max_context_chars_alias(self, mock_settings, mock_provider):
        """Test _calculate_max_context_chars alias."""
        with patch("victor.config.config_loaders.get_provider_limits") as mock_get_limits:
            mock_limits = MagicMock()
            mock_limits.context_window = 128000
            mock_get_limits.return_value = mock_limits

            result1 = calculate_max_context_chars(mock_settings, mock_provider, "test-model")
            result2 = _calculate_max_context_chars(mock_settings, mock_provider, "test-model")

            assert result1 == result2

    def test_infer_git_operation_alias(self):
        """Test _infer_git_operation alias."""
        args = {"path": "."}

        result1 = infer_git_operation("git_status", "git", args)
        result2 = _infer_git_operation("git_status", "git", args)

        assert result1 == result2

    def test_get_tool_status_message_alias(self):
        """Test _get_tool_status_message alias."""
        result1 = get_tool_status_message("read", {"path": "/test"}, presentation=self.presentation)
        result2 = _get_tool_status_message(
            "read", {"path": "/test"}, presentation=self.presentation
        )

        assert result1 == result2


# =============================================================================
# Integration Tests with Orchestrator Patterns
# =============================================================================


class TestOrchestratorIntegrationPatterns:
    """Tests simulating orchestrator usage patterns."""

    # Icon prefix used in tool status messages (using NullPresentationAdapter)
    ICON_PREFIX = "*"  # Running icon from NullPresentationAdapter

    @pytest.fixture(autouse=True)
    def setup_presentation(self):
        """Set up the NullPresentationAdapter for all tests."""
        self.presentation = NullPresentationAdapter()

    def test_context_calculation_flow(self, mock_settings, mock_provider):
        """Test typical context calculation flow in orchestrator."""
        with patch("victor.config.config_loaders.get_provider_limits") as mock_get_limits:
            mock_limits = MagicMock()
            mock_limits.context_window = 200000
            mock_get_limits.return_value = mock_limits

            # Simulating orchestrator initialization
            max_chars = calculate_max_context_chars(
                settings=mock_settings,
                provider=mock_provider,
                model="claude-3-opus-20240229",
            )

            # Context should be usable for message truncation
            assert max_chars > 0
            assert isinstance(max_chars, int)

    def test_git_tool_normalization_flow(self):
        """Test git tool call normalization flow."""
        # Simulating tool call from model
        tool_calls = [
            ("git_status", "git", {}),
            ("git_diff", "git", {"staged": True}),
            ("git_log", "git", {"num_commits": 5}),
        ]

        for original_name, canonical_name, original_args in tool_calls:
            args = infer_git_operation(original_name, canonical_name, original_args)

            # All should have operation inferred
            assert "operation" in args
            # Original args should be preserved
            for key in original_args:
                assert args[key] == original_args[key]

    def test_tool_execution_status_flow(self):
        """Test tool execution status message flow."""
        # Simulating multiple tool executions
        tool_executions = [
            ("execute_bash", {"command": "pytest tests/"}),
            ("read", {"path": "/src/main.py"}),
            ("edit_files", {"files": [{"path": "config.py"}]}),
            ("code_search", {"query": "class Agent"}),
        ]

        for tool_name, tool_args in tool_executions:
            message = get_tool_status_message(tool_name, tool_args, presentation=self.presentation)

            # All messages should start with emoji
            assert message.startswith(self.ICON_PREFIX)
            # Should contain relevant info
            assert tool_name in message or any(
                action in message
                for action in ["Listing", "Reading", "Editing", "Writing", "Searching"]
            )

    def test_large_scale_context_handling(self, mock_settings, mock_provider):
        """Test handling of large context values."""
        with patch("victor.config.config_loaders.get_provider_limits") as mock_get_limits:
            # Test with very large context window (like Claude 3's 200k)
            mock_limits = MagicMock()
            mock_limits.context_window = 200000
            mock_get_limits.return_value = mock_limits

            result = calculate_max_context_chars(
                settings=mock_settings, provider=mock_provider, model="claude-3-opus"
            )

            # Should be 200000 * 3.5 * 0.8 = 560000
            assert result == 560000

    def test_mixed_tool_args_types(self):
        """Test handling of various arg types in tool status."""
        # Test with complex nested args
        complex_args = {
            "files": [
                {"path": "a.py", "content": "code", "line_numbers": [1, 2, 3]},
                {"path": "b.py"},
            ],
            "options": {"recursive": True},
        }

        result = get_tool_status_message("edit_files", complex_args, presentation=self.presentation)
        assert "a.py" in result
        assert "b.py" in result
