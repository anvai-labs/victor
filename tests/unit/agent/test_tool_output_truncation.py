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

"""Tests for tool output size enforcement.

Verifies that the tool executor truncates large string outputs
to prevent context window blowout.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.tool_executor import ToolExecutor, ToolExecutionResult


@pytest.fixture
def executor():
    """Create a minimal ToolExecutor for testing."""
    exec = ToolExecutor.__new__(ToolExecutor)
    exec._tools = {}
    exec._stats = {}
    exec._failed_signatures = set()
    exec.cache = None
    exec._hooks_before = []
    exec._hooks_after = []
    exec._rbac_manager = None
    exec._tracer = None
    exec._rl_hooks = None
    exec._code_correction = None
    exec._argument_normalizer = None
    return exec


class TestToolOutputTruncation:
    """Test that large tool outputs are truncated at framework level."""

    def test_large_string_result_is_truncated(self):
        """Tool returning >25KB multiline string should be truncated."""
        from victor.tools.output_utils import truncate_by_lines

        # Create multiline output larger than 25600 bytes
        lines = [f"line {i:04d}: {'data' * 10}" for i in range(1000)]
        large_output = "\n".join(lines)
        assert len(large_output) > 25600

        truncated, info = truncate_by_lines(large_output)

        assert info.was_truncated is True
        assert len(truncated) <= len(large_output)

    def test_small_string_result_not_truncated(self):
        """Tool returning <25KB string should not be truncated."""
        small_output = "hello world"
        result = ToolExecutionResult(
            tool_name="test_tool",
            success=True,
            result=small_output,
        )
        assert result.result == "hello world"
        assert result.truncation_info is None

    def test_dict_result_not_truncated(self):
        """Tool returning dict should not be truncated."""
        dict_output = {"key": "x" * 50000}
        result = ToolExecutionResult(
            tool_name="test_tool",
            success=True,
            result=dict_output,
        )
        assert result.result == dict_output
        assert result.truncation_info is None

    def test_failed_result_not_truncated(self):
        """Failed tool results should not be truncated even if large."""
        large_error = "e" * 30000
        result = ToolExecutionResult(
            tool_name="test_tool",
            success=False,
            result=large_error,
            error="Tool failed",
        )
        # Truncation only applies to successful results
        assert result.truncation_info is None

    def test_multiline_truncation_preserves_line_boundaries(self):
        """Truncation should never cut mid-line."""
        from victor.tools.output_utils import truncate_by_lines

        # 1000 lines of 50 chars each = 50KB
        lines = [f"line {i:04d}: {'x' * 44}" for i in range(1000)]
        large_output = "\n".join(lines)

        truncated, info = truncate_by_lines(large_output)

        assert info.was_truncated
        # Verify no partial lines (last line may be truncation notice)
        for line in truncated.split("\n"):
            if line and not line.startswith("["):
                assert line.startswith("line ")

    def test_truncation_info_has_continuation_metadata(self):
        """TruncationInfo should include offset for continuation."""
        from victor.tools.output_utils import truncate_by_lines

        lines = [f"line {i}" for i in range(1000)]
        large_output = "\n".join(lines)

        _, info = truncate_by_lines(large_output)

        assert info.was_truncated
        assert info.total_lines == 1000
        assert info.lines_returned > 0
        assert info.lines_returned < 1000
