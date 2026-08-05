# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Tests for per-invocation effective access resolution.

Covers the regression where every ``shell readonly=True`` / ``code grep``
invocation was classified WRITE from the tool's static envelope — firing a
pre-tool checkpoint before pure reads and rendering a misleading MIXED badge.
"""

from victor.agent.services.tool_execution_runtime import ToolExecutionRuntime
from victor.agent.tool_execution.categorization import ToolCategory, categorize_tool_call
from victor.tools.base import AccessMode
from victor.tools.effective_access import (
    register_effective_access_resolver,
    resolve_effective_access,
)
from victor.ui.rendering.utils import get_tool_metadata_for_display


class TestResolvers:
    def test_shell_readonly_true_narrows(self):
        assert (
            resolve_effective_access("shell", {"readonly": True, "cmd": "grep foo"})
            is AccessMode.READONLY
        )

    def test_shell_readonly_false_does_not_narrow(self):
        assert resolve_effective_access("shell", {"readonly": False, "cmd": "grep foo"}) is None

    def test_shell_action_read_alone_does_not_narrow(self):
        # action="read" is declared intent, not enforcement — must not narrow.
        assert resolve_effective_access("shell", {"action": "read", "cmd": "grep foo"}) is None

    def test_code_readonly_subcommands_narrow(self):
        for cmd in ('code grep "x" src', 'search "auth" --mode literal', "metrics src/"):
            assert resolve_effective_access("code", {"cmd": cmd}) is AccessMode.READONLY, cmd

    def test_code_effectful_subcommands_do_not_narrow(self):
        for cmd in ('code python "print(1)"', "code test pytest", 'execute "x=1"'):
            assert resolve_effective_access("code", {"cmd": cmd}) is None, cmd

    def test_code_suspicious_operators_do_not_narrow(self):
        assert resolve_effective_access("code", {"cmd": "grep foo | sh"}) is None
        assert resolve_effective_access("code", {"cmd": "grep $(rm -rf /) ."}) is None

    def test_unknown_tool_and_empty_arguments(self):
        assert resolve_effective_access("write", {"path": "a"}) is None
        assert resolve_effective_access("shell", None) is None
        assert resolve_effective_access("code", {}) is None

    def test_resolver_exception_falls_back_to_static(self):
        def broken(_args):
            raise RuntimeError("boom")

        register_effective_access_resolver("broken_tool", broken)
        assert resolve_effective_access("broken_tool", {"x": 1}) is None

    def test_resolver_cannot_widen(self):
        register_effective_access_resolver("sneaky", lambda _a: AccessMode.EXECUTE)
        assert resolve_effective_access("sneaky", {}) is None


class TestCategorization:
    def test_readonly_shell_categorized_read_only(self):
        assert (
            categorize_tool_call("shell", {"readonly": True, "cmd": "grep foo"})
            is ToolCategory.READ_ONLY
        )

    def test_mutating_shell_stays_write(self):
        assert (
            categorize_tool_call("shell", {"readonly": False, "cmd": "rm -rf x"})
            is ToolCategory.WRITE
        )

    def test_code_grep_categorized_read_only(self):
        assert categorize_tool_call("code", {"cmd": 'grep "foo" .'}) is ToolCategory.READ_ONLY

    def test_bash_alias_normalized_and_narrowed(self):
        assert (
            categorize_tool_call("bash", {"readonly": True, "cmd": "ls"}) is ToolCategory.READ_ONLY
        )


class TestCheckpointTriggerSelection:
    """_first_write_tool_call drives 'Before tool X modifies files' checkpoints."""

    def test_readonly_batch_triggers_no_checkpoint(self):
        batch = [
            {"name": "shell", "arguments": {"readonly": True, "cmd": "grep foo"}},
            {"name": "code", "arguments": {"cmd": 'search "x" --mode literal'}},
            {"name": "read", "arguments": {"path": "a.py"}},
        ]
        assert ToolExecutionRuntime._first_write_tool_call(batch) is None

    def test_write_in_batch_still_triggers(self):
        batch = [
            {"name": "shell", "arguments": {"readonly": True, "cmd": "grep foo"}},
            {"name": "write", "arguments": {"path": "a.py", "content": "x"}},
        ]
        triggering = ToolExecutionRuntime._first_write_tool_call(batch)
        assert triggering is not None and triggering["name"] == "write"

    def test_mutating_shell_still_triggers(self):
        batch = [{"name": "shell", "arguments": {"readonly": False, "cmd": "touch x"}}]
        triggering = ToolExecutionRuntime._first_write_tool_call(batch)
        assert triggering is not None and triggering["name"] == "shell"


class TestDisplayOverride:
    def test_arguments_narrow_badge_to_readonly(self):
        display = get_tool_metadata_for_display("code", {"cmd": 'grep "x" .'})
        assert display["access_mode"] == "readonly"
        assert display["execution_category"] == "read_only"

    def test_no_arguments_keeps_static_metadata(self):
        with_args = get_tool_metadata_for_display("code", {"cmd": 'python "1/0"'})
        without_args = get_tool_metadata_for_display("code")
        assert with_args == without_args
