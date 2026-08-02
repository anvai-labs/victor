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

"""Tests for ToolResultDeduplicator."""

import pytest

from victor.agent.tool_result_deduplicator import ToolResultDeduplicator
from victor.config.orchestrator_constants import DeduplicationConfig
from victor.providers.base import Message


class TestDeduplicationConfig:
    def test_defaults(self):
        config = DeduplicationConfig()
        assert config.enabled is True
        assert config.min_content_chars_to_dedup == 500
        assert "read" in config.dedup_tool_names

    def test_custom_values(self):
        config = DeduplicationConfig(enabled=False, min_content_chars_to_dedup=100)
        assert config.enabled is False
        assert config.min_content_chars_to_dedup == 100


class TestShouldDeduplicate:
    def test_read_tool_yes(self):
        dedup = ToolResultDeduplicator()
        assert dedup.should_deduplicate("read", {"path": "/a.py"}) is True

    def test_write_tool_no(self):
        dedup = ToolResultDeduplicator()
        assert dedup.should_deduplicate("write", {"path": "/a.py"}) is False

    def test_unknown_tool_no(self):
        dedup = ToolResultDeduplicator()
        assert dedup.should_deduplicate("shell", {"cmd": "ls"}) is False

    def test_disabled_config(self):
        dedup = ToolResultDeduplicator(config=DeduplicationConfig(enabled=False))
        assert dedup.should_deduplicate("read", {"path": "/a.py"}) is False


class TestDeduplicateInPlace:
    def _make_read_msg(self, path, content_size=600):
        content = f'<TOOL_OUTPUT tool="read" path="{path}">' + "x" * content_size + "</TOOL_OUTPUT>"
        return Message(role="user", content=content)

    def test_single_dup(self):
        dedup = ToolResultDeduplicator()
        msgs = [
            self._make_read_msg("/src/foo.py"),
            Message(role="assistant", content="I see the file"),
            self._make_read_msg("/src/foo.py"),  # This is the new read (last msg)
        ]
        count = dedup.deduplicate_in_place(msgs, "read", {"path": "/src/foo.py"})
        assert count == 1
        assert "Previously read" in msgs[0].content

    def test_multiple_dups(self):
        dedup = ToolResultDeduplicator()
        msgs = [
            self._make_read_msg("/src/foo.py"),
            Message(role="assistant", content="ok"),
            self._make_read_msg("/src/foo.py"),
            Message(role="assistant", content="ok again"),
            self._make_read_msg("/src/foo.py"),  # newest
        ]
        count = dedup.deduplicate_in_place(msgs, "read", {"path": "/src/foo.py"})
        assert count == 2

    def test_no_match(self):
        dedup = ToolResultDeduplicator()
        msgs = [
            self._make_read_msg("/src/bar.py"),
            self._make_read_msg("/src/foo.py"),
        ]
        count = dedup.deduplicate_in_place(msgs, "read", {"path": "/src/foo.py"})
        # bar.py should not be touched
        assert count == 0
        assert "Previously read" not in msgs[0].content

    def test_small_content_skipped(self):
        dedup = ToolResultDeduplicator()
        small_msg = Message(
            role="user",
            content='<TOOL_OUTPUT tool="read" path="/src/foo.py">small</TOOL_OUTPUT>',
        )
        msgs = [small_msg, self._make_read_msg("/src/foo.py")]
        count = dedup.deduplicate_in_place(msgs, "read", {"path": "/src/foo.py"})
        assert count == 0  # Too small to dedup

    def test_no_path_no_dedup(self):
        dedup = ToolResultDeduplicator()
        msgs = [self._make_read_msg("/src/foo.py")]
        count = dedup.deduplicate_in_place(msgs, "read", {})
        assert count == 0


class TestStubCreation:
    def test_stub_format(self):
        dedup = ToolResultDeduplicator()
        msgs = [
            Message(
                role="user",
                content='<TOOL_OUTPUT tool="read" path="/src/foo.py">'
                + "line1\nline2\nline3\n" * 50
                + "</TOOL_OUTPUT>",
            ),
            Message(
                role="user",
                content='<TOOL_OUTPUT tool="read" path="/src/foo.py">new read</TOOL_OUTPUT>',
            ),
        ]
        dedup.deduplicate_in_place(msgs, "read", {"path": "/src/foo.py"})
        assert msgs[0].content.startswith("[Previously read: /src/foo.py")
        assert "lines" in msgs[0].content
