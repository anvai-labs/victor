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

"""Reader IR + loss-accounting tests."""

import json
from pathlib import Path

import pytest

from victor.agent.conversation.interop import (
    TranscriptFormat,
    UnsupportedFormatError,
    import_transcript,
)

FIXTURES = Path(__file__).parents[5] / "tests" / "fixtures" / "interop"


def fixture(name: str) -> Path:
    path = FIXTURES / name
    assert path.exists(), f"missing fixture {name}"
    return path


def roles(imported) -> list:
    return [m.role if isinstance(m.role, str) else m.role.value for m in imported.session.messages]


class TestClaudeReader:
    def test_minimal_ir(self):
        imported = import_transcript(fixture("claude_minimal.jsonl"))
        assert roles(imported) == ["user", "assistant", "user", "assistant"]
        assert imported.session.messages[0].content == "What is 2+2?"
        assert imported.session.provider == "claude-code"
        assert imported.session.session_id.startswith("imp_")

    def test_timestamps_preserved_and_ordered(self):
        imported = import_transcript(fixture("claude_minimal.jsonl"))
        stamps = [m.timestamp for m in imported.session.messages]
        assert stamps == sorted(stamps)
        assert all(b > a for a, b in zip(stamps, stamps[1:]))  # strictly increasing

    def test_tool_fidelity_and_loss_accounting(self):
        imported = import_transcript(fixture("claude_tools_thinking.jsonl"))
        report = imported.report
        # tool_use → role=tool_call with callable tool_calls
        call = next(m for m in imported.session.messages if m.role == "tool_call")
        assert call.tool_name == "bash"
        assert call.tool_calls[0]["name"] == "bash"
        assert json.loads(call.tool_calls[0]["arguments"]) == {"command": "pytest -q"}
        # tool_result → role=tool with matching call id
        result = next(m for m in imported.session.messages if m.role == "tool")
        assert result.tool_call_id == "toolu_01"
        assert result.content == "3 passed"
        # exact loss ledger
        assert report.tool_calls_preserved == 1
        assert report.thinking_blocks_dropped == 2  # signatures don't transfer
        assert report.encrypted_items_skipped == 0
        skipped_reasons = [s["reason"] for s in report.skipped]
        assert any("sidechain" in r for r in skipped_reasons)
        assert any("meta" in r for r in skipped_reasons)
        assert any("preamble" in r for r in skipped_reasons)

    def test_orphan_tool_result_becomes_trace(self):
        imported = import_transcript(fixture("claude_tools_thinking.jsonl"))
        orphans = [m for m in imported.session.messages if "tool_result" in m.content]
        assert any("late result" in m.content for m in orphans)
        assert any("orphan" in w for w in imported.report.warnings)

    def test_model_extracted_from_assistant(self):
        imported = import_transcript(fixture("claude_tools_thinking.jsonl"))
        assert imported.session.model == "glm-5.3"

    def test_malformed_line_skipped_not_fatal(self):
        imported = import_transcript(fixture("malformed_line.jsonl"))
        assert len(imported.session.messages) == 2
        assert len(imported.report.skipped) == 1

    def test_deterministic_session_id(self):
        one = import_transcript(fixture("claude_minimal.jsonl"))
        two = import_transcript(fixture("claude_minimal.jsonl"))
        assert one.session.session_id == two.session.session_id


class TestCodexReader:
    def test_minimal_ir(self):
        imported = import_transcript(fixture("codex_minimal.jsonl"))
        assert roles(imported) == ["user", "assistant", "user"]
        assert imported.session.provider == "codex"
        assert imported.session.model == "gpt-test"  # from turn_context

    def test_function_calls_preserved(self):
        imported = import_transcript(fixture("codex_function_calls.jsonl"))
        call = next(m for m in imported.session.messages if m.role == "tool_call")
        assert call.tool_name == "shell"
        assert json.loads(call.tool_calls[0]["arguments"]) == {"command": "ls"}
        result = next(m for m in imported.session.messages if m.role == "tool")
        assert result.tool_call_id == "fc_1"
        assert result.content == "file_a.py file_b.py"

    def test_developer_and_env_preamble_excluded(self):
        imported = import_transcript(fixture("codex_function_calls.jsonl"))
        texts = [m.content for m in imported.session.messages]
        assert not any("helpful agent" in t for t in texts)  # developer role skipped
        assert not any("environment_context" in t for t in texts)
        assert any("List the files" in t for t in texts)  # real user turn kept

    def test_encrypted_items_accounted(self):
        imported = import_transcript(fixture("codex_function_calls.jsonl"))
        assert imported.report.encrypted_items_skipped == 2  # reasoning + compacted


class TestChatGPTReader:
    def test_tree_walk(self):
        imported = import_transcript(fixture("chatgpt_tree.json"))
        assert roles(imported) == ["user", "assistant", "assistant"]
        assert imported.session.model == "gpt-5"  # default_model_slug honored
        assert imported.session.title == "Tree chat"

    def test_null_current_node_uses_deepest_branch(self):
        imported = import_transcript(fixture("chatgpt_null_current_node.json"))
        contents = [m.content for m in imported.session.messages]
        assert contents == ["Hello GPT", "Hi!", "How can I help?"]
        assert "abandoned branch" not in contents
        assert any("current_node" in w for w in imported.report.warnings)

    def test_conversations_array_imports_first(self):
        imported = import_transcript(fixture("chatgpt_conversations_array.json"))
        assert len(imported.session.messages) == 3
        assert any("first" in w.lower() for w in imported.report.warnings)


class TestDocumentReaders:
    def test_victor_native_round_shapes(self):
        imported = import_transcript(fixture("victor_native.json"))
        assert imported.session.session_id == "vic_fix_001"  # original id kept
        assert roles(imported) == ["user", "assistant", "tool"]
        assert imported.report.tool_calls_preserved == 1

    def test_victor_legacy_export_array(self):
        imported = import_transcript(fixture("victor_export_legacy_array.json"))
        assert imported.session.session_id == "vic_legacy_9"
        assert roles(imported) == ["user", "assistant"]

    def test_anthropic_blocks_flattened(self):
        imported = import_transcript(fixture("anthropic_messages.json"))
        assert [m.content for m in imported.session.messages] == ["hey", "hey back"]

    def test_generic(self):
        imported = import_transcript(fixture("generic_messages.json"))
        assert roles(imported) == ["user", "assistant"]


class TestFlagEnforcement:
    def test_explicit_format_mismatch_rejected(self):
        with pytest.raises(UnsupportedFormatError, match="does not match"):
            import_transcript(fixture("claude_minimal.jsonl"), TranscriptFormat.CODEX_ROLLOUT)

    def test_explicit_format_accepted_when_matching(self):
        imported = import_transcript(fixture("claude_minimal.jsonl"), TranscriptFormat.CLAUDE_CODE)
        assert imported.report.source_format == "claude"
