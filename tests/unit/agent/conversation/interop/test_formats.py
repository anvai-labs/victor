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

"""Detection matrix for transcript interop formats."""

from pathlib import Path

import pytest

from victor.agent.conversation.interop import (
    TranscriptFormat,
    UnsupportedFormatError,
    detect_format,
)

FIXTURES = Path(__file__).parents[5] / "tests" / "fixtures" / "interop"


def fixture(name: str) -> Path:
    path = FIXTURES / name
    assert path.exists(), f"missing fixture {name}"
    return path


class TestDetection:
    def test_claude_jsonl(self):
        result = detect_format(fixture("claude_minimal.jsonl"))
        assert result.format is TranscriptFormat.CLAUDE_CODE

    def test_claude_jsonl_with_leading_bookkeeping(self, tmp_path):
        target = tmp_path / "claude-modern.jsonl"
        target.write_text(
            '{"type":"mode","sessionId":"session-123"}\n'
            '{"type":"permission-mode","sessionId":"session-123"}\n'
            '{"type":"atis-latch","sessionId":"session-123"}\n'
            '{"type":"file-history-snapshot"}\n'
            '{"type":"user","uuid":"message-123","sessionId":"session-123",'
            '"message":{"role":"user","content":"hello"}}\n'
        )

        result = detect_format(target)

        assert result.format is TranscriptFormat.CLAUDE_CODE

    def test_codex_rollout(self):
        result = detect_format(fixture("codex_minimal.jsonl"))
        assert result.format is TranscriptFormat.CODEX_ROLLOUT

    def test_chatgpt_tree(self):
        assert detect_format(fixture("chatgpt_tree.json")).format is TranscriptFormat.CHATGPT_EXPORT

    def test_chatgpt_array(self):
        result = detect_format(fixture("chatgpt_conversations_array.json"))
        assert result.format is TranscriptFormat.CHATGPT_EXPORT

    def test_victor_native(self):
        result = detect_format(fixture("victor_native.json"))
        assert result.format is TranscriptFormat.VICTOR_JSON

    def test_victor_legacy_export_array(self):
        result = detect_format(fixture("victor_export_legacy_array.json"))
        assert result.format is TranscriptFormat.VICTOR_JSON
        assert any("legacy" in r for r in result.reasons)

    def test_anthropic(self):
        result = detect_format(fixture("anthropic_messages.json"))
        assert result.format is TranscriptFormat.ANTHROPIC_JSON

    def test_generic(self):
        result = detect_format(fixture("generic_messages.json"))
        assert result.format is TranscriptFormat.GENERIC_JSON

    def test_malformed_line_still_detects_jsonl(self):
        # one broken line between valid records must not break detection
        result = detect_format(fixture("malformed_line.jsonl"))
        assert result.format is TranscriptFormat.CLAUDE_CODE

    def test_empty_file_rejected(self):
        with pytest.raises(UnsupportedFormatError, match="empty"):
            detect_format(fixture("empty.json"))

    def test_missing_file_rejected(self):
        with pytest.raises(UnsupportedFormatError, match="not found"):
            detect_format(FIXTURES / "nope.json")

    def test_size_guard(self, tmp_path):
        import os

        from victor.agent.conversation.interop.formats import MAX_FILE_BYTES

        big = tmp_path / "big.jsonl"
        big.write_text("{}")
        try:
            os.truncate(big, MAX_FILE_BYTES + 1)
            with pytest.raises(UnsupportedFormatError, match="guard"):
                detect_format(big)
        finally:
            big.unlink(missing_ok=True)

    def test_unrecognizable_document(self, tmp_path):
        target = tmp_path / "noise.json"
        target.write_text('{"unexpected": true}')
        with pytest.raises(UnsupportedFormatError, match="detect"):
            detect_format(target)
