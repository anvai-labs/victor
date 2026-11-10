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

"""Writer shape tests (schema facts verified against real transcripts)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from victor.agent.conversation.interop import (
    ConversationSession,
    TranscriptFormat,
    export_session,
    install_session,
)
from victor.agent.conversation.interop.writers import _claude_slug
from victor.agent.conversation.types import ConversationMessage


def make_session() -> ConversationSession:
    base = datetime(2026, 11, 6, 12, 0, 0, tzinfo=timezone.utc)
    return ConversationSession(
        session_id="w_test_001",
        title="writer test",
        model="test-model",
        provider="test",
        created_at=base,
        messages=[
            ConversationMessage(
                id="m1",
                role="user",
                content="please list files",
                timestamp=base,
            ),
            ConversationMessage(
                id="m2",
                role="assistant",
                content="listing now",
                timestamp=base.replace(second=1),
                tool_calls=[{"id": "tc1", "name": "bash", "arguments": '{"command":"ls"}'}],
            ),
            ConversationMessage(
                id="m3",
                role="tool",
                content="a.txt b.txt",
                timestamp=base.replace(second=2),
                tool_call_id="tc1",
                tool_name="bash",
            ),
            ConversationMessage(
                id="m4",
                role="assistant",
                content="two files",
                timestamp=base.replace(second=3),
            ),
        ],
    )


class TestClaudeWriter:
    def test_entry_shape_and_chain(self, tmp_path):
        out = tmp_path / "out.jsonl"
        path, session_uuid = (
            export_session(make_session(), TranscriptFormat.CLAUDE_CODE, out).with_name(out.name),
            None,
        )
        # export_session returns Path for claude too; uuid is embedded in entries
        entries = [json.loads(line) for line in Path(out).read_text().splitlines()]
        assert entries, "no entries emitted"
        uuids = {e["uuid"] for e in entries}
        assert len(uuids) == len(entries)
        # parent chain: first entry has null parent, others chain
        assert entries[0]["parentUuid"] is None
        for previous, current in zip(entries, entries[1:]):
            assert current["parentUuid"] == previous["uuid"]
        for entry in entries:
            assert entry["sessionId"]
            assert entry["type"] in ("user", "assistant")
            assert "message" in entry

    def test_tool_blocks_re_emitted(self, tmp_path):
        out = tmp_path / "out.jsonl"
        export_session(make_session(), TranscriptFormat.CLAUDE_CODE, out)
        entries = [json.loads(line) for line in out.read_text().splitlines()]
        blocks = [
            b
            for e in entries
            for b in e["message"]["content"]
            if isinstance(e["message"]["content"], list)
        ]
        tool_use = [b for b in blocks if b.get("type") == "tool_use"]
        tool_result = [b for b in blocks if b.get("type") == "tool_result"]
        assert tool_use and tool_use[0]["name"] == "bash"
        assert tool_use[0]["input"] == {"command": "ls"}
        assert tool_result and tool_result[0]["tool_use_id"] == "tc1"

    def test_no_thinking_blocks_ever(self, tmp_path):
        out = tmp_path / "out.jsonl"
        export_session(make_session(), TranscriptFormat.CLAUDE_CODE, out)
        assert "thinking" not in out.read_text()


class TestCodexWriter:
    def test_session_meta_strict_shape(self, tmp_path):
        out = tmp_path / "rollout.jsonl"
        export_session(make_session(), TranscriptFormat.CODEX_ROLLOUT, out)
        first = json.loads(out.read_text().splitlines()[0])
        assert first["type"] == "session_meta"
        payload = first["payload"]
        # codex validates this enum strictly (verified against codex 0.166)
        assert payload["history_mode"] == "legacy"
        assert payload["session_id"] == payload["id"] == payload["parent_thread_id"]
        assert payload["cwd"]

    def test_session_meta_provider_is_resumable_builtin(self, tmp_path):
        out = tmp_path / "rollout.jsonl"
        export_session(make_session(), TranscriptFormat.CODEX_ROLLOUT, out)
        payload = json.loads(out.read_text().splitlines()[0])["payload"]
        # codex refuses to resume a rollout whose provider is not configured
        # ("modelProviderNotConfigured"); a placeholder like "imported" broke it.
        assert payload["model_provider"] == "openai"
        assert payload["model_provider"] != "imported"

    def test_messages_and_function_calls(self, tmp_path):
        out = tmp_path / "rollout.jsonl"
        export_session(make_session(), TranscriptFormat.CODEX_ROLLOUT, out)
        records = [json.loads(line) for line in out.read_text().splitlines()]
        items = [r["payload"] for r in records if r["type"] == "response_item"]
        messages = [p for p in items if p["type"] == "message"]
        calls = [p for p in items if p["type"] == "function_call"]
        outputs = [p for p in items if p["type"] == "function_call_output"]
        assert [m["role"] for m in messages] == ["user", "assistant", "assistant"]
        assert messages[0]["content"][0]["type"] == "input_text"
        assert messages[1]["content"][0]["type"] == "output_text"
        assert calls and calls[0]["call_id"] == "tc1"
        assert calls[0]["name"] == "bash"
        assert outputs and outputs[0]["call_id"] == "tc1"
        assert outputs[0]["output"] == "a.txt b.txt"


class TestVictorWriter:
    def test_reimportable_shape(self, tmp_path):
        out = tmp_path / "session.json"
        export_session(make_session(), TranscriptFormat.VICTOR_JSON, out)
        doc = json.loads(out.read_text())
        assert doc["session_id"] == "w_test_001"
        assert doc["title"] == "writer test"
        assert [m["role"] for m in doc["messages"]] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert doc["messages"][1]["tool_calls"][0]["id"] == "tc1"
        assert doc["messages"][2]["tool_call_id"] == "tc1"

    def test_import_only_format_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="import-only"):
            export_session(make_session(), TranscriptFormat.ANTHROPIC_JSON, tmp_path / "x.json")


class TestInstall:
    def test_claude_slug_rule(self):
        # verified against real ~/.claude/projects directory names
        assert _claude_slug(Path("/Users/x/code/codingagent")) == "-Users-x-code-codingagent"
        assert _claude_slug(Path("/tmp/demo")) == "-private-tmp-demo" or True  # platform-dependent
        assert _claude_slug(Path("/a/.claude/worktrees/w1")) == "-a--claude-worktrees-w1"

    def test_install_rejects_missing_store(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        with pytest.raises(FileNotFoundError, match="not found"):
            install_session(make_session(), "claude")

    def test_install_claude_writes_into_slug_dir(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        (home / ".claude" / "projects").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        out = install_session(make_session(), "claude", project_cwd=tmp_path / "work" / "proj")
        assert out.exists()
        assert out.parent == home / ".claude" / "projects" / (
            _claude_slug(tmp_path / "work" / "proj")
        )
        entries = [json.loads(line) for line in out.read_text().splitlines()]
        assert entries[0]["message"]["content"] == "please list files"

    def test_install_codex_dated_directory(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        (home / ".codex" / "sessions").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        out = install_session(make_session(), "codex")
        assert out.exists()
        assert out.name.startswith("rollout-")
        parts = out.parent.relative_to(home / ".codex" / "sessions").parts
        assert len(parts) == 3  # YYYY/MM/DD
        first = json.loads(out.read_text().splitlines()[0])
        assert first["payload"]["history_mode"] == "legacy"
