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

"""Address-based import: find/list sessions in the claude/codex stores."""

import json

import pytest

from victor.agent.conversation.interop.stores import (
    SessionAddressError,
    find_transcript,
    list_store_sessions,
    resolve_import_target,
)

CLAUDE_ID = "aaaaaaaa-bbbb-4ccc-8ddd-000000000001"
CODEX_ID = "eeeeeeee-ffff-4aaa-8bbb-000000000002"


@pytest.fixture
def fake_stores(tmp_path, monkeypatch):
    """Fake ~/.claude/projects and ~/.codex/sessions under a temp HOME."""
    home = tmp_path / "home"
    work = tmp_path / "work" / "proj"
    work.mkdir(parents=True)

    claude_dir = home / ".claude" / "projects" / (str(work).replace("/", "-").replace(".", "-"))
    claude_dir.mkdir(parents=True)
    entry = {
        "type": "user",
        "message": {"role": "user", "content": "fix the login bug please"},
        "uuid": "u1",
        "sessionId": CLAUDE_ID,
    }
    (claude_dir / f"{CLAUDE_ID}.jsonl").write_text(json.dumps(entry) + "\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    return {"home": home, "work": work, "claude_dir": claude_dir}


@pytest.fixture
def fake_codex_store(tmp_path, monkeypatch):
    home = tmp_path / "home"
    rollout_dir = home / ".codex" / "sessions" / "2026" / "11" / "07"
    rollout_dir.mkdir(parents=True)
    lines = [
        json.dumps(
            {
                "timestamp": "2026-11-07T12:00:00Z",
                "ordinal": 0,
                "type": "session_meta",
                "payload": {
                    "session_id": CODEX_ID,
                    "id": CODEX_ID,
                    "cwd": "/tmp",
                    "history_mode": "legacy",
                },
            }
        ),
        json.dumps(
            {
                "timestamp": "2026-11-07T12:00:01Z",
                "ordinal": 1,
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "m1",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "diagnose the flaky test"}],
                },
            }
        ),
    ]
    name = f"rollout-2026-11-07T12-00-00-{CODEX_ID}.jsonl"
    (rollout_dir / name).write_text("\n".join(lines) + "\n")
    monkeypatch.setenv("HOME", str(home))
    return {"home": home, "rollout": rollout_dir / name}


class TestFindTranscript:
    def test_claude_by_session_id(self, fake_stores):
        path = find_transcript("claude", CLAUDE_ID)
        assert path.is_file()
        assert path.name == f"{CLAUDE_ID}.jsonl"

    def test_codex_by_filename_id(self, fake_codex_store):
        path = find_transcript("codex", CODEX_ID)
        assert path.is_file()
        assert CODEX_ID in path.name

    def test_missing_session_rejected(self, fake_stores):
        with pytest.raises(SessionAddressError, match="No claude session"):
            find_transcript("claude", "12345678-dead-4beef-8bad-0000000000ff")

    def test_path_characters_in_id_rejected(self, fake_stores):
        with pytest.raises(SessionAddressError, match="Invalid session id"):
            find_transcript("claude", "../../etc/passwd")
        with pytest.raises(SessionAddressError, match="Invalid session id"):
            find_transcript("codex", "a/b/c")

    def test_unknown_source_rejected(self, fake_stores):
        with pytest.raises(SessionAddressError, match="Unknown source"):
            find_transcript("gemini", CLAUDE_ID)


class TestResolveImportTarget:
    def test_address_resolves_to_store_file(self, fake_stores):
        path, source = resolve_import_target(f"claude:{CLAUDE_ID}")
        assert source == "claude"
        assert path.name == f"{CLAUDE_ID}.jsonl"

    def test_plain_path_passthrough(self, fake_stores, tmp_path):
        target = tmp_path / "some.jsonl"
        target.write_text("{}")
        path, source = resolve_import_target(str(target))
        assert path == target
        assert source == ""

    def test_malformed_address_rejected(self, fake_stores):
        with pytest.raises(SessionAddressError, match="Invalid session address"):
            resolve_import_target("claude:../escape")

    def test_codex_address(self, fake_codex_store):
        path, source = resolve_import_target(f"codex:{CODEX_ID}")
        assert source == "codex"
        assert path.is_file()


class TestListStoreSessions:
    def test_claude_listing_with_preview(self, fake_stores):
        entries = list_store_sessions("claude", limit=5)
        assert len(entries) == 1
        assert entries[0].session_id == CLAUDE_ID
        assert entries[0].title == "fix the login bug please"
        assert entries[0].source == "claude"

    def test_codex_listing_with_preview(self, fake_codex_store):
        entries = list_store_sessions("codex", limit=5)
        assert len(entries) == 1
        assert entries[0].session_id == CODEX_ID
        assert entries[0].title == "diagnose the flaky test"

    def test_listing_missing_store_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))  # no stores at all
        with pytest.raises(SessionAddressError, match="not found"):
            list_store_sessions("claude")

    def test_listing_newest_first_and_bounded(self, fake_stores):
        import time

        # add older + newer sessions; listing must lead with the newest
        older = fake_stores["claude_dir"] / "00000000-old0-4old-8old-0000000000old.jsonl"
        older.write_text("{}")
        older_mtime = time.time() - 5000
        import os

        os.utime(older, (older_mtime, older_mtime))
        entries = list_store_sessions("claude", limit=1)
        assert len(entries) == 1
        assert entries[0].session_id == CLAUDE_ID


class TestAddressImportEndToEnd:
    def test_import_by_address_via_reader(self, fake_stores):
        """/import claude:<id> path: address → file → IR."""
        from victor.agent.conversation.interop import import_transcript

        path, _source = resolve_import_target(f"claude:{CLAUDE_ID}")
        imported = import_transcript(path)
        assert imported.report.source_format == "claude"
        assert imported.session.messages[0].content == "fix the login bug please"
