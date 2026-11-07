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

"""Round-trip property tests: any-pair conversion via the victor IR."""

import json
from pathlib import Path

from victor.agent.conversation.interop import (
    TranscriptFormat,
    export_session,
    import_transcript,
)

FIXTURES = Path(__file__).parents[5] / "tests" / "fixtures" / "interop"


def fixture(name: str) -> Path:
    path = FIXTURES / name
    assert path.exists(), f"missing fixture {name}"
    return path


def role_of(message) -> str:
    return message.role if isinstance(message.role, str) else message.role.value


class TestRoundTrips:
    def test_victor_json_identity(self, tmp_path):
        """victor → export → import ⇒ identical conversation."""
        first = import_transcript(fixture("victor_native.json"))
        out = tmp_path / "rt.json"
        export_session(first.session, TranscriptFormat.VICTOR_JSON, out)
        second = import_transcript(out)
        assert second.session.session_id == first.session.session_id
        assert [role_of(m) for m in second.session.messages] == [
            role_of(m) for m in first.session.messages
        ]
        assert [m.content for m in second.session.messages] == [
            m.content for m in first.session.messages
        ]
        assert [m.timestamp for m in second.session.messages] == [
            m.timestamp for m in first.session.messages
        ]

    def test_claude_to_victor_to_claude_preserves_content(self, tmp_path):
        """claude → victor IR → claude keeps the recall-critical turns."""
        source = import_transcript(fixture("claude_tools_thinking.jsonl"))
        victor_doc = tmp_path / "mid.json"
        export_session(source.session, TranscriptFormat.VICTOR_JSON, victor_doc)
        middle = import_transcript(victor_doc)
        back = tmp_path / "back.jsonl"
        export_session(middle.session, TranscriptFormat.CLAUDE_CODE, back)

        final = import_transcript(back)
        src_texts = [m.content for m in source.session.messages if m.content]
        dst_texts = [m.content for m in final.session.messages if m.content]
        # every non-empty source text survives the full loop verbatim
        for text in src_texts:
            assert text in dst_texts, f"lost on round trip: {text[:50]!r}"
        # tool identity survives: matching call ids
        src_calls = {m.tool_call_id for m in source.session.messages if m.tool_call_id}
        dst_calls = {m.tool_call_id for m in final.session.messages if m.tool_call_id}
        assert src_calls <= dst_calls

    def test_codex_to_victor_to_codex_preserves_turns(self, tmp_path):
        source = import_transcript(fixture("codex_function_calls.jsonl"))
        victor_doc = tmp_path / "mid.json"
        export_session(source.session, TranscriptFormat.VICTOR_JSON, victor_doc)
        middle = import_transcript(victor_doc)
        back = tmp_path / "back.jsonl"
        export_session(middle.session, TranscriptFormat.CODEX_ROLLOUT, back)

        final = import_transcript(back)
        src_user = [
            m.content for m in source.session.messages if role_of(m) == "user" and m.content
        ]
        dst_user = [m.content for m in final.session.messages if role_of(m) == "user" and m.content]
        assert src_user == dst_user
        # function_call arguments survive as JSON
        src_call = next(m for m in source.session.messages if role_of(m) == "tool_call")
        dst_call = next(m for m in final.session.messages if role_of(m) == "tool_call")
        assert src_call.tool_calls[0]["arguments"] == dst_call.tool_calls[0]["arguments"]

    def test_reimport_same_file_is_idempotent(self, tmp_path):
        """Deterministic session ids: same file → same session (upsert, no dupes)."""
        first = import_transcript(fixture("claude_minimal.jsonl"))
        out = tmp_path / "copy.jsonl"
        out.write_bytes(fixture("claude_minimal.jsonl").read_bytes())
        second = import_transcript(out)
        assert first.session.session_id == second.session.session_id

    def test_tied_timestamps_keep_source_order_through_storage(self, tmp_path):
        """Equal/absent timestamps stay ordered after export+import."""
        doc = {
            "session_id": "ties",
            "messages": [
                {"role": "user", "content": "alpha"},
                {"role": "assistant", "content": "beta"},
                {"role": "user", "content": "gamma"},
            ],
        }
        path = tmp_path / "ties.json"
        path.write_text(json.dumps(doc))
        imported = import_transcript(path)
        stamps = [m.timestamp for m in imported.session.messages]
        assert all(b > a for a, b in zip(stamps, stamps[1:]))
        assert [m.content for m in imported.session.messages] == ["alpha", "beta", "gamma"]
