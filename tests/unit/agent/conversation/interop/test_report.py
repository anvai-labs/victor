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

"""Mirror suite for victor.agent.conversation.interop.report."""

from victor.agent.conversation.interop.report import ConversionReport


class TestConversionReport:
    def test_counters_and_message_count(self):
        report = ConversionReport(source_format="claude", target_format="victor")
        report.count_role("user")
        report.count_role("user")
        report.count_role("assistant")
        assert report.message_count == 3
        assert report.role_counts == {"user": 2, "assistant": 1}

    def test_skip_and_warning_dedup(self):
        report = ConversionReport()
        report.note_skip(3, "line", "malformed JSON: oops")
        report.note_skip(4, "line", "line exceeds guard")
        report.note_warning("same warning")
        report.note_warning("same warning")
        assert len(report.skipped) == 2
        assert report.skipped[0] == {"index": 3, "type": "line", "reason": "malformed JSON: oops"}
        assert report.warnings == ["same warning"]

    def test_to_dict_round_trip_fields(self):
        report = ConversionReport(
            source_format="codex",
            target_format="victor",
            entries_read=9,
            tool_calls_preserved=2,
            thinking_blocks_dropped=1,
            encrypted_items_skipped=2,
            model="gpt",
            session_id="imp_x",
        )
        report.count_role("user")
        data = report.to_dict()
        assert data["source_format"] == "codex"
        assert data["message_count"] == 1
        assert data["encrypted_items_skipped"] == 2
        assert data["session_id"] == "imp_x"
        assert data["role_counts"] == {"user": 1}

    def test_render_summary_loss_ledger(self):
        report = ConversionReport(source_format="claude", target_format="victor")
        report.count_role("user")
        report.count_role("assistant")
        report.thinking_blocks_dropped = 3
        report.truncations = 1
        report.skipped.append({"index": 0, "type": "line", "reason": "x"})
        text = report.render_summary()
        assert "Format: claude → victor" in text
        assert "1 assistant · 1 user" in text
        assert "3 thinking blocks dropped" in text
        assert "1 truncations" in text
        assert "1 entries skipped" in text

    def test_render_summary_clean_conversion(self):
        report = ConversionReport(source_format="victor", target_format="victor")
        report.count_role("user")
        text = report.render_summary()
        assert "Lossy" not in text
