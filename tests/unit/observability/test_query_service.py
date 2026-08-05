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

"""Tests for observability query service file-backed readers."""

import json
from datetime import datetime, timedelta

from victor.observability.query_service import EventFilters, QueryService


class TestQueryServiceFileReaders:
    """Covers QueryService paths that read events/sessions from files."""

    async def test_query_events_from_jsonl(self, tmp_path) -> None:
        service = QueryService(project_root=tmp_path)
        logs_dir = tmp_path / ".victor" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        older = (now - timedelta(minutes=1)).isoformat()
        recent = now.isoformat()
        usage_log = logs_dir / "usage.jsonl"
        usage_log.write_text(
            "invalid-json\n"
            + json.dumps(
                {
                    "id": "e1",
                    "event_type": "info",
                    "timestamp": recent,
                    "session_id": "s1",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "id": "e2",
                    "event_type": "warning",
                    "timestamp": older,
                    "session_id": "s2",
                    "level": "warning",
                }
            )
            + "\n"
        )

        events = await service._query_events_from_jsonl(
            limit=10,
            offset=0,
            filters=None,
        )
        assert len(events) == 2
        assert [event.id for event in events] == ["e1", "e2"]

        filtered = await service._query_events_from_jsonl(
            limit=10,
            offset=0,
            filters=EventFilters(event_types=["warning"]),
        )
        assert len(filtered) == 1
        assert filtered[0].id == "e2"

    async def test_get_sessions_from_json_skips_invalid_and_returns_valid(self, tmp_path) -> None:
        service = QueryService(project_root=tmp_path)
        sessions_dir = service.paths.sessions_dir
        sessions_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now().isoformat()
        session_file = sessions_dir / "session-good.json"
        session_file.write_text(
            json.dumps(
                {
                    "session_id": "session-good",
                    "metadata": {
                        "created_at": now,
                        "updated_at": now,
                        "message_count": 3,
                        "provider": "provider-a",
                        "model": "model-b",
                    },
                }
            )
        )
        (sessions_dir / "session-bad.json").write_text("{not json}")

        sessions = await service._get_sessions_from_json(limit=10, offset=0)
        assert len(sessions) == 1
        assert sessions[0].id == "session-good"
        assert sessions[0].message_count == 3

    async def test_query_events_from_jsonl_applies_additional_filters(self, tmp_path) -> None:
        service = QueryService(project_root=tmp_path)
        logs_dir = tmp_path / ".victor" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        recent = now.isoformat()
        older = (now - timedelta(minutes=30)).isoformat()
        earliest = (now - timedelta(hours=2)).isoformat()
        usage_log = logs_dir / "usage.jsonl"
        usage_log.write_text(
            json.dumps(
                {
                    "id": "e1",
                    "event_type": "tool",
                    "timestamp": recent,
                    "session_id": "s1",
                    "tool_name": "read",
                    "status": "ok",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "id": "e2",
                    "event_type": "tool_error",
                    "timestamp": older,
                    "session_id": "s2",
                    "tool_name": "write",
                    "status": "error",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "id": "e3",
                    "event_type": "tool",
                    "timestamp": earliest,
                    "session_id": "s1",
                    "tool_name": "write",
                    "status": "ok",
                }
            )
            + "\n"
        )

        filtered = await service._query_events_from_jsonl(
            limit=10,
            offset=0,
            filters=EventFilters(
                session_ids=["s1"],
                tool_names=["read"],
                start_time=now - timedelta(minutes=10),
                end_time=now + timedelta(minutes=1),
                severity="info",
                search_query="read",
            ),
        )
        assert len(filtered) == 1
        assert filtered[0].id == "e1"
