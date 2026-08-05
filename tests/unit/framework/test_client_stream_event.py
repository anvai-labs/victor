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

"""Tests for ``VictorClient._to_stream_event`` tool-result normalization.

The producer nests the rich tool-result payload under
``metadata["tool_result"]``; ``_to_stream_event`` must flatten it so UI consumers
read one predictable, flat shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from victor.framework.client import _to_stream_event
from victor.framework.events import EventType


@dataclass
class _FakeEvent:
    type: Any
    metadata: Optional[Dict[str, Any]] = None
    result: Any = None
    success: bool = True
    tool_name: Optional[str] = None
    content: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None


def test_flattens_nested_tool_result_payload() -> None:
    event = _FakeEvent(
        type=EventType.TOOL_RESULT,
        tool_name="read",
        result="placeholder",
        success=True,
        metadata={
            "tool_result": {
                "name": "read",
                "success": True,
                "elapsed": 0.02,
                "was_pruned": True,
                "original_result": "real output",
                "arguments": {"path": "x.py"},
                "follow_up_suggestions": [{"suggestion": "next"}],
            }
        },
    )

    out = _to_stream_event(event)

    assert out.event_type == "tool_result"
    md = out.metadata
    assert md["elapsed"] == 0.02
    assert md["was_pruned"] is True
    assert md["original_result"] == "real output"
    assert md["follow_up_suggestions"] == [{"suggestion": "next"}]
    assert md["arguments"] == {"path": "x.py"}
    # The flat payload is exposed identically on ``result`` and ``metadata``.
    assert out.result is md


def test_tool_result_without_nested_payload_is_safe() -> None:
    event = _FakeEvent(
        type=EventType.TOOL_RESULT,
        tool_name="grep",
        result="3 matches",
        success=True,
        metadata={},
    )

    out = _to_stream_event(event)

    assert out.event_type == "tool_result"
    assert out.success is True
    # No telemetry present, but the call must not raise and result is preserved.
    assert out.metadata.get("result") == "3 matches"
