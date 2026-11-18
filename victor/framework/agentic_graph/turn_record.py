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

"""Durable ACT observations for the StateGraph agentic loop.

``TurnResult`` remains the live runtime primitive returned by ACT.  It may carry
provider-specific metadata and run-local objects, and therefore is not a durable
graph-state value.  ``TurnRecord`` is the explicit JSON-safe projection needed by
EVALUATE, publication, and checkpoint/resume.  A run-owned channel retains only
the latest completed record for timeout recovery; it never transports live
objects through a module-global ContextVar.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

TURN_RECORD_SCHEMA_VERSION = "victor.turn.record.v1"

# Exact lowercase field names that must never enter durable turn state.  This is
# deliberately a fail-closed denylist for common credential/raw-provider fields;
# unknown object types are rejected separately by _durable_value().
_NON_DURABLE_FIELD_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "client_secret",
        "credential",
        "credentials",
        "password",
        "provider_response",
        "raw_provider_response",
        "raw_response",
        "refresh_token",
        "secret",
        "virtual_key",
    }
)


def _durable_value(value: Any) -> Any:
    """Return a JSON-compatible projection without accepting runtime handles."""
    if isinstance(value, float) and not math.isfinite(value):
        raise TypeError("Non-finite floats are not JSON-compatible")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _durable_value(value.value)
    if isinstance(value, dict):
        durable: Dict[str, Any] = {}
        for key, item in value.items():
            durable_key = str(key)
            if durable_key.lower() in _NON_DURABLE_FIELD_NAMES:
                raise ValueError(f"Refusing durable turn field {durable_key!r}")
            durable[durable_key] = _durable_value(item)
        return durable
    if isinstance(value, (set, frozenset)):
        projected = [_durable_value(item) for item in value]
        return sorted(
            projected,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(value, (list, tuple)):
        return [_durable_value(item) for item in value]
    if isinstance(value, BaseModel):
        return _durable_value(value.model_dump(mode="json"))
    raise TypeError(f"Value of type {type(value).__name__} is not JSON-compatible")


class TurnResponseRecord(BaseModel):
    """Durable completion-response projection used by turn evaluation."""

    content: str = ""
    role: str = "assistant"
    tool_calls: Optional[List[Dict[str, Any]]] = None
    stop_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_response(cls, response: Any) -> "TurnResponseRecord":
        tool_calls = getattr(response, "tool_calls", None)
        stop_reason = getattr(response, "stop_reason", None)
        usage = getattr(response, "usage", None)
        model = getattr(response, "model", None)
        metadata = getattr(response, "metadata", None)
        content = getattr(response, "content", "")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            raise TypeError(
                f"Completion content of type {type(content).__name__} is not JSON-compatible"
            )
        return cls(
            content=content,
            role=str(_durable_value(getattr(response, "role", "assistant")) or "assistant"),
            tool_calls=(
                _durable_value(tool_calls) if isinstance(tool_calls, (list, tuple)) else None
            ),
            stop_reason=(
                _durable_value(stop_reason) if isinstance(stop_reason, (str, Enum)) else None
            ),
            usage=_durable_value(usage) if isinstance(usage, dict) else None,
            model=_durable_value(model) if isinstance(model, (str, Enum)) else None,
            metadata=_durable_value(metadata) if isinstance(metadata, dict) else {},
        )

    def to_completion_response(self) -> Any:
        from victor.core.provider_messages import CompletionResponse

        return CompletionResponse(
            content=self.content,
            role=self.role,
            tool_calls=self.tool_calls,
            stop_reason=self.stop_reason,
            usage=self.usage,
            model=self.model,
            metadata=self.metadata,
        )


class TurnRecord(BaseModel):
    """Versioned, serializable observation of one completed ACT boundary."""

    schema_version: Literal["victor.turn.record.v1"] = TURN_RECORD_SCHEMA_VERSION
    response: TurnResponseRecord
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)
    follow_up_suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    has_tool_calls: bool = False
    tool_calls_count: int = 0
    all_tools_blocked: bool = False
    is_qa_response: bool = False
    disposition: str = "continue"
    tool_signatures: List[str] = Field(default_factory=list)

    @classmethod
    def from_turn_result(cls, result: Any) -> "TurnRecord":
        disposition = getattr(result, "disposition", None)
        from victor.agent.services.turn_execution_runtime import TurnDisposition

        if disposition is None:
            disposition_value = TurnDisposition.CONTINUE.value
        elif isinstance(disposition, TurnDisposition):
            disposition_value = disposition.value
        elif isinstance(disposition, str) and disposition in {
            member.value for member in TurnDisposition
        }:
            disposition_value = disposition
        else:
            raise TypeError(f"Unknown turn disposition: {disposition!r}")
        tool_results = getattr(result, "tool_results", None)
        follow_ups = getattr(result, "follow_up_suggestions", None)
        signatures = getattr(result, "tool_signatures", None)
        return cls(
            response=TurnResponseRecord.from_response(getattr(result, "response", None)),
            tool_results=(
                _durable_value(tool_results) if isinstance(tool_results, (list, tuple)) else []
            ),
            follow_up_suggestions=(
                _durable_value(follow_ups) if isinstance(follow_ups, (list, tuple)) else []
            ),
            has_tool_calls=bool(getattr(result, "has_tool_calls", False)),
            tool_calls_count=int(getattr(result, "tool_calls_count", 0) or 0),
            all_tools_blocked=bool(getattr(result, "all_tools_blocked", False)),
            is_qa_response=bool(getattr(result, "is_qa_response", False)),
            disposition=disposition_value,
            tool_signatures=(
                sorted(_durable_value(signatures))
                if isinstance(signatures, (set, frozenset, list, tuple))
                else []
            ),
        )

    def to_turn_result(self) -> Any:
        from victor.agent.services.turn_execution_runtime import (
            TurnDisposition,
            TurnResult,
        )

        return TurnResult(
            response=self.response.to_completion_response(),
            tool_results=self.tool_results,
            follow_up_suggestions=self.follow_up_suggestions,
            has_tool_calls=self.has_tool_calls,
            tool_calls_count=self.tool_calls_count,
            all_tools_blocked=self.all_tools_blocked,
            is_qa_response=self.is_qa_response,
            disposition=TurnDisposition(self.disposition),
        )

    @property
    def content(self) -> str:
        return self.response.content


@dataclass
class TurnRecordChannel:
    """Latest-record holder owned by one StateGraph run or stream.

    The agentic-loop graph executes ACT/EVALUATE sequentially.  Parallel team or
    workflow graphs must not use this channel; their state values are their own
    durable observation contracts.
    """

    _latest: Optional[TurnRecord] = None

    def publish(self, record: TurnRecord) -> None:
        self._latest = record.model_copy(deep=True)

    @property
    def latest(self) -> Optional[TurnRecord]:
        return self._latest.model_copy(deep=True) if self._latest is not None else None

    def clear(self) -> None:
        self._latest = None
