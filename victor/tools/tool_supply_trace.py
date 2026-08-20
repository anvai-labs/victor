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

"""Per-turn tool-supply telemetry.

Records how the *registered* tool set is narrowed to the tools the model actually
receives, stage by stage, so over-restriction becomes observable. Today the only
signal is a single ``logger.info`` line in the selection runtime, which cannot
answer "which gate dropped ``code_search`` this turn?". A :class:`ToolSupplyTrace`
captures that funnel and is emitted once per turn on the observability bus under
the :data:`TOOL_SUPPLY_TOPIC` topic (mirroring the existing ``tool.intent`` event).

This module is pure data + a thin builder. It performs no selection and changes no
behavior; it only observes the existing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Observability-bus topic for the per-turn tool-supply event.
TOOL_SUPPLY_TOPIC = "tool.supply"


def _tool_names(tools: Any) -> Tuple[str, ...]:
    """Best-effort extraction of tool names from a tool list (or None)."""
    if not tools:
        return ()
    names: List[str] = []
    for t in tools:
        name = getattr(t, "name", None)
        if name is None and isinstance(t, str):
            name = t
        names.append(str(name) if name is not None else "?")
    return tuple(names)


@dataclass(frozen=True)
class GateRecord:
    """One narrowing stage in the tool-supply funnel.

    A stage may *drop* tools (removed from the callable set) or *demote* them
    (kept but with a terser schema). ``dropped`` is computed from the name sets
    before/after the stage; ``demoted_to_stub`` is supplied explicitly by stages
    that tier schemas rather than remove tools.
    """

    name: str
    in_count: int
    out_count: int
    dropped: Tuple[str, ...] = ()
    demoted_to_stub: Tuple[str, ...] = ()
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "in_count": self.in_count,
            "out_count": self.out_count,
            "dropped": list(self.dropped),
            "demoted_to_stub": list(self.demoted_to_stub),
            "reason": self.reason,
        }


@dataclass
class ToolSupplyTrace:
    """Mutable per-turn builder; serialized to an event payload at turn end.

    Usage (instrumentation only — never alters the value flowing through)::

        trace = ToolSupplyTrace.begin(registered_tools)
        tools = trace.record("intent_filter", before=prev, after=tools)
        ...
        trace.finalize(dispatched=final_tools)
    """

    registered: Tuple[str, ...] = ()
    candidates: Tuple[str, ...] = ()
    stages: List[GateRecord] = field(default_factory=list)
    dispatched: Tuple[str, ...] = ()
    profile: Optional[str] = None
    skipped: bool = False
    skip_reason: str = ""
    # Correlation spine (R1): joins this offered-trace to the turn's invoked
    # (tool.intent) and resulted (rl_outcome) records.
    session_id: str = ""
    turn_id: str = ""
    request_id: str = ""

    @classmethod
    def begin(cls, registered: Any = None) -> "ToolSupplyTrace":
        trace = cls(registered=_tool_names(registered))
        trace._capture_correlation()
        return trace

    def _capture_correlation(self) -> None:
        """Stamp the live correlation spine (best-effort; telemetry is non-critical)."""
        try:
            from victor.core.context import get_request_id, get_session_id, get_turn_id

            self.session_id = get_session_id() or ""
            self.turn_id = get_turn_id() or ""
            self.request_id = get_request_id() or ""
        except Exception:
            pass

    def set_candidates(self, tools: Any) -> Any:
        """Record the candidate set produced by candidate generation."""
        self.candidates = _tool_names(tools)
        return tools

    def record(
        self,
        name: str,
        before: Any,
        after: Any,
        *,
        reason: str = "",
        demoted: Sequence[str] = (),
    ) -> Any:
        """Append a :class:`GateRecord` for one stage and return ``after`` unchanged.

        Returning ``after`` lets call sites wrap a transform without restructuring:
        ``tools = trace.record("kv_sort", prev, transform(prev))``.
        """
        before_names = _tool_names(before)
        after_names = _tool_names(after)
        after_set = set(after_names)
        dropped = tuple(n for n in before_names if n not in after_set)
        self.stages.append(
            GateRecord(
                name=name,
                in_count=len(before_names),
                out_count=len(after_names),
                dropped=dropped,
                demoted_to_stub=tuple(demoted),
                reason=reason,
            )
        )
        return after

    def mark_skipped(self, reason: str) -> "ToolSupplyTrace":
        """Record that the entire tool set was withheld this turn (e.g. Q&A gate)."""
        self.skipped = True
        self.skip_reason = reason
        self.dispatched = ()
        return self

    def finalize(self, dispatched: Any) -> "ToolSupplyTrace":
        self.dispatched = _tool_names(dispatched)
        return self

    def to_payload(self) -> Dict[str, Any]:
        """Serialize for the observability event (JSON-friendly)."""
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "request_id": self.request_id,
            "registered_count": len(self.registered),
            "candidate_count": len(self.candidates),
            "dispatched_count": len(self.dispatched),
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "profile": self.profile,
            "candidates": list(self.candidates),
            "dispatched": list(self.dispatched),
            "stages": [s.to_dict() for s in self.stages],
        }
