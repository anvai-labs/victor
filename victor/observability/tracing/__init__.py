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

"""Execution tracing for debugging and observability.

This package provides structured tracers for monitoring execution flow:
- ExecutionTracer: Tracks execution spans with parent-child relationships
- ToolCallTracer: Tracks tool calls with execution span linkage

SOLID Principles:
- SRP: Each tracer handles one concern
- OCP: Extensible via event emission
- LSP: All tracers implement ITracer protocol
- ISP: Focused interfaces for specific tracing needs
- DIP: Depends on EventBus abstraction

Usage:
    from victor.observability.tracing import ExecutionTracer, ToolCallTracer
    from victor.core.events import ObservabilityBus as EventBus

    event_bus = EventBus()
    exec_tracer = ExecutionTracer(event_bus)
    tool_tracer = ToolCallTracer(event_bus)

    # Start execution span
    span_id = exec_tracer.start_span("agent-1", "agent")

    # Record tool call
    call_id = tool_tracer.record_call("read_file", {"file": "test.py"}, span_id)

    # Complete tool call
    tool_tracer.complete_call(call_id, result="file content")

    # End span
    exec_tracer.end_span(span_id, status="success")
"""

from victor.observability.tracing.execution import ExecutionSpan, ExecutionTracer
from victor.observability.tracing.tool_calls import (
    ToolCallRecord,
    ToolCallTracer,
)

__all__ = [
    "ExecutionTracer",
    "ExecutionSpan",
    "ToolCallTracer",
    "ToolCallRecord",
]
