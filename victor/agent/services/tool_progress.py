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

"""One rule for "this turn's tool calls were all rejected" (LOOP-04).

Both live loops feed ``TurnResult.all_tools_blocked`` into the shared
``SpinDetector``, which nudges after two consecutive rejected batches and stops
the run at its configured limit. The buffered loop used to derive the flag from
tool-pipeline state alone and the streaming loop never set it, so a model
repeating unknown, denied or already-failed calls in streaming mode ran until
the iteration cap. This predicate reads the processed tool-result dicts both
loops already produce, so the same batch is classified the same way in either
mode.

Only outcomes where the runtime refused the call before it could do any work
count. Redirects that hand back cached or narrowed content (duplicate reads,
redundant calls, broad reads) are informative and do not. The streaming-only
``tool_blocked`` outcome is excluded: its tracker keeps its own consecutive and
total limits and forces a final answer, and counting it here as well would end
the run in the same turn, before the model could give that answer.

A skipped call that has no provider call id never becomes a processed result
(it is dropped so history holds no orphan tool message). When calls were
dispatched but nothing came back, every one of them was such a skip, so the
batch counts as rejected.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

REJECTED_OUTCOME_KINDS = frozenset(
    {
        "budget_exhausted",
        "invalid_tool_call",
        "invalid_tool_name",
        "middleware_blocked",
        "middleware_error",
        "missing_tool_name",
        "permission_denied",
        "repeated_failure",
        "tool_unavailable",
    }
)


def _rejected(result: Any) -> bool:
    if not isinstance(result, Mapping):
        return False
    return (
        bool(result.get("skipped"))
        and not result.get("success")
        and result.get("outcome_kind") in REJECTED_OUTCOME_KINDS
    )


def batch_all_rejected(tool_results: Iterable[Any], *, dispatched: int = 0) -> bool:
    """True when the runtime refused every call in the batch.

    Args:
        tool_results: Processed result dicts for the batch.
        dispatched: Number of calls handed to tool execution. When it is positive
            and no results came back, all of them were dropped id-less skips.
    """
    results = list(tool_results or ())
    if not results:
        return dispatched > 0
    return all(_rejected(result) for result in results)
