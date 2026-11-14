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
redundant calls, broad reads) are informative and do not.
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
        "tool_blocked",
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


def batch_all_rejected(tool_results: Iterable[Any]) -> bool:
    """True when the batch is non-empty and the runtime refused every call in it."""
    results = list(tool_results or ())
    return bool(results) and all(_rejected(result) for result in results)
