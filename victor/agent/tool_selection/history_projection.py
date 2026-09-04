# Copyright 2025 Vijaykumar Singh <vijaykumar@anvaiops.com>
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

"""History projection for tool selection.

Lives beside its consumer (the tool selector) rather than in
turn_execution_runtime — that hotspot has a size-ratchet cap and already
regrew once after decomposition.
"""

from __future__ import annotations

import copy

from typing import Any, Dict, List, Optional


def _selector_history_projection(messages: List[Any]) -> Optional[List[Dict[str, Any]]]:
    """Project Message objects to exactly the keys the tool selector reads.

    Replaces a per-iteration ``model_dump()`` of the entire history, which
    pydantic-serialized every message on every model turn while the selector
    only ever accesses ``role``, full ``content``, and ``tool_calls`` names
    (co-design review U1-1). Content must remain untruncated here — the
    selection cache-key builder applies its own truncation downstream.

    Args:
        messages: Live Message objects from the chat context.

    Returns:
        Plain dicts with role/content (+ tool_calls when present, list-copied
        so the projection cannot be used to mutate the live history).
    """
    if not messages:
        return None  # matches the call site's previous `else None` branch
    projected: List[Dict[str, Any]] = []
    for msg in messages:
        entry: Dict[str, Any] = {
            "role": getattr(msg, "role", None),
            "content": getattr(msg, "content", ""),
        }
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            # Deep-copied: a shallow list copy left the inner tool_call dicts
            # aliased to the live history (adversarial-review finding).
            entry["tool_calls"] = copy.deepcopy(tool_calls)
        projected.append(entry)
    return projected
