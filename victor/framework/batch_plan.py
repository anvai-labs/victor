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

"""Batch execution types (FEP-0041 Phase 1).

The PLAN step may emit a BatchPlan instead of a single-action plan: a
sequential chain of tool calls with data-flow references between steps.
The ACT step detects a BatchPlan and executes all steps in one turn —
no LLM round-trips between calls. EVALUATE runs once after the batch.

The design is deliberately NOT code generation (the Cloudflare Code Mode
approach): victor's tools need real environment access, and the existing
ToolPipeline already handles validation, HITL gates, and retries. The
batch is a structured plan, not a script.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Input reference syntax: $stepN.key (e.g., $step0.file_path)
_INPUT_REF_PATTERN = re.compile(r"^\$step(\d+)(?:\.(\S+))?$")


@dataclass
class BatchStep:
    """One step in a batch plan.

    ``input_refs`` maps input-key -> reference: ``{"file_path":
    "$step0.file_path"}`` means this step's ``file_path`` input comes
    from step 0's output key ``file_path``. Resolved by the batch
    executor before each step runs.
    """

    tool_name: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    input_refs: Dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class BatchPlan:
    """A PLAN-step output: a sequential chain of tool calls.

    The LLM emits this instead of a single-action plan when the task is
    a well-defined chain. The ACT step executes all steps in one turn;
    EVALUATE sees the batch results as a single evaluation unit.
    """

    steps: List[BatchStep] = field(default_factory=list)
    rationale: str = ""
    stop_on_error: bool = True


@dataclass
class BatchStepResult:
    """One step's execution outcome (for the audit log and EVALUATE)."""

    step_index: int
    tool_name: str
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class BatchResult:
    """The aggregate outcome of a batch execution."""

    plan: BatchPlan
    step_results: List[BatchStepResult] = field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        return bool(self.step_results) and all(r.success for r in self.step_results)

    @property
    def error(self) -> Optional[str]:
        """The first error (None if all succeeded)."""
        for r in self.step_results:
            if not r.success:
                return r.error
        return None


def resolve_input_refs(step: BatchStep, prior_outputs: List[Any]) -> Dict[str, Any]:
    """Resolve a step's input_refs against prior step outputs.

    ``$stepN.key`` looks up ``prior_outputs[N]`` and extracts ``key``
    (a dict lookup; non-dict prior outputs resolve to the whole value
    when key is absent). Returns the merged inputs dict.
    """
    resolved = dict(step.inputs)
    for input_key, ref in step.input_refs.items():
        match = _INPUT_REF_PATTERN.match(ref)
        if not match:
            continue  # malformed refs pass through as-is; the tool will error
        step_num = int(match.group(1))
        output_key = match.group(2)  # None when the ref is just $stepN
        if step_num >= len(prior_outputs):
            continue  # out-of-range; the tool will error on missing input
        prior = prior_outputs[step_num]
        if output_key is None:
            resolved[input_key] = prior  # whole-value reference ($stepN)
        elif isinstance(prior, dict) and output_key in prior:
            resolved[input_key] = prior[output_key]
        elif not isinstance(prior, dict):
            resolved[input_key] = prior  # scalar prior: use whole value
    return resolved


def parse_batch_plan(data: Dict[str, Any]) -> Optional[BatchPlan]:
    """Parse a batch plan from the PLAN step's structured output.

    Returns None if the dict doesn't look like a batch plan (no 'steps'
    key, or steps is empty) — callers fall through to the normal plan
    path.
    """
    steps_data = data.get("steps")
    if not isinstance(steps_data, list) or not steps_data:
        return None
    steps: List[BatchStep] = []
    for step_data in steps_data:
        if not isinstance(step_data, dict):
            continue
        tool_name = step_data.get("tool_name") or step_data.get("tool")
        if not tool_name:
            continue
        steps.append(
            BatchStep(
                tool_name=tool_name,
                inputs=step_data.get("inputs", {}),
                input_refs=step_data.get("input_refs", {}),
                description=step_data.get("description", ""),
            )
        )
    if not steps:
        return None
    return BatchPlan(
        steps=steps,
        rationale=data.get("rationale", ""),
        stop_on_error=data.get("stop_on_error", True),
    )
