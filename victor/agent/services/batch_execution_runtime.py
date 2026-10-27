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

"""Batch execution runtime (FEP-0041 Phase 2).

Executes a BatchPlan: for each step, resolves its input_refs against
prior step outputs, calls the tool pipeline, and collects results. The
AgenticLoop's ACT step dispatches here when the PLAN output is a
BatchPlan instead of a single-action plan.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from victor.core.batch_plan import (
    BatchPlan,
    BatchResult,
    BatchStepResult,
    resolve_input_refs,
)

if TYPE_CHECKING:
    from victor.agent.tool_pipeline import ToolPipeline

logger = logging.getLogger(__name__)


class BatchExecutionRuntime:
    """Sequential batch executor: PLAN's chain → one ACT turn.

    Reuses the existing ToolPipeline for each step (validation, retries,
    HITL gates all apply). stop_on_error=True (default) halts at the
    first failure; EVALUATE sees the partial results + error.
    """

    def __init__(self, tool_pipeline: "ToolPipeline") -> None:
        self._pipeline = tool_pipeline

    async def execute(
        self, plan: BatchPlan, context: Optional[Dict[str, Any]] = None
    ) -> BatchResult:
        """Execute all steps in the batch plan sequentially."""
        context = context or {}
        results: List[BatchStepResult] = []
        prior_outputs: List[Any] = []

        for i, step in enumerate(plan.steps):
            logger.info(
                "batch step %d/%d: %s%s",
                i + 1,
                len(plan.steps),
                step.tool_name,
                f" ({step.description})" if step.description else "",
            )
            try:
                resolved = resolve_input_refs(step, prior_outputs)
                output = await self._execute_step(step.tool_name, resolved, context)
                results.append(
                    BatchStepResult(
                        step_index=i,
                        tool_name=step.tool_name,
                        success=True,
                        output=output,
                    )
                )
                prior_outputs.append(self._extract_output(output))
            except Exception as exc:
                logger.warning(
                    "batch step %d/%d FAILED: %s — %s",
                    i + 1,
                    len(plan.steps),
                    step.tool_name,
                    exc,
                )
                results.append(
                    BatchStepResult(
                        step_index=i,
                        tool_name=step.tool_name,
                        success=False,
                        error=str(exc),
                    )
                )
                prior_outputs.append(None)
                if plan.stop_on_error:
                    break

        return BatchResult(plan=plan, step_results=results)

    async def _execute_step(
        self, tool_name: str, inputs: Dict[str, Any], context: Dict[str, Any]
    ) -> Any:
        """Execute one tool call through the existing pipeline."""
        if not hasattr(self._pipeline, "_execute_single_call"):
            raise RuntimeError(
                f"ToolPipeline does not expose _execute_single_call; "
                f"cannot execute batch step '{tool_name}'"
            )
        tool_call = {"name": tool_name, "arguments": inputs}
        result = await self._pipeline._execute_single_call(tool_call, context)
        if hasattr(result, "success") and not result.success:
            error = getattr(result, "error", "tool call failed")
            raise RuntimeError(f"{tool_name}: {error}")
        return result

    @staticmethod
    def _extract_output(result: Any) -> Any:
        """Extract the usable output from a ToolCallResult for input_refs.

        The pipeline returns a ToolCallResult; downstream steps reference
        its output field. If the result has an ``output`` attribute,
        use it; otherwise return the result itself.
        """
        output = getattr(result, "output", None)
        if output is not None:
            return output
        content = getattr(result, "content", None)
        if content is not None:
            return content
        return result
