# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Calibration adapter for ``EnhancedCompletionEvaluator`` (EVR-3, ADR-009).

The rubric↔enhanced parity question — "does the default completion evaluator
actually judge completion better than the rubric judge?" — has never been
measured: the calibration harness scores judges, and ``enhanced`` has never
been wrapped as one. This adapter closes that gap by scoring the PRODUCTION
evaluator on the same blinded (prompt, transcript, workspace) views as every
other calibration judge.

Fidelity contract (what makes the measurement honest):

- **Perception** comes from the production ``PerceptionIntegration.perceive``
  on the task prompt — the same component the live loop uses — so requirement
  extraction and task-typing are the real thing, not a hand-rolled stand-in.
- **action_result** is a real ``TurnResult`` shaped like the live loop's final
  turn: response content = the transcript's final message, no pending tool
  calls (evaluation in the live loop happens on the no-tool final turn).
- **Evaluator flags** mirror ``AgenticLoopConfig`` defaults (all detection
  paths enabled).

Known fidelity deltas (documented, inherent to offline calibration):

- ``spin_detector``/``fulfillment_detector`` are None — both need live
  multi-turn loop state a static transcript cannot supply. The evaluator's
  spin/fulfillment fast paths therefore do not fire; what is measured is the
  requirement-validation + completion-scoring + keyword cascade.
- Prior-turn tool activity is visible to the evaluator only through ``state``
  (see ``_TOOL_EVIDENCE_KEY``), not a live SpinDetector's counters.

Record both deltas in FINDINGS whenever a run using this judge is reported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from victor.evaluation.judge_calibration_harness import (
    CalibrationJudge,
    PersistentLoopRunner,
    Transcript,
)

_TOOL_EVIDENCE_KEY = "_successful_tool_evidence"


def make_enhanced_judge(
    *,
    evaluator: Optional[Any] = None,
    perception: Optional[Any] = None,
) -> CalibrationJudge:
    """Wrap the production enhanced-completion evaluator as a CalibrationJudge.

    Zero LLM calls; deterministic given the transcript. Score is the binary
    completion verdict (COMPLETE → 1.0), matching the harness's gold scope.
    """
    from victor.framework.enhanced_completion_evaluation import (
        EnhancedCompletionEvaluator,
    )
    from victor.framework.perception_integration import PerceptionIntegration

    # Mirror AgenticLoopConfig defaults: every detection path enabled.
    enhanced = evaluator or EnhancedCompletionEvaluator(
        enable_requirement_validation=True,
        enable_completion_scoring=True,
        enable_context_keywords=True,
    )
    perceiver = perception or PerceptionIntegration(memory_coordinator=None)
    runner = PersistentLoopRunner()

    def judge(prompt: str, transcript: Transcript, workspace: Path) -> float:
        async def _run() -> float:
            from victor.agent.services.turn_execution_runtime import TurnResult
            from victor.providers.base import CompletionResponse

            perceived = await perceiver.perceive(prompt)
            tool_steps = transcript.tool_steps()
            # The live loop evaluates the FINAL turn: content, no pending tool
            # calls. Prior tool activity is evidence carried in state.
            action_result = TurnResult(
                response=CompletionResponse(content=transcript.final_message or ""),
                tool_results=[{"tool": step.content, "success": True} for step in tool_steps],
                has_tool_calls=False,
                tool_calls_count=0,
            )
            state: dict[str, Any] = {
                "task_type": getattr(perceived, "task_type", None) or "general",
                "workspace": str(workspace),
            }
            if tool_steps:
                state[_TOOL_EVIDENCE_KEY] = True
            result = await enhanced.evaluate(
                perception=perceived,
                action_result=action_result,
                state=state,
                fulfillment_detector=None,
                spin_detector=None,
            )
            decision = getattr(result, "decision", None)
            value = getattr(decision, "value", decision)
            return 1.0 if str(value).lower() == "complete" else 0.0

        return runner.run(_run())

    return judge
