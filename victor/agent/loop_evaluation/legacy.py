"""Legacy loop evaluator — thin wrapper around ContinuationStrategy.

Zero behavior change from the current code path.  All 23 parameters that
``ContinuationStrategy.determine_continuation_action`` expects are marshalled
from the unified ``LoopContext`` and the returned ``ContinuationDirective`` is
converted to a ``LoopDecision``.

This backend is the default when ``USE_STATEGRAPH_AGENTIC_LOOP`` is OFF.
"""

from __future__ import annotations

import logging
from typing import Optional

from victor.agent.loop_evaluation.protocol import (
    LoopContext,
    LoopDecision,
    LoopEvaluator,
)

logger = logging.getLogger(__name__)


class LegacyEvaluator(LoopEvaluator):
    """Wraps ``ContinuationStrategy`` to implement the ``LoopEvaluator`` protocol.

    Creating a fresh ``ContinuationStrategy`` per call mirrors the current
    ``IntentClassificationHandler._determine_action`` behaviour — the strategy
    is stateless between turns.
    """

    def evaluate(self, ctx: LoopContext) -> LoopDecision:
        from victor.agent.continuation_strategy import ContinuationStrategy
        from victor.agent.continuation_contract import ContinuationStatePatch

        strategy = ContinuationStrategy(runtime_intelligence=ctx.runtime_intelligence)

        directive = strategy.determine_continuation_action(
            intent_result=ctx.intent_result,
            is_analysis_task=ctx.is_analysis_task,
            is_action_task=ctx.is_action_task,
            content_length=ctx.content_length,
            full_content=ctx.full_content,
            continuation_prompts=ctx.continuation_prompts,
            asking_input_prompts=ctx.asking_input_prompts,
            one_shot_mode=ctx.one_shot_mode,
            mentioned_tools=ctx.mentioned_tools,
            max_prompts_summary_requested=ctx.max_prompts_summary_requested,
            settings=ctx.settings,
            rl_coordinator=ctx.rl_coordinator,
            provider_name=ctx.provider_name,
            model=ctx.model,
            tool_budget=ctx.tool_budget,
            unified_tracker_config=ctx.unified_tracker_config,
            task_completion_signals=ctx.task_completion_signals,
            compaction_occurred=ctx.compaction_occurred,
            compaction_messages_removed=ctx.compaction_messages_removed,
            degraded_resume_state=ctx.degraded_resume_state,
            resume_summary=ctx.resume_summary,
            force_tool_execution_attempts=ctx.force_tool_execution_attempts,
            query_classification=ctx.query_classification,
            plan_step_count=ctx.plan_step_count,
        )

        patch: Optional[ContinuationStatePatch] = getattr(directive, "state_patch", None)

        return LoopDecision(
            action=directive.action,
            reason=directive.reason,
            message=directive.message,
            confidence=1.0,
            source="legacy",
            state_patch=patch,
            extracted_call=getattr(directive, "extracted_call", None),
            mentioned_tools_override=getattr(directive, "mentioned_tools", None),
            set_final_summary_requested=bool(patch.final_summary_requested if patch else False),
            set_max_prompts_summary_requested=bool(
                patch.max_prompts_summary_requested if patch else False
            ),
        )
