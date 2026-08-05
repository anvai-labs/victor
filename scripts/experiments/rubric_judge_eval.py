# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""EVR-3 effectiveness comparison: LLM rubric judge vs the heuristic baseline.

Runs both completion judges over a small human-labeled set and reports which agrees better with the
ground-truth completion label — accuracy + Cohen's κ (EVR-2). This measures the crux: *does the
task-adaptive LLM rubric judge make better completion decisions than the deterministic baseline?*

The buffered `EnhancedCompletionEvaluator` is deeply coupled to `_evaluate`, so this A/B compares the
two *rubric* judges (the part that differs); a full end-to-end turn-count comparison needs the
strategy plumbed through chat construction (a follow-up).

Usage:
    python scripts/experiments/rubric_judge_eval.py            # heuristic baseline only (offline)
    python scripts/experiments/rubric_judge_eval.py --live     # also run the LLM judge via z.ai
"""

from __future__ import annotations

import argparse
import asyncio
import os

from victor.evaluation.judge_calibration import cohens_kappa
from victor.framework.rubric_completion import (
    AsyncRubricCompletionEvaluator,
    LLMRubricJudge,
    RubricCompletionEvaluator,
)

# Human-labeled completion set: (task_family, answer_content, is_complete_truth).
# Complete = a genuine, sufficient final answer; not-complete = premature / partial / intent-only.
LABELED: list[tuple[str, str, bool]] = [
    (
        "lookup",
        "TurnResult is defined in victor/agent/services/turn_execution_runtime.py:70. It has 7 "
        "fields: response, tool_results, follow_up_suggestions, has_tool_calls, tool_calls_count, "
        "all_tools_blocked, is_qa_response.",
        True,
    ),
    ("lookup", "Let me search the repo for the TurnResult class.", False),
    ("qa", "6 times 7 is 42.", True),
    ("qa", "I'll calculate that for you.", False),
    (
        "coding",
        "Fixed the bug in parser.py: the off-by-one in tokenize() now uses <= instead of <, and the "
        "test suite passes (12/12). The diff edits line 88.",
        True,
    ),
    (
        "coding",
        "I found the parser. Now I need to read it to understand the bug.",
        False,
    ),
    (
        "analysis",
        "## Summary\nThe auth flow has 3 stages: login, token refresh, and logout. The refresh path "
        "(auth/refresh.py) lacks a retry, which is the reported flakiness. Recommend adding a bounded "
        "retry with backoff.",
        True,
    ),
    (
        "analysis",
        "Here are the files. Would you like me to analyze the auth flow next?",
        False,
    ),
]


async def _evaluate_judge(evaluator, *, is_async: bool) -> list[bool]:
    """Return the judge's complete/incomplete decision for each labeled example."""
    decisions = []
    for task_family, content, _truth in LABELED:
        if is_async:
            result = await evaluator.aevaluate(task_family=task_family, content=content, context={})
        else:
            result = evaluator.evaluate(task_family=task_family, content=content, context={})
        decisions.append(bool(result.complete))
    return decisions


def _report(name: str, decisions: list[bool]) -> None:
    truth = [t for _, _, t in LABELED]
    correct = sum(1 for d, t in zip(decisions, truth) if d == t)
    acc = correct / len(truth)
    kappa = cohens_kappa(truth, decisions)
    print(f"\n=== {name} ===")
    print(
        f"accuracy vs truth: {correct}/{len(truth)} = {acc:.2f}   Cohen's κ vs truth: {kappa:.3f}"
    )
    for (tf, content, t), d in zip(LABELED, decisions):
        flag = "ok " if d == t else "XX "
        print(f"  {flag} truth={int(t)} judge={int(d)}  [{tf}] {content[:60]!r}")


def _build_zai_complete_fn():
    # Authenticate via the `victor auth env -p zai` bridge: run
    #   eval "$(victor auth env -p zai 2>/dev/null | grep '^export')"
    # before this script so the credentials are in the environment. ZAIProvider's resolver
    # (get_api_key("zai")) then picks them up the same way `victor chat` does — api_key=None lets
    # the resolver consult env + keyring.
    from victor.providers.base import Message
    from victor.providers.zai_provider import ZAIProvider

    model = os.environ.get("VICTOR_ZAI_MODEL", "glm-5.2")
    provider = ZAIProvider(api_key=os.environ.get("ZAI_API_KEY"), coding_plan=True, model=model)

    async def complete(prompt: str) -> str:
        resp = await provider.chat(
            [Message(role="user", content=prompt)],
            model=model,
            temperature=0.0,
            max_tokens=400,
        )
        return resp.content or ""

    return complete


async def main(live: bool) -> None:
    # Baseline: the deterministic heuristic rubric judge (no LLM).
    heuristic = await _evaluate_judge(RubricCompletionEvaluator(), is_async=False)
    _report("Heuristic rubric judge (baseline, no LLM)", heuristic)

    if live:
        complete_fn = _build_zai_complete_fn()
        llm = await _evaluate_judge(
            AsyncRubricCompletionEvaluator(LLMRubricJudge(complete_fn)), is_async=True
        )
        _report("LLM rubric judge (z.ai)", llm)
    else:
        print("\n(LLM judge skipped — pass --live to run it via z.ai)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="run the LLM judge via z.ai")
    args = parser.parse_args()
    asyncio.run(main(args.live))
