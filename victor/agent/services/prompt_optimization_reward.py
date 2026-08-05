# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Prompt-candidate reward attribution (FEP-0017).

Emits a ``PROMPT_CANDIDATE_USED`` RL event for each prompt candidate served
during a turn, carrying the turn's completion score so the ``prompt_optimizer``
learner can update the candidate's Thompson posterior. Extracted into its own
module to avoid growing the ``runtime_intelligence`` hotspot.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# A turn counts as a positive prompt-quality outcome when its completion score
# is at least this. Derived from the SCORE, deliberately NOT from the agentic
# loop's COMPLETE flag: a mid-task CONTINUE turn is not a prompt failure, and
# scoring COMPLETE-only would bias the Thompson posterior toward turn position
# rather than prompt quality (every non-terminal turn would otherwise penalize
# the served candidate). See FEP-0017.
REWARD_SUCCESS_THRESHOLD = 0.5


def reward_success_from_score(completion_score: float) -> bool:
    """Derive the per-turn prompt-quality success from the completion score.

    Returns True when the turn made at-least-mediocre progress. Centralized so
    the agentic loop and tests share one success policy.
    """
    try:
        return float(completion_score) >= REWARD_SUCCESS_THRESHOLD
    except (TypeError, ValueError):
        return False


def emit_prompt_candidate_outcome(
    identities: Iterable[Any],
    *,
    completion_score: float,
    provider: str = "",
    model: str = "",
    task_type: str = "default",
    session_id: Optional[str] = None,
) -> None:
    """Attribute a turn's outcome to the prompt candidates served this turn.

    Emits one ``PROMPT_CANDIDATE_USED`` RL event per served candidate identity
    so the prompt_optimizer learner can update each candidate's Thompson
    posterior (alpha/beta/sample_count). Non-blocking — mirrors the existing RL
    emission pattern (response_quality). No-op when no identity is supplied.

    Success is derived from ``completion_score`` (via ``reward_success_from_score``),
    NOT from whether the agentic loop reached COMPLETE — see
    ``REWARD_SUCCESS_THRESHOLD`` for the rationale.

    Args:
        identities: Served ``PromptOptimizationIdentity`` objects for the turn
            (only those carrying a ``prompt_candidate_hash`` are rewardable).
        completion_score: The turn's final completion score in [0, 1].
    """
    rewardable = [
        identity for identity in identities if getattr(identity, "prompt_candidate_hash", None)
    ]
    if not rewardable:
        return
    success = reward_success_from_score(completion_score)
    try:
        from victor.framework.rl.hooks import RLEvent, RLEventType, get_rl_hooks

        hooks = get_rl_hooks()
    except Exception:
        return
    if hooks is None:
        return
    for identity in rewardable:
        try:
            event = RLEvent(
                type=RLEventType.PROMPT_CANDIDATE_USED,
                provider=getattr(identity, "provider", None) or provider or "",
                model=model or "",
                task_type=task_type,
                success=success,
                quality_score=completion_score,
                metadata={
                    "prompt_section": getattr(identity, "prompt_section_name", None)
                    or getattr(identity, "section_name", None),
                    "prompt_candidate_hash": identity.prompt_candidate_hash,
                    "session_id": session_id,
                },
            )
            hooks.emit(event)
        except Exception as exc:
            logger.debug("prompt candidate outcome emission failed: %s", exc)
