# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for prompt-candidate reward attribution (FEP-0017)."""

from unittest.mock import MagicMock

import pytest

from victor.agent.services.prompt_optimization_reward import (
    REWARD_SUCCESS_THRESHOLD,
    emit_prompt_candidate_outcome,
    reward_success_from_score,
)
from victor.agent.services.runtime_intelligence import PromptOptimizationIdentity


def test_reward_success_from_score_threshold():
    """Success is derived from the completion score, not the COMPLETE flag."""
    assert reward_success_from_score(0.0) is False
    assert reward_success_from_score(0.49) is False
    assert reward_success_from_score(REWARD_SUCCESS_THRESHOLD) is True
    assert reward_success_from_score(0.85) is True
    # Robust to bad input.
    assert reward_success_from_score(None) is False  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "score,expected_success",
    [(0.85, True), (0.9, True), (0.3, False), (0.1, False)],
)
def test_emitted_success_follows_score_not_caller(monkeypatch, score, expected_success):
    """The emitted event's success is derived from the score — a low-progress
    turn must not register as success regardless of what the caller 'feels'
    (guards against the turn-position bias fixed in FEP-0017)."""
    captured: list = []
    fake_hooks = MagicMock()
    fake_hooks.emit = lambda ev: captured.append(ev)
    monkeypatch.setattr("victor.framework.rl.hooks.get_rl_hooks", lambda: fake_hooks)

    emit_prompt_candidate_outcome(
        [
            PromptOptimizationIdentity(
                provider="zai",
                prompt_candidate_hash="h1",
                section_name="GROUNDING_RULES",
                prompt_section_name="GROUNDING_RULES",
            )
        ],
        completion_score=score,
    )

    assert len(captured) == 1
    assert captured[0].success is expected_success
    assert captured[0].quality_score == score


def test_emits_one_event_per_rewardable_identity(monkeypatch):
    """One PROMPT_CANDIDATE_USED event per served identity carrying the outcome."""
    from victor.framework.rl.hooks import RLEventType

    identities = [
        PromptOptimizationIdentity(
            provider="zai",
            prompt_candidate_hash="h1",
            section_name="GROUNDING_RULES",
            prompt_section_name="GROUNDING_RULES",
        ),
        PromptOptimizationIdentity(
            provider="zai",
            prompt_candidate_hash="h2",
            section_name="COMPLETION_GUIDANCE",
            prompt_section_name="COMPLETION_GUIDANCE",
        ),
        # No hash → not rewardable, must be skipped.
        PromptOptimizationIdentity(provider="zai", section_name="X"),
    ]

    captured: list = []
    fake_hooks = MagicMock()
    fake_hooks.emit = lambda ev: captured.append(ev)
    monkeypatch.setattr("victor.framework.rl.hooks.get_rl_hooks", lambda: fake_hooks)

    emit_prompt_candidate_outcome(
        identities,
        completion_score=0.85,
        provider="zai",
        model="glm-5.2",
        task_type="action",
        session_id="sess-1",
    )

    assert len(captured) == 2  # only the two with hashes
    emitted_hashes = set()
    for ev in captured:
        assert ev.type is RLEventType.PROMPT_CANDIDATE_USED
        assert ev.success is True  # 0.85 >= threshold
        assert ev.quality_score == 0.85
        assert ev.metadata["session_id"] == "sess-1"
        emitted_hashes.add(ev.metadata["prompt_candidate_hash"])
    assert emitted_hashes == {"h1", "h2"}


def test_noop_without_rewardable_identities():
    """No identities (or none with a hash) → no emission, no raise."""
    emit_prompt_candidate_outcome([], completion_score=0.5)
    emit_prompt_candidate_outcome(
        [PromptOptimizationIdentity(provider="zai", section_name="X")],
        completion_score=0.5,
    )
