# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for /prompt-optimize slash command inspection helpers."""

from __future__ import annotations

import io
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch

from rich.console import Console

from victor.framework.rl.learners.prompt_optimizer import (
    PromptCandidate,
    PromptOptimizerLearner,
)
from victor.ui.slash.commands.prompt_optimize import PromptOptimizeCommand
from victor.ui.slash.protocol import CommandContext


def _make_context(args):
    stream = io.StringIO()
    console = Console(file=stream, width=120, force_terminal=False, color_system=None)
    return (
        CommandContext(console=console, settings=SimpleNamespace(), agent=None, args=args),
        stream,
    )


def _make_learner():
    db = sqlite3.connect(":memory:")
    learner = PromptOptimizerLearner(name="prompt_optimizer", db_connection=db)
    first = PromptCandidate(
        section_name="GROUNDING_RULES",
        provider="default",
        text="first prompt",
        text_hash="hash0001aaaa",
        generation=1,
        parent_hash="base0000aaaa",
    )
    second = PromptCandidate(
        section_name="GROUNDING_RULES",
        provider="default",
        text="second prompt",
        text_hash="hash0002bbbb",
        generation=2,
        parent_hash="hash0001aaaa",
        is_active=True,
        benchmark_passed=True,
    )
    learner._candidates[learner._candidate_key("GROUNDING_RULES", "default")] = [
        first,
        second,
    ]
    return learner


def test_prompt_optimize_show_candidate_text():
    learner = _make_learner()
    command = PromptOptimizeCommand()
    ctx, stream = _make_context(
        ["--show", "GROUNDING_RULES", "--provider", "default", "--ordinal", "2"]
    )

    with patch(
        "victor.framework.rl.coordinator.get_rl_coordinator",
        return_value=SimpleNamespace(
            db_path="/tmp/victor.db",
            get_learner=lambda name: learner,
        ),
    ):
        command.execute(ctx)

    output = stream.getvalue()
    assert "second prompt" in output
    assert "GROUNDING_RULES" in output
    assert "hash0002" in output


def test_prompt_optimize_diff_candidates():
    learner = _make_learner()
    command = PromptOptimizeCommand()
    ctx, stream = _make_context(
        [
            "--diff",
            "GROUNDING_RULES",
            "--provider",
            "default",
            "--from",
            "1",
            "--to",
            "2",
        ]
    )

    with patch(
        "victor.framework.rl.coordinator.get_rl_coordinator",
        return_value=SimpleNamespace(
            db_path="/tmp/victor.db",
            get_learner=lambda name: learner,
        ),
    ):
        command.execute(ctx)

    output = stream.getvalue()
    assert "--- GROUNDING_RULES:default:1" in output
    assert "+++ GROUNDING_RULES:default:2" in output
    assert "-first prompt" in output
    assert "+second prompt" in output


def test_prompt_optimize_baseline_map_includes_scoped_sections():
    from victor.agent.prompt_section_registry import get_section_registry

    section_text = PromptOptimizeCommand._get_section_text()
    registry_evolvable = get_section_registry().get_evolvable_sections()

    assert "CONCISE_MODE_GUIDANCE" in section_text
    assert section_text["CONCISE_MODE_GUIDANCE"]
    assert "PARALLEL_READ_GUIDANCE" in section_text
    assert section_text["PARALLEL_READ_GUIDANCE"]
    assert "LARGE_FILE_PAGINATION_GUIDANCE" in section_text
    assert section_text["LARGE_FILE_PAGINATION_GUIDANCE"]
    assert "GROUNDING_RULES_EXTENDED" in section_text
    assert section_text["GROUNDING_RULES_EXTENDED"]
    assert registry_evolvable.issubset(set(section_text))


def test_prompt_optimize_uses_active_session_provider_for_evolution():
    learner = _make_learner()
    learner.evolve = (
        lambda section, current, provider="default", query=None, on_phase=None: PromptCandidate(
            section_name=section,
            provider=provider,
            text=current + " evolved",
            text_hash="hash9999cccc",
            generation=3,
            parent_hash="hash0002bbbb",
        )
    )
    command = PromptOptimizeCommand()
    stream = io.StringIO()
    console = Console(file=stream, width=120, force_terminal=False, color_system=None)
    ctx = CommandContext(
        console=console,
        settings=SimpleNamespace(
            provider=SimpleNamespace(default_provider="openai", default_model="gpt-5")
        ),
        agent=SimpleNamespace(provider_name="zai", model="glm-5.1"),
        args=["GROUNDING"],
    )

    with patch(
        "victor.framework.rl.coordinator.get_rl_coordinator",
        return_value=SimpleNamespace(
            db_path="/tmp/victor.db",
            get_learner=lambda name: learner,
        ),
    ):
        command.execute(ctx)

    output = stream.getvalue()
    assert "zai" in output
    assert "glm-5.1" in output


def test_prompt_optimize_show_defaults_to_active_session_provider():
    learner = _make_learner()
    learner._candidates[learner._candidate_key("GROUNDING_RULES", "anthropic")] = [
        PromptCandidate(
            section_name="GROUNDING_RULES",
            provider="anthropic",
            text="anthropic prompt",
            text_hash="anthropic1234",
            generation=2,
            parent_hash="base0000aaaa",
            is_active=True,
            benchmark_passed=True,
        )
    ]
    command = PromptOptimizeCommand()
    stream = io.StringIO()
    console = Console(file=stream, width=120, force_terminal=False, color_system=None)
    ctx = CommandContext(
        console=console,
        settings=SimpleNamespace(
            provider=SimpleNamespace(default_provider="openai", default_model="gpt-5")
        ),
        agent=SimpleNamespace(provider_name="anthropic", model="claude-sonnet-4-6"),
        args=["--show", "GROUNDING_RULES", "--ordinal", "2"],
    )

    with patch(
        "victor.framework.rl.coordinator.get_rl_coordinator",
        return_value=SimpleNamespace(
            db_path="/tmp/victor.db",
            get_learner=lambda name: learner,
        ),
    ):
        command.execute(ctx)

    output = stream.getvalue()
    assert "anthropic prompt" in output
