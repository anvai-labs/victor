# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for prompt-candidate binding: baseline sentinel + provider fallback."""

from types import SimpleNamespace

from victor.agent.optimization_injector import BASELINE_CANDIDATE_HASH, OptimizationInjector

SECTION = "ASI_TOOL_EFFECTIVENESS_GUIDANCE"


def _seed_text() -> str:
    from victor.agent.prompt_section_registry import get_section_registry

    return next(s.default_text for s in get_section_registry().get_all() if s.name == SECTION)


def test_baseline_sentinel_resolves_to_seed():
    """Binding the __baseline__ sentinel serves the section's seed text (no
    stored candidate, no Thompson pollution) — enables evolved-vs-seed."""
    inj = OptimizationInjector()
    inj.bind_prompt_candidate(
        section_name=SECTION,
        prompt_candidate_hash=BASELINE_CANDIDATE_HASH,
        provider="zai",
        strict=True,
    )
    payload = inj._resolve_bound_candidate_payload(SECTION, "zai")

    assert payload is not None
    assert payload["source"] == "baseline"
    assert payload["prompt_candidate_hash"] == BASELINE_CANDIDATE_HASH
    assert payload["text"] == _seed_text()


def test_provider_fallback_finds_cross_provider_candidate(monkeypatch):
    """A default-profile run (e.g. ollama) binding a candidate stored under a
    different provider (e.g. zai) must resolve via the cross-provider fallback."""
    inj = OptimizationInjector()
    candidate = SimpleNamespace(
        text="evolved",
        text_hash="h1",
        section_name=SECTION,
        provider="zai",
        strategy_name="gepa",
        strategy_chain="gepa",
    )
    fake_learner = SimpleNamespace(
        get_candidate=lambda **kw: None,  # provider-specific miss
        find_candidate_any_provider=lambda **kw: candidate,  # fallback hit
    )
    fake_coord = SimpleNamespace(get_learner=lambda name: fake_learner)
    monkeypatch.setattr("victor.agent.services.rl_runtime.get_rl_coordinator", lambda: fake_coord)

    inj.bind_prompt_candidate(
        section_name=SECTION,
        prompt_candidate_hash="h1",
        provider="ollama",  # wrong provider on purpose
        strict=True,
    )
    payload = inj._resolve_bound_candidate_payload(SECTION, "ollama")

    assert payload is not None
    assert payload["prompt_candidate_hash"] == "h1"
    assert payload["provider"] == "zai"  # found via cross-provider fallback
    assert payload["source"] == "bound_candidate"


def test_missing_candidate_returns_none_when_not_strict(monkeypatch):
    """Non-strict binding of a missing candidate resolves to None (graceful)."""
    inj = OptimizationInjector()
    fake_learner = SimpleNamespace(
        get_candidate=lambda **kw: None,
        find_candidate_any_provider=lambda **kw: None,
    )
    fake_coord = SimpleNamespace(get_learner=lambda name: fake_learner)
    monkeypatch.setattr("victor.agent.services.rl_runtime.get_rl_coordinator", lambda: fake_coord)

    inj.bind_prompt_candidate(
        section_name=SECTION,
        prompt_candidate_hash="nope",
        provider="ollama",
        strict=False,
    )
    assert inj._resolve_bound_candidate_payload(SECTION, "ollama") is None


def test_missing_candidate_raises_when_strict(monkeypatch):
    """Strict binding of a missing candidate raises (no silent false-positive)."""
    import pytest

    inj = OptimizationInjector()
    fake_learner = SimpleNamespace(
        get_candidate=lambda **kw: None,
        find_candidate_any_provider=lambda **kw: None,
    )
    fake_coord = SimpleNamespace(get_learner=lambda name: fake_learner)
    monkeypatch.setattr("victor.agent.services.rl_runtime.get_rl_coordinator", lambda: fake_coord)

    with pytest.raises(ValueError):
        inj.bind_prompt_candidate(
            section_name=SECTION,
            prompt_candidate_hash="nope",
            provider="ollama",
            strict=True,  # resolves at bind time → raises
        )
