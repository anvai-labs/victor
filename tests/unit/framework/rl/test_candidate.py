# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the extracted prompt-candidate data model and its re-export shim."""

from victor.framework.rl.learners import candidate as cm
from victor.framework.rl.learners.candidate import (
    PromptCandidate,
    PromptCandidateBenchmarkDecision,
    PromptCandidateBenchmarkSyncResult,
)


def _candidate(**kw):
    base = {
        "section_name": "GROUNDING_RULES",
        "text": "t",
        "text_hash": "h",
        "generation": 0,
        "parent_hash": "",
    }
    base.update(kw)
    return PromptCandidate(**base)


class TestReExportBackCompat:
    def test_symbols_reexported_with_identity(self):
        from victor.framework.rl.learners import prompt_optimizer as po

        for name in (
            "PromptCandidate",
            "PromptCandidateBenchmarkDecision",
            "PromptCandidateBenchmarkSyncResult",
        ):
            assert getattr(po, name) is getattr(cm, name), name


class TestPosteriorMath:
    def test_update_moves_mean(self):
        c = _candidate()
        assert c.mean == 0.5  # prior alpha=beta=1
        c.update(True)  # alpha=2, beta=1
        assert c.mean == 2 / 3
        c.update(False)  # alpha=2, beta=2
        assert c.mean == 0.5
        assert c.sample_count == 2

    def test_sample_is_bounded_and_seedable(self):
        c = _candidate()
        # Inject a seeded RNG so the draw is deterministic and reproducible.
        import random

        c._rng = random.Random(1234)
        draw = c.sample()
        assert 0.0 <= draw <= 1.0
        c2 = _candidate()
        c2._rng = random.Random(1234)
        assert c2.sample() == draw

    def test_staleness_decay_shrinks_effective_confidence(self):
        # A high-sample, high-alpha candidate should have its posterior pulled
        # back toward the prior — the mean of many samples stays below the
        # undecayed alpha/(alpha+beta).
        strong = _candidate(alpha=50.0, beta_val=2.0, sample_count=500)
        rng_draws = []
        import random

        strong._rng = random.Random(0)
        for _ in range(2000):
            rng_draws.append(strong.sample())
        avg = sum(rng_draws) / len(rng_draws)
        undecayed = 50.0 / 52.0
        assert avg < undecayed  # decay increased uncertainty


class TestBenchmarkSyncResult:
    def test_to_dict_roundtrips_decisions(self):
        decision = PromptCandidateBenchmarkDecision(
            prompt_candidate_hash="abc",
            section_name="GROUNDING_RULES",
            provider="ollama",
            score=0.8,
            passed=True,
            recorded=True,
            rank=1,
            benchmark_score=0.8,
            benchmark_runs=3,
            promoted=False,
            paired_contrast=None,
        )
        result = PromptCandidateBenchmarkSyncResult(
            decisions=[decision], best_prompt_candidate_hash="abc"
        )
        payload = result.to_dict()
        assert payload["best_prompt_candidate_hash"] == "abc"
        assert payload["decisions"][0]["prompt_candidate_hash"] == "abc"
        assert payload["decisions"][0]["paired_contrast"] is None
