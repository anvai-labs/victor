# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Prompt-candidate data model for prompt evolution.

Extracted from ``prompt_optimizer.py`` (which re-exports these names for
backward compatibility) so the candidate model — the evolved section text plus
its Thompson-sampling posterior and benchmark-sync records — lives apart from
the learner that manages it. Pure and stdlib-only (no learner/provider/DB
coupling), which keeps the Bayesian sampling math independently testable.

Contents:
- ``PromptCandidate`` — one evolved section candidate with Beta posteriors.
- ``PromptCandidateBenchmarkDecision`` / ``PromptCandidateBenchmarkSyncResult``
  — records of syncing a candidate-bound benchmark suite back into learner state.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PromptCandidate:
    """An evolved prompt section candidate with Bayesian scoring.

    Candidates are scoped to (section_name, provider) so each provider
    can evolve independently. A cheap model may need more explicit guidance
    while a stronger model benefits from concise prompts.
    """

    section_name: str
    text: str
    text_hash: str
    generation: int
    parent_hash: str
    provider: str = "default"  # Provider scope (e.g., "xai", "anthropic", "ollama")
    scores: Dict[str, float] = field(default_factory=dict)
    alpha: float = 1.0
    beta_val: float = 1.0
    sample_count: int = 0
    benchmark_score: float = 0.0
    benchmark_runs: int = 0
    benchmark_passed: bool = False
    is_active: bool = False
    strategy_name: str = "gepa"
    strategy_chain: str = "gepa"
    instance_scores: Dict[str, float] = field(default_factory=dict)
    coverage_count: int = 0
    is_on_frontier: bool = True
    char_length: int = 0
    requires_benchmark: bool = False
    # Independent per-instance RNG (decoupled from the global random module so
    # candidates sample independently and tests don't share global RNG state).
    _rng: random.Random = field(default_factory=random.Random, repr=False, compare=False)

    def sample(self) -> float:
        """Thompson Sampling: draw from Beta distribution with staleness decay.

        Candidates with many samples have their posteriors slightly decayed
        toward uncertainty (0.5), giving newer candidates a fair chance.
        Decay factor: 0.95^(samples/20) — halves certainty after ~280 samples.
        """
        decay = 0.95 ** (self.sample_count / 20.0)
        # Decay posteriors toward prior (1,1) — increases uncertainty
        eff_alpha = 1.0 + (self.alpha - 1.0) * decay
        eff_beta = 1.0 + (self.beta_val - 1.0) * decay
        return self._rng.betavariate(max(eff_alpha, 0.01), max(eff_beta, 0.01))

    def update(self, success: bool) -> None:
        """Update Beta posteriors."""
        if success:
            self.alpha += 1.0
        else:
            self.beta_val += 1.0
        self.sample_count += 1

    @property
    def mean(self) -> float:
        """Posterior mean."""
        return self.alpha / (self.alpha + self.beta_val)


@dataclass
class PromptCandidateBenchmarkDecision:
    """One benchmark-suite update applied to a prompt candidate."""

    prompt_candidate_hash: str
    section_name: str
    provider: str
    score: float
    passed: bool
    recorded: bool
    rank: int
    benchmark_score: float = 0.0
    benchmark_runs: int = 0
    promoted: bool = False
    # None when the suite carried no baseline arm — an observation, not a contrast.
    paired_contrast: Optional[Any] = None


@dataclass
class PromptCandidateBenchmarkSyncResult:
    """Summary of syncing a prompt-candidate benchmark suite into learner state."""

    decisions: List[PromptCandidateBenchmarkDecision] = field(default_factory=list)
    best_prompt_candidate_hash: Optional[str] = None
    approved_prompt_candidate_hash: Optional[str] = None
    promoted_prompt_candidate_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize sync results for CLI output and saved artifacts."""
        return {
            "best_prompt_candidate_hash": self.best_prompt_candidate_hash,
            "approved_prompt_candidate_hash": self.approved_prompt_candidate_hash,
            "promoted_prompt_candidate_hash": self.promoted_prompt_candidate_hash,
            "decisions": [
                {
                    "prompt_candidate_hash": decision.prompt_candidate_hash,
                    "section_name": decision.section_name,
                    "provider": decision.provider,
                    "score": decision.score,
                    "passed": decision.passed,
                    "recorded": decision.recorded,
                    "rank": decision.rank,
                    "benchmark_score": decision.benchmark_score,
                    "benchmark_runs": decision.benchmark_runs,
                    "promoted": decision.promoted,
                    "paired_contrast": (
                        decision.paired_contrast.to_dict()
                        if decision.paired_contrast is not None
                        else None
                    ),
                }
                for decision in self.decisions
            ],
        }
