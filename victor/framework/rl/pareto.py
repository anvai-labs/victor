"""Pareto frontier for GEPA v2 prompt candidate selection.

Implements the paper's core algorithm: candidates survive on the frontier
if they are best on at least one evaluation instance. Selection is
coverage-proportional (probability = unique wins / total instances).

Pareto dominance: A dominates B if A >= B on ALL instances AND strictly
better on at least one. A candidate is non-dominated (on the frontier)
if no other candidate dominates it.

Reference: GEPA (ICLR 2026) — arxiv:2507.19457

Current repo behavior:
- per-instance scores are updated from runtime RL outcomes and from benchmark
  artifacts only when those artifacts identify the evaluated prompt candidate
  hash explicitly
- merge is used as a fallback path when normal mutation fails to produce a
  novel candidate
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvaluationInstance:
    """A specific evaluation context for per-instance scoring.

    Instances are keyed by (tool_type, provider, section_name) so that
    GEPA can evolve prompts that specialize per-context.
    """

    instance_id: str  # e.g., "exploration::ollama::GROUNDING_RULES"
    tool_type: str  # "exploration" | "mutation" | "execution" | "analysis"
    provider: str
    section_name: str


@dataclass
class ParetoEntry:
    """A candidate on the Pareto frontier with per-instance scores."""

    text_hash: str
    text: str
    generation: int
    instance_scores: Dict[str, float] = field(default_factory=dict)
    coverage_count: int = 0  # Number of instances where this is best
    char_length: int = 0
    token_count: int = 0  # Estimated token count for efficiency tracking


class ParetoFrontier:
    """Maintains a Pareto front of prompt candidates.

    Thread-safe for single-writer usage (the prompt optimizer learner).
    """

    def __init__(self, max_candidates: int = 20):
        self._candidates: List[ParetoEntry] = []
        self._max_candidates = max_candidates
        # All known instance IDs
        self._instances: set = set()
        # Injectable, reproducible RNG (honours VICTOR_RL_SEED) instead of the
        # global random module, so Pareto sampling is deterministic when seeded
        # and never couples to other components' global RNG state.
        from victor.framework.rl.base import resolve_rl_rng_seed

        self._rng = random.Random(resolve_rl_rng_seed("pareto_frontier"))

    @property
    def size(self) -> int:
        return len(self._candidates)

    def add_candidate(
        self,
        text_hash: str,
        text: str,
        generation: int,
        instance_scores: Optional[Dict[str, float]] = None,
        token_count: int = 0,
    ) -> bool:
        """Add a candidate to the pool. Returns True if it joins the frontier.

        Rejects candidates if pool is at capacity AND the new candidate
        is dominated by all existing members.
        """
        scores = instance_scores or {}
        self._instances.update(scores.keys())

        entry = ParetoEntry(
            text_hash=text_hash,
            text=text,
            generation=generation,
            instance_scores=dict(scores),
            char_length=len(text),
            token_count=token_count,
        )

        # Check if already present
        for existing in self._candidates:
            if existing.text_hash == text_hash:
                existing.instance_scores.update(scores)
                self._recompute_coverage()
                return True

        # Check dominance — reject if dominated by everyone
        if self._candidates and all(self._dominates(c, entry) for c in self._candidates):
            return False

        self._candidates.append(entry)

        # Prune dominated candidates
        self.prune_dominated()

        # Enforce capacity limit — remove lowest-coverage candidate
        while len(self._candidates) > self._max_candidates:
            min_cov = min(self._candidates, key=lambda c: c.coverage_count)
            self._candidates.remove(min_cov)
            logger.debug(
                "Pareto capacity limit: removed gen-%d (coverage=%d)",
                min_cov.generation,
                min_cov.coverage_count,
            )

        self._recompute_coverage()
        return True

    def update_instance_score(self, text_hash: str, instance_id: str, score: float) -> None:
        """Update a candidate's score on a specific evaluation instance."""
        self._instances.add(instance_id)
        for candidate in self._candidates:
            if candidate.text_hash == text_hash:
                candidate.instance_scores[instance_id] = score
                break
        self._recompute_coverage()

    def prune_dominated(self) -> List[ParetoEntry]:
        """Remove candidates dominated by another. Return the pruned list."""
        if len(self._candidates) <= 1:
            return []

        pruned: List[ParetoEntry] = []
        surviving: List[ParetoEntry] = []

        for candidate in self._candidates:
            dominated = False
            for other in self._candidates:
                if other is candidate:
                    continue
                if self._dominates(other, candidate):
                    dominated = True
                    break
            if dominated:
                pruned.append(candidate)
            else:
                surviving.append(candidate)

        if pruned:
            self._candidates = surviving
            self._recompute_coverage()
            logger.debug(
                "Pareto pruned %d dominated candidates, %d remain",
                len(pruned),
                len(surviving),
            )
        return pruned

    def select_parent(self) -> Optional[ParetoEntry]:
        """Coverage-proportional sampling for parent selection.

        Probability of selection is proportional to the number of
        instances where this candidate is the best scorer.
        """
        if not self._candidates:
            return None

        if len(self._candidates) == 1:
            return self._candidates[0]

        # Build coverage weights
        weights = [max(c.coverage_count, 1) for c in self._candidates]
        total = sum(weights)
        if total == 0:
            return self._rng.choice(self._candidates)

        # Weighted random selection
        r = self._rng.uniform(0, total)
        cumulative = 0.0
        for candidate, w in zip(self._candidates, weights):
            cumulative += w
            if r <= cumulative:
                return candidate
        return self._candidates[-1]

    def get_frontier(self) -> List[ParetoEntry]:
        """Return all non-dominated candidates (the Pareto front)."""
        return list(self._candidates)

    def get_best_overall(self) -> Optional[ParetoEntry]:
        """Return candidate with highest average score across instances."""
        if not self._candidates:
            return None
        return max(self._candidates, key=lambda c: self._avg_score(c))

    def attempt_merge(
        self,
        gepa_service: Any,
        *,
        section_name: str = "merged",
    ) -> Optional[ParetoEntry]:
        """Pick two random frontier members and merge via LLM.

        Paper: merge combines complementary strengths from candidates
        that excel on different problem types.
        """
        if len(self._candidates) < 2:
            return None

        a, b = self._rng.sample(self._candidates, 2)
        try:
            merged_text = gepa_service.merge(a.text, b.text, section_name=section_name)
            if merged_text and merged_text != a.text and merged_text != b.text:
                import hashlib

                merged_hash = hashlib.md5(merged_text.encode(), usedforsecurity=False).hexdigest()[
                    :12
                ]
                # Inherit average scores from parents
                merged_scores: Dict[str, float] = {}
                all_instances = set(a.instance_scores.keys()) | set(b.instance_scores.keys())
                for inst in all_instances:
                    sa = a.instance_scores.get(inst, 0)
                    sb = b.instance_scores.get(inst, 0)
                    merged_scores[inst] = (sa + sb) / 2

                entry = ParetoEntry(
                    text_hash=merged_hash,
                    text=merged_text,
                    generation=max(a.generation, b.generation) + 1,
                    instance_scores=merged_scores,
                    char_length=len(merged_text),
                )
                logger.info(
                    "GEPA merge: gen-%d x gen-%d → gen-%d (%d chars)",
                    a.generation,
                    b.generation,
                    entry.generation,
                    entry.char_length,
                )
                return entry
        except Exception as e:
            logger.debug("GEPA merge failed: %s", e)
        return None

    def get_instance_winners(self) -> Dict[str, tuple[str, float]]:
        """Return the best candidate hash and score for each known instance."""
        winners: Dict[str, tuple[str, float]] = {}
        for inst in sorted(self._instances):
            best_hash: Optional[str] = None
            best_score = -1.0
            for candidate in self._candidates:
                if inst not in candidate.instance_scores:
                    continue
                score = candidate.instance_scores[inst]
                if score > best_score:
                    best_hash = candidate.text_hash
                    best_score = score
            if best_hash is not None:
                winners[inst] = (best_hash, best_score)
        return winners

    def _dominates(self, a: ParetoEntry, b: ParetoEntry) -> bool:
        """Check if candidate A Pareto-dominates candidate B.

        A dominates B iff:
        - A >= B on ALL instances where both have scores
        - A > B on at least one instance
        """
        common = set(a.instance_scores.keys()) & set(b.instance_scores.keys())
        if not common:
            return False  # Can't compare without shared instances

        all_geq = True
        any_gt = False
        for inst in common:
            sa = a.instance_scores[inst]
            sb = b.instance_scores[inst]
            if sa < sb:
                all_geq = False
                break
            if sa > sb:
                any_gt = True

        return all_geq and any_gt

    def _recompute_coverage(self) -> None:
        """Recompute coverage count for all candidates.

        Coverage = number of instances where this candidate has the best score.
        """
        # Reset
        for c in self._candidates:
            c.coverage_count = 0

        # For each instance, find the best candidate
        for inst in self._instances:
            best_score = -1.0
            best_candidate: Optional[ParetoEntry] = None
            for c in self._candidates:
                score = c.instance_scores.get(inst, -1.0)
                if score > best_score:
                    best_score = score
                    best_candidate = c
            if best_candidate is not None and best_score >= 0:
                best_candidate.coverage_count += 1

    @staticmethod
    def _avg_score(entry: ParetoEntry) -> float:
        """Average score across all instances."""
        scores = entry.instance_scores.values()
        if not scores:
            return 0.0
        return sum(scores) / len(scores)
