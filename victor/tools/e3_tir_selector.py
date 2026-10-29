"""E3-TIR Tool Selector — Enhanced Experience Exploitation for tool selection.

Extends HybridToolSelector with three experience-based strategies:
1. **Demonstration warm-up**: Bootstrap Q-values from expert trajectories
2. **Self-play exploration**: Agent discovers tool combos autonomously
3. **Targeted exploration**: Actively probe under-utilized or stale tools

Dynamic scheduling adjusts the mix based on session phase:
- Early session: heavy demonstration + targeted exploration
- Mid session: self-play dominant (agent learns from own attempts)
- Late session: exploitation dominant (use learned Q-values)

Related research:
- Experience replay for LLM reasoning (arXiv:2507.07451)
- Mode collapse prevention in LLM agents (DAPO, DrGRPO)
- Warm-up paradigms for tool-integrated reasoning
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from victor.tools.experience_store import (
    ExperienceType,
    ToolExperience,
    ToolExperienceStore,
)

logger = logging.getLogger(__name__)


@dataclass
class E3TIRConfig:
    """Configuration for the E3-TIR selector."""

    # Demonstration warm-up
    demonstration_boost: float = 0.3  # Q-value boost for demonstrated tools
    min_demonstrations: int = 3  # Min demonstrations before trusting demo signal
    # R4: tool *success* is owned by the RL Q-value (hybrid_tool_selector); E3-TIR is
    # exploration scheduling only. Demonstration-reward reranking would re-score the
    # same success signal a second time (double-count), so it is OFF by default.
    enable_demonstration_reranking: bool = False

    # Self-play
    self_play_epsilon: float = 0.15  # Exploration rate during self-play
    self_play_novelty_bonus: float = 0.1  # Bonus for trying new tool combos

    # Targeted exploration
    underutilized_threshold: int = 5  # Uses below this = underutilized
    staleness_threshold: float = 3600.0  # Seconds before tool is "stale"
    exploration_slots: int = 2  # Max exploration tools to inject per selection

    # Dynamic scheduling
    warmup_turns: int = 5  # Turns of heavy demonstration/exploration
    exploitation_ramp: float = 0.1  # How fast to shift to exploitation (per turn)

    # Mode collapse prevention
    diversity_floor: float = 0.3  # Min diversity score before forcing exploration
    max_consecutive_same: int = 3  # Max times same tool can be picked consecutively


@dataclass
class SelectionPhase:
    """Current phase of the dynamic scheduler."""

    turn_count: int = 0
    demo_weight: float = 0.4  # Weight for demonstration influence
    self_play_weight: float = 0.3  # Weight for self-play exploration
    exploration_weight: float = 0.3  # Weight for targeted exploration

    def advance(self, config: E3TIRConfig) -> None:
        """Advance phase by one turn. Shifts toward exploitation."""
        self.turn_count += 1
        ramp = config.exploitation_ramp

        if self.turn_count <= config.warmup_turns:
            # Warm-up: heavy demo + exploration
            self.demo_weight = max(0.1, 0.4 - self.turn_count * ramp * 0.5)
            self.exploration_weight = max(0.1, 0.3 - self.turn_count * ramp * 0.3)
            self.self_play_weight = 1.0 - self.demo_weight - self.exploration_weight
        else:
            # Post-warmup: exploitation dominant
            self.demo_weight = 0.05
            self.exploration_weight = 0.1
            self.self_play_weight = 0.85


class E3TIRToolSelector:
    """Tool selector with E3-TIR experience exploitation.

    Wraps any base tool selection (semantic, hybrid, etc.) and applies
    experience-based reranking and exploration injection.

    Usage:
        store = ToolExperienceStore()
        selector = E3TIRToolSelector(store=store)

        # Select tools with experience-aware ranking
        tools = selector.select(available_tools, task_type, user_message)

        # Record outcome for learning
        selector.record_outcome("read_file", "coding", success=True, reward=0.8)
    """

    def __init__(
        self,
        store: Optional[ToolExperienceStore] = None,
        config: Optional[E3TIRConfig] = None,
    ) -> None:
        self._store = store or ToolExperienceStore()
        self._config = config or E3TIRConfig()
        self._phase = SelectionPhase()
        self._consecutive_tool: Dict[str, int] = {}  # tool -> consecutive count
        self._last_selected: Optional[str] = None

    @property
    def store(self) -> ToolExperienceStore:
        return self._store

    @property
    def phase(self) -> SelectionPhase:
        return self._phase

    def select(
        self,
        available_tools: List[str],
        task_type: str = "general",
        user_message: str = "",
        base_ranking: Optional[List[str]] = None,
        max_tools: int = 10,
    ) -> List[str]:
        """Select tools using E3-TIR experience exploitation.

        Args:
            available_tools: All available tool names
            task_type: Current task type for context
            user_message: User's message (for context)
            base_ranking: Optional pre-ranked list from semantic/hybrid selector
            max_tools: Maximum tools to return

        Returns:
            Reranked list of tool names
        """
        self._store.register_tools(available_tools)

        # Start from base ranking or available tools
        candidates = list(base_ranking or available_tools)

        # 1. Apply demonstration warm-up boost (R4: OFF by default — tool success is
        #    the RL Q-value's job in hybrid_tool_selector; reranking by demonstration
        #    reward here would double-count the same signal).
        if self._config.enable_demonstration_reranking:
            candidates = self._apply_demonstration_boost(candidates, task_type)

        # 2. Apply self-play exploration
        candidates = self._apply_self_play(candidates, task_type)

        # 3. Inject targeted exploration tools
        candidates = self._inject_exploration_tools(candidates, available_tools)

        # 4. Apply mode collapse prevention
        candidates = self._prevent_mode_collapse(candidates, available_tools)

        # Advance phase for next turn
        self._phase.advance(self._config)

        return candidates[:max_tools]

    def record_outcome(
        self,
        tool_name: str,
        task_type: str,
        success: bool,
        reward: float,
        experience_type: ExperienceType = ExperienceType.SELF_PLAY,
    ) -> None:
        """Record a tool execution outcome."""
        self._store.record_outcome(
            tool_name=tool_name,
            task_type=task_type,
            success=success,
            reward=reward,
            experience_type=experience_type,
        )

        # Track consecutive usage
        if tool_name == self._last_selected:
            self._consecutive_tool[tool_name] = self._consecutive_tool.get(tool_name, 0) + 1
        else:
            self._consecutive_tool = {tool_name: 1}
        self._last_selected = tool_name

    def add_demonstration(
        self,
        tool_name: str,
        task_type: str,
        reward: float = 1.0,
        trajectory: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add an expert demonstration for warm-up."""
        self._store.record(
            ToolExperience(
                tool_name=tool_name,
                task_type=task_type,
                experience_type=ExperienceType.DEMONSTRATION,
                success=True,
                reward=reward,
                trajectory=trajectory or [],
            )
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get E3-TIR metrics for monitoring."""
        return {
            "diversity_score": self._store.get_diversity_score(),
            "total_experiences": len(self._store),
            "underutilized_tools": len(
                self._store.get_underutilized_tools(self._config.underutilized_threshold)
            ),
            "stale_tools": len(self._store.get_stale_tools(self._config.staleness_threshold)),
            "phase": {
                "turn": self._phase.turn_count,
                "demo_weight": round(self._phase.demo_weight, 3),
                "self_play_weight": round(self._phase.self_play_weight, 3),
                "exploration_weight": round(self._phase.exploration_weight, 3),
            },
        }

    # ── Experience strategies ─────────────────────────────────────────

    def _apply_demonstration_boost(self, candidates: List[str], task_type: str) -> List[str]:
        """Boost tools that have strong demonstration evidence."""
        if self._phase.demo_weight < 0.05:
            return candidates  # Skip when demo influence is negligible

        demos = self._store.sample_experiences(
            experience_type=ExperienceType.DEMONSTRATION,
            task_type=task_type,
            limit=50,
        )

        if not demos:
            return candidates

        # Build demo score: tool → avg demo reward
        demo_scores: Dict[str, float] = {}
        demo_counts: Dict[str, int] = {}
        for d in demos:
            demo_scores[d.tool_name] = demo_scores.get(d.tool_name, 0) + d.reward
            demo_counts[d.tool_name] = demo_counts.get(d.tool_name, 0) + 1

        demo_avg = {
            name: demo_scores[name] / demo_counts[name]
            for name in demo_scores
            if demo_counts[name] >= self._config.min_demonstrations
        }

        if not demo_avg:
            return candidates

        # Rerank: move demo-boosted tools up
        boost = self._config.demonstration_boost * self._phase.demo_weight

        def sort_key(tool: str) -> float:
            idx = candidates.index(tool) if tool in candidates else len(candidates)
            position_score = 1.0 / (idx + 1)
            demo_bonus = demo_avg.get(tool, 0.0) * boost
            return -(position_score + demo_bonus)

        return sorted(candidates, key=sort_key)

    def _apply_self_play(self, candidates: List[str], task_type: str) -> List[str]:
        """Apply self-play exploration via epsilon-greedy with novelty bonus."""
        if self._phase.self_play_weight < 0.05:
            return candidates

        epsilon = self._config.self_play_epsilon * self._phase.self_play_weight

        if random.random() < epsilon:
            # Explore: shuffle a portion of candidates
            n = min(5, len(candidates))
            top = candidates[:n]
            random.shuffle(top)
            candidates = top + candidates[n:]
            logger.debug("E3-TIR: Self-play exploration shuffle applied")

        return candidates

    def _inject_exploration_tools(self, candidates: List[str], available: List[str]) -> List[str]:
        """Inject under-utilized or stale tools for targeted exploration."""
        if self._phase.exploration_weight < 0.05:
            return candidates

        underutilized = self._store.get_underutilized_tools(self._config.underutilized_threshold)
        stale = self._store.get_stale_tools(self._config.staleness_threshold)

        # Combine and filter to available tools
        explore_candidates = set(underutilized + stale) & set(available)
        explore_candidates -= set(candidates[: self._config.exploration_slots])

        if not explore_candidates:
            return candidates

        # Pick exploration tools (weighted by staleness)
        to_inject = random.sample(
            sorted(explore_candidates),
            min(self._config.exploration_slots, len(explore_candidates)),
        )

        # Inject near the end of the candidate list (don't displace top picks)
        inject_pos = max(3, len(candidates) - self._config.exploration_slots)
        for tool in to_inject:
            if tool not in candidates:
                candidates.insert(inject_pos, tool)
                logger.debug("E3-TIR: Injected exploration tool '%s'", tool)

        return candidates

    def _prevent_mode_collapse(self, candidates: List[str], available: List[str]) -> List[str]:
        """Prevent mode collapse by enforcing diversity floor.

        If the same tool has been selected too many times consecutively,
        or if overall diversity is below the floor, force diversification.
        """
        # Check consecutive usage
        for tool, count in self._consecutive_tool.items():
            if count >= self._config.max_consecutive_same and tool in candidates[:3]:
                # Move the over-used tool down
                candidates.remove(tool)
                candidates.insert(min(5, len(candidates)), tool)
                logger.debug("E3-TIR: Demoted '%s' (used %d consecutive times)", tool, count)

        # Check diversity floor
        diversity = self._store.get_diversity_score()
        if diversity < self._config.diversity_floor and len(available) > 3:
            # Force inject a rarely-used tool at position 2-3
            underutilized = self._store.get_underutilized_tools(
                self._config.underutilized_threshold
            )
            for tool in underutilized:
                if tool in available and tool not in candidates[:3]:
                    candidates.insert(2, tool)
                    logger.debug(
                        "E3-TIR: Diversity injection '%s' (diversity=%.2f)",
                        tool,
                        diversity,
                    )
                    break

        return candidates
