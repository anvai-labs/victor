# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hybrid tool selection strategy combining semantic and keyword approaches.

This module provides hybrid tool selection as part of HIGH-002:
Unified Tool Selection Architecture - Release 3, Phase 8.

Blends semantic similarity (ML-based, high quality) with keyword matching
(fast, reliable) to get best of both worlds.

RL Enhancement (Sprint 1):
Integrates ToolSelectorLearner for adaptive tool ranking based on historical
success rates. Uses contextual bandits with conservative exploration (ε=0.1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from victor.providers.base import ToolDefinition
from victor.tools.selection_filters import blend_tool_results, deduplicate_tools
from victor.config.tool_selection_defaults import HybridSelectorDefaults

if TYPE_CHECKING:
    from victor.agent.protocols import ToolSelectionContext, ToolSelectorFeatures
    from victor.framework.rl.learners.tool_selector import ToolSelectorLearner
    from victor.tools.keyword_tool_selector import KeywordToolSelector
    from victor.tools.semantic_selector import SemanticToolSelector

logger = logging.getLogger(__name__)


@dataclass
class HybridSelectorConfig:
    """Configuration for hybrid tool selector.

    Uses centralized defaults from HybridSelectorDefaults.

    Attributes:
        semantic_weight: Weight for semantic results (0.0-1.0)
        keyword_weight: Weight for keyword results (0.0-1.0)
        min_semantic_tools: Minimum tools from semantic selector
        min_keyword_tools: Minimum tools from keyword selector
        max_total_tools: Maximum total tools to return
        enable_rl: Enable RL-based tool ranking (default True)
        rl_boost_weight: Weight for RL boost in final ranking (0.0-0.5)
    """

    semantic_weight: float = HybridSelectorDefaults.SEMANTIC_WEIGHT
    keyword_weight: float = HybridSelectorDefaults.KEYWORD_WEIGHT
    min_semantic_tools: int = HybridSelectorDefaults.MIN_SEMANTIC_TOOLS
    min_keyword_tools: int = HybridSelectorDefaults.MIN_KEYWORD_TOOLS
    max_total_tools: int = HybridSelectorDefaults.MAX_TOTAL_TOOLS
    enable_rl: bool = True  # Enabled by default as per plan
    rl_boost_weight: float = HybridSelectorDefaults.RL_BOOST_WEIGHT

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.semantic_weight <= 1.0:
            raise ValueError(f"semantic_weight must be in [0.0, 1.0], got {self.semantic_weight}")
        if not 0.0 <= self.keyword_weight <= 1.0:
            raise ValueError(f"keyword_weight must be in [0.0, 1.0], got {self.keyword_weight}")
        if self.min_semantic_tools < 0:
            raise ValueError(f"min_semantic_tools must be >= 0, got {self.min_semantic_tools}")
        if self.min_keyword_tools < 0:
            raise ValueError(f"min_keyword_tools must be >= 0, got {self.min_keyword_tools}")
        if self.max_total_tools < 1:
            raise ValueError(f"max_total_tools must be >= 1, got {self.max_total_tools}")
        if not 0.0 <= self.rl_boost_weight <= 0.5:
            raise ValueError(f"rl_boost_weight must be in [0.0, 0.5], got {self.rl_boost_weight}")


class HybridToolSelector:
    """Blends semantic and keyword-based tool selection.

    Uses both semantic similarity (ML-based, high quality, 10-50ms) and
    keyword matching (registry-based, fast, <1ms) to select tools.

    Semantic results are prioritized (higher weight) but keyword results
    ensure core functionality is always available.

    Features:
    - Best of both worlds: quality + reliability
    - Configurable blending weights
    - Minimum tool guarantees per strategy
    - Supports all IToolSelector features

    HIGH-002 Release 3, Phase 8: Hybrid strategy implementation.
    """

    def __init__(
        self,
        semantic_selector: "SemanticToolSelector",
        keyword_selector: "KeywordToolSelector",
        config: Optional[HybridSelectorConfig] = None,
    ):
        """Initialize hybrid tool selector.

        Args:
            semantic_selector: Semantic tool selector instance
            keyword_selector: Keyword tool selector instance
            config: Optional hybrid selector configuration
        """
        self.semantic = semantic_selector
        self.keyword = keyword_selector
        self.config = config or HybridSelectorConfig()

        # RL learner (lazy-initialized via RLCoordinator)
        self._rl_learner: Optional["ToolSelectorLearner"] = None
        self._rl_init_attempted: bool = False

        logger.info(
            f"Initialized HybridToolSelector with semantic_weight={self.config.semantic_weight}, "
            f"keyword_weight={self.config.keyword_weight}, enable_rl={self.config.enable_rl}"
        )

    def _get_rl_learner(self) -> Optional["ToolSelectorLearner"]:
        """Get RL learner, initializing lazily via RLCoordinator.

        Returns:
            ToolSelectorLearner instance or None if RL disabled/unavailable
        """
        if not self.config.enable_rl:
            return None

        if self._rl_learner is not None:
            return self._rl_learner

        if self._rl_init_attempted:
            return None  # Already tried and failed

        self._rl_init_attempted = True
        try:
            from victor.framework.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            self._rl_learner = coordinator.get_learner("tool_selector")
            if self._rl_learner:
                logger.info("RL: ToolSelectorLearner initialized for hybrid selector")
            return self._rl_learner
        except Exception as e:
            logger.debug(f"RL: Failed to initialize tool selector learner: {e}")
            return None

    async def select_tools(
        self,
        prompt: str,
        context: "ToolSelectionContext",
    ) -> List[ToolDefinition]:
        """Select tools by blending semantic and keyword strategies with RL boost.

        Strategy:
        1. Get semantic results (ML-based, high quality)
        2. Get keyword results (registry-based, fast)
        3. Blend with configurable weights
        4. Apply RL boost (if enabled) based on learned Q-values
        5. Deduplicate
        6. Cap to max_total_tools

        Args:
            prompt: User message
            context: Tool selection context (conversation history, stage, etc.)

        Returns:
            Blended list of relevant ToolDefinition objects
        """
        # 1. Get semantic results
        semantic_tools = await self.semantic.select_tools(prompt, context)

        # 2. Get keyword results
        keyword_tools = await self.keyword.select_tools(prompt, context)

        logger.debug(
            f"Hybrid selection: {len(semantic_tools)} semantic tools, "
            f"{len(keyword_tools)} keyword tools"
        )

        # 3. Blend with weights (semantic first, then keyword)
        blended = blend_tool_results(
            semantic_tools=semantic_tools,
            keyword_tools=keyword_tools,
            semantic_weight=self.config.semantic_weight,
            keyword_weight=self.config.keyword_weight,
        )

        # 4. Apply RL boost to rerank based on learned Q-values
        task_type = getattr(context, "task_type", "default") if context else "default"
        blended = self._apply_rl_boost(blended, task_type)

        # 5. Deduplicate (blend_tool_results already does this, but be safe)
        blended = deduplicate_tools(blended)

        # 6. Ensure minimum tool requirements
        blended = self._ensure_minimum_tools(
            blended=blended,
            semantic_tools=semantic_tools,
            keyword_tools=keyword_tools,
        )

        # 7. Cap to max_total_tools
        if len(blended) > self.config.max_total_tools:
            logger.debug(
                f"Capping hybrid selection from {len(blended)} to {self.config.max_total_tools} tools"
            )
            blended = blended[: self.config.max_total_tools]

        tool_names = [t.name for t in blended]
        logger.info(f"Hybrid selection: {len(blended)} tools selected: {', '.join(tool_names)}")

        return blended

    def _apply_rl_boost(
        self,
        tools: List[ToolDefinition],
        task_type: str,
    ) -> List[ToolDefinition]:
        """Apply RL-based boost to rerank tools.

        Uses learned Q-values to boost tools with higher historical success rates.
        The boost is conservative (rl_boost_weight) to avoid disrupting good
        semantic/keyword selections.

        Args:
            tools: Current tool list (already blended)
            task_type: Task type for context-aware boosting

        Returns:
            Reranked tool list with RL boost applied
        """
        learner = self._get_rl_learner()
        if not learner or not tools:
            return tools

        try:
            # Check if we should explore (random boost) vs exploit (use Q-values)
            if learner.should_explore():
                # Exploration: small random shuffle of top tools
                import random

                top_k = HybridSelectorDefaults.EXPLORATION_TOP_K
                if len(tools) > top_k:
                    top_tools = tools[:top_k]
                    random.shuffle(top_tools)
                    tools = top_tools + tools[top_k:]
                    logger.debug(f"RL: Exploration mode - shuffled top {top_k} tools")
                return tools

            # Exploitation: boost based on learned Q-values
            tool_names = [t.name for t in tools]
            rankings = learner.get_tool_rankings(tool_names, task_type)

            if not rankings:
                return tools

            # Build Q-value lookup: {tool_name: (q_value, confidence)}
            q_lookup = {name: (q_val, conf) for name, q_val, conf in rankings}

            # Score each tool: original_position_score + rl_boost
            scored = []
            for i, tool in enumerate(tools):
                # Position score: higher = worse (1.0 for first, 0.5 for 10th, etc.)
                position_score = 1.0 / (i + 1)

                # RL boost from Q-value
                default_q = HybridSelectorDefaults.DEFAULT_Q_VALUE
                default_conf = HybridSelectorDefaults.DEFAULT_CONFIDENCE
                q_val, confidence = q_lookup.get(tool.name, (default_q, default_conf))
                rl_boost = q_val * confidence * self.config.rl_boost_weight

                # Combined score
                final_score = position_score + rl_boost
                scored.append((tool, final_score, q_val, confidence))

            # Sort by combined score (descending)
            scored.sort(key=lambda x: x[1], reverse=True)

            # Log significant reorderings
            reordered = [t[0] for t in scored]
            if reordered[:5] != tools[:5]:
                old_top5 = [t.name for t in tools[:5]]
                new_top5 = [t.name for t in reordered[:5]]
                logger.debug(f"RL: Reranked top-5 tools from {old_top5} to {new_top5}")

            return reordered

        except Exception as e:
            logger.debug(f"RL: Failed to apply boost, using original order: {e}")
            return tools

    def get_supported_features(self) -> "ToolSelectorFeatures":
        """Return features supported by hybrid tool selector.

        Hybrid selector supports all features (union of semantic + keyword).

        Returns:
            ToolSelectorFeatures with all features enabled
        """
        from victor.agent.protocols import ToolSelectorFeatures

        return ToolSelectorFeatures(
            supports_semantic_matching=True,
            supports_context_awareness=True,
            supports_cost_optimization=True,
            supports_usage_learning=True,
            supports_workflow_patterns=True,
            requires_embeddings=True,  # Semantic selector requires embeddings
        )

    def record_tool_execution(
        self,
        tool_name: str,
        success: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record tool execution (delegates to all selectors including RL).

        Both semantic and keyword selectors get execution feedback for learning.
        RL learner also records outcome for Q-value updates.

        Args:
            tool_name: Name of the tool that was executed
            success: Whether the execution succeeded
            context: Optional execution context with:
                - task_type: Task type (analysis, action, create, etc.)
                - task_completed: Whether overall task completed
                - grounding_score: Score from grounding verifier (0-1)
                - efficiency_score: Time/resource efficiency (0-1)
        """
        # Delegate to both selectors
        self.semantic.record_tool_execution(tool_name, success, context)
        self.keyword.record_tool_execution(tool_name, success, context)

        # Record to RL learner for Q-value updates
        self._record_rl_outcome(tool_name, success, context)

    def _record_rl_outcome(
        self,
        tool_name: str,
        success: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record outcome to RL learner for Q-value updates.

        Creates RLOutcome from execution context and records via coordinator.

        Args:
            tool_name: Name of the tool that was executed
            success: Whether the execution succeeded
            context: Optional execution context
        """
        learner = self._get_rl_learner()
        if not learner:
            return

        try:
            from victor.framework.rl.base import RLOutcome
            from victor.framework.rl.coordinator import get_rl_coordinator

            ctx = context or {}
            task_type = ctx.get("task_type", "default")

            # Build metadata for reward computation
            default_grounding = (
                HybridSelectorDefaults.DEFAULT_GROUNDING_SCORE_SUCCESS
                if success
                else HybridSelectorDefaults.DEFAULT_GROUNDING_SCORE_FAILURE
            )
            metadata = {
                "tool_name": tool_name,
                "tool_success": success,
                "task_completed": ctx.get("task_completed", success),
                "grounding_score": ctx.get("grounding_score", default_grounding),
                "efficiency_score": ctx.get(
                    "efficiency_score", HybridSelectorDefaults.DEFAULT_EFFICIENCY_SCORE
                ),
            }

            # Create outcome
            outcome = RLOutcome(
                provider="tool_selector",  # Not used, but required
                model="hybrid",
                task_type=task_type,
                success=success,
                quality_score=metadata["grounding_score"],
                metadata=metadata,
            )

            # Record via coordinator (ensures shared rl_outcomes table is updated)
            coordinator = get_rl_coordinator()
            coordinator.record_outcome("tool_selector", outcome, vertical="coding")

            logger.debug(
                f"RL: Recorded tool execution for '{tool_name}' "
                f"(success={success}, task_type={task_type})"
            )

        except Exception as e:
            logger.debug(f"RL: Failed to record tool outcome: {e}")

    async def close(self) -> None:
        """Cleanup resources (delegates to both selectors).

        Ensures both semantic and keyword selectors clean up properly.
        """
        # Close both selectors
        await self.semantic.close()
        await self.keyword.close()

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _ensure_minimum_tools(
        self,
        blended: List[ToolDefinition],
        semantic_tools: List[ToolDefinition],
        keyword_tools: List[ToolDefinition],
    ) -> List[ToolDefinition]:
        """Ensure minimum tool requirements are met.

        If blending doesn't include enough semantic or keyword tools,
        add more from the respective sources.

        Args:
            blended: Current blended tool list
            semantic_tools: Original semantic results
            keyword_tools: Original keyword results

        Returns:
            Blended list with minimum requirements met
        """
        existing_names = {t.name for t in blended}

        # Count how many semantic and keyword tools are in blended
        semantic_count = sum(1 for t in blended if t.name in {s.name for s in semantic_tools})
        keyword_count = sum(1 for t in blended if t.name in {k.name for k in keyword_tools})

        # Add more semantic tools if below minimum
        if semantic_count < self.config.min_semantic_tools:
            needed = self.config.min_semantic_tools - semantic_count
            for tool in semantic_tools:
                if tool.name not in existing_names:
                    blended.append(tool)
                    existing_names.add(tool.name)
                    needed -= 1
                    if needed == 0:
                        break

        # Add more keyword tools if below minimum
        if keyword_count < self.config.min_keyword_tools:
            needed = self.config.min_keyword_tools - keyword_count
            for tool in keyword_tools:
                if tool.name not in existing_names:
                    blended.append(tool)
                    existing_names.add(tool.name)
                    needed -= 1
                    if needed == 0:
                        break

        return blended
