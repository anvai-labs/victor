"""Skill auto-selection via hybrid embedding + edge LLM.

Matches user messages to the best skill using:
1. Embedding similarity (StaticEmbeddingCollection, ~5ms)
2. Edge LLM fallback for ambiguous matches (~500ms, optional)

Usage:
    matcher = SkillMatcher()
    await matcher.initialize(registry)
    result = await matcher.match("fix the failing test")
    if result:
        skill, score = result
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from victor_contracts.skills import SkillDefinition

logger = logging.getLogger(__name__)

PHASE_ORDER = {"diagnostic": 0, "action": 1, "verification": 2, "documentation": 3}

# Lazy imports to avoid circular deps and heavy imports at module level
FeatureFlag: Any = None
decide_sync: Any = None


def _ensure_edge_imports() -> bool:
    """Lazily import edge model dependencies."""
    global FeatureFlag, decide_sync
    if FeatureFlag is not None:
        return True
    try:
        from victor.core.feature_flags import FeatureFlag as _FF
        from victor.agent.edge_model import decide_sync as _ds

        FeatureFlag = _FF
        decide_sync = _ds
        return True
    except ImportError:
        return False


class SkillMatcher:
    """Hybrid embedding + edge LLM skill auto-selector.

    Primary: StaticEmbeddingCollection cosine similarity (~5ms, free)
    Fallback: Edge LLM decision for ambiguous zone (~500ms, optional)

    Thresholds:
        score > high_threshold  → Use skill directly (high confidence)
        low < score < high      → Edge LLM decides (if enabled)
        score < low_threshold   → No skill (normal agent behavior)
    """

    def __init__(
        self,
        high_threshold: float = 0.65,
        low_threshold: float = 0.45,
        use_edge_fallback: bool = True,
        chain_margin: float = 0.15,
    ):
        from victor.storage.embeddings.collections import StaticEmbeddingCollection

        self._collection = StaticEmbeddingCollection(
            name="skill_definitions",
            cache_dir=Path.home() / ".victor" / "embeddings",
        )
        self._skills: Dict[str, SkillDefinition] = {}
        self._high_threshold = high_threshold
        self._low_threshold = low_threshold
        self._use_edge_fallback = use_edge_fallback
        self._chain_margin = chain_margin
        self._initialized = False

    @property
    def initialized(self) -> bool:
        """Whether the matcher has loaded and embedded its skill catalog."""
        return self._initialized

    @property
    def skills(self) -> Dict[str, SkillDefinition]:
        """Return a copy of the registered skill catalog keyed by name."""
        return dict(self._skills)

    async def initialize(self, registry: Any) -> None:
        """Pre-embed all skills from the registry.

        Uses StaticEmbeddingCollection with hash-based caching —
        only recomputes embeddings when skills change.
        """
        from victor.storage.embeddings.collections import CollectionItem

        skills = registry.list_all()
        self._skills = {s.name: s for s in skills}

        items = [
            CollectionItem(
                id=skill.name,
                text=f"{skill.name}: {skill.description}. Tags: {', '.join(sorted(skill.tags))}",
                metadata={"category": skill.category},
            )
            for skill in skills
        ]

        await self._collection.initialize(items)
        self._initialized = True
        logger.info("SkillMatcher initialized with %d skills", len(skills))

    async def match(self, user_message: str) -> Optional[Tuple[SkillDefinition, float]]:
        """Find best matching skill for a user message.

        Returns:
            (skill, score) tuple or None if no match above threshold.
        """
        if not self._initialized or not self._skills:
            return None

        results = await self._collection.search(
            user_message, top_k=3, threshold=self._low_threshold
        )
        if not results:
            return None

        top_item, top_score = results[0]
        skill = self._skills.get(top_item.id)
        if not skill:
            return None

        # High confidence — use directly
        if top_score >= self._high_threshold:
            logger.debug("Skill match (high): %s score=%.3f", skill.name, top_score)
            return (skill, top_score)

        # Ambiguous zone — try edge LLM fallback
        if self._use_edge_fallback:
            edge_result = self._edge_llm_decide(user_message, results)
            if edge_result:
                return edge_result

        # Fallback: use embedding top-1 if above low threshold
        logger.debug("Skill match (embedding): %s score=%.3f", skill.name, top_score)
        return (skill, top_score)

    def match_sync(self, user_message: str) -> Optional[Tuple[SkillDefinition, float]]:
        """Synchronous skill matching using collection's sync search."""
        if not self._initialized or not self._skills:
            return None

        results = self._collection.search_sync(user_message, top_k=3, threshold=self._low_threshold)
        if not results:
            return None

        top_item, top_score = results[0]
        skill = self._skills.get(top_item.id)
        if not skill:
            return None

        if top_score >= self._high_threshold:
            return (skill, top_score)

        if self._use_edge_fallback:
            edge_result = self._edge_llm_decide(user_message, results)
            if edge_result:
                return edge_result

        return (skill, top_score)

    def match_multiple_sync(
        self,
        user_message: str,
        max_skills: int = 3,
    ) -> List[Tuple[SkillDefinition, float]]:
        """Find multiple matching skills, ordered by phase then score.

        Returns:
            List of (skill, score) tuples ordered for execution.
            Empty list if no matches above threshold.
        """
        if not self._initialized or not self._skills:
            return []

        results = self._collection.search_sync(
            user_message,
            top_k=min(max_skills + 2, 6),
            threshold=self._low_threshold,
        )
        if not results:
            return []

        # Resolve to SkillDefinition objects
        candidates = []
        for item, score in results:
            skill = self._skills.get(item.id)
            if skill:
                candidates.append((skill, score))

        if not candidates:
            return []

        # A skill selection needs a high-confidence anchor. Without one, a long
        # message full of generic terms can pull several marginal skills past
        # the low threshold and inject an irrelevant chain (seen live:
        # migrate_database → setup_ci_pipeline → optimize_performance on a
        # UX-plan continuation turn). Precision beats recall for prompt
        # injection — in the ambiguous zone, defer to the edge arbiter or
        # select nothing.
        top_score = candidates[0][1]
        if top_score < self._high_threshold:
            if self._use_edge_fallback:
                edge_result = self._edge_llm_decide(user_message, results)
                if edge_result:
                    return [edge_result]
            return []

        # Companions ride only within a fixed relevance margin of the anchor —
        # a confident top match must not drag distant also-rans along.
        chain = [
            (skill, score) for skill, score in candidates if score >= top_score - self._chain_margin
        ]
        ordered = self._order_by_phase(chain)
        return ordered[:max_skills]

    @staticmethod
    def _order_by_phase(
        candidates: List[Tuple[SkillDefinition, float]],
    ) -> List[Tuple[SkillDefinition, float]]:
        """Order skills by phase (diagnostic→action→verification→documentation),
        then by score within same phase."""
        return sorted(
            candidates,
            key=lambda pair: (
                PHASE_ORDER.get(getattr(pair[0], "phase", "action"), 1),
                -pair[1],
            ),
        )

    def _edge_llm_decide(
        self,
        user_message: str,
        candidates: List[Tuple[Any, float]],
    ) -> Optional[Tuple[SkillDefinition, float]]:
        """Use edge LLM to resolve ambiguous skill selection.

        Only called when embedding score is in the ambiguous zone
        (between low and high thresholds).
        """
        if not _ensure_edge_imports():
            return None

        try:
            if not FeatureFlag.USE_EDGE_MODEL.is_enabled():
                return None

            skill_list = "\n".join(
                f"- {item.id}: {self._skills[item.id].description}"
                for item, _score in candidates
                if item.id in self._skills
            )

            prompt = (
                f"Given these skills:\n{skill_list}\n\n"
                f"Which skill best matches this request? "
                f"Reply with just the skill name, or 'none'.\n\n"
                f"Request: {user_message}"
            )

            decision = decide_sync(prompt)
            chosen_name = decision.strip().strip("'\"").lower()

            if chosen_name != "none" and chosen_name in self._skills:
                logger.debug("Skill match (edge LLM): %s", chosen_name)
                return (self._skills[chosen_name], 0.80)
        except Exception:
            logger.debug("Edge LLM skill fallback failed", exc_info=True)

        return None
