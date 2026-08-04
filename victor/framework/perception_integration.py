# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Perception Integration - Reuses existing Victor components.

This module provides a lightweight integration layer that combines:
- ActionIntent detection (victor/agent/action_authorizer.py)
- Task complexity analysis (victor/agent/task_analyzer.py)
- Requirement extraction (new feature)
- Context retrieval (victor/storage/memory/unified.py)

Design Principle: EXTEND existing components, don't duplicate.

Based on research from:
- arXiv:2603.07379 - Agentic RAG Taxonomy
- arXiv:2601.03192 - MemRL (Episodic Memory)

Example:
    from victor.framework.perception_integration import (
        PerceptionIntegration,
        Perception,
    )

    integration = PerceptionIntegration(
        memory_coordinator=unified_memory,
        enable_requirement_extraction=True,
    )

    result = await integration.perceive(
        query="Fix the authentication bug",
        context={"project": "myapp"}
    )
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.agent.action_authorizer import (
    ActionIntent,
    IntentClassification,
    detect_intent,
)
from victor.agent.task_analyzer import TaskAnalysis, TaskAnalyzer
from victor.framework.request_scope_heuristics import (
    contains_keyword_marker,
    conversation_history_has_explicit_target,
    has_ambiguous_target_reference,
)
from victor.framework.runtime_evaluation_policy import RuntimeEvaluationPolicy
from victor.framework.task.protocols import TaskComplexity

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.storage.memory.unified import UnifiedMemoryCoordinator


class RequirementType(Enum):
    """Types of requirements extracted from user queries."""

    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    CONSTRAINT = "constraint"
    PREFERENCE = "preference"
    QUALITY = "quality"


@dataclass
class Requirement:
    """A requirement extracted from user query."""

    type: RequirementType
    description: str
    priority: int = 3
    source: str = "explicit"
    acceptance_criteria: Optional[str] = None


@dataclass
class SimilarExperience:
    """Similar past experience from memory."""

    task_id: str
    description: str
    similarity_score: float
    outcome: bool
    lessons_learned: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)


@dataclass
class Perception:
    """Integrated perception result.

    Combines ActionIntent, TaskComplexity, and extracted requirements.
    Enhanced with TaskAnalysis rich fields (budget, coordination hints).
    """

    # From existing components
    intent: ActionIntent
    complexity: TaskComplexity
    task_analysis: TaskAnalysis

    # New features
    requirements: List[Requirement] = field(default_factory=list)
    similar_experiences: List[SimilarExperience] = field(default_factory=list)

    # Metadata
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    needs_clarification: bool = False
    clarification_reason: Optional[str] = None
    clarification_prompt: Optional[str] = None

    @property
    def tool_budget(self) -> Optional[int]:
        """Tool budget from TaskAnalysis, if available."""
        if self.task_analysis and hasattr(self.task_analysis, "tool_budget"):
            return self.task_analysis.tool_budget
        return None

    @property
    def should_spawn_team(self) -> bool:
        """Whether this task warrants multi-agent coordination."""
        if self.task_analysis and hasattr(self.task_analysis, "should_spawn_team"):
            return self.task_analysis.should_spawn_team
        return False

    @property
    def coordination_suggestion(self) -> Optional[str]:
        """Team formation suggestion from TaskAnalysis."""
        if self.task_analysis and hasattr(self.task_analysis, "coordination_suggestion"):
            return self.task_analysis.coordination_suggestion
        return None

    @property
    def task_type(self) -> str:
        """Map intent to task type string."""
        return self.intent.value

    @property
    def primary_intent(self) -> str:
        """Get primary intent as string."""
        if self.task_analysis:
            return self.task_analysis.task_type or self.intent.value
        return ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "intent": self.intent.value,
            "complexity": self.complexity.value,
            "task_type": self.task_type,
            "primary_intent": self.primary_intent,
            "tool_budget": self.tool_budget,
            "should_spawn_team": self.should_spawn_team,
            "coordination_suggestion": self.coordination_suggestion,
            "requirements": [
                {
                    "type": r.type.value,
                    "description": r.description,
                    "priority": r.priority,
                    "source": r.source,
                    "acceptance_criteria": r.acceptance_criteria,
                }
                for r in self.requirements
            ],
            "similar_experiences_count": len(self.similar_experiences),
            "confidence": self.confidence,
            "metadata": self.metadata,
            "needs_clarification": self.needs_clarification,
            "clarification_reason": self.clarification_reason,
            "clarification_prompt": self.clarification_prompt,
        }


class PerceptionIntegration:
    """Integration layer for perception using existing Victor components.

    Reuses:
    - ActionIntent detection (action_authorizer.py)
    - Task complexity analysis (task_analyzer.py)
    - Unified memory coordinator (unified.py)

    Adds:
    - Requirement extraction (new)
    - Similar experience retrieval (new)
    """

    def __init__(
        self,
        memory_coordinator: Optional["UnifiedMemoryCoordinator"] = None,
        enable_requirement_extraction: bool = True,
        enable_similarity_search: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize perception integration.

        Args:
            memory_coordinator: Unified memory for context retrieval
            enable_requirement_extraction: Extract requirements from queries
            enable_similarity_search: Search for similar past experiences
            config: Additional configuration
        """
        self.memory_coordinator = memory_coordinator
        self.enable_requirement_extraction = enable_requirement_extraction
        self.enable_similarity_search = enable_similarity_search
        self.config = config or {}
        self._evaluation_policy = RuntimeEvaluationPolicy.from_config(self.config)

        logger.info(
            f"PerceptionIntegration initialized "
            f"(requirements={enable_requirement_extraction}, "
            f"similarity={enable_similarity_search})"
        )

    async def perceive(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Perception:
        """Perceive user intent using existing Victor components.

        Supports multi-turn context via conversation_history parameter,
        inspired by Collaborative Proactive Management (arXiv:2510.05110).

        Args:
            query: User's natural language query
            context: Additional context (project, files, etc.)
            conversation_history: Prior turns for multi-turn context

        Returns:
            Perception object with integrated understanding
        """
        logger.debug(f"Perceiving: {query[:100]}...")

        # Spin telemetry (loop guards): perception is called exactly once per loop
        # iteration in healthy runs, so counting here — the single seam both the
        # legacy and StateGraph loops share — surfaces a perception/restart spin
        # the iteration counter and post-ACT SpinDetector cannot see.
        from victor.framework.loop.guards import note_perception

        note_perception()

        # REUSE: Detect intent using existing ActionIntent detector
        intent_result = detect_intent(query)
        intent = intent_result.intent

        # REUSE: Analyze task complexity using existing TaskAnalyzer.
        # Offload to a worker thread: analyze() → classify_sync → embed_text_sync
        # (model.encode) is CPU-bound and SYNC; running it inline blocks the event
        # loop, which is why per-turn/total timeouts could never fire on a spinning
        # loop. to_thread keeps the loop responsive so deadlines and cancellation
        # work. The embedding already guards model.encode with a threading.Lock,
        # so offloading is thread-safe.
        analyzer = TaskAnalyzer()
        analysis_context = dict(context or {})
        if conversation_history:
            analysis_context["history"] = conversation_history
        task_analysis = await asyncio.to_thread(analyzer.analyze, query, context=analysis_context)
        complexity = task_analysis.complexity

        # NEW: Extract requirements (if enabled)
        requirements = []
        if self.enable_requirement_extraction:
            requirements = self._extract_requirements(query, context)

        # NEW: Retrieve similar experiences (if enabled)
        similar_experiences = []
        if self.enable_similarity_search and self.memory_coordinator:
            similar_experiences = await self._retrieve_similar_experiences(query, intent, context)

        # Calculate calibrated confidence
        confidence = self._calculate_confidence(
            intent,
            complexity,
            requirements,
            similar_experiences,
            intent_confidence=intent_result.confidence,
        )
        clarification = self._assess_clarification_need(
            query=query,
            task_analysis=task_analysis,
            confidence=confidence,
            context=context,
            conversation_history=conversation_history,
        )

        perception = Perception(
            intent=intent,
            complexity=complexity,
            task_analysis=task_analysis,
            requirements=requirements,
            similar_experiences=similar_experiences,
            confidence=confidence,
            metadata={
                "query_length": len(query),
                "has_context": context is not None,
            },
            needs_clarification=clarification["needs_clarification"],
            clarification_reason=clarification["clarification_reason"],
            clarification_prompt=clarification["clarification_prompt"],
        )

        logger.debug(
            f"Perception complete: intent={intent.value}, "
            f"complexity={complexity.value}, "
            f"confidence={confidence:.2f}"
        )

        return perception

    def _assess_clarification_need(
        self,
        query: str,
        task_analysis: TaskAnalysis,
        confidence: float,
        context: Optional[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Optional[str] | bool]:
        """Detect when execution confidence is too weak to proceed safely."""
        message = query.strip()
        if not message:
            return self._evaluation_policy.empty_request_decision().to_dict()

        message_lower = message.lower()
        action_markers = (
            "fix",
            "add",
            "update",
            "implement",
            "refactor",
            "create",
            "write",
            "edit",
            "change",
            "remove",
            "delete",
            "rename",
            "optimize",
            "benchmark",
        )
        is_action_task = bool(
            getattr(task_analysis, "is_action_task", False)
            or contains_keyword_marker(message_lower, action_markers)
        )
        if not is_action_task:
            return {
                "needs_clarification": False,
                "clarification_reason": None,
                "clarification_prompt": None,
            }

        # Self-contained generation tasks (create a checklist / report / guide) name their
        # output artifact explicitly in the task description and do not require a file or
        # code-component target to proceed.  Applying the underspecified-target check to
        # these tasks produces false positives that abort otherwise-valid plan steps.
        #
        # Heuristic: the task uses a creation verb AND its object is a document/list artifact
        # AND there are no file-modification signals ("in file", "in src/", explicit paths).
        _CREATION_ARTIFACT_PATTERNS = re.compile(
            r"\b(?:create|generate|write|build|produce|compile|draft)\b"
            r".{0,60}"
            r"\b(?:checklist|best.practices|report|guide|overview|summary|documentation|"
            r"list\s+of|inventory|catalog|manifest|template|skeleton|blueprint)\b",
            re.IGNORECASE,
        )
        _FILE_MODIFICATION_SIGNALS = re.compile(
            r"\b(?:in\s+(?:file|src|lib|tests?|crates?|module)|"
            r"\.(?:rs|py|ts|js|go|java|cpp|c|h)\b|"
            r"edit\s+|modify\s+|delete\s+|remove\s+the\s+(?:file|function|method|class))\b",
            re.IGNORECASE,
        )
        if _CREATION_ARTIFACT_PATTERNS.search(message) and not _FILE_MODIFICATION_SIGNALS.search(
            message
        ):
            return {
                "needs_clarification": False,
                "clarification_reason": None,
                "clarification_prompt": None,
            }

        target_present = bool(
            self._extract_required_files(message)
            or self._extract_scope_hints(message)
            or self._context_has_explicit_target(context)
            or conversation_history_has_explicit_target(conversation_history)
        )
        ambiguous_reference = has_ambiguous_target_reference(message_lower)
        threshold = self._evaluation_policy.clarification_confidence_threshold

        if ambiguous_reference and not target_present:
            effective_confidence = max(0.0, confidence - 0.35)
            if effective_confidence <= threshold:
                return self._evaluation_policy.underspecified_target_decision(
                    effective_confidence
                ).to_dict()

        if getattr(task_analysis, "requires_confirmation", False) and not target_present:
            effective_confidence = max(0.0, confidence - 0.2)
            if effective_confidence <= threshold:
                return self._evaluation_policy.confirmation_required_decision(
                    effective_confidence
                ).to_dict()

        return {
            "needs_clarification": False,
            "clarification_reason": None,
            "clarification_prompt": None,
        }

    @property
    def evaluation_policy(self) -> RuntimeEvaluationPolicy:
        """Expose the canonical runtime evaluation policy."""
        return self._evaluation_policy

    def _extract_required_files(self, query: str) -> List[str]:
        """Extract explicit file paths from the query."""
        matches = re.findall(
            r"(?:^|\s|[\"'\-])((?:\.{0,2}/)?[\w./-]+/[\w.-]+\.[a-z]{1,10})(?:\s|[\"']|$|[,;:.\)]|\Z)",
            query,
            flags=re.IGNORECASE,
        )
        return list(dict.fromkeys(matches))

    def _extract_scope_hints(self, query: str) -> List[str]:
        """Extract bounded scope hints like backticked identifiers."""
        hints = re.findall(r"`([^`]{3,80})`", query)
        return list(dict.fromkeys(hints))

    def _context_has_explicit_target(self, context: Optional[Dict[str, Any]]) -> bool:
        """Check for explicit target artifacts supplied through structured context."""
        if not context:
            return False
        target_keys = ("file", "file_path", "path", "target", "component", "symbol")
        return any(bool(context.get(key)) for key in target_keys)

    def _extract_requirements(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
    ) -> List[Requirement]:
        """Extract requirements from query.

        New feature - not duplicated in existing code.
        """
        requirements = []

        # Extract explicit requirements
        requirement_indicators = [
            "must",
            "should",
            "need to",
            "require",
            "ensure",
            "guarantee",
        ]

        query_lower = query.lower()
        for indicator in requirement_indicators:
            if indicator in query_lower:
                sentences = query.split(".")
                for sentence in sentences:
                    if indicator in sentence.lower():
                        requirements.append(
                            Requirement(
                                type=RequirementType.FUNCTIONAL,
                                description=sentence.strip(),
                                priority=1,
                                source="explicit",
                            )
                        )
                        break  # One requirement per indicator

        # Extract quality requirements
        quality_keywords = {
            "test": "Include tests",
            "fast": "Optimize for performance",
            "secure": "Follow security best practices",
            "clean": "Maintain code quality",
        }

        for keyword, criterion in quality_keywords.items():
            if keyword in query_lower:
                requirements.append(
                    Requirement(
                        type=RequirementType.QUALITY,
                        description=criterion,
                        priority=3,
                        source="inferred",
                        acceptance_criteria=f"Code includes {keyword}",
                    )
                )

        return requirements

    async def _retrieve_similar_experiences(
        self,
        query: str,
        intent: ActionIntent,
        context: Optional[Dict[str, Any]],
    ) -> List[SimilarExperience]:
        """Retrieve similar experiences from memory.

        New feature - leverages existing UnifiedMemoryCoordinator.
        """
        if not self.memory_coordinator:
            return []

        try:
            # Search across all memory types
            results = await self.memory_coordinator.search_all(
                query=query,
                limit=5,
            )

            return [
                SimilarExperience(
                    task_id=r.id,
                    description=str(r.content)[:200],
                    similarity_score=r.relevance,
                    outcome=True,  # Assume successful if in memory
                    lessons_learned=[],
                    tools_used=[],
                )
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Failed to retrieve similar experiences: {e}")
            return []

    def _calculate_confidence(
        self,
        intent: ActionIntent,
        complexity: TaskComplexity,
        requirements: List[Requirement],
        similar_experiences: List[SimilarExperience],
        intent_confidence: float = 0.0,
    ) -> float:
        """Calculate calibrated confidence in perception.

        Uses multi-signal fusion inspired by Truth-Aligned Uncertainty
        Estimation (arXiv:2604.00445): combine independent confidence
        signals with diminishing returns rather than linear addition.

        Args:
            intent: Detected intent
            complexity: Task complexity
            requirements: Extracted requirements
            similar_experiences: Past similar tasks
            intent_confidence: Raw confidence from intent detector
        """
        signals: List[float] = []

        # Signal 1: Intent clarity (0.0 - 1.0)
        if intent != ActionIntent.AMBIGUOUS:
            # Use detector's own confidence if available
            intent_signal = max(intent_confidence, 0.7)
            signals.append(intent_signal)
        else:
            signals.append(0.3)

        # Signal 2: Complexity clarity (0.0 - 1.0)
        # Complex/generation tasks are harder but clearer in intent
        complexity_signal = {
            TaskComplexity.SIMPLE: 0.8,
            TaskComplexity.MEDIUM: 0.7,
            TaskComplexity.COMPLEX: 0.6,
        }.get(complexity, 0.5)
        signals.append(complexity_signal)

        # Signal 3: Requirements specificity (0.0 - 1.0)
        if requirements:
            # More specific requirements = higher confidence
            req_signal = min(0.5 + len(requirements) * 0.1, 0.9)
            signals.append(req_signal)

        # Signal 4: Experience support (0.0 - 1.0)
        if similar_experiences:
            avg_sim = sum(e.similarity_score for e in similar_experiences) / len(
                similar_experiences
            )
            signals.append(avg_sim)

        # Combine with geometric mean for calibration
        # (avoids overconfidence from adding independent signals)
        if not signals:
            return 0.5

        product = 1.0
        for s in signals:
            product *= s
        calibrated = product ** (1.0 / len(signals))

        return min(calibrated, 1.0)


# Convenience function
async def perceive(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    memory_coordinator: Optional["UnifiedMemoryCoordinator"] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> Perception:
    """Quick perception without managing integration instance.

    Args:
        query: User query
        context: Execution context
        memory_coordinator: Optional memory coordinator
        conversation_history: Prior turns for multi-turn context

    Returns:
        Perception result
    """
    integration = PerceptionIntegration(memory_coordinator=memory_coordinator)
    return await integration.perceive(query, context, conversation_history)
