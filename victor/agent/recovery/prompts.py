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

"""Model-specific prompt registry with learned templates.

This module provides intelligent prompt template selection based on:
- Model/provider characteristics
- Failure type
- Historical effectiveness (Q-learning)

Template evolution:
- Templates are versioned
- Successful templates can be promoted
- Templates can be specialized per model
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from victor.agent.recovery.protocols import (
    FailureType,
    PromptTemplate,
    QLearningStore,
    RecoveryContext,
)

logger = logging.getLogger(__name__)


@dataclass
class TemplateStats:
    """Statistics for a prompt template."""

    template_id: str
    usage_count: int = 0
    success_count: int = 0
    total_quality_improvement: float = 0.0
    last_used: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        if self.usage_count == 0:
            return 0.5
        return self.success_count / self.usage_count

    @property
    def avg_quality_improvement(self) -> float:
        if self.usage_count == 0:
            return 0.0
        return self.total_quality_improvement / self.usage_count

    @property
    def effectiveness_score(self) -> float:
        """Combined effectiveness score."""
        # Blend success rate with quality improvement
        return 0.7 * self.success_rate + 0.3 * min(1.0, self.avg_quality_improvement)


class ModelSpecificPromptRegistry:
    """Registry for model-specific prompt templates with learning.

    Features:
    - Model pattern matching for template selection
    - Q-learning-based template ranking
    - Template versioning and evolution
    - Persistent storage of effectiveness metrics

    Follows:
    - Single Responsibility: Only manages prompt templates
    - Open/Closed: New templates can be registered without modification
    - Dependency Inversion: Depends on QLearningStore protocol
    """

    # Default templates for each failure type
    DEFAULT_TEMPLATES: List[PromptTemplate] = [
        # Empty Response Templates
        PromptTemplate(
            id="empty_gentle_v1",
            name="Empty Response - Gentle",
            template=(
                "Your response was empty. Please provide a response. "
                "You can either make a tool call or provide text content."
            ),
            failure_types=[FailureType.EMPTY_RESPONSE],
        ),
        PromptTemplate(
            id="empty_specific_v1",
            name="Empty Response - Specific",
            template=(
                "No response received. Please either:\n"
                "1. Call a tool ({available_tools}) to gather information, OR\n"
                "2. Provide your analysis based on what you know.\n\n"
                "Respond NOW."
            ),
            failure_types=[FailureType.EMPTY_RESPONSE],
        ),
        PromptTemplate(
            id="empty_assertive_v1",
            name="Empty Response - Assertive",
            template=(
                "CRITICAL: Empty response detected. You MUST respond with one of:\n\n"
                "OPTION A - Make a tool call:\n"
                '```json\n{{"name": "{suggested_tool}", "arguments": {{}}}}\n```\n\n'
                "OPTION B - Provide text response:\n"
                "Start your response with 'Based on my analysis...' and provide content."
            ),
            failure_types=[FailureType.EMPTY_RESPONSE],
        ),
        # Ollama-specific empty response template
        PromptTemplate(
            id="empty_ollama_v1",
            name="Empty Response - Ollama",
            template=(
                "No response. To call a tool, respond with JSON:\n"
                '{{"name": "read", "arguments": {{"path": "..."}}}}\n\n'
                "Or provide your text answer directly."
            ),
            failure_types=[FailureType.EMPTY_RESPONSE],
            provider_patterns=["ollama"],
        ),
        # Stuck Loop Templates
        PromptTemplate(
            id="stuck_direct_v1",
            name="Stuck Loop - Direct",
            template=(
                "You appear to be stuck in a planning loop. You keep describing what "
                "you will do but are not making actual tool calls.\n\n"
                "Please either:\n"
                "1. Make an ACTUAL tool call NOW (not just describe it), OR\n"
                "2. Provide your response based on what you already know."
            ),
            failure_types=[FailureType.STUCK_LOOP],
        ),
        PromptTemplate(
            id="stuck_assertive_v1",
            name="Stuck Loop - Assertive",
            template=(
                "STUCK LOOP DETECTED. Your responses are not making progress.\n\n"
                "DO NOT describe what you will do. DO one of these:\n"
                "- CALL a tool: {available_tools}\n"
                "- PROVIDE your answer NOW\n\n"
                "Action, not words."
            ),
            failure_types=[FailureType.STUCK_LOOP],
        ),
        PromptTemplate(
            id="stuck_force_v1",
            name="Stuck Loop - Force Completion",
            template=(
                "FINAL WARNING: You have been planning without executing.\n\n"
                "Provide your FINAL RESPONSE now. No more tool calls will be allowed. "
                "Summarize what you know and answer the user's question."
            ),
            failure_types=[FailureType.STUCK_LOOP],
        ),
        # Qwen-specific stuck loop template
        PromptTemplate(
            id="stuck_qwen_v1",
            name="Stuck Loop - Qwen",
            template=(
                "Stop planning. Act NOW.\n\n"
                "Either call a tool or give your final answer. "
                "No more 'let me' or 'I will' - just DO IT."
            ),
            failure_types=[FailureType.STUCK_LOOP],
            model_patterns=["qwen*"],
        ),
        # Hallucinated Tool Templates
        PromptTemplate(
            id="hallucinate_gentle_v1",
            name="Hallucinated Tool - Gentle",
            template=(
                "You mentioned {mentioned_tools} but did not make an actual tool call. "
                "Please ACTUALLY call the tool now, or provide your analysis."
            ),
            failure_types=[FailureType.HALLUCINATED_TOOL],
        ),
        PromptTemplate(
            id="hallucinate_critical_v1",
            name="Hallucinated Tool - Critical",
            template=(
                "CRITICAL: You mentioned {mentioned_tools} but did NOT execute a tool call.\n\n"
                "Your response contained TEXT describing what you would do, "
                "but no actual tool invocation.\n\n"
                "You MUST respond with an ACTUAL tool call. "
                "If you cannot call tools, provide your answer NOW."
            ),
            failure_types=[FailureType.HALLUCINATED_TOOL],
        ),
        # Timeout Templates
        PromptTemplate(
            id="timeout_warning_v1",
            name="Timeout - Warning",
            template=(
                "TIME LIMIT APPROACHING ({remaining_time} remaining).\n\n"
                "Provide your response NOW. Summarize your findings and "
                "answer the user's question. Do NOT call any more tools."
            ),
            failure_types=[FailureType.TIMEOUT_APPROACHING],
        ),
        PromptTemplate(
            id="timeout_final_v1",
            name="Timeout - Final",
            template=(
                "TIME LIMIT REACHED. Provide your FINAL ANSWER NOW.\n\n"
                "Based on what you've learned:\n"
                "{progress_summary}\n\n"
                "Give your best response immediately."
            ),
            failure_types=[FailureType.TIMEOUT_APPROACHING],
        ),
    ]

    def __init__(
        self,
        q_store: Optional[QLearningStore] = None,
        db_path: Optional[Path] = None,
    ):
        self._q_store = q_store
        self._templates: Dict[str, PromptTemplate] = {}
        self._stats: Dict[str, TemplateStats] = {}
        self._db_path = db_path

        # Register default templates
        for template in self.DEFAULT_TEMPLATES:
            self.register_template(template)

        # Load persisted stats if db_path provided
        if db_path:
            self._load_stats()

    def register_template(self, template: PromptTemplate) -> None:
        """Register a prompt template."""
        self._templates[template.id] = template
        if template.id not in self._stats:
            self._stats[template.id] = TemplateStats(template_id=template.id)

    def get_template(
        self,
        context: RecoveryContext,
        escalation_level: int = 0,
    ) -> Tuple[PromptTemplate, Dict[str, str]]:
        """Get best template for context.

        Args:
            context: Recovery context
            escalation_level: 0=gentle, 1=specific, 2+=assertive

        Returns:
            Tuple of (template, format_kwargs)
        """
        failure_type = context.failure_type
        provider = context.provider_name.lower()
        model = context.model_name.lower()

        # Find matching templates
        candidates: List[Tuple[PromptTemplate, float]] = []

        for template in self._templates.values():
            if failure_type not in template.failure_types:
                continue

            # Calculate match score
            score = 0.0

            # Provider match
            if template.provider_patterns:
                if any(fnmatch.fnmatch(provider, p) for p in template.provider_patterns):
                    score += 0.3
                else:
                    continue  # Provider-specific but doesn't match

            # Model match
            if template.model_patterns:
                if any(fnmatch.fnmatch(model, p) for p in template.model_patterns):
                    score += 0.3
                else:
                    continue  # Model-specific but doesn't match

            # Effectiveness score
            stats = self._stats.get(template.id)
            if stats:
                score += stats.effectiveness_score * 0.4

            # Q-learning boost if available
            if self._q_store:
                state_key = f"{context.to_state_key()}:template={template.id}"
                from victor.agent.recovery.protocols import RecoveryAction

                q_value = self._q_store.get_q_value(
                    state_key, RecoveryAction.RETRY_WITH_TEMPLATE.name
                )
                score += q_value * 0.2

            candidates.append((template, score))

        if not candidates:
            # Return a generic template
            generic = PromptTemplate(
                id="generic_fallback",
                name="Generic Fallback",
                template="Please continue with your response or provide your analysis.",
                failure_types=[failure_type],
            )
            return generic, {}

        # Sort by score and select based on escalation
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Group by assertiveness level (based on name convention)
        gentle = [c for c in candidates if "gentle" in c[0].name.lower()]
        specific = [
            c
            for c in candidates
            if "specific" in c[0].name.lower() or "direct" in c[0].name.lower()
        ]
        assertive = [
            c
            for c in candidates
            if "assertive" in c[0].name.lower()
            or "critical" in c[0].name.lower()
            or "force" in c[0].name.lower()
            or "final" in c[0].name.lower()
        ]
        _other = [c for c in candidates if c not in gentle + specific + assertive]  # noqa: F841

        # Select based on escalation level
        if escalation_level == 0 and gentle:
            selected = gentle[0][0]
        elif escalation_level == 1 and (specific or gentle):
            selected = (specific or gentle)[0][0]
        elif assertive:
            selected = assertive[0][0]
        else:
            selected = candidates[0][0]

        # Prepare format kwargs
        format_kwargs = self._prepare_format_kwargs(context)

        return selected, format_kwargs

    def _prepare_format_kwargs(self, context: RecoveryContext) -> Dict[str, str]:
        """Prepare format kwargs for template."""
        kwargs: Dict[str, str] = {}

        # Mentioned tools
        if context.mentioned_tools:
            kwargs["mentioned_tools"] = ", ".join(context.mentioned_tools)
        else:
            kwargs["mentioned_tools"] = "tools"

        # Available tools (simplified)
        kwargs["available_tools"] = "read, search, ls"
        kwargs["suggested_tool"] = "read"

        # Time remaining
        remaining = context.session_idle_timeout - context.elapsed_time_seconds
        kwargs["remaining_time"] = f"{remaining:.0f}s" if remaining > 0 else "no time"

        # Progress summary
        if context.tool_calls_made > 0:
            kwargs["progress_summary"] = f"You made {context.tool_calls_made} tool calls."
        else:
            kwargs["progress_summary"] = "No tool calls were made."

        return kwargs

    def record_usage(
        self,
        template_id: str,
        success: bool,
        quality_improvement: float = 0.0,
        context: Optional[RecoveryContext] = None,
    ) -> None:
        """Record template usage for learning.

        Args:
            template_id: ID of the template used
            success: Whether recovery was successful
            quality_improvement: Change in quality score
            context: Optional recovery context for Q-learning
        """
        if template_id not in self._stats:
            self._stats[template_id] = TemplateStats(template_id=template_id)

        stats = self._stats[template_id]
        stats.usage_count += 1
        stats.last_used = datetime.now()

        if success:
            stats.success_count += 1
        stats.total_quality_improvement += quality_improvement

        # Update template's built-in metrics
        if template_id in self._templates:
            template = self._templates[template_id]
            template.usage_count = stats.usage_count
            template.success_count = stats.success_count
            template.avg_quality_improvement = stats.avg_quality_improvement

        # Update Q-learning if available
        if self._q_store and context:
            state_key = f"{context.to_state_key()}:template={template_id}"
            reward = 0.5 + quality_improvement if success else -0.3
            from victor.agent.recovery.protocols import RecoveryAction

            self._q_store.update_q_value(
                state_key,
                RecoveryAction.RETRY_WITH_TEMPLATE.name,
                reward,
            )

        # Persist stats
        if self._db_path:
            self._save_stats(template_id)

        logger.debug(
            f"Template {template_id} usage recorded: "
            f"success={success}, quality_improvement={quality_improvement:.2f}, "
            f"total_uses={stats.usage_count}, success_rate={stats.success_rate:.2f}"
        )

    def get_template_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all templates."""
        return {
            tid: {
                "name": self._templates.get(
                    tid,
                    PromptTemplate(id=tid, name="Unknown", template="", failure_types=[]),
                ).name,
                "usage_count": stats.usage_count,
                "success_rate": stats.success_rate,
                "avg_quality_improvement": stats.avg_quality_improvement,
                "effectiveness_score": stats.effectiveness_score,
                "last_used": stats.last_used.isoformat() if stats.last_used else None,
            }
            for tid, stats in self._stats.items()
        }

    def get_best_templates(
        self,
        failure_type: FailureType,
        limit: int = 5,
    ) -> List[Tuple[str, float]]:
        """Get best performing templates for a failure type.

        Returns:
            List of (template_id, effectiveness_score)
        """
        matching = [
            (tid, self._stats[tid].effectiveness_score)
            for tid, template in self._templates.items()
            if failure_type in template.failure_types and tid in self._stats
        ]
        matching.sort(key=lambda x: x[1], reverse=True)
        return matching[:limit]

    def evolve_template(
        self,
        parent_id: str,
        new_template: str,
        reason: str = "manual_evolution",
    ) -> Optional[str]:
        """Create a new version of a template.

        Args:
            parent_id: ID of parent template
            new_template: New template text
            reason: Reason for evolution

        Returns:
            ID of new template, or None if parent not found
        """
        if parent_id not in self._templates:
            return None

        parent = self._templates[parent_id]
        new_version = parent.version + 1
        new_id = f"{parent_id.rsplit('_v', 1)[0]}_v{new_version}"

        new = PromptTemplate(
            id=new_id,
            name=f"{parent.name} (v{new_version})",
            template=new_template,
            failure_types=parent.failure_types,
            provider_patterns=parent.provider_patterns,
            model_patterns=parent.model_patterns,
            version=new_version,
            parent_id=parent_id,
        )

        self.register_template(new)
        logger.info(f"Evolved template {parent_id} -> {new_id}: {reason}")
        return new_id

    def _load_stats(self) -> None:
        """Load stats from SQLite database."""
        if not self._db_path or not self._db_path.exists():
            return

        try:
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS template_stats (
                    template_id TEXT PRIMARY KEY,
                    usage_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    total_quality_improvement REAL DEFAULT 0.0,
                    last_used TEXT
                )
            """)

            cursor.execute("SELECT * FROM template_stats")
            for row in cursor.fetchall():
                tid, usage, success, quality, last_used = row
                self._stats[tid] = TemplateStats(
                    template_id=tid,
                    usage_count=usage,
                    success_count=success,
                    total_quality_improvement=quality,
                    last_used=datetime.fromisoformat(last_used) if last_used else None,
                )

            conn.close()
        except Exception as e:
            logger.warning(f"Failed to load template stats: {e}")

    def _save_stats(self, template_id: str) -> None:
        """Save stats for a template to SQLite database."""
        if not self._db_path:
            return

        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS template_stats (
                    template_id TEXT PRIMARY KEY,
                    usage_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    total_quality_improvement REAL DEFAULT 0.0,
                    last_used TEXT
                )
            """)

            stats = self._stats[template_id]
            cursor.execute(
                """
                INSERT OR REPLACE INTO template_stats
                (template_id, usage_count, success_count, total_quality_improvement, last_used)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    template_id,
                    stats.usage_count,
                    stats.success_count,
                    stats.total_quality_improvement,
                    stats.last_used.isoformat() if stats.last_used else None,
                ),
            )

            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to save template stats: {e}")
