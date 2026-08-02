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

"""Enhanced conversation management for victor-research using ConversationCoordinator.

This module provides research-specific conversation management features using
the framework's ConversationCoordinator for better context tracking and
summarization.

Design Pattern: Extension + Delegation
- Provides research-specific conversation management
- Delegates to framework ConversationCoordinator
- Tracks research-specific context (experiments, data sources, findings, etc.)

Integration Point:
    Use in ResearchAssistant for enhanced conversation tracking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from victor_contracts import ConversationCoordinator, ConversationStats, TurnType

logger = logging.getLogger(__name__)


@dataclass
class ResearchContext:
    """Research-specific conversation context.

    Tracks:
    - Research questions and hypotheses
    - Data sources used
    - Experiments conducted
    - Findings and insights
    - Papers and references analyzed
    - Next steps and action items

    Attributes:
        research_questions: List of research questions explored
        hypotheses: List of hypotheses tested
        data_sources: List of data sources consulted
        experiments: List of experiments performed
        findings: List of key findings discovered
        references: List of papers and references analyzed
        next_steps: List of planned next steps
    """

    research_questions: List[str] = field(default_factory=list)
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    data_sources: List[Dict[str, Any]] = field(default_factory=list)
    experiments: List[Dict[str, Any]] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    references: List[Dict[str, Any]] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "research_questions": self.research_questions,
            "hypotheses": self.hypotheses,
            "data_sources": self.data_sources,
            "experiments": self.experiments,
            "findings": self.findings,
            "references": self.references,
            "next_steps": self.next_steps,
        }

    def add_research_question(self, question: str) -> None:
        """Record a research question.

        Args:
            question: Research question being explored
        """
        if question not in self.research_questions:
            self.research_questions.append(question)
            logger.debug(f"Recorded research question: {question}")

    def add_hypothesis(self, hypothesis: str, status: str = "untested") -> None:
        """Record a hypothesis.

        Args:
            hypothesis: Hypothesis statement
            status: Status (untested, testing, confirmed, rejected)
        """
        self.hypotheses.append(
            {
                "statement": hypothesis,
                "status": status,
            }
        )
        logger.debug(f"Recorded hypothesis: {hypothesis} (status={status})")

    def add_data_source(self, source: str, source_type: str) -> None:
        """Record a data source.

        Args:
            source: Data source identifier
            source_type: Type (paper, dataset, web, api, etc.)
        """
        self.data_sources.append(
            {
                "source": source,
                "type": source_type,
            }
        )
        logger.debug(f"Recorded data source: {source} ({source_type})")

    def add_experiment(self, experiment: str, result: str) -> None:
        """Record an experiment.

        Args:
            experiment: Experiment description
            result: Result or outcome
        """
        self.experiments.append(
            {
                "description": experiment,
                "result": result,
            }
        )
        logger.debug(f"Recorded experiment: {experiment}")

    def add_finding(self, finding: str, category: str = "general") -> None:
        """Record a finding.

        Args:
            finding: Key finding or insight
            category: Category (general, methodological, data-related)
        """
        self.findings.append(
            {
                "finding": finding,
                "category": category,
            }
        )
        logger.debug(f"Recorded finding: {finding}")


class EnhancedResearchConversationManager:
    """Enhanced conversation manager for Research using ConversationCoordinator.

    Provides:
    - Standard conversation tracking via ConversationCoordinator
    - Research-specific context tracking (questions, hypotheses, experiments, findings)
    - Automatic summarization of research work
    - Research-focused conversation history

    Example:
        manager = EnhancedResearchConversationManager()

        # Add a user message
        manager.add_message("user", "What's the relationship between X and Y?", TurnType.USER)

        # Track research elements
        manager.track_research_question("How does X affect Y?")
        manager.track_experiment("Regression analysis of X vs Y", "Significant correlation found")

        # Get conversation summary
        summary = manager.get_research_summary()
    """

    def __init__(
        self,
        max_history_turns: int = 50,
        summarization_threshold: int = 40,
        enable_deduplication: bool = True,
        enable_statistics: bool = True,
    ):
        """Initialize the enhanced conversation manager.

        Args:
            max_history_turns: Maximum turns to keep in history
            summarization_threshold: Turns before triggering summarization
            enable_deduplication: Whether to enable message deduplication
            enable_statistics: Whether to track conversation statistics
        """
        self._conversation_coordinator = ConversationCoordinator(
            max_history_turns=max_history_turns,
            summarization_threshold=summarization_threshold,
            enable_deduplication=enable_deduplication,
            enable_statistics=enable_statistics,
        )

        self._research_context = ResearchContext()

        logger.info(
            f"EnhancedResearchConversationManager initialized with "
            f"max_turns={max_history_turns}"
        )

    # =========================================================================
    # Message Management (delegates to ConversationCoordinator)
    # =========================================================================

    def add_message(
        self,
        role: str,
        content: str,
        turn_type: TurnType,
        metadata: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Add a message to the conversation.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            turn_type: Type of turn
            metadata: Optional metadata
            tool_calls: Optional tool calls made in this turn

        Returns:
            Turn ID for the added message
        """
        return self._conversation_coordinator.add_message(
            role, content, turn_type, metadata, tool_calls
        )

    def get_history(
        self,
        max_turns: Optional[int] = None,
        include_system: bool = True,
        include_tool: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get conversation history.

        Args:
            max_turns: Maximum number of turns to return
            include_system: Whether to include system messages
            include_tool: Whether to include tool messages

        Returns:
            List of message dictionaries
        """
        return self._conversation_coordinator.get_history(max_turns, include_system, include_tool)

    def clear_history(self, keep_summaries: bool = True) -> None:
        """Clear conversation history.

        Args:
            keep_summaries: Whether to keep conversation summaries
        """
        self._conversation_coordinator.clear_history(keep_summaries)
        if not keep_summaries:
            self._research_context = ResearchContext()
        logger.info("Conversation history cleared")

    # =========================================================================
    # Research-Specific Context Tracking
    # =========================================================================

    def track_research_question(self, question: str) -> None:
        """Track a research question.

        Args:
            question: Research question being explored
        """
        self._research_context.add_research_question(question)

    def track_hypothesis(self, hypothesis: str, status: str = "untested") -> None:
        """Track a hypothesis.

        Args:
            hypothesis: Hypothesis statement
            status: Status (untested, testing, confirmed, rejected)
        """
        self._research_context.add_hypothesis(hypothesis, status)

    def track_data_source(self, source: str, source_type: str) -> None:
        """Track a data source.

        Args:
            source: Data source identifier
            source_type: Type (paper, dataset, web, api, etc.)
        """
        self._research_context.add_data_source(source, source_type)

    def track_experiment(self, experiment: str, result: str) -> None:
        """Track an experiment.

        Args:
            experiment: Experiment description
            result: Result or outcome
        """
        self._research_context.add_experiment(experiment, result)

    def track_finding(self, finding: str, category: str = "general") -> None:
        """Track a finding.

        Args:
            finding: Key finding or insight
            category: Category
        """
        self._research_context.add_finding(finding, category)

    # =========================================================================
    # Summarization
    # =========================================================================

    def needs_summarization(self) -> bool:
        """Check if conversation needs summarization.

        Returns:
            True if summarization is recommended
        """
        return self._conversation_coordinator.needs_summarization()

    def add_summary(self, summary: str) -> None:
        """Add a conversation summary.

        Args:
            summary: Summary text
        """
        self._conversation_coordinator.add_summary(summary)

    def get_research_summary(self) -> str:
        """Get a research-focused conversation summary.

        Returns:
            Formatted summary of research work done
        """
        parts = []

        ctx = self._research_context

        # Research questions
        if ctx.research_questions:
            parts.append("## Research Questions")
            for q in ctx.research_questions:
                parts.append(f"- {q}")
            parts.append("")

        # Hypotheses
        if ctx.hypotheses:
            parts.append("## Hypotheses")
            for h in ctx.hypotheses:
                parts.append(f"- {h['statement']} ({h['status']})")
            parts.append("")

        # Data sources
        if ctx.data_sources:
            parts.append("## Data Sources")
            for ds in ctx.data_sources:
                parts.append(f"- {ds['source']} ({ds['type']})")
            parts.append("")

        # Experiments
        if ctx.experiments:
            parts.append("## Experiments")
            for exp in ctx.experiments:
                parts.append(f"- {exp['description']}: {exp['result']}")
            parts.append("")

        # Findings
        if ctx.findings:
            parts.append("## Key Findings")
            for f in ctx.findings:
                category = f"[{f['category']}]" if f.get("category") != "general" else ""
                parts.append(f"- {category} {f['finding']}")
            parts.append("")

        # Conversation stats
        stats = self._conversation_coordinator.get_stats()
        parts.append("## Conversation Stats")
        parts.append(f"- Total turns: {stats.total_turns}")
        parts.append(f"- User turns: {stats.user_turns}")
        parts.append(f"- Assistant turns: {stats.assistant_turns}")
        parts.append(f"- Tool calls: {stats.tool_calls}")

        return "\n".join(parts)

    # =========================================================================
    # Statistics and Observability
    # =========================================================================

    def get_stats(self) -> ConversationStats:
        """Get conversation statistics.

        Returns:
            ConversationStats object
        """
        return self._conversation_coordinator.get_stats()

    def get_research_context(self) -> ResearchContext:
        """Get the research context.

        Returns:
            ResearchContext object
        """
        return self._research_context

    def get_observability_data(self) -> Dict[str, Any]:
        """Get observability data for dashboard integration.

        Returns:
            Dictionary with observability data
        """
        conv_obs = self._conversation_coordinator.get_observability_data()

        return {
            **conv_obs,
            "research_context": self._research_context.to_dict(),
            "vertical": "research",
        }

    def get_conversation_coordinator(self) -> ConversationCoordinator:
        """Get the underlying ConversationCoordinator.

        Returns:
            ConversationCoordinator instance
        """
        return self._conversation_coordinator


__all__ = [
    "ResearchContext",
    "EnhancedResearchConversationManager",
]
