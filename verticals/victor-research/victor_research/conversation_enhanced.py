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
from typing import Any, Dict, List

from victor_contracts.verticals import VerticalConversationManager

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


class EnhancedResearchConversationManager(VerticalConversationManager):
    """Coordinator-backed conversation manager with the research context."""

    def __init__(
        self,
        max_history_turns: int = 50,
        summarization_threshold: int = 40,
        enable_deduplication: bool = True,
        enable_statistics: bool = True,
    ):
        super().__init__(
            ResearchContext,
            "research_context",
            "research",
            max_history_turns=max_history_turns,
            summarization_threshold=summarization_threshold,
            enable_deduplication=enable_deduplication,
            enable_statistics=enable_statistics,
        )

    def get_research_context(self) -> ResearchContext:
        return self.get_context()

    def track_research_question(self, question: str) -> None:
        """Track a research question.

        Args:
            question: Research question being explored
        """
        self._context.add_research_question(question)

    def track_hypothesis(self, hypothesis: str, status: str = "untested") -> None:
        """Track a hypothesis.

        Args:
            hypothesis: Hypothesis statement
            status: Status (untested, testing, confirmed, rejected)
        """
        self._context.add_hypothesis(hypothesis, status)

    def track_data_source(self, source: str, source_type: str) -> None:
        """Track a data source.

        Args:
            source: Data source identifier
            source_type: Type (paper, dataset, web, api, etc.)
        """
        self._context.add_data_source(source, source_type)

    def track_experiment(self, experiment: str, result: str) -> None:
        """Track an experiment.

        Args:
            experiment: Experiment description
            result: Result or outcome
        """
        self._context.add_experiment(experiment, result)

    def track_finding(self, finding: str, category: str = "general") -> None:
        """Track a finding.

        Args:
            finding: Key finding or insight
            category: Category
        """
        self._context.add_finding(finding, category)

    # =========================================================================
    # Summarization
    # =========================================================================

    def get_research_summary(self) -> str:
        """Get a research-focused conversation summary.

        Returns:
            Formatted summary of research work done
        """
        parts = []

        ctx = self._context

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


__all__ = [
    "ResearchContext",
    "EnhancedResearchConversationManager",
]
