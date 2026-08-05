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

"""Task type hint capability for framework-level task hints.

This module provides centralized task type hints for common task types
that can be used across verticals.

Design Pattern: Capability Provider + Registry
- Centralized task type hints
- Domain-specific hint extensions
- Tool budget recommendations
- Priority tool mappings

Integration Point:
    Use in orchestrator for prompt building and tool budget allocation

Example:
    capability = TaskTypeHintCapabilityProvider()
    hints = capability.get_hints()

    # Get hint for specific task type
    edit_hint = capability.get_hint("edit")

Phase 1: Promote Generic Capabilities to Framework
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum


class TaskCategory(Enum):
    """Categories of task types."""

    ANALYSIS = "analysis"  # Understanding and analyzing
    CREATION = "creation"  # Creating new content
    MODIFICATION = "modification"  # Modifying existing content
    DEBUGGING = "debugging"  # Finding and fixing bugs
    REFACTORING = "refactoring"  # Improving code structure
    TESTING = "testing"  # Writing and running tests
    DOCUMENTATION = "documentation"  # Writing documentation
    DEPLOYMENT = "deployment"  # Deploying applications


@dataclass
class TaskTypeHint:
    """Hint for a specific task type.

    Attributes:
        task_type: Task type identifier (e.g., "edit", "search")
        category: Task category this belongs to
        hint: Prompt hint text to include in system prompt
        description: Human-readable description
        tool_budget: Recommended tool budget for this task type
        priority_tools: Tools to prioritize for this task
        keywords: Keywords that suggest this task type
        token_budget: Max tokens for response generation (NEW)
        context_budget: Max context tokens to include (NEW)
        skip_planning: Skip planning phase for direct execution (NEW)
        skip_evaluation: Skip evaluation/verification steps (NEW)
        temperature_override: LLM temperature for this task type; None = provider default (NEW)
    """

    task_type: str
    category: TaskCategory
    hint: str
    description: str
    tool_budget: int = 15
    priority_tools: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    token_budget: Optional[int] = None  # NEW: Max response tokens
    context_budget: Optional[int] = None  # NEW: Max context tokens
    skip_planning: bool = False  # NEW: Skip planning phase
    skip_evaluation: bool = False  # NEW: Skip LLM evaluation
    temperature_override: Optional[float] = None  # NEW: Task-specific LLM temperature

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_type": self.task_type,
            "category": self.category.value,
            "hint": self.hint,
            "description": self.description,
            "tool_budget": self.tool_budget,
            "priority_tools": self.priority_tools,
            "keywords": self.keywords,
            "token_budget": self.token_budget,
            "context_budget": self.context_budget,
            "skip_planning": self.skip_planning,
            "skip_evaluation": self.skip_evaluation,
            "temperature_override": self.temperature_override,
        }


class TaskTypeHintCapabilityProvider:
    """Generic task type hint capability provider.

    Provides centralized task type hints for:
    - Edit tasks (modify code)
    - Search tasks (find information)
    - Debug tasks (fix bugs)
    - Refactor tasks (improve code)
    - Test tasks (verify behavior)

    These hints help the orchestrator provide better guidance and
    allocate appropriate tool budgets for different task types.

    Attributes:
        custom_hints: Optional custom hints to add/override defaults
    """

    # Standard task type hints
    STANDARD_HINTS: Dict[str, TaskTypeHint] = {
        "general": TaskTypeHint(
            task_type="general",
            category=TaskCategory.ANALYSIS,
            hint="[GENERAL] Moderate exploration. Use 3-6 tool calls. Answer concisely.",
            description="General purpose task handling",
            tool_budget=8,
            priority_tools=["read", "grep", "ls"],
            keywords=["help", "what", "how", "explain"],
        ),
        "search": TaskTypeHint(
            task_type="search",
            category=TaskCategory.ANALYSIS,
            hint="[SEARCH] Use grep/ls for exploration. Summarize findings after 2-4 calls.",
            description="Searching for information in codebase",
            tool_budget=6,
            priority_tools=["grep", "ls", "read"],
            keywords=["find", "search", "where", "locate", "look for"],
            token_budget=500,
            skip_planning=True,
            temperature_override=0.3,  # Focused retrieval
        ),
        "create": TaskTypeHint(
            task_type="create",
            category=TaskCategory.CREATION,
            hint="[CREATE] Read 1-2 relevant files for context, then create. Follow existing patterns.",
            description="Creating new files or content",
            tool_budget=5,
            priority_tools=["read", "write", "ls"],
            keywords=["create", "new", "add", "generate", "make"],
            token_budget=800,
            context_budget=2000,
            skip_planning=True,
            skip_evaluation=True,
            temperature_override=0.2,  # Low variance for correct generation
        ),
        "edit": TaskTypeHint(
            task_type="edit",
            category=TaskCategory.MODIFICATION,
            hint="[EDIT] Read target file first, then modify. Focused changes only. Verify after.",
            description="Editing existing files",
            tool_budget=5,
            priority_tools=["read", "edit", "write"],
            keywords=["edit", "modify", "change", "update", "fix"],
            token_budget=600,
            context_budget=1500,
            skip_evaluation=True,
            temperature_override=0.4,  # Balance precision with natural edits
        ),
        "debug": TaskTypeHint(
            task_type="debug",
            category=TaskCategory.DEBUGGING,
            hint="[DEBUG] Investigate errors systematically. Check logs, stack traces, and related code.",
            description="Finding and fixing bugs",
            tool_budget=12,
            priority_tools=["read", "grep", "shell", "test"],
            keywords=["debug", "fix", "error", "bug", "broken", "fail"],
            temperature_override=0.1,  # Precise error analysis
        ),
        "refactor": TaskTypeHint(
            task_type="refactor",
            category=TaskCategory.REFACTORING,
            hint="[REFACTOR] Understand existing structure first. Make incremental improvements.",
            description="Improving code structure without changing behavior",
            tool_budget=10,
            priority_tools=["read", "grep", "edit"],
            keywords=["refactor", "improve", "restructure", "clean up"],
            temperature_override=0.4,
        ),
        "test": TaskTypeHint(
            task_type="test",
            category=TaskCategory.TESTING,
            hint="[TEST] Write tests for existing behavior. Run tests to verify.",
            description="Writing and running tests",
            tool_budget=8,
            priority_tools=["read", "write", "shell", "test"],
            keywords=["test", "verify", "validate", "check"],
            temperature_override=0.2,  # Deterministic test generation
        ),
        "analyze": TaskTypeHint(
            task_type="analyze",
            category=TaskCategory.ANALYSIS,
            hint="[ANALYZE] Examine content carefully. Read related files. Provide structured findings.",
            description="Deep analysis of code or data",
            tool_budget=12,
            priority_tools=["read", "grep", "search"],
            keywords=["analyze", "understand", "explain", "review"],
            temperature_override=0.6,  # Broader reasoning for analysis
        ),
        "deploy": TaskTypeHint(
            task_type="deploy",
            category=TaskCategory.DEPLOYMENT,
            hint="[DEPLOY] Verify configuration before deploying. Check for existing resources.",
            description="Deploying applications or infrastructure",
            tool_budget=8,
            priority_tools=["read", "shell", "docker"],
            keywords=["deploy", "ship", "release", "provision"],
            temperature_override=0.3,  # Focused execution
        ),
        "document": TaskTypeHint(
            task_type="document",
            category=TaskCategory.DOCUMENTATION,
            hint="[DOCUMENT] Read code to understand functionality. Write clear, concise docs.",
            description="Writing documentation",
            tool_budget=6,
            priority_tools=["read", "write"],
            keywords=["document", "docs", "readme", "comment"],
        ),
    }

    def __init__(self, custom_hints: Optional[Dict[str, TaskTypeHint]] = None):
        """Initialize the task type hint capability provider.

        Args:
            custom_hints: Optional custom hints to add/override defaults
        """
        self._custom_hints = custom_hints or {}
        self._hint_cache: Optional[Dict[str, TaskTypeHint]] = None

    def get_hints(self, category: Optional[TaskCategory] = None) -> Dict[str, TaskTypeHint]:
        """Get all task type hints, optionally filtered by category.

        Args:
            category: Optional category to filter by

        Returns:
            Dictionary mapping task type to TaskTypeHint
        """
        if self._hint_cache is None:
            self._build_cache()

        if category:
            return {k: v for k, v in self._hint_cache.items() if v.category == category}
        return self._hint_cache.copy()

    def _build_cache(self) -> None:
        """Build the hint cache."""
        self._hint_cache = self.STANDARD_HINTS.copy()
        self._hint_cache.update(self._custom_hints)

    def get_hint(self, task_type: str) -> Optional[TaskTypeHint]:
        """Get a specific task type hint.

        Args:
            task_type: Task type name

        Returns:
            TaskTypeHint or None if not found
        """
        hints = self.get_hints()
        return hints.get(task_type.lower())

    def get_hint_for_keywords(self, keywords: List[str]) -> Optional[TaskTypeHint]:
        """Get the best matching hint based on keywords.

        Args:
            keywords: List of keywords to match against

        Returns:
            Best matching TaskTypeHint or None
        """
        hints = self.get_hints()
        best_match = None
        best_score = 0

        for hint in hints.values():
            score = sum(
                1 for keyword in keywords if any(kw in keyword.lower() for kw in hint.keywords)
            )
            if score > best_score:
                best_score = score
                best_match = hint

        return best_match if best_score > 0 else None

    def get_tool_budget(self, task_type: str) -> int:
        """Get recommended tool budget for a task type.

        Args:
            task_type: Task type name

        Returns:
            Recommended tool budget
        """
        hint = self.get_hint(task_type)
        return hint.tool_budget if hint else 15

    def get_priority_tools(self, task_type: str) -> List[str]:
        """Get priority tools for a task type.

        Args:
            task_type: Task type name

        Returns:
            List of priority tool names
        """
        hint = self.get_hint(task_type)
        return hint.priority_tools if hint else []

    def get_prompt_text(self, task_type: str) -> str:
        """Get the hint text for a task type.

        Args:
            task_type: Task type name

        Returns:
            Hint text for inclusion in prompts
        """
        hint = self.get_hint(task_type)
        return hint.hint if hint else ""

    def get_all_prompt_text(self) -> str:
        """Get all hint text formatted for prompts.

        Returns:
            Formatted hint text for all task types
        """
        hints = self.get_hints()
        lines = ["# Task Type Guidance", ""]

        for hint in hints.values():
            lines.append(f"## {hint.task_type.upper()}")
            lines.append(f"{hint.description}")
            lines.append(f"Hint: {hint.hint}")
            if hint.priority_tools:
                lines.append(f"Priority tools: {', '.join(hint.priority_tools)}")
            lines.append("")

        return "\n".join(lines)

    def add_hint(self, hint: TaskTypeHint) -> None:
        """Add a custom task type hint.

        Args:
            hint: TaskTypeHint to add
        """
        self._custom_hints[hint.task_type] = hint
        self._hint_cache = None

    def remove_hint(self, task_type: str) -> bool:
        """Remove a custom hint by task type.

        Args:
            task_type: Task type to remove

        Returns:
            True if hint was removed, False if not found
        """
        if task_type in self._custom_hints:
            del self._custom_hints[task_type]
            self._hint_cache = None
            return True
        return False

    def list_task_types(self) -> List[str]:
        """List all available task types.

        Returns:
            List of task type names
        """
        hints = self.get_hints()
        return list(hints.keys())

    def list_categories(self) -> List[TaskCategory]:
        """List all task categories.

        Returns:
            List of TaskCategory enum values
        """
        return list(TaskCategory)

    def clear_cache(self) -> None:
        """Clear the hint cache."""
        self._hint_cache = None


# Pre-configured task type hints for common vertical types
class TaskTypeHintPresets:
    """Pre-configured task type hints for common verticals."""

    @staticmethod
    def coding() -> TaskTypeHintCapabilityProvider:
        """Get task hints optimized for coding vertical."""
        # Coding uses all standard hints
        custom_hints = {
            "review": TaskTypeHint(
                task_type="review",
                category=TaskCategory.ANALYSIS,
                hint="[REVIEW] Examine code for quality, security, and best practices.",
                description="Code review",
                tool_budget=10,
                priority_tools=["read", "grep"],
                keywords=["review", "audit", "inspect"],
            ),
        }
        return TaskTypeHintCapabilityProvider(custom_hints=custom_hints)

    @staticmethod
    def devops() -> TaskTypeHintCapabilityProvider:
        """Get task hints optimized for DevOps vertical."""
        custom_hints = {
            "provision": TaskTypeHint(
                task_type="provision",
                category=TaskCategory.DEPLOYMENT,
                hint="[PROVISION] Set up infrastructure. Verify existing resources first.",
                description="Provisioning infrastructure",
                tool_budget=10,
                priority_tools=["shell", "docker", "read"],
                keywords=["provision", "setup", "configure"],
            ),
            "monitor": TaskTypeHint(
                task_type="monitor",
                category=TaskCategory.ANALYSIS,
                hint="[MONITOR] Check system status and logs. Identify issues.",
                description="Monitoring systems",
                tool_budget=6,
                priority_tools=["shell", "read", "grep"],
                keywords=["monitor", "status", "health", "logs"],
            ),
        }
        return TaskTypeHintCapabilityProvider(custom_hints=custom_hints)

    @staticmethod
    def research() -> TaskTypeHintCapabilityProvider:
        """Get task hints optimized for research vertical."""
        custom_hints = {
            "investigate": TaskTypeHint(
                task_type="investigate",
                category=TaskCategory.ANALYSIS,
                hint="[INVESTIGATE] Use web_search and web_fetch. Verify multiple sources.",
                description="Investigating a topic",
                tool_budget=10,
                priority_tools=["web_search", "web_fetch"],
                keywords=["investigate", "research", "look into"],
            ),
            "synthesize": TaskTypeHint(
                task_type="synthesize",
                category=TaskCategory.ANALYSIS,
                hint="[SYNTHESIZE] Combine information from sources. Cite references.",
                description="Synthesizing information",
                tool_budget=8,
                priority_tools=["read", "write"],
                keywords=["synthesize", "summarize", "combine"],
            ),
        }
        return TaskTypeHintCapabilityProvider(custom_hints=custom_hints)

    @staticmethod
    def data_analysis() -> TaskTypeHintCapabilityProvider:
        """Get task hints optimized for data analysis vertical."""
        custom_hints = {
            "explore": TaskTypeHint(
                task_type="explore",
                category=TaskCategory.ANALYSIS,
                hint="[EXPLORE] Load and examine data. Check types and distributions.",
                description="Exploring data",
                tool_budget=8,
                priority_tools=["read", "shell", "write"],
                keywords=["explore", "examine", "look at"],
            ),
            "visualize": TaskTypeHint(
                task_type="visualize",
                category=TaskCategory.CREATION,
                hint="[VISUALIZE] Create charts and graphs. Understand data first.",
                description="Creating visualizations",
                tool_budget=6,
                priority_tools=["read", "write"],
                keywords=["visualize", "plot", "chart", "graph"],
            ),
        }
        return TaskTypeHintCapabilityProvider(custom_hints=custom_hints)

    @staticmethod
    def rag() -> TaskTypeHintCapabilityProvider:
        """Get task hints optimized for RAG vertical."""
        custom_hints = {
            "query": TaskTypeHint(
                task_type="query",
                category=TaskCategory.ANALYSIS,
                hint="[QUERY] Use RAG tools to search and retrieve relevant context.",
                description="Querying RAG system",
                tool_budget=6,
                priority_tools=["rag_search", "rag_query"],
                keywords=["query", "search", "find"],
            ),
            "answer": TaskTypeHint(
                task_type="answer",
                category=TaskCategory.ANALYSIS,
                hint="[ANSWER] Base answer on retrieved context. Cite sources.",
                description="Answering questions with RAG",
                tool_budget=5,
                priority_tools=["rag_query", "read"],
                keywords=["answer", "respond", "explain"],
            ),
        }
        return TaskTypeHintCapabilityProvider(custom_hints=custom_hints)


__all__ = [
    "TaskTypeHintCapabilityProvider",
    "TaskTypeHintPresets",
    "TaskCategory",
    "TaskTypeHint",
]
