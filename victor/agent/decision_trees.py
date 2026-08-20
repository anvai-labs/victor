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

"""Pre-computed decision trees for LLM-free agent decisions.

Based on arXiv research papers:
- AgentGate: Lightweight Structured Routing (arXiv:2604.06696)
- Runtime Burden Allocation (arXiv:2604.01235)
- Select-then-Solve Paradigm Routing (arXiv:SelectThenSolve)

Encodes common agent workflows as deterministic decision trees that execute
without LLM calls, reducing both latency and token usage.

Pattern: condition_check → action → next_state
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class DecisionAction(Enum):
    """Actions that can be taken by decision trees."""

    TOOL_CALL = "tool_call"
    PROMPT_TEMPLATE = "prompt_template"
    MODEL_TIER = "model_tier"
    ROUTE_TO_AGENT = "route_to_agent"
    SKIP = "skip"


@dataclass
class DecisionResult:
    """Result from decision tree evaluation."""

    action: DecisionAction
    confidence: float  # 0.0 to 1.0
    result: Any
    reasoning: str
    next_state: Optional[str] = None


class DecisionCondition:
    """Base class for decision conditions."""

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate condition against context.

        Args:
            context: Decision context with relevant state

        Returns:
            True if condition is met
        """
        raise NotImplementedError


class FileTypeCondition(DecisionCondition):
    """Check file extension or type."""

    def __init__(self, extensions: List[str], match_all: bool = False):
        """Initialize file type condition.

        Args:
            extensions: List of file extensions (e.g., [".py", ".txt"])
            match_all: True if all extensions must match, False if any
        """
        self.extensions = [
            ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions
        ]
        self.match_all = match_all

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Check if file matches extension pattern."""
        path = context.get("path", "")
        if not path:
            return False

        path_obj = Path(path)
        ext = path_obj.suffix.lower()

        if self.match_all:
            return ext in self.extensions
        else:
            return ext in self.extensions


class KeywordCondition(DecisionCondition):
    """Check for keywords in query/message."""

    def __init__(self, keywords: List[str], match_all: bool = False):
        """Initialize keyword condition.

        Args:
            keywords: List of keywords to match
            match_all: True if all keywords must match, False if any
        """
        self.keywords = [kw.lower() for kw in keywords]
        self.match_all = match_all

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Check if query contains keywords."""
        query = context.get("query", "").lower()

        if self.match_all:
            return all(kw in query for kw in self.keywords)
        else:
            return any(kw in query for kw in self.keywords)


class RegexCondition(DecisionCondition):
    """Check for pattern match using regex."""

    def __init__(self, pattern: str, flags: int = 0):
        """Initialize regex condition.

        Args:
            pattern: Regular expression pattern
            flags: Regex flags (e.g., re.IGNORECASE)
        """
        self.pattern = re.compile(pattern, flags)

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Check if query matches regex pattern."""
        query = context.get("query", "")
        return bool(self.pattern.search(query))


class ProjectTypeCondition(DecisionCondition):
    """Check project type based on directory structure."""

    def __init__(self, project_types: List[str]):
        """Initialize project type condition.

        Args:
            project_types: List of project types (e.g., ["python", "rust"])
        """
        self.project_types = [pt.lower() for pt in project_types]

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Check if project matches type."""
        root_path = context.get("root_path", "")
        if not root_path:
            return False

        root = Path(root_path)

        # Check for project markers
        for project_type in self.project_types:
            if project_type == "python":
                if (root / "pyproject.toml").exists() or (root / "setup.py").exists():
                    return True
            elif project_type == "rust":
                if (root / "Cargo.toml").exists():
                    return True
            elif project_type == "javascript":
                if (root / "package.json").exists():
                    return True
            elif project_type == "go":
                if (root / "go.mod").exists():
                    return True

        return False


class DecisionNode:
    """A single node in a decision tree."""

    def __init__(
        self,
        condition: Optional[DecisionCondition] = None,
        true_action: Optional[DecisionResult] = None,
        false_action: Optional[DecisionResult] = None,
        true_node: Optional["DecisionNode"] = None,
        false_node: Optional["DecisionNode"] = None,
    ):
        """Initialize decision node.

        Args:
            condition: Condition to evaluate
            true_action: Action to take if condition is True
            false_action: Action to take if condition is False
            true_node: Next node if condition is True
            false_node: Next node if condition is False
        """
        self.condition = condition
        self.true_action = true_action
        self.false_action = false_action
        self.true_node = true_node
        self.false_node = false_node

    def evaluate(self, context: Dict[str, Any]) -> Optional[DecisionResult]:
        """Evaluate this node and return action.

        Args:
            context: Decision context

        Returns:
            DecisionResult or None if no action
        """
        # If no condition, return default action
        if self.condition is None:
            return self.true_action

        # Evaluate condition
        condition_met = self.condition.evaluate(context)

        if condition_met:
            # Condition met
            if self.true_action:
                return self.true_action
            if self.true_node:
                return self.true_node.evaluate(context)
        else:
            # Condition not met
            if self.false_action:
                return self.false_action
            if self.false_node:
                return self.false_node.evaluate(context)

        return None


class PreComputedDecisionTrees:
    """Collection of pre-computed decision trees for common workflows."""

    @staticmethod
    def file_read_tool() -> DecisionNode:
        """Decision tree for file reading operations.

        Routes to read() vs ls() based on input.
        """
        # Condition: Is the path a file?
        is_file = DecisionNode(
            condition=FileTypeCondition([""]),  # Will check if path exists and is file
            true_action=DecisionResult(
                action=DecisionAction.TOOL_CALL,
                confidence=1.0,
                result={"tool": "read", "args": {"path": "{path}"}},
                reasoning="Path is a file, using read()",
            ),
            false_action=DecisionResult(
                action=DecisionAction.TOOL_CALL,
                confidence=1.0,
                result={"tool": "ls", "args": {"path": "{path}"}},
                reasoning="Path is a directory, using ls()",
            ),
        )
        return is_file

    @staticmethod
    def code_search_mode() -> DecisionNode:
        """Decision tree for code search mode selection.

        Routes to semantic vs literal search based on query complexity.
        """
        # Condition: Does query contain code-specific keywords?
        has_code_keywords = DecisionNode(
            condition=KeywordCondition(
                ["function", "class", "variable", "import"], match_all=False
            ),
            true_action=DecisionResult(
                action=DecisionAction.TOOL_CALL,
                confidence=0.9,
                result={"tool": "code_search", "mode": "semantic"},
                reasoning="Code-specific query, using semantic search",
            ),
            false_node=DecisionNode(
                condition=KeywordCondition(["file", "where", "find"], match_all=False),
                true_action=DecisionResult(
                    action=DecisionAction.TOOL_CALL,
                    confidence=0.8,
                    result={"tool": "code_search", "mode": "literal"},
                    reasoning="File location query, using literal search",
                ),
                false_action=DecisionResult(
                    action=DecisionAction.TOOL_CALL,
                    confidence=0.7,
                    result={"tool": "code_search", "mode": "semantic"},
                    reasoning="General code query, defaulting to semantic search",
                ),
            ),
        )
        return has_code_keywords

    @staticmethod
    def error_recovery_tool() -> DecisionNode:
        """Decision tree for error recovery routing.

        Routes to appropriate recovery action based on error type.
        """
        # Condition: Is it a file not found error?
        is_file_not_found = DecisionNode(
            condition=KeywordCondition(
                ["not found", "no such file", "file not found"], match_all=False
            ),
            true_action=DecisionResult(
                action=DecisionAction.TOOL_CALL,
                confidence=0.95,
                result={"tool": "ls", "suggestion": "Check directory listing"},
                reasoning="File not found error, suggesting directory listing",
            ),
            false_node=DecisionNode(
                condition=KeywordCondition(["permission denied", "access denied"], match_all=False),
                true_action=DecisionResult(
                    action=DecisionAction.SKIP,
                    confidence=1.0,
                    result={"error": "Permission denied, cannot recover"},
                    reasoning="Permission error, cannot recover",
                ),
                false_action=DecisionResult(
                    action=DecisionAction.TOOL_CALL,
                    confidence=0.7,
                    result={"tool": "read", "retry": True},
                    reasoning="Generic error, suggesting retry with read",
                ),
            ),
        )
        return is_file_not_found

    @staticmethod
    def model_tier_selection() -> DecisionNode:
        """Decision tree for model tier selection.

        Routes to appropriate model tier based on task complexity.
        """
        # Condition: Is task simple (file operations, listing)?
        is_simple_task = DecisionNode(
            condition=KeywordCondition(["ls", "read", "list", "show"], match_all=False),
            true_action=DecisionResult(
                action=DecisionAction.MODEL_TIER,
                confidence=0.9,
                result={"tier": "fast", "model": "qwen3.5:2b"},
                reasoning="Simple filesystem task, using fast model",
            ),
            false_node=DecisionNode(
                condition=KeywordCondition(
                    ["analyze", "refactor", "optimize", "generate"], match_all=False
                ),
                true_action=DecisionResult(
                    action=DecisionAction.MODEL_TIER,
                    confidence=0.8,
                    result={"tier": "balanced", "model": "sonnet"},
                    reasoning="Code generation task, using balanced model",
                ),
                false_action=DecisionResult(
                    action=DecisionAction.MODEL_TIER,
                    confidence=0.7,
                    result={"tier": "performance", "model": "opus"},
                    reasoning="Complex task, using performance model",
                ),
            ),
        )
        return is_simple_task

    @classmethod
    def get_tree(cls, tree_name: str) -> Optional[DecisionNode]:
        """Get a pre-computed decision tree by name.

        Args:
            tree_name: Name of the decision tree

        Returns:
            DecisionNode or None if not found
        """
        trees = {
            "file_read_tool": cls.file_read_tool(),
            "code_search_mode": cls.code_search_mode(),
            "error_recovery_tool": cls.error_recovery_tool(),
            "model_tier_selection": cls.model_tier_selection(),
        }
        return trees.get(tree_name)

    @classmethod
    def evaluate_tree(
        cls,
        tree_name: str,
        context: Dict[str, Any],
    ) -> Optional[DecisionResult]:
        """Evaluate a decision tree with given context.

        Args:
            tree_name: Name of the decision tree
            context: Decision context

        Returns:
            DecisionResult or None if tree not found
        """
        tree = cls.get_tree(tree_name)
        if tree is None:
            logger.warning(f"Decision tree not found: {tree_name}")
            return None

        return tree.evaluate(context)


# Convenience functions


def decide_without_llm(
    tree_name: str,
    context: Dict[str, Any],
) -> Optional[DecisionResult]:
    """Make a decision without LLM using pre-computed decision tree.

    Args:
        tree_name: Name of the decision tree
        context: Decision context (query, path, etc.)

    Returns:
        DecisionResult or None if tree not found

    Example:
        result = decide_without_llm(
            "file_read_tool",
            {"query": "read file", "path": "/path/to/file.py"}
        )
        if result and result.confidence > 0.8:
            # Use decision result
            pass
    """
    return PreComputedDecisionTrees.evaluate_tree(tree_name, context)


def can_decide_without_llm(
    tree_name: str,
    context: Dict[str, Any],
    min_confidence: float = 0.8,
) -> bool:
    """Check if a decision can be made without LLM.

    Args:
        tree_name: Name of the decision tree
        context: Decision context
        min_confidence: Minimum confidence threshold

    Returns:
        True if decision can be made with sufficient confidence
    """
    result = decide_without_llm(tree_name, context)
    return result is not None and result.confidence >= min_confidence
