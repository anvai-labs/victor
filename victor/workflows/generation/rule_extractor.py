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

"""Rule-based requirement extraction (fallback).

This module provides a fallback extraction system using keyword matching,
regex patterns, and heuristics. Used when LLM extraction is unavailable or fails.

Design Pattern: Pattern Matching
- Uses regex for common workflow patterns
- Keyword-based detection for structure
- Heuristic confidence scoring
- No LLM dependency

Example:
    from victor.workflows.generation.rule_extractor import RuleBasedExtractor

    extractor = RuleBasedExtractor()
    requirements = extractor.extract("Run tests and if they pass, deploy")
    # Detects: sequential execution with conditional branch
"""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Optional

from victor.workflows.generation.requirements import (
    BranchRequirement,
    ContextRequirements,
    ExtractionMetadata,
    FunctionalRequirements,
    LoopRequirement,
    QualityRequirements,
    StructuralRequirements,
    TaskRequirement,
    WorkflowRequirements,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Rule-Based Extractor
# =============================================================================


class RuleBasedExtractor:
    """Fallback extractor using rules and patterns.

    Uses keyword matching, regex patterns, and heuristics to extract
    requirements when LLM extraction is unavailable or failed.

    Attributes:
        _task_verbs: Common task verbs to detect
        _tool_names: Known tool names
        _parallel_keywords: Keywords indicating parallel execution
        _conditional_keywords: Keywords indicating conditional branches

    Example:
        extractor = RuleBasedExtractor()
        requirements = extractor.extract("Run tests and if they pass, deploy")
        # Returns WorkflowRequirements with conditional structure
    """

    def __init__(self) -> None:
        """Initialize the rule-based extractor."""
        # Common task verbs
        self._task_verbs = [
            "analyze",
            "deploy",
            "test",
            "fix",
            "research",
            "summarize",
            "create",
            "generate",
            "review",
            "build",
            "compile",
            "document",
            "install",
            "configure",
            "monitor",
            "validate",
            "execute",
            "process",
            "transform",
            "extract",
            "load",
            "train",
            "evaluate",
            "predict",
        ]

        # Known tool names (from tool registry)
        self._tool_names = [
            "bash",
            "code_search",
            "file_read",
            "web_search",
            "git",
            "pytest",
            "npm",
            "docker",
            "kubectl",
            "terraform",
            "ansible",
            "jira",
            "slack",
            "github",
            "gitlab",
        ]

        # Structure keywords
        self._parallel_keywords = [
            "and",
            "simultaneously",
            "in parallel",
            "concurrently",
            "at the same time",
        ]

        self._conditional_keywords = [
            "if",
            "when",
            "unless",
            "otherwise",
            "else",
            "whether",
        ]

        self._loop_keywords = [
            "repeat",
            "retry",
            "until",
            "while",
            "for each",
            "iterate",
        ]

        # Vertical detection patterns
        self._vertical_patterns = {
            "coding": [
                r"\bcode\b",
                r"\bbug\b",
                r"\btest\b",
                r"\bdebug\b",
                r"\brefactor\b",
                r"\bimplement\b",
            ],
            "devops": [
                r"\bdeploy\b",
                r"\bbuild\b",
                r"\bci/cd\b",
                r"\binfrastructure\b",
                r"\bkubernetes\b",
                r"\bdocker\b",
            ],
            "research": [
                r"\bresearch\b",
                r"\binvestigate\b",
                r"\banalyze\b",
                r"\bfind\b",
                r"\bexplore\b",
                r"\bstudy\b",
            ],
            "rag": [
                r"\bdocument\b",
                r"\bsearch\b",
                r"\bindex\b",
                r"\bembeddings?\b",
                r"\bvector\b",
                r"\bknowledge\b",
            ],
            "dataanalysis": [
                r"\bdata\b",
                r"\banalyze\b",
                r"\bvisuali[sz]e\b",
                r"\bstatistic\b",
                r"\bplot\b",
                r"\bchart\b",
            ],
        }

    def extract(self, description: str) -> WorkflowRequirements:
        """Extract requirements using rule-based patterns.

        Args:
            description: Natural language workflow description

        Returns:
            WorkflowRequirements with extracted information
        """
        logger.info("Using rule-based extraction")

        # Detect execution order
        execution_order = self._detect_execution_order(description)

        # Extract tasks using verb patterns
        tasks = self._extract_tasks(description)

        # Extract tool mentions
        tools = self._extract_tools(description, tasks)

        # Detect conditional keywords
        branches = self._detect_branches(description)

        # Detect loop patterns
        loops = self._detect_loops(description)

        # Infer dependencies
        dependencies = self._infer_dependencies(tasks)

        # Detect vertical
        vertical = self._detect_vertical(description)

        # Build requirements
        functional = FunctionalRequirements(
            tasks=tasks,
            tools=tools,
            success_criteria=self._extract_success_criteria(description),
        )

        structural = StructuralRequirements(
            execution_order=execution_order,
            dependencies=dependencies,
            branches=branches,
            loops=loops,
        )

        quality = QualityRequirements(
            max_cost_tier="MEDIUM",
            retry_policy="retry",
        )

        context = ContextRequirements(
            vertical=vertical,
            environment="local",
        )

        # Calculate confidence (rule-based is lower confidence)
        confidence_scores = {
            "functional": 0.6 if tasks else 0.0,
            "structural": 0.5 if execution_order else 0.0,
            "quality": 0.3,
            "context": 0.7 if vertical else 0.0,
        }

        metadata = ExtractionMetadata(
            extraction_method="rules",
            extraction_time=0.1,  # Very fast
            confidence=sum(confidence_scores.values()) / len(confidence_scores),
        )

        return WorkflowRequirements(
            description=description,
            functional=functional,
            structural=structural,
            quality=quality,
            context=context,
            confidence_scores=confidence_scores,
            metadata=metadata,
        )

    def _detect_execution_order(self, text: str) -> str:
        """Detect execution order from keywords.

        Args:
            text: Input text to analyze

        Returns:
            Execution order (sequential, parallel, conditional)
        """
        text_lower = text.lower()

        # Check for conditional first (highest priority)
        if any(kw in text_lower for kw in self._conditional_keywords):
            return "conditional"

        # Check for parallel
        if any(kw in text_lower for kw in self._parallel_keywords):
            return "parallel"

        # Default to sequential
        return "sequential"

    def _extract_tasks(self, text: str) -> List[TaskRequirement]:
        """Extract tasks using verb-noun patterns.

        Args:
            text: Input text to analyze

        Returns:
            List of extracted tasks
        """
        tasks = []
        seen_tasks = set()

        # Pattern: verb + noun phrase
        # "analyze code", "run tests", "deploy to staging"
        for i, verb in enumerate(self._task_verbs):
            # Pattern with flexible matching
            pattern = rf"\b{verb}\s+([^.!?;]+)(?:[.!?;]|$)"
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                task_desc = match.group(0).strip()

                # Avoid duplicates
                if task_desc.lower() in seen_tasks:
                    continue
                seen_tasks.add(task_desc.lower())

                # Determine task type
                task_type = self._infer_task_type(task_desc)

                # Determine role if agent task
                role = self._infer_role(task_desc, task_type)

                tasks.append(
                    TaskRequirement(
                        id=f"task_{i}",
                        description=task_desc,
                        task_type=task_type,
                        role=role if task_type == "agent" else None,
                    )
                )

        # Fallback: split by common delimiters if no tasks found
        if not tasks:
            # Try splitting by comma, "and", "then"
            parts = re.split(r",\s*|\s+and\s+|\s+then\s+", text)
            for i, part in enumerate(parts):
                part = part.strip()
                if len(part) > 3:  # Minimum length threshold
                    tasks.append(
                        TaskRequirement(
                            id=f"task_{i}",
                            description=part,
                            task_type="agent",
                            role="executor",
                        )
                    )

        return tasks

    def _infer_task_type(self, description: str) -> str:
        """Infer task type from description.

        Args:
            description: Task description

        Returns:
            Task type (agent, compute, condition, etc.)
        """
        desc_lower = description.lower()

        # Compute tasks (direct tool usage)
        compute_verbs = ["run", "execute", "compile", "build", "deploy"]
        if any(desc_lower.startswith(v) for v in compute_verbs):
            return "compute"

        # Condition tasks
        if any(kw in desc_lower for kw in ["check", "verify", "validate"]):
            return "condition"

        # Default to agent
        return "agent"

    def _infer_role(self, description: str, task_type: str) -> Optional[str]:
        """Infer agent role from task description.

        Args:
            description: Task description
            task_type: Task type

        Returns:
            Role name or None
        """
        if task_type != "agent":
            return None

        desc_lower = description.lower()

        # Research roles
        if any(
            kw in desc_lower for kw in ["research", "investigate", "analyze", "explore", "study"]
        ):
            return "researcher"

        # Execution roles
        if any(kw in desc_lower for kw in ["fix", "implement", "build", "create", "deploy"]):
            return "executor"

        # Planning roles
        if any(kw in desc_lower for kw in ["plan", "design", "architect"]):
            return "planner"

        # Review roles
        if any(kw in desc_lower for kw in ["review", "check", "validate"]):
            return "reviewer"

        # Writing roles
        if any(kw in desc_lower for kw in ["write", "document", "summarize"]):
            return "writer"

        # Default
        return "executor"

    def _extract_tools(self, text: str, tasks: List[TaskRequirement]) -> Dict[str, List[str]]:
        """Extract tool names from text.

        Args:
            text: Input text
            tasks: Extracted tasks

        Returns:
            Dict mapping task IDs to tool lists
        """
        tools = {}
        text_lower = text.lower()

        # Find mentioned tools
        mentioned_tools = []
        for tool in self._tool_names:
            if tool in text_lower:
                mentioned_tools.append(tool)

        # Associate tools with tasks (heuristic: all tasks get mentioned tools)
        # In a real implementation, this would be more sophisticated
        for task in tasks:
            if mentioned_tools:
                tools[task.id] = mentioned_tools

        return tools

    def _detect_branches(self, text: str) -> List[BranchRequirement]:
        """Detect conditional branches from text.

        Args:
            text: Input text

        Returns:
            List of detected branches
        """
        branches = []

        # Pattern: "if X then Y else Z"
        pattern = r"if\s+(.+?)\s+then\s+(.+?)(?:\s+else\s+(.+?))?(?:[.!?;]|$)"
        matches = re.finditer(pattern, text, re.IGNORECASE)

        for i, match in enumerate(matches):
            condition = match.group(1).strip()
            true_branch = match.group(2).strip()
            false_branch = match.group(3).strip() if match.group(3) else "end"

            branches.append(
                BranchRequirement(
                    condition_id=f"branch_{i}",
                    condition=condition,
                    true_branch=true_branch,
                    false_branch=false_branch,
                    condition_type="data_check",
                )
            )

        return branches

    def _detect_loops(self, text: str) -> List[LoopRequirement]:
        """Detect loop patterns from text.

        Args:
            text: Input text

        Returns:
            List of detected loops
        """
        loops = []

        # Pattern: "repeat X until Y"
        pattern = r"(?:repeat|retry)\s+(.+?)\s+(?:until|while)\s+(.+?)(?:[.!?;]|$)"
        matches = re.finditer(pattern, text, re.IGNORECASE)

        for i, match in enumerate(matches):
            task_desc = match.group(1).strip()
            exit_condition = match.group(2).strip()

            loops.append(
                LoopRequirement(
                    loop_id=f"loop_{i}",
                    task_to_repeat=task_desc,
                    exit_condition=exit_condition,
                    max_iterations=3,
                )
            )

        return loops

    def _infer_dependencies(self, tasks: List[TaskRequirement]) -> Dict[str, List[str]]:
        """Infer task dependencies from order.

        Args:
            tasks: List of tasks

        Returns:
            Dict mapping task IDs to dependency IDs
        """
        dependencies = {}

        # Simple heuristic: each task depends on the previous one
        for i in range(1, len(tasks)):
            task_id = tasks[i].id
            dep_id = tasks[i - 1].id
            dependencies[task_id] = [dep_id]

        return dependencies

    def _detect_vertical(self, text: str) -> str:
        """Detect domain vertical from text.

        Args:
            text: Input text

        Returns:
            Vertical name (coding, devops, etc.)
        """
        text_lower = text.lower()

        # Score each vertical
        scores = {}
        for vertical, patterns in self._vertical_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1
            scores[vertical] = score

        # Return vertical with highest score
        if scores:
            best_vertical = max(scores, key=scores.get)
            if scores[best_vertical] > 0:
                return best_vertical

        # Default to coding
        return "coding"

    def _extract_success_criteria(self, text: str) -> List[str]:
        """Extract success criteria from text.

        Args:
            text: Input text

        Returns:
            List of success criteria
        """
        criteria = []

        # Pattern: "ensure X", "verify X", "make sure X"
        patterns = [
            r"ensure\s+([^.!?]+)",
            r"verify\s+([^.!?]+)",
            r"make sure\s+([^.!?]+)",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                criteria.append(match.group(1).strip())

        return criteria
