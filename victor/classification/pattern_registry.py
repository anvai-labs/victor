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

"""Unified pattern registry for task classification.

This module provides a single source of truth for all classification patterns,
eliminating duplication between TaskTypeClassifier and TaskComplexityService.

Design Principles:
- Single Responsibility: Only defines patterns, no classification logic
- Open/Closed: New patterns can be added without modifying existing code
- Patterns are used by both semantic classifiers and nudge engine
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Pattern

from victor.framework.task.protocols import TaskComplexity

# Adapter-local pre-map (tool-supply P7 follow-on) for the handful of members that have
# NO alias or canonical name in the *default* (core) ``TaskTypeRegistry``. Everything
# else is either a core type, an existing core alias (``implement`` -> create,
# ``plan`` / ``architecture`` -> design), or a vertical-scoped type that resolves to
# itself (kubernetes, terraform, data_profiling, …). These six are the genuine orphans;
# they map to the closest core type so ``to_canonical()`` never yields an unregistered
# name. ``bug_fix`` / ``issue_resolution`` follow the registry's existing
# ``bugfix`` / ``fix_bug`` -> edit decision for consistency.
_CANONICAL_OVERRIDES: Dict[str, str] = {
    "bug_fix": "edit",
    "issue_resolution": "edit",
    "data_analysis": "analyze",
    "security": "analyze",
    "explain": "analyze",
    "general_query": "general",
}


class TaskType(Enum):
    """Prompt-classification task labels (an *adapter* over the canonical registry).

    These labels are the fine-grained output of pattern classification. The single
    authority for task-type identity/aliasing is
    ``victor.framework.task_types.TaskTypeRegistry``; use :meth:`to_canonical` to
    collapse a member onto a registered canonical name (tool-supply P7 follow-on)
    rather than treating this enum as a parallel taxonomy.

    Semantic Variants (all different purposes):
    - TaskType (here): Prompt classification labels (this adapter)
    - TrackerTaskType: Progress tracking with milestones
    - LoopDetectorTaskType: Loop detection thresholds
    - ClassifierTaskType: Unified classification output
    - FrameworkTaskType: Framework-level task abstraction
    """

    # Core task types
    EDIT = "edit"
    SEARCH = "search"
    CREATE = "create"
    CREATE_SIMPLE = "create_simple"
    ANALYZE = "analyze"
    DESIGN = "design"
    GENERAL = "general"
    ACTION = "action"
    ANALYSIS_DEEP = "analysis_deep"

    # Bug fix / issue resolution
    BUG_FIX = "bug_fix"
    ISSUE_RESOLUTION = "issue_resolution"

    # DevOps task types
    INFRASTRUCTURE = "infrastructure"
    CI_CD = "ci_cd"
    DOCKERFILE = "dockerfile"
    DOCKER_COMPOSE = "docker_compose"
    KUBERNETES = "kubernetes"
    TERRAFORM = "terraform"
    MONITORING = "monitoring"

    # Coding granular types
    CODE_GENERATION = "code_generation"
    REFACTOR = "refactor"
    DEBUG = "debug"
    TEST = "test"

    # Data Analysis task types
    DATA_ANALYSIS = "data_analysis"
    VISUALIZATION = "visualization"
    DATA_PROFILING = "data_profiling"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"

    # Research task types
    FACT_CHECK = "fact_check"
    LITERATURE_REVIEW = "literature_review"
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    TREND_RESEARCH = "trend_research"
    TECHNICAL_RESEARCH = "technical_research"
    GENERAL_QUERY = "general_query"

    # Implementation task types
    IMPLEMENT = "implement"
    SECURITY = "security"
    ARCHITECTURE = "architecture"
    PLAN = "plan"
    EXPLAIN = "explain"

    def to_canonical(self) -> str:
        """Canonical task-type name in the ``TaskTypeRegistry`` (tool-supply P7).

        Collapses this adapter enum onto the single registry taxonomy. Core members
        and existing core aliases resolve directly; vertical-scoped members
        (kubernetes, terraform, …) resolve to themselves (they are registered under
        their vertical); the genuine orphans route through ``_CANONICAL_OVERRIDES``.
        """
        from victor.framework.task_types import canonicalize_task_type

        return canonicalize_task_type(self.value, overrides=_CANONICAL_OVERRIDES)

    @classmethod
    def from_canonical(cls, name: str) -> "Optional[TaskType]":
        """Best-effort reverse map: a canonical name -> the member that yields it.

        Several members may share a canonical name (e.g. bug_fix/issue_resolution ->
        edit); the first definition-order member is returned, which is canonical-stable
        on round trip.
        """
        from victor.framework.task_types import canonicalize_task_type

        target = canonicalize_task_type(name, overrides=_CANONICAL_OVERRIDES)
        for member in cls:
            if member.to_canonical() == target:
                return member
        return None


@dataclass(frozen=True)
class ClassificationPattern:
    """Unified pattern definition used by all classifiers.

    This replaces both NudgeRule patterns and complexity PATTERNS.
    Each pattern maps to both a TaskType (fine-grained) and TaskComplexity (coarse).

    Attributes:
        name: Unique identifier for debugging/logging
        regex: Regular expression pattern to match
        semantic_intent: Phrase for embedding similarity search
        task_type: Fine-grained task type classification
        complexity: Coarse complexity level for budgeting
        confidence: Classification confidence (0.0 to 1.0)
        override: If True, always use this classification when pattern matches
        priority: Higher priority patterns are checked first (default 50)
    """

    name: str
    regex: str
    semantic_intent: str
    task_type: TaskType
    complexity: TaskComplexity
    confidence: float = 0.9
    override: bool = False
    priority: int = 50

    @property
    def compiled_pattern(self) -> Pattern:
        """Get compiled regex pattern (cached)."""
        return re.compile(self.regex, re.IGNORECASE)


# Task type to complexity mapping (default mapping)
TASK_TYPE_TO_COMPLEXITY: Dict[TaskType, TaskComplexity] = {
    # Simple tasks
    TaskType.SEARCH: TaskComplexity.SIMPLE,
    TaskType.DESIGN: TaskComplexity.SIMPLE,
    TaskType.GENERAL: TaskComplexity.SIMPLE,
    # Medium tasks
    TaskType.ANALYZE: TaskComplexity.MEDIUM,
    TaskType.EDIT: TaskComplexity.MEDIUM,
    # Complex tasks
    TaskType.CREATE: TaskComplexity.COMPLEX,
    TaskType.REFACTOR: TaskComplexity.COMPLEX,
    TaskType.DEBUG: TaskComplexity.COMPLEX,
    # Generation tasks
    TaskType.CREATE_SIMPLE: TaskComplexity.GENERATION,
    TaskType.CODE_GENERATION: TaskComplexity.GENERATION,
    # Action tasks (multi-step)
    TaskType.ACTION: TaskComplexity.ACTION,
    TaskType.BUG_FIX: TaskComplexity.ACTION,
    TaskType.ISSUE_RESOLUTION: TaskComplexity.ACTION,
    TaskType.TEST: TaskComplexity.ACTION,
    # Analysis tasks
    TaskType.ANALYSIS_DEEP: TaskComplexity.ANALYSIS,
    TaskType.DATA_ANALYSIS: TaskComplexity.ANALYSIS,
    TaskType.STATISTICAL_ANALYSIS: TaskComplexity.ANALYSIS,
    # DevOps tasks
    TaskType.INFRASTRUCTURE: TaskComplexity.COMPLEX,
    TaskType.CI_CD: TaskComplexity.COMPLEX,
    TaskType.KUBERNETES: TaskComplexity.COMPLEX,
    TaskType.TERRAFORM: TaskComplexity.COMPLEX,
    TaskType.DOCKERFILE: TaskComplexity.MEDIUM,
    TaskType.DOCKER_COMPOSE: TaskComplexity.MEDIUM,
    TaskType.MONITORING: TaskComplexity.COMPLEX,
    # Visualization
    TaskType.VISUALIZATION: TaskComplexity.MEDIUM,
    TaskType.DATA_PROFILING: TaskComplexity.MEDIUM,
    TaskType.CORRELATION_ANALYSIS: TaskComplexity.MEDIUM,
    TaskType.REGRESSION: TaskComplexity.COMPLEX,
    TaskType.CLUSTERING: TaskComplexity.COMPLEX,
    TaskType.TIME_SERIES: TaskComplexity.COMPLEX,
    # Research tasks
    TaskType.FACT_CHECK: TaskComplexity.MEDIUM,
    TaskType.LITERATURE_REVIEW: TaskComplexity.ANALYSIS,
    TaskType.COMPETITIVE_ANALYSIS: TaskComplexity.ANALYSIS,
    TaskType.TREND_RESEARCH: TaskComplexity.ANALYSIS,
    TaskType.TECHNICAL_RESEARCH: TaskComplexity.ANALYSIS,
    TaskType.GENERAL_QUERY: TaskComplexity.MEDIUM,
    # Implementation tasks
    TaskType.IMPLEMENT: TaskComplexity.COMPLEX,
    TaskType.SECURITY: TaskComplexity.ACTION,
    TaskType.ARCHITECTURE: TaskComplexity.COMPLEX,
    TaskType.PLAN: TaskComplexity.MEDIUM,
    TaskType.EXPLAIN: TaskComplexity.SIMPLE,
}


# Single source of truth for all classification patterns
# Organized by category for maintainability
PATTERNS: Dict[str, ClassificationPattern] = {}


def _register_patterns():
    """Register all patterns. Called on module load."""
    global PATTERNS

    patterns_list = [
        # =================================================================
        # Bug Fix / Issue Resolution (SWE-bench style) - Highest Priority
        # =================================================================
        ClassificationPattern(
            name="github_issue_format",
            regex=r"###\s*(Description|Expected\s+behavior|Actual\s+behavior|Steps)",
            semantic_intent="fix github issue bug report error",
            task_type=TaskType.BUG_FIX,
            complexity=TaskComplexity.ACTION,
            confidence=1.0,
            override=True,
            priority=100,
        ),
        ClassificationPattern(
            name="steps_to_reproduce",
            regex=r"\bSteps\s+to\s+Reproduce\b",
            semantic_intent="reproduce bug steps error",
            task_type=TaskType.BUG_FIX,
            complexity=TaskComplexity.ACTION,
            confidence=1.0,
            override=True,
            priority=100,
        ),
        ClassificationPattern(
            name="fix_the_issue",
            regex=r"\b(fix|resolve|address)\s+(this|the)\s+(issue|bug|problem)\b",
            semantic_intent="fix resolve bug issue problem",
            task_type=TaskType.BUG_FIX,
            complexity=TaskComplexity.ACTION,
            confidence=1.0,
            override=True,
            priority=95,
        ),
        ClassificationPattern(
            name="traceback_error",
            regex=r"\bTraceback\b[\s\S]*(Error|Exception)\b",
            semantic_intent="traceback error exception python",
            task_type=TaskType.BUG_FIX,
            complexity=TaskComplexity.ACTION,
            confidence=1.0,
            override=True,
            priority=100,
        ),
        ClassificationPattern(
            name="expected_vs_actual",
            regex=r"\bexpected\s+.+\s+but\s+(got|received|returns?)\b",
            semantic_intent="expected actual mismatch bug",
            task_type=TaskType.BUG_FIX,
            complexity=TaskComplexity.ACTION,
            confidence=0.95,
            override=False,
            priority=90,
        ),
        ClassificationPattern(
            name="failed_to_converge",
            regex=r"\bfailed\s+to\s+(converge|complete|work)\b",
            semantic_intent="failed error converge work",
            task_type=TaskType.BUG_FIX,
            complexity=TaskComplexity.ACTION,
            confidence=0.9,
            override=False,
            priority=85,
        ),
        ClassificationPattern(
            name="bug_behavior",
            regex=r"\b(unexpected|wrong|incorrect)\s+(behavior|result|output|error)\b",
            semantic_intent="unexpected wrong behavior bug",
            task_type=TaskType.BUG_FIX,
            complexity=TaskComplexity.COMPLEX,
            confidence=0.95,
            override=False,
            priority=85,
        ),
        ClassificationPattern(
            name="test_fails",
            regex=r"\b(test|tests?)\s+(fails?|failing|broken)\b",
            semantic_intent="test fails failing broken",
            task_type=TaskType.BUG_FIX,
            complexity=TaskComplexity.ACTION,
            confidence=1.0,
            override=True,
            priority=95,
        ),
        # =================================================================
        # Analysis / Understanding (read-only operations)
        # =================================================================
        ClassificationPattern(
            name="read_and_explain",
            regex=r"^read\s+(and\s+)?explain",
            semantic_intent="read explain understand code",
            task_type=TaskType.ANALYZE,
            complexity=TaskComplexity.MEDIUM,
            confidence=1.0,
            override=True,
            priority=90,
        ),
        ClassificationPattern(
            name="explain_code_in",
            regex=r"^explain\s+(the\s+)?code\s+in",
            semantic_intent="explain code understanding",
            task_type=TaskType.ANALYZE,
            complexity=TaskComplexity.MEDIUM,
            confidence=1.0,
            override=True,
            priority=90,
        ),
        ClassificationPattern(
            name="analyze_code_in",
            regex=r"^analyze\s+(the\s+)?code\s+in",
            semantic_intent="analyze code review",
            task_type=TaskType.ANALYZE,
            complexity=TaskComplexity.MEDIUM,
            confidence=1.0,
            override=True,
            priority=90,
        ),
        ClassificationPattern(
            name="review_code_in",
            regex=r"^review\s+(the\s+)?code\s+in",
            semantic_intent="review code quality",
            task_type=TaskType.ANALYZE,
            complexity=TaskComplexity.MEDIUM,
            confidence=1.0,
            override=True,
            priority=90,
        ),
        ClassificationPattern(
            name="count_query",
            regex=r"^(count|how\s+many)\s+",
            semantic_intent="count how many metrics",
            task_type=TaskType.ANALYZE,
            complexity=TaskComplexity.SIMPLE,
            confidence=0.9,
            override=False,
            priority=80,
        ),
        ClassificationPattern(
            name="comprehensive_analysis",
            regex=r"\b(comprehensive|thorough|detailed|full)\s+(analysis|review|audit)\b",
            semantic_intent="comprehensive thorough analysis audit",
            task_type=TaskType.ANALYSIS_DEEP,
            complexity=TaskComplexity.ANALYSIS,
            confidence=1.0,
            override=True,
            priority=85,
        ),
        ClassificationPattern(
            name="architecture_review",
            regex=r"\barchitecture\s+(review|analysis|overview)\b",
            semantic_intent="architecture review system design",
            task_type=TaskType.ANALYSIS_DEEP,
            complexity=TaskComplexity.ANALYSIS,
            confidence=1.0,
            override=True,
            priority=85,
        ),
        ClassificationPattern(
            name="security_audit",
            regex=r"\b(security|vulnerability)\s+(audit|scan|review)\b",
            semantic_intent="security vulnerability audit",
            task_type=TaskType.ANALYSIS_DEEP,
            complexity=TaskComplexity.ANALYSIS,
            confidence=0.95,
            override=False,
            priority=85,
        ),
        # =================================================================
        # Search / Find operations
        # =================================================================
        ClassificationPattern(
            name="find_inheritance",
            regex=r"find\s+(all\s+)?class(es)?\s+(that\s+)?inherit",
            semantic_intent="find classes inherit subclass",
            task_type=TaskType.SEARCH,
            complexity=TaskComplexity.SIMPLE,
            confidence=1.0,
            override=True,
            priority=85,
        ),
        ClassificationPattern(
            name="list_subclasses",
            regex=r"list\s+(all\s+)?(subclass|class)",
            semantic_intent="list subclasses classes",
            task_type=TaskType.SEARCH,
            complexity=TaskComplexity.SIMPLE,
            confidence=0.9,
            override=False,
            priority=80,
        ),
        ClassificationPattern(
            name="find_definitions",
            regex=r"\bfind\s+(all\s+)?(classes?|functions?|methods?)\b",
            semantic_intent="find classes functions definitions",
            task_type=TaskType.SEARCH,
            complexity=TaskComplexity.SIMPLE,
            confidence=0.8,
            override=False,
            priority=75,
        ),
        ClassificationPattern(
            name="where_query",
            regex=r"\bwhere\s+(is|are|does)\b",
            semantic_intent="where locate find",
            task_type=TaskType.SEARCH,
            complexity=TaskComplexity.SIMPLE,
            confidence=0.8,
            override=False,
            priority=70,
        ),
        ClassificationPattern(
            name="git_read",
            regex=r"\bgit\s+(status|log|branch|diff|show)\b",
            semantic_intent="git status log branch view",
            task_type=TaskType.SEARCH,
            complexity=TaskComplexity.SIMPLE,
            confidence=1.0,
            override=True,
            priority=85,
        ),
        # =================================================================
        # Edit / Modify operations
        # =================================================================
        ClassificationPattern(
            name="edit_command",
            regex=r"^edit\s+[\w/\\.-]+\.(py|js|ts|go|rs|java|yaml|json|md)",
            semantic_intent="edit modify file",
            task_type=TaskType.EDIT,
            complexity=TaskComplexity.MEDIUM,
            confidence=1.0,
            override=True,
            priority=90,
        ),
        ClassificationPattern(
            name="refactor",
            regex=r"\brefactor\b",
            semantic_intent="refactor restructure code",
            task_type=TaskType.REFACTOR,
            complexity=TaskComplexity.COMPLEX,
            confidence=0.9,
            override=False,
            priority=80,
        ),
        ClassificationPattern(
            name="debug_issue",
            regex=r"\bdebug\s+(the\s+)?(issue|problem|error|bug|code)\b",
            semantic_intent="debug issue problem error",
            task_type=TaskType.DEBUG,
            complexity=TaskComplexity.COMPLEX,
            confidence=0.95,
            override=False,
            priority=80,
        ),
        ClassificationPattern(
            name="write_tests",
            regex=r"\b(write|create|add)\s+(unit\s+)?(tests?|test\s+cases?)\b",
            semantic_intent="write create unit tests",
            task_type=TaskType.TEST,
            complexity=TaskComplexity.ACTION,
            confidence=0.95,
            override=False,
            priority=80,
        ),
        # =================================================================
        # Create / Generate operations
        # =================================================================
        ClassificationPattern(
            name="simple_create",
            regex=r"(create|write|make)\s+a?\s*simple\s+",
            semantic_intent="create simple function code",
            task_type=TaskType.CREATE_SIMPLE,
            complexity=TaskComplexity.GENERATION,
            confidence=0.9,
            override=False,
            priority=80,
        ),
        ClassificationPattern(
            name="just_function",
            regex=r"^just\s+(write|create|make)\s+",
            semantic_intent="just create write simple",
            task_type=TaskType.CREATE_SIMPLE,
            complexity=TaskComplexity.GENERATION,
            confidence=0.9,
            override=False,
            priority=80,
        ),
        ClassificationPattern(
            name="generate_code",
            regex=r"\b(create|write|generate)\s+(a\s+)?(simple\s+)?(function|script|code)\b",
            semantic_intent="create write generate function script",
            task_type=TaskType.CODE_GENERATION,
            complexity=TaskComplexity.GENERATION,
            confidence=1.0,
            override=False,
            priority=75,
        ),
        ClassificationPattern(
            name="implement_feature",
            regex=r"\b(implement|add|create)\s+(a\s+)?(new\s+)?(feature|system|module)\b",
            semantic_intent="implement add create feature module",
            task_type=TaskType.CREATE,
            complexity=TaskComplexity.COMPLEX,
            confidence=0.9,
            override=False,
            priority=75,
        ),
        # =================================================================
        # Action / Execution operations
        # =================================================================
        ClassificationPattern(
            name="git_commit",
            regex=r"\b(commit|push|pull|merge)\s+(all|the|these|my)?\s*(changes?|files?)?\b",
            semantic_intent="git commit push changes",
            task_type=TaskType.ACTION,
            complexity=TaskComplexity.ACTION,
            confidence=1.0,
            override=True,
            priority=90,
        ),
        ClassificationPattern(
            name="git_command",
            regex=r"\bgit\s+(add|commit|push|pull|merge|rebase|stash)\b",
            semantic_intent="git add commit push",
            task_type=TaskType.ACTION,
            complexity=TaskComplexity.ACTION,
            confidence=1.0,
            override=True,
            priority=90,
        ),
        ClassificationPattern(
            name="create_pr",
            regex=r"\b(create|make|open)\s+(a\s+)?(pr|pull\s*request)\b",
            semantic_intent="create pull request PR",
            task_type=TaskType.ACTION,
            complexity=TaskComplexity.ACTION,
            confidence=1.0,
            override=True,
            priority=90,
        ),
        ClassificationPattern(
            name="run_tests",
            regex=r"\b(run|execute)\s+(the\s+)?(tests?|pytest|unittest|jest|mocha)\b",
            semantic_intent="run execute tests pytest",
            task_type=TaskType.ACTION,
            complexity=TaskComplexity.ACTION,
            confidence=1.0,
            override=True,
            priority=90,
        ),
        ClassificationPattern(
            name="test_command",
            regex=r"\bpytest\b|\bnpm\s+test\b|\bcargo\s+test\b",
            semantic_intent="pytest npm cargo test",
            task_type=TaskType.ACTION,
            complexity=TaskComplexity.ACTION,
            confidence=1.0,
            override=True,
            priority=90,
        ),
        ClassificationPattern(
            name="build_deploy",
            regex=r"\b(build|compile|deploy)\s+(the\s+)?(project|app|code)\b",
            semantic_intent="build compile deploy project",
            task_type=TaskType.ACTION,
            complexity=TaskComplexity.ACTION,
            confidence=0.9,
            override=False,
            priority=85,
        ),
        ClassificationPattern(
            name="web_search_action",
            regex=r"\b(perform|do|run)\s+(a\s+)?(web\s*search|websearch)\b",
            semantic_intent="web search internet",
            task_type=TaskType.ACTION,
            complexity=TaskComplexity.ACTION,
            confidence=1.0,
            override=True,
            priority=85,
        ),
        # =================================================================
        # Design / Planning (no tools needed)
        # =================================================================
        ClassificationPattern(
            name="explain_what_does",
            regex=r"^explain\s+what\s+",
            semantic_intent="explain what concept",
            task_type=TaskType.DESIGN,
            complexity=TaskComplexity.SIMPLE,
            confidence=1.0,
            override=True,
            priority=85,
        ),
        ClassificationPattern(
            name="what_does",
            regex=r"^what\s+does\s+",
            semantic_intent="what does explain",
            task_type=TaskType.DESIGN,
            complexity=TaskComplexity.SIMPLE,
            confidence=1.0,
            override=True,
            priority=85,
        ),
        ClassificationPattern(
            name="without_code",
            regex=r"without\s+(reading|looking\s+at)\s+(any\s+)?(code|files)",
            semantic_intent="without reading code conceptual",
            task_type=TaskType.DESIGN,
            complexity=TaskComplexity.SIMPLE,
            confidence=1.0,
            override=True,
            priority=90,
        ),
        # =================================================================
        # DevOps: Infrastructure
        # =================================================================
        ClassificationPattern(
            name="kubernetes_config",
            regex=r"\b(kubernetes|k8s)\s+(deployment|service|ingress|config|manifest|pod|statefulset)\b",
            semantic_intent="kubernetes k8s deployment config",
            task_type=TaskType.KUBERNETES,
            complexity=TaskComplexity.COMPLEX,
            confidence=1.0,
            override=True,
            priority=85,
        ),
        ClassificationPattern(
            name="terraform_infra",
            regex=r"\b(terraform|tf)\s+(module|resource|plan|apply|config)\b",
            semantic_intent="terraform infrastructure config",
            task_type=TaskType.TERRAFORM,
            complexity=TaskComplexity.COMPLEX,
            confidence=1.0,
            override=True,
            priority=85,
        ),
        ClassificationPattern(
            name="dockerfile_create",
            regex=r"\b(create|write|optimize)\s+(a\s+)?dockerfile\b",
            semantic_intent="create dockerfile container",
            task_type=TaskType.DOCKERFILE,
            complexity=TaskComplexity.MEDIUM,
            confidence=1.0,
            override=True,
            priority=85,
        ),
        ClassificationPattern(
            name="docker_compose",
            regex=r"\b(docker[-_]?compose|compose\.ya?ml)\b",
            semantic_intent="docker compose container",
            task_type=TaskType.DOCKER_COMPOSE,
            complexity=TaskComplexity.MEDIUM,
            confidence=0.9,
            override=False,
            priority=80,
        ),
        ClassificationPattern(
            name="helm_chart",
            regex=r"\bhelm\s+(chart|template|install|upgrade)\b",
            semantic_intent="helm chart kubernetes",
            task_type=TaskType.KUBERNETES,
            complexity=TaskComplexity.COMPLEX,
            confidence=1.0,
            override=True,
            priority=85,
        ),
        # =================================================================
        # DevOps: CI/CD
        # =================================================================
        ClassificationPattern(
            name="github_actions",
            regex=r"\bgithub\s+(actions?|workflow)\b",
            semantic_intent="github actions workflow CI",
            task_type=TaskType.CI_CD,
            complexity=TaskComplexity.COMPLEX,
            confidence=1.0,
            override=True,
            priority=85,
        ),
        ClassificationPattern(
            name="ci_cd_pipeline",
            regex=r"\b(ci[/-]?cd|pipeline|jenkins|gitlab[-_]?ci|circleci)\b",
            semantic_intent="CI CD pipeline jenkins",
            task_type=TaskType.CI_CD,
            complexity=TaskComplexity.COMPLEX,
            confidence=0.9,
            override=False,
            priority=80,
        ),
        # =================================================================
        # Data Analysis
        # =================================================================
        ClassificationPattern(
            name="statistical_analysis",
            regex=r"\b(statistical|stats)\s+(analysis|test|correlation|regression)\b",
            semantic_intent="statistical analysis correlation",
            task_type=TaskType.STATISTICAL_ANALYSIS,
            complexity=TaskComplexity.ANALYSIS,
            confidence=1.0,
            override=True,
            priority=85,
        ),
        ClassificationPattern(
            name="analyze_data",
            regex=r"\banalyze\s+(the\s+)?(data|dataset|dataframe|csv|excel)\b",
            semantic_intent="analyze data dataset",
            task_type=TaskType.DATA_ANALYSIS,
            complexity=TaskComplexity.ANALYSIS,
            confidence=1.0,
            override=True,
            priority=85,
        ),
        ClassificationPattern(
            name="create_visualization",
            regex=r"\b(create|generate|make)\s+(a\s+)?(chart|graph|plot|visualization|dashboard)\b",
            semantic_intent="create chart graph visualization",
            task_type=TaskType.VISUALIZATION,
            complexity=TaskComplexity.MEDIUM,
            confidence=1.0,
            override=True,
            priority=85,
        ),
        # =================================================================
        # Research
        # =================================================================
        ClassificationPattern(
            name="fact_check",
            regex=r"\b(fact[-_]?check|verify|validate)\s+(the\s+)?(claim|statement|information|facts?)\b",
            semantic_intent="fact check verify validate",
            task_type=TaskType.FACT_CHECK,
            complexity=TaskComplexity.MEDIUM,
            confidence=1.0,
            override=True,
            priority=85,
        ),
        ClassificationPattern(
            name="literature_review",
            regex=r"\b(literature|systematic)\s+(review|survey|analysis)\b",
            semantic_intent="literature review systematic",
            task_type=TaskType.LITERATURE_REVIEW,
            complexity=TaskComplexity.ANALYSIS,
            confidence=1.0,
            override=True,
            priority=85,
        ),
        ClassificationPattern(
            name="competitive_analysis",
            regex=r"\b(competitive|competitor)\s+(analysis|comparison|review)\b",
            semantic_intent="competitive analysis comparison",
            task_type=TaskType.COMPETITIVE_ANALYSIS,
            complexity=TaskComplexity.ANALYSIS,
            confidence=1.0,
            override=True,
            priority=85,
        ),
        ClassificationPattern(
            name="trend_research",
            regex=r"\b(trend|market)\s+(analysis|research|report)\b",
            semantic_intent="trend market analysis research",
            task_type=TaskType.TREND_RESEARCH,
            complexity=TaskComplexity.ANALYSIS,
            confidence=1.0,
            override=True,
            priority=85,
        ),
        # =================================================================
        # Additional SWE-bench style patterns
        # =================================================================
        ClassificationPattern(
            name="memory_leak",
            regex=r"\b(memory\s+leak|OOM|out\s+of\s+memory|memory\s+usage\s+grows)\b",
            semantic_intent="memory leak OOM out of memory grows",
            task_type=TaskType.BUG_FIX,
            complexity=TaskComplexity.ACTION,
            confidence=1.0,
            override=True,
            priority=95,
        ),
        ClassificationPattern(
            name="bug_prefix",
            regex=r"(?m)^BUG:\s+",  # Multiline mode for ^ to match line start
            semantic_intent="BUG bug fix issue",
            task_type=TaskType.BUG_FIX,
            complexity=TaskComplexity.ACTION,
            confidence=1.0,
            override=True,
            priority=100,
        ),
        ClassificationPattern(
            name="null_pointer",
            regex=r"\b(null\s*pointer|NullPointerException|None\s+type|NoneType|AttributeError:\s+.*None)\b",
            semantic_intent="null pointer exception none type error",
            task_type=TaskType.BUG_FIX,
            complexity=TaskComplexity.ACTION,
            confidence=1.0,
            override=True,
            priority=95,
        ),
        ClassificationPattern(
            name="encoding_unicode_error",
            regex=r"\b(UnicodeDecodeError|UnicodeEncodeError|encoding\s+bug|encode|decode)\b",
            semantic_intent="unicode encoding decode error",
            task_type=TaskType.BUG_FIX,
            complexity=TaskComplexity.ACTION,
            confidence=0.95,
            override=False,
            priority=90,
        ),
        # =================================================================
        # Security / Vulnerability patterns
        # =================================================================
        ClassificationPattern(
            name="security_vulnerability",
            regex=r"\b(security\s+vulnerability|CVE-|critical\s+security|vulnerability\s+found)\b",
            semantic_intent="security vulnerability CVE critical",
            task_type=TaskType.SECURITY,
            complexity=TaskComplexity.ACTION,
            confidence=1.0,
            override=True,
            priority=100,
        ),
        ClassificationPattern(
            name="fix_vulnerability",
            regex=r"\b(fix|patch|resolve)\s+(the\s+)?(vulnerability|security\s+issue|injection)\b",
            semantic_intent="fix patch vulnerability security",
            task_type=TaskType.BUG_FIX,
            complexity=TaskComplexity.ACTION,
            confidence=1.0,
            override=True,
            priority=95,
        ),
        # =================================================================
        # Implementation / Feature patterns
        # =================================================================
        ClassificationPattern(
            name="add_endpoint",
            regex=r"\b(add|create|implement)\s+(a\s+)?(new\s+)?(API|REST|endpoint|route)\b",
            semantic_intent="add create API REST endpoint route",
            task_type=TaskType.IMPLEMENT,
            complexity=TaskComplexity.COMPLEX,
            confidence=0.95,
            override=False,
            priority=80,
        ),
        ClassificationPattern(
            name="add_validation",
            regex=r"\b(add|implement|create)\s+(input\s+)?validation\b",
            semantic_intent="add implement input validation",
            task_type=TaskType.IMPLEMENT,
            complexity=TaskComplexity.MEDIUM,
            confidence=0.9,
            override=False,
            priority=75,
        ),
        ClassificationPattern(
            name="extend_implementation",
            regex=r"\b(extend|enhance|expand)\s+(the\s+)?(current|existing)?\s*(implementation|fix|code)\b",
            semantic_intent="extend enhance expand implementation",
            task_type=TaskType.EDIT,
            complexity=TaskComplexity.MEDIUM,
            confidence=0.9,
            override=False,
            priority=75,
        ),
        # =================================================================
        # Architecture / Migration patterns
        # =================================================================
        ClassificationPattern(
            name="migrate_architecture",
            regex=r"\b(migrate|migration)\s+(from|to)\s+\w+\b",
            semantic_intent="migrate migration architecture",
            task_type=TaskType.ARCHITECTURE,
            complexity=TaskComplexity.COMPLEX,
            confidence=0.95,
            override=False,
            priority=85,
        ),
        ClassificationPattern(
            name="microservices_pattern",
            regex=r"\b(microservice|monolith|service\s+boundaries)\b",
            semantic_intent="microservices monolith architecture",
            task_type=TaskType.ARCHITECTURE,
            complexity=TaskComplexity.COMPLEX,
            confidence=0.9,
            override=False,
            priority=80,
        ),
        ClassificationPattern(
            name="create_migration_plan",
            regex=r"\b(create|make)\s+(a\s+)?(migration|rollout|phased)\s+plan\b",
            semantic_intent="create migration rollout plan",
            task_type=TaskType.PLAN,
            complexity=TaskComplexity.COMPLEX,
            confidence=0.95,
            override=False,
            priority=80,
        ),
        # =================================================================
        # Performance / Investigation patterns
        # =================================================================
        ClassificationPattern(
            name="performance_investigation",
            regex=r"\b(performance|profil|bottleneck|slow|degrad|latency)\b.*\b(investigat|analyz|identif|review)\b",
            semantic_intent="performance profile bottleneck investigate",
            task_type=TaskType.DEBUG,
            complexity=TaskComplexity.COMPLEX,
            confidence=0.9,
            override=False,
            priority=80,
        ),
        ClassificationPattern(
            name="investigate_issue",
            regex=r"\b(investigate|investigation)\s+(the\s+)?(issue|problem|error|root\s+cause)\b",
            semantic_intent="investigate issue problem root cause",
            task_type=TaskType.DEBUG,
            complexity=TaskComplexity.COMPLEX,
            confidence=0.9,
            override=False,
            priority=80,
        ),
        ClassificationPattern(
            name="investigation_needed",
            regex=r"\b(investigation\s+needed|needs?\s+investigation)\b",
            semantic_intent="investigation needed required",
            task_type=TaskType.DEBUG,
            complexity=TaskComplexity.COMPLEX,
            confidence=0.9,
            override=False,
            priority=80,
        ),
        ClassificationPattern(
            name="performance_degraded",
            regex=r"\b(response\s+times?\s+degraded|performance\s+degraded|slow\s+after)\b",
            semantic_intent="performance degraded slow response",
            task_type=TaskType.DEBUG,
            complexity=TaskComplexity.COMPLEX,
            confidence=0.9,
            override=False,
            priority=80,
        ),
        ClassificationPattern(
            name="clarification_question",
            regex=r"\b(i'm\s+confused|can\s+you\s+(show|explain)|how\s+does.*work|what\s+happens)\b",
            semantic_intent="confused clarification explain show",
            task_type=TaskType.ANALYZE,
            complexity=TaskComplexity.MEDIUM,
            confidence=0.8,
            override=False,
            priority=75,
        ),
        # =================================================================
        # Check / Analyze patterns (ambiguous)
        # =================================================================
        ClassificationPattern(
            name="check_config",
            regex=r"\b(check|verify|look\s+at)\s+(the\s+)?(config|settings|configuration)\b",
            semantic_intent="check verify config settings",
            task_type=TaskType.ANALYZE,
            complexity=TaskComplexity.SIMPLE,
            confidence=0.8,
            override=False,
            priority=70,
        ),
        ClassificationPattern(
            name="fix_any_issues",
            regex=r"\b(fix|address|correct)\s+(any|all)?\s*(issues?|problems?)\s+(you\s+)?find\b",
            semantic_intent="fix any issues problems find",
            task_type=TaskType.BUG_FIX,
            complexity=TaskComplexity.ACTION,
            confidence=0.9,
            override=False,
            priority=85,
        ),
        ClassificationPattern(
            name="look_at_and_fix",
            regex=r"\b(look\s+at|examine)\s+.*?\s+and\s+(fix|address|correct)\b",
            semantic_intent="look at examine fix address",
            task_type=TaskType.BUG_FIX,
            complexity=TaskComplexity.ACTION,
            confidence=0.9,
            override=False,
            priority=85,
        ),
        # =================================================================
        # Simple operations
        # =================================================================
        ClassificationPattern(
            name="list_files",
            regex=r"\b(list|show|display)\s+.*?(files?|directories?|folders?)\b",
            semantic_intent="list show files directories",
            task_type=TaskType.SEARCH,
            complexity=TaskComplexity.SIMPLE,
            confidence=1.0,
            override=False,
            priority=70,
        ),
        ClassificationPattern(
            name="git_status",
            regex=r"\bgit\s+(status|log|branch)\b",
            semantic_intent="git status log branch",
            task_type=TaskType.SEARCH,
            complexity=TaskComplexity.SIMPLE,
            confidence=1.0,
            override=False,
            priority=70,
        ),
        ClassificationPattern(
            name="pwd",
            regex=r"\bpwd\b|\bcurrent\s+(directory|dir|folder)\b",
            semantic_intent="current directory pwd",
            task_type=TaskType.SEARCH,
            complexity=TaskComplexity.SIMPLE,
            confidence=1.0,
            override=False,
            priority=70,
        ),
    ]

    # Build dictionary keyed by name
    PATTERNS = {p.name: p for p in patterns_list}


def get_patterns_by_complexity(
    complexity: TaskComplexity,
) -> List[ClassificationPattern]:
    """Get all patterns that map to a complexity level.

    Args:
        complexity: The complexity level to filter by

    Returns:
        List of patterns with matching complexity
    """
    return [p for p in PATTERNS.values() if p.complexity == complexity]


def get_patterns_by_task_type(task_type: TaskType) -> List[ClassificationPattern]:
    """Get all patterns that map to a task type.

    Args:
        task_type: The task type to filter by

    Returns:
        List of patterns with matching task type
    """
    return [p for p in PATTERNS.values() if p.task_type == task_type]


def get_patterns_sorted_by_priority() -> List[ClassificationPattern]:
    """Get all patterns sorted by priority (highest first).

    Returns:
        List of patterns sorted by priority descending
    """
    return sorted(PATTERNS.values(), key=lambda p: p.priority, reverse=True)


def match_first_pattern(text: str) -> Optional[ClassificationPattern]:
    """Find the first matching pattern in priority order.

    Args:
        text: Text to match against patterns

    Returns:
        First matching ClassificationPattern or None
    """
    for pattern in get_patterns_sorted_by_priority():
        if pattern.compiled_pattern.search(text):
            return pattern
    return None


def match_all_patterns(text: str) -> List[ClassificationPattern]:
    """Find all matching patterns.

    Args:
        text: Text to match against patterns

    Returns:
        List of all matching patterns sorted by priority
    """
    matches = []
    for pattern in PATTERNS.values():
        if pattern.compiled_pattern.search(text):
            matches.append(pattern)
    return sorted(matches, key=lambda p: p.priority, reverse=True)


# Initialize patterns on module load
_register_patterns()
