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

"""BenchmarkVertical - Domain-specific vertical for AI coding benchmarks.

Follows the same pattern as CodingAssistant, ResearchAssistant, etc.
Provides tools, stages, and workflows optimized for benchmark evaluation.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from victor.core.verticals.base import StageDefinition, VerticalBase, VerticalConfig
from victor.tools.tool_names import ToolNames

if TYPE_CHECKING:
    from victor.core.verticals.protocols import ModeConfigProviderProtocol
    from victor.core.vertical_types import TieredToolConfig


class BenchmarkVertical(VerticalBase):
    """Vertical for AI coding benchmark evaluation.

    This vertical configures agents for evaluating coding tasks from benchmarks
    like SWE-bench, HumanEval, and MBPP. It provides:

    - Optimized tool selection for code analysis and modification
    - Benchmark-specific stages (UNDERSTANDING → ANALYSIS → IMPLEMENTATION → VERIFICATION)
    - Workflows for common benchmark patterns
    - Metrics collection through framework observability
    """

    name = "benchmark"
    description = "AI coding benchmark evaluation and performance testing"
    version = "1.0.0"

    # Benchmark-specific stages
    STAGE_UNDERSTANDING = "UNDERSTANDING"
    STAGE_ANALYSIS = "ANALYSIS"
    STAGE_IMPLEMENTATION = "IMPLEMENTATION"
    STAGE_VERIFICATION = "VERIFICATION"

    @classmethod
    def get_tools(cls) -> List[str]:
        """Tools optimized for benchmark task execution.

        Returns a curated set of tools for:
        - Code reading and understanding
        - Code search and navigation
        - Code modification
        - Test execution and verification
        """
        return [
            # Core reading/navigation (always needed)
            ToolNames.READ,
            ToolNames.LS,
            ToolNames.GREP,
            # Code-specific tools
            ToolNames.CODE_SEARCH,  # Semantic code search
            ToolNames.SYMBOL,  # Find symbol definitions
            ToolNames.REFS,  # Find references
            # Modification tools
            ToolNames.WRITE,
            ToolNames.EDIT,
            # Execution/verification
            ToolNames.SHELL,
            ToolNames.TEST,
            # Git for patch generation (unified git tool)
            ToolNames.GIT,
            ToolNames.DIFF,  # Create patches
        ]

    @classmethod
    def get_system_prompt(cls) -> str:
        """System prompt optimized for benchmark task execution."""
        return """You are an expert software engineer solving coding tasks from benchmarks.

Your approach:
1. UNDERSTAND the problem completely before making changes
2. ANALYZE the codebase to find relevant files and understand context
3. IMPLEMENT the solution with minimal, focused changes
4. VERIFY your changes work correctly

Guidelines:
- Read and understand the issue/task description carefully
- Search the codebase to find all relevant files
- Make the smallest change that solves the problem
- Follow the existing code style and patterns
- Generate clean, applicable patches when requested
- Verify your solution by running tests if available

You are being evaluated on:
- Correctness: Does your solution pass the tests?
- Efficiency: Did you find the right files quickly?
- Quality: Is your code clean and maintainable?
"""

    @classmethod
    def get_skills(cls) -> list:
        """Skills for benchmark task execution."""
        from victor_contracts.skills import SkillDefinition

        return [
            SkillDefinition(
                name="swe_bench_solve",
                description="Solve a SWE-bench coding task: read issue, find files, implement fix, verify with tests",
                category="benchmark",
                prompt_fragment=(
                    "SKILL: swe_bench_solve\n"
                    "1. Read the issue/task description to understand the requirement\n"
                    "2. Search the codebase for relevant files using grep and code_search\n"
                    "3. Read the relevant source files and test files\n"
                    "4. Implement the minimal fix that addresses the issue\n"
                    "5. Run the test suite to verify the fix passes\n"
                    "6. Generate a clean diff/patch of the changes"
                ),
                required_tools=["read", "grep", "edit", "shell"],
                optional_tools=["code_search", "symbol", "refs", "git", "test"],
                tags=frozenset({"benchmark", "swe-bench", "bug-fix"}),
                max_tool_calls=40,
            ),
            SkillDefinition(
                name="benchmark_analyze",
                description="Analyze benchmark results: parse scores, compare runs, identify patterns",
                category="benchmark",
                prompt_fragment=(
                    "SKILL: benchmark_analyze\n"
                    "1. Read the benchmark output/results files\n"
                    "2. Parse pass/fail counts and identify failing cases\n"
                    "3. Compare against previous runs if available\n"
                    "4. Identify patterns in failures (common modules, error types)\n"
                    "5. Summarize findings with actionable next steps"
                ),
                required_tools=["read", "grep", "shell"],
                optional_tools=["ls"],
                tags=frozenset({"benchmark", "analysis", "metrics"}),
                max_tool_calls=15,
            ),
        ]

    @classmethod
    def get_stages(cls) -> Dict[str, StageDefinition]:
        """Benchmark-specific conversation stages.

        Stages map to benchmark task phases and influence tool selection.
        """
        return {
            cls.STAGE_UNDERSTANDING: StageDefinition(
                name=cls.STAGE_UNDERSTANDING,
                description="Understanding the task/issue",
                tools={ToolNames.READ, ToolNames.GREP, ToolNames.LS},
                keywords=[
                    "understand",
                    "issue",
                    "problem",
                    "task",
                    "bug",
                    "error",
                    "failing",
                ],
                next_stages={cls.STAGE_ANALYSIS},
            ),
            cls.STAGE_ANALYSIS: StageDefinition(
                name=cls.STAGE_ANALYSIS,
                description="Analyzing codebase for relevant files",
                tools={
                    ToolNames.CODE_SEARCH,
                    ToolNames.SYMBOL,
                    ToolNames.REFS,
                    ToolNames.GREP,
                },
                keywords=[
                    "find",
                    "search",
                    "locate",
                    "where",
                    "definition",
                    "references",
                    "usage",
                ],
                next_stages={cls.STAGE_IMPLEMENTATION},
            ),
            cls.STAGE_IMPLEMENTATION: StageDefinition(
                name=cls.STAGE_IMPLEMENTATION,
                description="Implementing the solution",
                tools={ToolNames.EDIT, ToolNames.WRITE, ToolNames.READ},
                keywords=[
                    "fix",
                    "change",
                    "modify",
                    "update",
                    "add",
                    "remove",
                    "implement",
                    "patch",
                ],
                next_stages={cls.STAGE_VERIFICATION},
            ),
            cls.STAGE_VERIFICATION: StageDefinition(
                name=cls.STAGE_VERIFICATION,
                description="Verifying the solution",
                tools={ToolNames.TEST, ToolNames.SHELL, ToolNames.GIT},
                keywords=[
                    "test",
                    "verify",
                    "check",
                    "run",
                    "validate",
                    "confirm",
                ],
                next_stages=set(),  # Terminal stage
            ),
        }

    @classmethod
    def get_provider_hints(cls) -> Dict[str, Any]:
        """Hints for LLM provider configuration."""
        return {
            "temperature": 0.2,  # Lower temperature for deterministic coding
            "max_tokens": 4096,  # Sufficient for code generation
            "stop_sequences": [],  # Let model complete naturally
        }

    @classmethod
    def get_evaluation_criteria(cls) -> List[str]:
        """Evaluation criteria for benchmark tasks."""
        return [
            "Correctness: Solution passes all provided tests",
            "Efficiency: Minimal tool calls to find solution",
            "Quality: Clean, maintainable code following project style",
            "Completeness: All aspects of the issue are addressed",
        ]

    @classmethod
    def get_tiered_tool_config(cls) -> Optional["TieredToolConfig"]:
        """Tiered tool configuration for benchmark evaluation.

        Returns tool tiers optimized for benchmark task execution:
        - Mandatory: Always available (read, ls, grep)
        - Core: Vertical-specific essentials (code_search, edit)
        - Semantic: Selected based on task similarity

        Returns:
            TieredToolConfig with proper tool tiers for benchmark evaluation
        """
        from victor.core.vertical_types import TieredToolConfig

        return TieredToolConfig(
            mandatory={
                ToolNames.READ,
                ToolNames.LS,
                ToolNames.GREP,
            },
            vertical_core={
                ToolNames.CODE_SEARCH,
                ToolNames.EDIT,
                ToolNames.WRITE,
                ToolNames.SHELL,
            },
            semantic_pool={
                ToolNames.SYMBOL,
                ToolNames.REFS,
                ToolNames.TEST,
                ToolNames.GIT,
                ToolNames.DIFF,
            },
            readonly_only_for_analysis=False,
        )

    @classmethod
    def customize_config(cls, config: VerticalConfig) -> VerticalConfig:
        """Apply benchmark-specific configuration tweaks."""
        # Add benchmark-specific metadata
        config.metadata["tool_budget"] = 30  # Recommended for complex benchmark tasks
        config.metadata["thinking"] = True  # Enable extended thinking for complex reasoning

        return config
